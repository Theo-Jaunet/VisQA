# coding=utf-8
# Copyright 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn import KLDivLoss
from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model import GQAModel
from tasks.gqa_data import GQADataset, GQATorchDataset, GQAEvaluator
# export PYTHONPATH=$PYTHONPATH:./src
from tensorboardX import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import string
import pickle
import time

from plot_util import draw_histogram

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)

# ? DEBUG CORENTIN ****************************************************************
def draw_bboxes(bboxes, ax, color='rand', alpha=None, label=None):
    used_colors = []
    for i, box in enumerate(bboxes):
        if sum(box) != 0:
            if color == 'none':
                c = np.array([0.5, 0.5, 0.5])
            elif color == 'rand':
                c = np.random.rand(3,)
            else:
                c = color[i]
            used_colors.append(c)
            x = int(box[0])
            y = int(box[1])
            w = int(box[2])
            h = int(box[3])
            transparency = None if alpha is None else alpha[i]
            rect = patches.Rectangle((x, y), w, h, linewidth=2.5,
                                    edgecolor=c, facecolor='none', alpha=transparency)
            if label is not None:
                plt.text(x, y, label[i])
            ax.add_patch(rect)
# ? DEBUG CORENTIN ****************************************************************

class GQA:
    def __init__(self):
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            self.new_ans_label = load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        # self.KL_loss = nn.KLDivLoss(reduction='none')
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

        # Tensorboard
        self.boards_dir = os.path.join('boards', self.output)
        if not os.path.exists(self.boards_dir):
            os.makedirs(self.boards_dir)
        self.writerTbrd = SummaryWriter(self.boards_dir)

        # get Glove projection for all answers
        if args.answer_loss == 'glove':
            path_glove = './data/GloVe/GloVeDict.pkl'
            with open(path_glove, 'rb') as f:
                glove_dic = pickle.load(f)
            glove_dim = glove_dic['the'].shape[-1]
            print("Loading Glove%d answer's vector" % glove_dim)
            self.labelans2glove = []
            self.valid_ans_embed = [1] * len(self.train_tuple.dataset.label2ans)
            for label, ans in enumerate(self.train_tuple.dataset.label2ans):
                ans = ans.split(' ')
                glove_ans = []
                for w in ans:
                    #print(w)
                    try:
                        glove_ans.append(glove_dic[w])
                    except KeyError:
                        #print('Full ans: %s' % ans)
                        #input(' ')
                        self.valid_ans_embed[label] = 0
                        glove_ans.append(np.zeros(glove_dim))
                #print(glove_ans)
                glove_ans = torch.tensor(glove_ans).mean(-2)    
                self.labelans2glove.append(torch.tensor(glove_ans))
            #print(self.labelans2glove)
            print('Ratio of valid ans embedding: %f' % (float(sum(self.valid_ans_embed))/len(self.valid_ans_embed)))
            self.labelans2glove = torch.stack(self.labelans2glove).float().cuda()
            self.cosineSim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

# ? DEBUG CORENTIN ****************************************************************
    def check_pointer_manually(self, train_tuple):
        IMAGE_PATH = 'data/gqa/images'
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        for i, (ques_id, feats, boxes, sent, target, iou_question, iou_answer)\
                 in iter_wrapper(enumerate(loader)):

            for batch_index in range(len(ques_id)):
                datum = dset.id2datum[ques_id[batch_index]]
                # Load image
                im_id = datum['image_id']
                im_path = os.path.join(IMAGE_PATH, '%s.jpg' % im_id)
                image_pil = Image.open(im_path)
                im = np.array(image_pil, dtype=np.uint8)
                height = image_pil.height
                width = image_pil.width
                # Load annotations
                question = datum['sent']
                question_pointer = datum['pointer']['question']
                answer = list(datum['label'])[0]
                answer_pointer = datum['pointer']['answer']
                # * * Display pointer and bboxes
                # Create plot
                fig = plt.figure()
                plt.suptitle(question)
                plt.title(answer)
                ax = fig.add_subplot(1, 1, 1)
                # draw image
                ax.imshow(im)
                # draw detected boxes
                def iou_preprocess(iou):
                    TRESHOLD = 0.1
                    TOPK = 5
                    # norm_iou = np.exp(iou) / np.sum(np.exp(iou), axis=0)  #iou / (iou.sum() + 1e-9)
                    # f_iou = norm_iou * (iou.sum() >= TRESHOLD)
                    sorted_idx = np.argsort(iou)[::-1]
                    iou_topk = iou
                    iou_topk[sorted_idx[TOPK:]] = -1e9
                    f_iou = np.exp(iou_topk) / np.sum(np.exp(iou_topk), axis=0)  #iou / (iou.sum() + 1e-9)
                    f_iou = f_iou * (iou_topk.clip(min=0).sum() >= TRESHOLD)
                    return f_iou

                detected_bboxe = boxes[batch_index] * torch.tensor([width, height, width, height]).float()
                total_iou_per_object = np.zeros((boxes.size(1)))
                for _, pointer in question_pointer.items():
                    iou = np.array(pointer['iou'])
                    f_iou = iou_preprocess(iou)                    
                    total_iou_per_object += f_iou
                for _, pointer in answer_pointer.items():
                    iou = np.array(pointer['iou'])
                    f_iou = iou_preprocess(iou)
                    total_iou_per_object += f_iou
                intensity = total_iou_per_object.clip(min=0, max=1)
                c = [np.array([0, 0, 1]) for j in range(boxes.size(1))]
                draw_bboxes(detected_bboxe.numpy(),
                            ax, color=c, alpha=intensity)
                # draw pointer boxes
                for word_id, pointer in question_pointer.items():
                    bboxe = pointer['boxe']
                    bboxe = [bboxe[0]*width,
                             bboxe[1]*height,
                             bboxe[2]*width,
                             bboxe[3]*height]
                    c = [np.array([1,0,0])]
                    draw_bboxes([bboxe], ax, color=c, label=['q_%s' % word_id])
                for word_id, pointer in answer_pointer.items():
                    bboxe = pointer['boxe']
                    bboxe = [bboxe[0]*width,
                             bboxe[1]*height,
                             bboxe[2]*width,
                             bboxe[3]*height]
                    c = [np.array([0,1,0])]
                    draw_bboxes([bboxe], ax, color=c, label=['a_%s' % word_id])
                plt.savefig('check_pointer_%s_sftmx.jpg' % im_id)
                plt.close()
                # * * Retrieve statistics


                # input('Press ENTER for next image')
    def pointer_stats(self, train_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        # Stat watchers
        pointer_per_question = []
        nb_words_per_question = []
        pointed_words = {}
        max_iou = []
        top5_cumiou = []
        top10_cumiou = []
        total_cumiou = []

        start = time.time()
        for i, (ques_id, feats, boxes, sent, target, iou_question, iou_answer)\
                 in iter_wrapper(enumerate(loader)):

            for batch_index in range(len(ques_id)):
                datum = dset.id2datum[ques_id[batch_index]]
            
                # Load annotations
                question = datum['sent']
                question_pointer = datum['pointer']['question']
                answer = list(datum['label'])[0]
                answer_pointer = datum['pointer']['answer']
                # parse question and answer
                parsed_question = question.translate(str.maketrans('', '', string.punctuation)).split(' ')
                parsed_answer = answer.translate(str.maketrans('', '', string.punctuation)).split(' ')
                # Stats words
                pointer_per_question.append(len(question_pointer) + len(answer_pointer))
                nb_words_per_question.append(len(parsed_question) + len(parsed_answer))
                def add2dic(pointer, parsed_sent):
                    for w_idx in pointer:
                        if ':' in w_idx:
                            indexes = w_idx.split(':')
                            for j in range(int(indexes[0]), int(indexes[1])):
                                word = parsed_sent[j]
                                if word in pointed_words:
                                    pointed_words[word] += 1
                                else:
                                    pointed_words[word] = 1
                        else:    
                            word = parsed_sent[int(w_idx)]
                            if word in pointed_words:
                                pointed_words[word] += 1
                            else:
                                pointed_words[word] = 1
                add2dic(question_pointer, parsed_question)
                add2dic(answer_pointer, parsed_answer)
                # Stats IoU
                    # max
                max_iou_question = iou_question.max(-1)[0]
                max_iou += max_iou_question[max_iou_question > 0].flatten().tolist()
                max_iou_answer = iou_answer.max(-1)[0]
                max_iou += max_iou_answer[max_iou_answer > 0].flatten().tolist()
                    # top5
                top5_cumiou_question = iou_question.topk(5)[0].sum(-1)
                top5_cumiou_answer = iou_answer.topk(5)[0].sum(-1)
                top5_cumiou += top5_cumiou_question[top5_cumiou_question > 0].flatten().tolist()
                top5_cumiou += top5_cumiou_answer[top5_cumiou_answer > 0].flatten().tolist()
                    # top10
                top10_cumiou_question = iou_question.topk(10)[0].sum(-1)
                top10_cumiou_answer = iou_answer.topk(10)[0].sum(-1)
                top10_cumiou += top10_cumiou_question[top10_cumiou_question > 0].flatten().tolist()
                top10_cumiou += top10_cumiou_answer[top10_cumiou_answer > 0].flatten().tolist()
                    # total
                total_cumiou_question = iou_question.sum(-1) 
                total_cumiou_answer = iou_answer.sum(-1)
                total_cumiou += total_cumiou_question[total_cumiou_question > 0].flatten().tolist()
                total_cumiou += total_cumiou_answer[total_cumiou_answer > 0].flatten().tolist()
        
        elapsed_time = time.time() - start
        print("Time: %.1fmin" % (elapsed_time/60))
        # Save into pickle dic
        dic = {'pointer_per_question': pointer_per_question,
               'nb_words_per_question': nb_words_per_question,
               'pointed_words': pointed_words,
               'max_iou': max_iou,
               'top5_cumiou': top5_cumiou,
               'top10_cumiou': top10_cumiou,
               'total_cumiou': total_cumiou}
        with open('stats_pointer.pickle', 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

# ? DEBUG CORENTIN ****************************************************************


# ? Manual evaluation (for matching) **********************************************
    def eval_matching_manual(self, eval_tuple):
        IMAGE_PATH = 'data/gqa/images'
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, _, iou_question, iou_answer = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                iou_question, iou_answer = iou_question.cuda(), iou_answer.cuda()
                logit, iou_target, iou_pred = self.model(feats, boxes, sent, iou_question, iou_answer)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            for batch_index in range(len(ques_id)):
                # Retrieve info + prediction
                qid = ques_id[batch_index]
                datum = dset.id2datum[qid]
                question = datum['sent']
                answer_pred = quesid2ans[qid]
                # Load image
                im_id = datum['image_id']
                im_path = os.path.join(IMAGE_PATH, '%s.jpg' % im_id)
                image_pil = Image.open(im_path)
                im = np.array(image_pil, dtype=np.uint8)
                height = image_pil.height
                width = image_pil.width
                detected_bboxe = boxes[batch_index].cpu() * torch.tensor([width, height, width, height]).float()
                # Display iou prediction
                for w in range(iou_pred[batch_index].size(0)):
                    fig = plt.figure()
                    plt.suptitle('Q:%s __ Idx:%d' % (question, w))
                    plt.title(answer_pred)
                    # all predicted bboxes
                    ax = fig.add_subplot(2, 1, 1)
                    ax.imshow(im)
                    c = [np.array([0, 0, 1]) for j in range(boxes.size(1))]
                    draw_bboxes(detected_bboxe.numpy(), ax, color=c)
                    # matched bboxes
                    iou = iou_pred[batch_index, w]
                    iou_norm = iou / (iou.sum() + 1e-9)
                    ax = fig.add_subplot(2, 1, 2)
                    ax.imshow(im)
                    c = [np.array([0, 0, 1]) for j in range(boxes.size(1))]
                    draw_bboxes(detected_bboxe.numpy(), ax, color=c, alpha=iou_norm)
                    plt.savefig('t0.4_pred_pointer_%s_%d.jpg' % (im_id, w))
                    plt.close()
                input('Press ENTER for next image')

            

# ? Manual evaluation (for matching) **********************************************

    def gqa_analysis(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        for i, (ques_id, feats, boxes, sent, target, iou_question, iou_answer, sem_question_words, sem_answer_words, bboxes_words,)\
             in iter_wrapper(enumerate(loader)):
            
            with torch.no_grad():
                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                iou_question, iou_answer = iou_question.cuda(), iou_answer.cuda()
                sem_question_words, sem_answer_words, bboxes_words = sem_question_words.cuda(), sem_answer_words.cuda(), bboxes_words.cuda()
                logit, iou_target, iou_score, lang_feat, vis_feat, tkn_sent = self.model(
                    feats, boxes, sent, iou_question, iou_answer,
                    sem_question_words, sem_answer_words, bboxes_words,
                    verbose=True)
                for i in range(lang_feat.size(0)):
                    len_sent = len(tkn_sent[i])
                    self.writerTbrd.add_embedding(
                        lang_feat[i, :len_sent],
                        metadata=tkn_sent[i]
                    )
                    pass

    def extract_maps(self, eval_tuple):
        self.model.eval()
        #self.model.cpu()
        dset, loader, evaluator = eval_tuple
        timer = time.time()
        att_maps = None
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, target, iou_question, iou_answer, sem_question_words, sem_answer_words, bboxes_words = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                iou_question, iou_answer = iou_question.cuda(), iou_answer.cuda()
                sem_question_words, sem_answer_words, bboxes_words = sem_question_words.cuda(), sem_answer_words.cuda(), bboxes_words.cuda()
                logit, iou_target, iou_score, activations = self.model(feats, boxes, sent, iou_question, iou_answer,
                                                         sem_question_words, sem_answer_words, bboxes_words)
                score, label = logit.max(1)

            # init according to model's architecture
            activations['cross'] = [item for sublist in activations['cross'] for item in sublist]  # flatten
            if att_maps is None:
                print('map shape', activations['lang'][0].shape)
                n_head = activations['lang'][0].shape[1]
                att_maps = {'lang':[torch.zeros((n_head)) for t in range(len(activations['lang']))],
                            'vis':[torch.zeros((n_head))for t in range(len(activations['vis']))],
                            'cross':[torch.zeros((n_head))for t in range(len(activations['cross']))]}
                map_names = []
            for maptype in ['lang', 'vis', 'cross']:
                for idx, maps in enumerate(activations[maptype]):
                    if maptype == 'cross':
                        sub_id = idx % 4
                        sub_maptype = ['xvl', 'xlv', 'xl', 'xv']
                        name = '%s%d'%(sub_maptype[sub_id], idx)
                    else:
                        name = '%s%d'%(maptype[0], idx)
                    map_names.append(name)
                    d_b, d_h, d_1, d_2 = maps.shape  # [batch, head, d1, d2]
                    head_max = torch.max(maps.view(d_b, d_h, d_1 * d_2), dim=-1).values  # [batch X heads]
                    head_max = head_max.sum(0) # sum over batch: [heads]
                    att_maps[maptype][idx] += head_max.cpu().data
        
        all_max_maps = torch.cat([torch.stack(att_maps['lang']), torch.stack(att_maps['vis']), torch.stack(att_maps['cross'])]).numpy() # [layer, heads]
        all_max_maps = all_max_maps /  len(dset)
        print("MAX ATT divided", all_max_maps)
        # Draw histogram
        FIG_PATH = self.output 
        draw_histogram(all_max_maps, path=FIG_PATH, labels=map_names)
        input('_')
    
        print('Processes set in %ds'%(time.time()-timer))

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        optim_steps = 0
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, iou_question, iou_answer, sem_question_words, sem_answer_words, bboxes_words,)\
                 in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                # DEBUG: print pointer (set batch size to 1)
                # print(dset.id2datum[ques_id[0]]['sent'])
                # print(dset.id2datum[ques_id[0]]['label'])
                # q_pointer = dset.id2datum[ques_id[0]]['pointer']['question']
                # for w_index in q_pointer:
                #     print(w_index)

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                iou_question, iou_answer = iou_question.cuda(), iou_answer.cuda()
                sem_question_words, sem_answer_words, bboxes_words = sem_question_words.cuda(), sem_answer_words.cuda(), bboxes_words.cuda()
                logit, iou_target, iou_score = self.model(feats, boxes, sent, iou_question, iou_answer,
                                                         sem_question_words, sem_answer_words, bboxes_words)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)
                #print('CE', loss.item())

                if args.answer_loss == 'glove':
                    gold_glove = (self.labelans2glove.unsqueeze(0) * target.unsqueeze(-1)).sum(1)
                    #gold_ans = self.train_tuple.dataset.label2ans[target.argmax(dim=1)[0]]
                    #print('gold:', gold_ans)
                    pred_glove = (self.labelans2glove.unsqueeze(0) * torch.softmax(logit, dim=1).unsqueeze(-1)).sum(1)
                    #pred_ans = self.train_tuple.dataset.label2ans[logit.argmax(dim=1)[0]]
                    #print('pred:', pred_ans)
                    sim_answer =  self.cosineSim(gold_glove, pred_glove).mean()
                    loss += - 10 * sim_answer
                    #print('Similarity', sim_answer)
                    #input(' ')

                if optim_steps % 1000 == 0:
                    self.writerTbrd.add_scalar('vqa_loss_train', loss.item(), optim_steps)

                # task_pointer = 'KLDiv'
                ALPHA = args.alpha_pointer
                def iou_preprocess(iou, obj_conf=None):
                    TRESHOLD = 0.1
                    TOPK = 3
                    # norm_iou = np.exp(iou) / np.sum(np.exp(iou), axis=0)  #iou / (iou.sum() + 1e-9)
                    # f_iou = norm_iou * (iou.sum() >= TRESHOLD)
                    sorted_values = torch.sort(iou, descending=True, dim=-1)[0]
                    t_top = sorted_values[:, :, TOPK-1]
                    iou_topk = iou.masked_fill(iou < t_top.unsqueeze(-1), -1e9)
                    f_iou = torch.softmax(iou_topk, dim=-1)
                    treshold_mask = (iou_topk.clamp(min=.0).sum(-1) >= TRESHOLD).float()
                    if args.task_pointer == 'KLDiv':
                        return f_iou, treshold_mask
                    elif args.task_pointer == 'Triplet':
                        # Remove top10 most similar objects
                        t_bot = sorted_values[:, :, 10]
                        iou_botk = (iou < t_bot.unsqueeze(-1)).float()
                        # Take topk most confident objects 
                        conf_top = torch.sort(obj_conf.unsqueeze(1) * iou_botk, descending=True, dim=-1)[0][:, :, TOPK-1]
                        conf_mask = obj_conf.unsqueeze(1).expand(-1, iou.size(1), -1) >= conf_top.unsqueeze(-1)
                        neg_score = iou_botk * conf_mask.float()                    
                        return f_iou, treshold_mask, neg_score

                if args.task_pointer == 'KLDiv':
                    iou_target_preprocess, treshold_mask = iou_preprocess(iou_target)
                    loss_pointer_fct = KLDivLoss(reduction='none')
                    iou_pred = torch.log_softmax(iou_score, dim=-1)
                    matching_loss = loss_pointer_fct(input=iou_pred, target=iou_target_preprocess)
                    matching_loss = ALPHA * (matching_loss.sum(-1) * treshold_mask).sum() / ((treshold_mask).sum() + 1e-9)
                    if optim_steps % 1000 == 0:
                        self.writerTbrd.add_scalar('pointer_loss_train', matching_loss.item(), optim_steps)
                    loss += matching_loss

                # ? by Corentin: Matching loss
                # def iou_preprocess(iou):
                #     TRESHOLD = 0.1
                #     TOPK = 1
                #     # norm_iou = np.exp(iou) / np.sum(np.exp(iou), axis=0)  #iou / (iou.sum() + 1e-9)
                #     # f_iou = norm_iou * (iou.sum() >= TRESHOLD)
                #     t = torch.sort(iou, descending=True, dim=-1)[0][:, :, TOPK-1]
                #     iou_topk = iou.masked_fill(iou < t.unsqueeze(-1), -1e9)
                #     f_iou = torch.softmax(iou_topk, dim=-1)
                #     treshold_mask = (iou_topk.clamp(min=.0).sum(-1) >= TRESHOLD).float()
                #     return f_iou, treshold_mask
                # # discard iou_target when total iou is under treshold
                # # it includes unsupervised datum
                # iou_target_preprocess, treshold_mask = iou_preprocess(iou_target)
                # iou_pred = torch.log_softmax(iou_pred, dim=-1)
                # # KL loss
                # matching_loss = []
                # matching_loss = self.KL_loss(input=iou_pred, target=iou_target_preprocess)
                # matching_loss = (matching_loss.sum(-1) * treshold_mask).sum() / treshold_mask.sum()
                # if optim_steps % 1000 == 0:
                #     self.writerTbrd.add_scalar('pointer_loss_train', matching_loss.item(), optim_steps)
                # ALPHA = 5.0
                # loss += ALPHA * matching_loss
                # ? **************************

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()
                optim_steps += 1

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

                # if self.valid_tuple is not None and optim_steps % 1152 == 0:  # Do Validation
                #     valid_score = self.evaluate(eval_tuple)
                #     fastepoch = int(optim_steps / 1152)
                #     print("fastEpoch %d: Valid %0.2f\n" % (fastepoch, valid_score * 100.,))
    
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                self.writerTbrd.add_scalar('vqa_acc_valid', valid_score, epoch)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None, iou=False):
        self.model.eval()
        #self.model.cpu()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        quesid2iou = {}
        timer = time.time()
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, target, iou_question, iou_answer, sem_question_words, sem_answer_words, bboxes_words = datum_tuple
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                iou_question, iou_answer = iou_question.cuda(), iou_answer.cuda()
                sem_question_words, sem_answer_words, bboxes_words = sem_question_words.cuda(), sem_answer_words.cuda(), bboxes_words.cuda()
                logit, iou_target, iou_score = self.model(feats, boxes, sent, iou_question, iou_answer,
                                                         sem_question_words, sem_answer_words, bboxes_words)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
                quesid2iou[qid] = None #iou_pred
        print('Processes set in %ds'%(time.time()-timer))
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        if iou is True:
            return quesid2ans, quesid2iou
        else:
            return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target, iou_question, iou_answer, sem_question_words, sem_answer_words, bboxes_words,)\
            in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)

    def finetune(self, train_tuple, eval_tuple):
        # log
        output_1 = os.path.join(self.output, 'finetune_1')
        os.makedirs(output_1, exist_ok=True)
        output_2 = os.path.join(self.output, 'finetune_2')
        os.makedirs(output_2, exist_ok=True)

        # Tensorboard
        boards_dir_1 = os.path.join(self.boards_dir, 'finetune_1')
        if not os.path.exists(boards_dir_1):
            os.makedirs(boards_dir_1)
        boards_dir_2 = os.path.join(self.boards_dir, 'finetune_2')
        if not os.path.exists(boards_dir_2):
            os.makedirs(boards_dir_2)

        # Params
        lr_1 = args.lr
        lr_2 = args.lr / 10
        epochs_1 = 4 #int(args.epochs / 3)
        epochs_2 =args.epochs - epochs_1

        # Step 0: evaluate pretraining
        if self.valid_tuple is not None:  # Do Validation
           valid_score = self.evaluate(eval_tuple)
           print("Before finetune: Valid %0.2f\n" % (valid_score * 100.))

        # Step 0.1: finetune new ans only
        # new_ans_params = []
        # for name, p in self.model.named_parameters():
        #     if "logit_fc.3" in name:
        #         for idx in range(p.size(0)):
        #             if idx in self.new_ans_label:
        #                 new_ans_params.append({'params': p[idx]})

        # args.epochs = epochs_0
        # from lxrt.optimization import BertAdam
        # self.optim = BertAdam(new_ans_params,
        #                       lr=lr_1,
        #                       warmup=0.0,
        #                       t_total=-1)
        # print('### Start finetuning new ans...')
        # self.train(train_tuple, eval_tuple)

        # First step, only updates answer head

        #self.optim = torch.optim.Adamax(list(self.model.parameters()), lr_1)
        #self.optim = torch.optim.SGD(list(self.model.parameters()), lr_1)
        args.epochs = epochs_1
        batch_per_epoch = len(self.train_tuple.loader)
        t_total = int(batch_per_epoch * epochs_1)
        print("Total Iters: %d" % t_total)
        from lxrt.optimization import BertAdam
        self.optim = BertAdam(list(self.model.parameters()),
                              lr=lr_1,
                              warmup=0.0,#!0.034
                              t_total=-1)
        # loaded_optim = torch.load("%s_LXRT.pth" % args.load_lxmert_qa)['optimizer']
        # self.optim.load_state_dict(loaded_optim)
        # for group in loaded_optim.param_groups:
        #     for p in group['params']:
        #         if p in loaded_optim['state']:
        #             self.optim.state[p] = loaded_optim.state[p]

        self.writerTbrd = SummaryWriter(boards_dir_1)
        self.output = output_1

        for name, p in self.model.named_parameters():
            if "logit_fc" in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

        print('### Start finetuning step 1...')
        self.train(train_tuple, eval_tuple)

        # Second step, finetune all
        for name, p in self.model.named_parameters():
            p.requires_grad = True

        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * epochs_2)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=lr_2,
                                  warmup=0.1,
                                  t_total=t_total,
                                  lr_min=1e-7)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), lr_2)
        args.epochs = epochs_2
        self.writerTbrd = SummaryWriter(boards_dir_2)
        self.output = output_2

        print('### Start finetuning step 2...')
        self.train(train_tuple, eval_tuple)

if __name__ == "__main__":
    # Build Class
    gqa = GQA()

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    if args.test_pointer is not None:
        args.fast = args.tiny = False
        eval_tuple = get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False)
        gqa.eval_matching_manual(eval_tuple)

    elif args.test is not None: # Test or Train
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
        if 'valid' in args.test:
            result = gqa.evaluate(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'valid_predict.json')
            )
            print(result)
    else:

        # gqa.check_pointer_manually(gqa.train_tuple)
        # gqa.pointer_stats(gqa.train_tuple)

        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")

        gqa.train(gqa.train_tuple, gqa.valid_tuple)
        #gqa.finetune(gqa.train_tuple, gqa.valid_tuple)
        # gqa.gqa_analysis(gqa.train_tuple, gqa.valid_tuple)

 
