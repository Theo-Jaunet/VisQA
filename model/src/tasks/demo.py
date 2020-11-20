"""
by Corentin

VQA demo:

Take one image (pre-processed) and one question and predict the answer

export PYTHONPATH=$PYTHONPATH:./src
"""

# python lib
import json
import time
#
import random
import os
import pickle
# python plot
import tkinter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pandas as pd
import seaborn as sn

# python numerical computation
import numpy as np
import torch

# nlp lib
import nltk
from nltk.corpus import stopwords

# my dependencies
# from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from src.tasks.gqa_data import GQABufferLoader
from src.tasks.gqa_model import GQAModel
import argparse
from param import args


# useful fonctions ----------------------------------
def get_alignment_from_attmap(att_map, boxes, obj_class, tkn_sent):
    """
    Extract word object alignment from one attention map (torch tensor)
    return:
        word2bbox: dictionnary mapping a word with a visual object (its bounding boxe
        'xywh' and its class)
    """
    alignment = torch.argmax(att_map, dim=-1).numpy()
    word2bbox = {wrd: {'xywh': boxes[0, alignment[w_i]].numpy(), 'class': obj_class[alignment[w_i]]} \
                 for w_i, wrd in enumerate(tkn_sent)}
    return word2bbox


def get_k_dist_from_attmaps(att_maps, lang_mask, vis_mask):
    """
    Return distribution of k for each attention map in att_maps
    """

    k_dist = {}
    for maptype in ['lang', 'vis', 'vl', 'lv', 'vv', 'll']:
        n_layers = len(att_maps[maptype])
        k_dist[maptype] = [None] * n_layers
        # Set order of masks (it depends of the direction of the attention)
        if maptype in ['lang', 'll']:
            mask_send = lang_mask
            mask_receive = lang_mask
        elif maptype in ['lv']:
            mask_send = lang_mask
            mask_receive = vis_mask
        elif maptype in ['vis', 'vv']:
            mask_send = vis_mask
            mask_receive = vis_mask
        elif maptype in ['vl']:
            mask_send = vis_mask
            mask_receive = lang_mask
        for l_i in range(n_layers):
            n_heads = len(att_maps[maptype][l_i].squeeze())
            k_dist[maptype][l_i] = [None] * n_heads
            for n_i in range(n_heads):
                this_map = att_maps[maptype][l_i].squeeze()[n_i]  # squeeze because batch size is one
                d_1, d_2 = this_map.shape
                # assert masks dimension match with attention map
                assert mask_send.shape[0] == d_2
                assert mask_receive.shape[0] == d_1
                # compute k
                treshold = 0.9
                tkn_sorted = this_map.contiguous().view((d_1, d_2)).sort(dim=-1, descending=True).values
                tkn_exp = tkn_sorted.unsqueeze(1).expand((-1, d_2, -1))
                triang_mask = torch.ones((d_2, d_2)).tril(diagonal=0).unsqueeze(0)
                cumsum = (triang_mask * tkn_exp).sum(-1)  # [d_1, d_2]
                k = (cumsum <= treshold).sum(-1) + 1  # [d_1, 1]
                # normalize k with the number of non masked tokens
                true_n_tkn = mask_send.sum(-1)  # number of tokens after masking
                k = k.view((d_1, 1)).float() / true_n_tkn.unsqueeze(-1).float() * 100
                # remove masked tokens
                k = torch.masked_select(k.squeeze(), mask_receive.bool())
                k_dist[maptype][l_i][n_i] = k.numpy()
    return k_dist


def empty_mask():
    head_mask = {}
    for maptype in ['lang', 'vis', 'vl', 'lv', 'vv', 'll']:

        if maptype == 'lang':
            n_layers = args.llayers
        elif maptype == 'vis':
            n_layers = args.rlayers
        else:
            n_layers = args.xlayers

        head_mask[maptype] = torch.zeros((n_layers, args.n_head))
    return head_mask


# demo class ------------------------------------------

class Demo_data():
    """
    Class to handle the data used for the demo.
    Load all the data in RAM.
    """

    def __init__(self, cfg):
        self.img_dst = {}  # {img_id: features, ...}
        self.gqa_buffer_loader = GQABufferLoader()
        self.cfg = cfg

        vocab_path = cfg["object_classes_oracle"] if cfg["oracle"] else cfg["object_classes"]
        print("Object vocab path is %s" % vocab_path)
        with open(vocab_path, 'r') as f:
            self.object_classes = f.read().split('\n')

    def load_all(self, cfg):
        """
        Load all pre-trained Faster-RCNN embeddings for all images.
        """

        #
        #   DEPRECATED
        #

        timer = time.time()
        img_data = []
        img_data.extend(self.gqa_buffer_loader.load_data(cfg['data_split'], -1, path=cfg['feats_dir']))
        for img_datum in img_data:
            img_id = img_datum['img_id']
            self.img_dst[img_id] = img_datum
        print("%d images loaded in %ds" % (len(self.img_dst), time.time() - timer))
        print("Available images: ", self.img_dst.keys())

        with open(cfg["object_classes"], 'r') as f:
            self.object_classes = f.read().split('\n')

    def check_img(self, img_id):
        return img_id in self.img_dst

    def get_random(self):
        return random.choice(list(self.img_dst.keys()))  # +'.jpg'

    def proc_img_feat(self, img_feat, img_feat_pad_size):
        if img_feat.shape[0] > img_feat_pad_size:
            img_feat = img_feat[:img_feat_pad_size]

        img_feat = np.pad(
            img_feat,
            ((0, img_feat_pad_size - img_feat.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        return img_feat

    def get_feats(self, img_id):
        """
        Get features of image 'img_id'.
        Also do some pre-processing.
        """

        if self.cfg['oracle']:  # load from pickle file
            load_path = os.path.join(self.cfg['oracle_dir'], "%s.pickle" % img_id)
            with open(load_path, 'rb') as handle:
                img_info = pickle.load(handle)
                img_info['boxes'] = img_info['boxes'][:, :2320].astype(np.float32)
                img_info['features'] = img_info['features'][:, :2320].astype(np.float32)
        else:
            load_path = os.path.join(self.cfg['rcnn_dir'], "%s.pickle" % img_id)
            with open(load_path, 'rb') as handle:
                img_info = pickle.load(handle)
                img_info['boxes'] = img_info['boxes'].copy()[:, :2048].astype(np.float32)
                img_info['features'] = img_info['features'].copy()[:, :2048].astype(np.float32)

        # Deprecated :
        # else:
        #     img_info = self.img_dst[img_id]  # load from RAM

        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        objects_id = img_info['objects_id'].copy()
        obj_class = []
        for obj_id in objects_id:
            obj_class.append(self.object_classes[obj_id])

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        # np.testing.assert_array_less(boxes, 1 + 1e-5) #TODO Not useful for demo
        # np.testing.assert_array_less(-boxes, 0 + 1e-5)

        # Padding, because each image does not necessary have same amount of
        # object (e.g. oracle)
        visual_attention_mask = np.concatenate((np.ones(min(obj_num, 36)), np.zeros(max(0, 36 - obj_num))))
        boxes = self.proc_img_feat(boxes, 36)
        feats = self.proc_img_feat(feats, 36)

        return feats, boxes, obj_class, visual_attention_mask, obj_num, img_w, img_h


class Demo_display():
    """
    Class for displaying images and other informations.
    Use matplotlib and seaborn.
    """

    def __init__(self, data_path):

        self.data_path = data_path
        self.img_displayed = None  # display one images one by one
        self.ignore_tkn = ["[SEP]", "?"] + list(stopwords.words('english'))

    def draw_k_dist(self, k_dist):
        n_heads = len(k_dist['lang'][0])
        for maptype in ['lang', 'vis', 'vl', 'lv', 'vv', 'll']:

            n_layers = len(k_dist[maptype])
            fig, axes = plt.subplots(nrows=n_layers, ncols=n_heads, figsize=(50, 35))
            for layer_id in range(len(k_dist[maptype])):
                for head_id, head_dist in enumerate(k_dist[maptype][layer_id]):
                    # print("Plotting layer %d head %d..." % (layer_id, head_id))
                    timer = time.time()
                    # display med k
                    median_k = np.median(head_dist)
                    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                    axes[layer_id, head_id].text(0.05, 0.95, 'median_k: %.2f$' % median_k,
                                                 transform=axes[layer_id, head_id].transAxes, fontsize=6,
                                                 verticalalignment='top', bbox=props)
                    # compute metastable state
                    n_tokens = 100  # range_bins[maptype][1]  # max nb of tokens
                    if median_k < (1 / 8 * n_tokens):
                        # class IV
                        color = 'b'
                    elif median_k < (1 / 4 * n_tokens):
                        # class III
                        color = 'g'
                    elif median_k < (1 / 2 * n_tokens):
                        # class II
                        color = 'orange'
                    elif median_k >= (1 / 2 * n_tokens):
                        color = 'r'
                        # class I
                    # plot dist
                    x_data, y_data = sn.distplot(head_dist, \
                                                 hist_kws={'range': (0, 100)}, \
                                                 bins=10, \
                                                 rug=False, \
                                                 hist=False, \
                                                 ax=axes[layer_id, head_id], \
                                                 kde_kws={"shade": True}, \
                                                 color=color).get_lines()[0].get_data()  # range_bins[maptype]
                    axes[layer_id, head_id].set_title('%s_%d_%d' % (maptype, layer_id, head_id))

                    # print("Done in %.2f seconds!"%(time.time() - timer))
            plt.suptitle(maptype)
            plt.show(block=False)

    def draw_heat_map(self, k_dist):

        all_k_median = []
        maps_label = []
        n_heads = len(k_dist['lang'][0])

        count_layer = 0
        for maptype in ['lang', 'vis', 'vl', 'lv', 'vv', 'll']:
            n_layers = len(k_dist[maptype])
            for layer_id in range(len(k_dist[maptype])):
                maps_label.append('%s_%d' % (maptype, layer_id))
                all_k_median.append([None] * n_heads)
                for head_id, head_dist in enumerate(k_dist[maptype][layer_id]):
                    # display med k
                    median_k = np.median(head_dist)
                    all_k_median[count_layer][head_id] = median_k
                count_layer += 1
        all_k_median = np.array(all_k_median)

        sn.heatmap(all_k_median.transpose(), annot=False, xticklabels=maps_label, center=50.0, vmin=0, vmax=100)

        plt.tight_layout()
        plt.suptitle("Attention head activation (measured with k_median)")
        plt.title("Row are heads, columns are layers")
        plt.show(block=False)

    def show_image(self, img_id, word2box=None):

        img_path = os.path.join(self.data_path, img_id)
        if os.path.exists(img_path):
            if (img_path == self.img_displayed) and (word2box is None):
                # do nothing
                pass
            else:
                plt.close()
                # load image
                im = np.array(Image.open(img_path), dtype=np.uint8)
                # Create figure and axes
                fig, ax = plt.subplots(1)
                # Display the image
                ax.imshow(im)

                if word2box is not None:
                    width = im.shape[0]
                    height = im.shape[1]
                    already_written = []
                    for word, roi in word2box.items():
                        box = roi['xywh']
                        obj_class = roi['class']

                        if word in self.ignore_tkn:
                            continue

                        # get coordinates in pixel            
                        box_xy = (box[0] * width, box[1] * height)
                        box_w = box[2] * width
                        box_h = box[3] * height
                        # Create a Rectangle patch
                        c = np.random.rand(3, )
                        rect = patches.Rectangle(box_xy, box_w, box_h, linewidth=2, edgecolor=c, facecolor='none')
                        # Add the patch to the Axes
                        ax.add_patch(rect)
                        # write label
                        text_xy = (box_xy[0] + 5, box_xy[1] + 5)
                        while text_xy in already_written:
                            text_xy = (text_xy[0] + 50, text_xy[1])
                        ax.annotate("%s/%s" % (word, obj_class), text_xy, color=c, weight='bold',
                                    fontsize=12)
                        already_written.append(text_xy)

                plt.show(block=False)
                self.img_displayed = img_path
        else:
            print("%s does not exist... I cannot display the image!" % img_path)


class Demo():
    """
    Main class for the demo.
    """

    def __init__(self, mode=None):
        self.model = None  # pretrained VQA model
        self.cfg = None  # demo configs
        self.mode = mode
        self.load_config()
        self.data_loader = ""  # my data loader (not pytorch one)
        # self.data_loader = Demo_data(self.cfg)  # my data loader (not pytorch one)
        self.label_to_ans = {}  # add dictionnary mapping ans_id to answer
        self.displayer = Demo_display(data_path=self.cfg['images_dir'])
        self.load_model()

    def load_config(self):
        """
        Load demo config from a json file and update LXMERT cfg
        /!\ LXMERT config is defined in src/params.py with argparse
        """
        with open('model/src/tasks/demo_cfg.json', 'r') as f:
            self.cfg = json.load(f)

        if self.mode is not None:
            if self.mode in ['lxmert_tiny', 'lxmert_tiny_init_oracle_pretrain', 'lxmert_tiny_init_oracle_scratch']:
                self.cfg["tiny_lxmert"] = 1
                self.cfg["oracle"] = 0
            elif self.mode in ['lxmert']:
                self.cfg["tiny_lxmert"] = 0
                self.cfg["oracle"] = 0
            elif self.mode in ['tiny_oracle']:
                self.cfg["tiny_lxmert"] = 1
                self.cfg["oracle"] = 1
            if "lxmert" in self.mode:
                self.cfg['type'] = "rcnn"
            else:
                self.cfg['type'] = "oracle"
            self.initConf("model/src/pretrain/" + self.mode)
        else:

            # modify LXMERT config accoring to the demo cfg:
            # (manually modify argument in args)
            if self.cfg['ecai_lxmert']:
                args.task_pointer = 'KLDiv'
            else:
                args.task_pointer = 'none'
            if self.cfg['tiny_lxmert']:
                args.n_head = 4
                args.hidden_size = 128
                args.from_scratch = True
            if self.cfg['oracle']:
                args.visual_feat_dim = 2320
                self.cfg['data_split'] = 'val'
                args.from_scratch = True
        print("Config loaded!")

    def initConf(self, path):
        # self.data_loader = Demo_data(self.cfg)

        args.task_pointer = 'KLDiv'
        args.n_head = 12
        args.hidden_size = 768
        args.from_scratch = False

        args.visual_feat_dim = 2048

        if self.cfg['ecai_lxmert']:
            args.task_pointer = 'KLDiv'
        else:
            args.task_pointer = 'none'
        if self.cfg['tiny_lxmert']:
            args.n_head = 4
            args.hidden_size = 128
            args.from_scratch = True
        if self.cfg['type'] == "oracle":
            args.visual_feat_dim = 2320

            # if self.cfg['oracle']:
            #     args.visual_feat_dim = 2320
            #     self.cfg['data_split'] = 'val'
            args.from_scratch = True

        self.cfg["pretrained_model_lxmert"] = path
        self.cfg["pretrained_model_tiny_lxmert"] = path
        self.cfg["pretrained_model_tiny_lxmert_oracle"] = path
        print("Config loaded!")

    def load_model(self):
        """
        Load the pre-trained VQA model
        """

        # update data, to allow multiple load_model() calls
        self.data_loader = Demo_data(self.cfg)

        # load answer dict
        with open(self.cfg['answers_dict'], 'r') as f:
            self.label_to_ans = json.load(f)

        # load architecture
        self.model = GQAModel(self.cfg['num_answers'])

        # load pretrained weights
        if self.cfg['ecai_lxmert']:
            if self.cfg['tiny_lxmert']:
                print("Sorry, there is no tiny version of ecai lxmert. Change config in src/task/demo_cfg.json!")
                exit(0)
            # Load raw pretrained model
            path = self.cfg['pretrained_model_ecai']
            _ = load_lxmert_qa(path, self.model,
                               label2ans=self.label_to_ans)

        else:
            print(self.cfg)
            # Load finetuned model
            if self.cfg['tiny_lxmert']:
                if self.cfg['oracle']:
                    path = self.cfg['pretrained_model_tiny_lxmert_oracle']
                else:
                    path = self.cfg['pretrained_model_tiny_lxmert']
            else:
                if self.cfg['oracle']:
                    print('Oracle model is only available in tiny version!')
                    exit(0)
                else:
                    path = self.cfg['pretrained_model_lxmert']
            print("Load model's weights from %s" % path)
            state_dict = torch.load("%s.pth" % path, map_location=torch.device('cpu'))
            for key in list(state_dict.keys()):
                if '.module' in key:
                    state_dict[key.replace('.module', '')] = state_dict.pop(key)
            self.model.load_state_dict(state_dict, strict=False)

        # To GPU
        # self.model = self.model.cuda()

        print("Model loaded!")

    def load_data(self):
        """
        Load all the data (GQA testdev) in RAM
        """

        if self.cfg['oracle']:
            print('Oracle data do not need to be loaded in RAM')
        else:
            print('load_data is deprecated.')
        # else:
        #     self.data_loader.load_all(self.cfg)

    def img_available(self, image):
        img_id = image.split('.')[0]
        return self.data_loader.check_img(img_id)

    def get_random_img(self):
        return self.data_loader.get_random()

    def display_image(self, img_id, word2box=None):
        self.displayer.show_image(img_id + '.jpg', word2box)

    def display_k_dist(self, k_dist, compact):
        if compact:
            self.displayer.draw_heat_map(k_dist)
        else:
            self.displayer.draw_k_dist(k_dist)

    def ask(self, question, image, head_mask, show_heads=False, force_attmaps=None):
        """
        Ask a question about an (pre-processed) image,
        @input:
            *question: string (e.g. "What color is the blue car?")
            *image: path of the image (e.g "./2383391.jpg")
            *mask: dictionnary assigning a binary masking to each attention head
            *show_heads=True: display the heads attention.
            *force_attmaps: do inference while forcing attention maps values. Must be logprob. The whole batch will be forced accordingly
        @return:
            *top_prediction: (top_answer, predicted_score)
            *five_predictions: top five predictions [(1_answer, 1_score),...,(5_answer, 5_score)]
            *attention_heads: visualization of attention heads
            *word2box: dictionnary mapping question's words to bounding boxes
            *          {"word":[x,y,w,h]}. Coordinates are relative to the dimension of the image.
            *k_dist: distribution of k for each attention head (dictionnary)
        """
        self.model.eval()

        # Load image features
        img_id = image.split('.')[0]
        feats, boxes, obj_class, visual_attention_mask, obj_num, width, height = self.data_loader.get_feats(img_id)

        # Reshape data in a batch of size 1 and turn them to tensor
        feats = torch.from_numpy(feats).unsqueeze(0)
        boxes = torch.from_numpy(boxes).unsqueeze(0)
        visual_attention_mask = torch.from_numpy(visual_attention_mask).unsqueeze(0)
        question = [question]

        # We do not use these variables, so we define dummy values
        iou_question = torch.zeros((1, 20, 36))
        iou_answer = torch.zeros((1, 1, 36))
        sem_question_words = torch.zeros((1, 20, 36))
        sem_answer_words = torch.zeros((1, 1, 36))
        bboxes_words = torch.zeros((1, 20 + 1, 4))
        vis_mask = torch.from_numpy(np.concatenate((np.ones(min(obj_num, 36)), np.zeros(max(0, 36 - obj_num)))))

        # To GPU
        # feats, boxes, visual_attention_mask = feats.cuda(), boxes.cuda(), visual_attention_mask.cuda()
        # iou_question, iou_answer = iou_question.cuda(), iou_answer.cuda()
        # sem_question_words, sem_answer_words, bboxes_words = sem_question_words.cuda(), sem_answer_words.cuda(), bboxes_words.cuda()

        #* Forcing attention maps: example!
        #* 1) I simulate an attention map extracted from another question
        # n_layers = {'lang':9, 'vis':5, 'vl':5, 'lv':5, 'vv':5, 'll':5}
        # dim = {'lang':(20,20), 'vis':(36,36), 'vl':(20,36), 'lv':(36,20), 'vv':(36,36), 'll':(20,20)} # (receive, send)
        # force_attmaps = {}
        # for maptype in ['lang', 'vis', 'vl', 'lv', 'vv', 'll']:
        #     force_attmaps[maptype] = []
        #     for layer in range(n_layers[maptype]):
        #         heads_attmap = []
        #         for head in range(4):
        #             heads_attmap.append(torch.log(torch.rand(dim[maptype]).softmax(dim=-1)+1e-9))    
        #         force_attmaps[maptype].append(heads_attmap)
        #* 2) be sure to have logprob (not softmax)
        #* 3) If you want, you can let some heads free. To do so, assign them a None value.
        #* These heads will be computed as usual.
        #force_attmaps['lang'][0][0] = None

        # Inference
        with torch.no_grad():
            """
            logit: prediction of the model
            tkn_sent: tokenized sentence
            att_maps: attention maps
            """

            logit, _, _, _, _, tkn_sent, att_maps, lang_mask, score_srt, label_srt = self.model(feats, boxes, question,
                                                                                                iou_question,
                                                                                                iou_answer,
                                                                                                sem_question_words,
                                                                                                sem_answer_words,
                                                                                                bboxes_words,
                                                                                                visual_attention_mask,
                                                                                                verbose=True,
                                                                                                head_mask=head_mask,
                                                                                                force_attmaps=force_attmaps,)

        # Extract alignment for attention map 'vl' layer 3 head 0
        # word2bbox = get_alignment_from_attmap(
        #     att_maps['vl'][3].cpu().squeeze().sum(0),
        #     boxes, obj_class, tkn_sent[0])
        word2bbox = get_alignment_from_attmap(
            att_maps['vl'][3].squeeze().sum(0),
            boxes, obj_class, tkn_sent[0])

        # Extract k_dist
        k_dist = get_k_dist_from_attmaps(att_maps, lang_mask.squeeze(), vis_mask)
        # k_dist = get_k_dist_from_attmaps(att_maps, lang_mask.cpu().squeeze(), vis_mask)

        # compute prediction
        # logit = torch.softmax(logit, dim=-1)
        # score, label = logit.max(1)
        # top_prediction = (self.label_to_ans[label[0].numpy()], score[0])
        # score_srt, label_srt = torch.sort(logit.squeeze(), descending=True, dim=-1)
        five_predictions = [(self.label_to_ans[label_srt[i].numpy()], score_srt[i]) for i in range(5)]
        attention_heads = att_maps


        # textual and visual input labels
        bboxes_pxl = (boxes.squeeze()[:obj_num] * torch.tensor([width, height, width, height]).unsqueeze(
            0).float()).short().tolist()
        input_labels = {'textual': tkn_sent[0], 'visual': obj_class, 'bboxes': bboxes_pxl}

        # Input size
        input_size = {'textual': len(tkn_sent[0]), 'visual': obj_num}
        return five_predictions, attention_heads, word2bbox, k_dist, input_labels, input_size


if __name__ == "__main__":

    # * Display config
    display_k_dist = True
    compact_k_dist = True  # compact=False will display all the k ditribution
    display_alignment = False
    # * /

    my_demo = Demo()
    my_demo.load_data()
    my_demo.load_model()

    image = None

    while (True):

        if image == None:
            image = input("Image? (ex:'n520071.jpg').......")
        else:
            keep = input("Keep the same image? [y/n]")
            if keep in "no":
                image = input("Image?.......")

        while not my_demo.img_available(image):
            if image == 'random':
                image = my_demo.get_random_img()
                print("Random image is %s" % image)
                break
            image = input("%s is not available, please provide a new image:" % image)

        my_demo.display_image(image)
        question = input("Question?.......")

        head_mask = empty_mask()
        # uncomment to allow head masking
        # head_mask['vl'] += 1  # mask all vl layers
        # head_mask['lang'][3,3] += 1# mask the head 3 in lang layer 3

        top_prediction, five_predictions, attention_heads, alignment, k_dist, input_labels, input_size \
            = my_demo.ask(question, image, head_mask)

        # display
        if display_alignment:
            my_demo.display_image(image, word2box=alignment)
        if display_k_dist:
            my_demo.display_k_dist(k_dist, compact=compact_k_dist)
        print(">Predicted answer:", top_prediction)
        print("")
