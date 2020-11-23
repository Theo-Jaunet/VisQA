# coding=utf-8
# Copyright 2019 project LXRT.

import torch.nn as nn
import torch

import math
from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()


        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)
        if args.task_pointer != 'none':
            self.matching_decoder = MatchingDecoderLV(metric='sdp')

    def forward(self, feat, pos, sent, iou_question, iou_answer, sem_question_words, sem_answer_words, bboxes_words,
                visual_attention_mask, head_mask=None, verbose=False, force_attmaps=None):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        x, lang_feat, vis_feat, iou_target, tkn_sent,input_mask, att_maps = self.lxrt_encoder(
            sent, (feat, pos),
            head_mask=head_mask,
            iou_question=iou_question, iou_answer=iou_answer,
            sem_question_words=sem_question_words,
            sem_answer_words=sem_answer_words,
            bboxes_words=bboxes_words, visual_attention_mask=visual_attention_mask, force_attmaps=force_attmaps)
        logit = self.logit_fc(x)

        iou_pred = None
        if args.task_pointer != 'none':
            iou_pred = self.matching_decoder(lang_feat, vis_feat)
        logit = torch.softmax(logit, dim=-1)

        # score, label = logit.max(1)
        score_srt, label_srt = torch.sort(logit.squeeze(), descending=True, dim=-1)

        if verbose:
            return logit, iou_target, iou_pred, lang_feat, vis_feat, tkn_sent, att_maps, input_mask,score_srt, label_srt
            # return logit, iou_target, iou_pred, lang_feat, vis_feat, tkn_sent, att_maps, input_mask,score,label,score_srt, label_srt
        else:
            return logit, iou_target, iou_pred#, att_maps


class MatchingDecoderLV(nn.Module):
    """Decode language->vision matching from language embeddings"""
    def __init__(self, metric):
        super(MatchingDecoderLV, self).__init__()
        HIDDEN_DECODER_SIZE = 256
        hid_dim = 768
        self.lang_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, HIDDEN_DECODER_SIZE)
        )
        self.vis_proj = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, HIDDEN_DECODER_SIZE)
        )
        self.metric = metric
        assert metric in ['sdp', 'cosine']

    def forward(self, lang_feat, vis_feat):
        # Projection
        l = self.lang_proj(lang_feat)
        v = self.vis_proj(vis_feat)
        if self.metric == 'sdp':
            scaled_dot_product = torch.einsum('bld, bnd -> bln', l, v) / math.sqrt(v.size(-1))
            # in [0, 1]
            matching = scaled_dot_product  #torch.relu(scaled_dot_product)  #! RELU
        elif self.metric == 'cosine':
            l_norm2 = l / torch.norm(l, p=2, dim=-1).unsqueeze(-1)
            v_norm2 = v / torch.norm(v, p=2, dim=-1).unsqueeze(-1)
            dot_product = torch.einsum('bld, bnd -> bln', l_norm2, v_norm2)
            matching = dot_product
        return matching
