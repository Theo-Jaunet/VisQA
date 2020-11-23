# coding=utf-8
# Copyright 2019 project LXRT.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import torch
import torch.nn as nn

from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTFeatureExtraction as VisualBertForLXRFeature, VISUAL_CONFIG


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_sents_to_features(sents, max_seq_length, tokenizer, iou_question=None, iou_answer=None,
                              sem_question_words=None, sem_answer_words=None, bboxes_words=None):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    tkn_sent = []
    iou_target = None
    if iou_question is not None:
        iou_target = torch.zeros(
            (len(sents), max_seq_length, iou_question.size(-1)),
            device=iou_question.device)

    for (i, sent) in enumerate(sents):
        tokens_a = tokenizer.tokenize(sent.strip())

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        # # ? by Corentin **********************
        # # update iou question
        # # handle punctuation cases
        # # note: idx = 0 is for [CLS] token

        # # Add answer word iou at [CLS] and dummy zeros at [SEP]
        # iou_target[i, 0] = iou_answer[i, 0]
        # diff = 0
        # for idx, tkn in enumerate(tokens_a):
        #     if "," in tkn:
        #         # , token receives dummy iou
        #         diff += 1
        #     elif "'" in tkn:
        #         # ' token receives dummy iou
        #         # token after "'" receives same iou as token before "'"
        #         iou_target[i, idx+2] = iou_question[i, idx - diff - 1]
        #         diff += 2
        #     elif "-" in tkn:
        #         # the second part of the '-' word is given the same iou
        #         # - token receives dummy iou
        #         iou_target[i, idx+2] = iou_question[i, idx - diff - 1]
        #         diff += 2
        #     elif "##" in tkn:  # word piece
        #         # ##token is given the iou of the word before it
        #         iou_target[i, idx+1] = iou_question[i, idx - diff - 1]
        #         diff += 1
        #     else:
        #         # copy iou at the right index
        #         iou_target[i, idx+1] = iou_question[i, idx - diff]
        # # ? *********************************

        # update iou question
        # handle punctuation cases
        # note: idx = 0 is for [CLS] token
        if iou_question is not None:
            iou = torch.cat([iou_answer[i], iou_question[i]])
            
            sem_question = sem_question_words[i]
            sem_answer = sem_answer_words[i]
            sem = torch.cat([sem_answer, sem_question])

            iou_weight = 0.5
            norm_iou = iou / (torch.norm(iou, p=2, dim=1).unsqueeze(-1) + 1e-9)
            norm_sem = sem / (torch.norm(sem, p=2, dim=1).unsqueeze(-1) + 1e-9)
            score = iou_weight * norm_iou + (1 - iou_weight) * norm_sem 

            iou_target = torch.zeros(
                (len(sents), max_seq_length, iou_question.size(-1)),
                device=iou_question.device)
            words_anchor = torch.zeros((len(sents), max_seq_length, 4), device=iou_question.device)
            # Add answer word iou at [CLS] and dummy zeros at [SEP]
            iou_target[i, 0] = score[0]
            words_anchor[i, 0] = bboxes_words[i, 0]
            diff = 0
            for idx, tkn in enumerate(tokens_a):
                if "," in tkn:
                    # , token receives dummy iou
                    diff += 1
                elif "'" in tkn:
                    # ' token receives dummy iou
                    # token after "'" receives same iou as token before "'"
                    iou_target[i, idx+2] = score[idx + 1 - diff - 1]
                    words_anchor[i, idx+2] = bboxes_words[i, idx + 1 - diff - 1]
                    diff += 2
                elif "-" in tkn:
                    # the second part of the '-' word is given the same iou
                    # - token receives dummy iou
                    iou_target[i, idx+2] = score[idx + 1 - diff - 1]
                    words_anchor[i, idx+2] = bboxes_words[i, idx + 1 - diff - 1]
                    diff += 2
                elif "##" in tkn:  # word piece
                    # ##token is given the iou of the word before it
                    iou_target[i, idx+1] = score[idx + 1 - diff - 1]
                    words_anchor[i, idx+1] = bboxes_words[i, idx + 1 - diff - 1]
                    diff += 1
                else:
                    # copy iou at the right index
                    iou_target[i, idx+1] = score[idx + 1 - diff]  #* +1: because score[0] is answer
                    words_anchor[i, idx+1] = bboxes_words[i, idx + 1 - diff]
                # ? *********************************

        # Keep segment id which allows loading BERT-weights.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        tkn_sent.append(tokens)
        segment_ids = [0] * len(tokens)

        # DEBUG: print pointed tokens
        # for idx, tkn in enumerate(tokens):
        #     if iou_target[i, idx].sum() > 0:
        #         print(tkn)
        # print('----------------------------------') 

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features, iou_target, tkn_sent


def set_visual_config(args):
    VISUAL_CONFIG.l_layers = args.llayers
    VISUAL_CONFIG.x_layers = args.xlayers
    VISUAL_CONFIG.r_layers = args.rlayers
    VISUAL_CONFIG.visual_feat_dim = args.visual_feat_dim

class LXRTEncoder(nn.Module):
    def __init__(self, args, max_seq_length, mode='x', already_converted=False):
        super().__init__()

        self.args = args

        self.max_seq_length = max_seq_length
        set_visual_config(args)

        # Using the bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased",
            do_lower_case=True
        )

        # Build LXRT Model
        self.model = VisualBertForLXRFeature.from_pretrained(
            "bert-base-uncased",
            mode='lxr'
        )

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_bert_weights)

        self.already_converted = already_converted

    def multi_gpu(self):
        self.model = nn.DataParallel(self.model)

    @property
    def dim(self):
        return self.args.hidden_size

    def forward(self, sents, feats, head_mask, visual_attention_mask=None,
                iou_question=None, iou_answer=None, sem_question_words=None, sem_answer_words=None, bboxes_words=None,
                pool_only=False, force_attmaps=None):
        if not self.already_converted:
            train_features, iou_target, tkn_sent = convert_sents_to_features(
                sents, self.max_seq_length, self.tokenizer, iou_question,
                iou_answer, sem_question_words, sem_answer_words, bboxes_words)
        else:
            train_features, iou_target, tkn_sent = sents, None, None

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        # input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda(non_blocking=True)
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        # input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda(non_blocking=True)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        # segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda(non_blocking=True)

        (lang_feat, vis_feat), output, att_maps = self.model(   
            input_ids, segment_ids, input_mask,
            head_mask=head_mask,
            visual_feats=feats,
            visual_attention_mask=visual_attention_mask, force_attmaps=force_attmaps)
        return output, lang_feat, vis_feat, iou_target, tkn_sent, input_mask, att_maps

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_LXRT.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load LXMERT pre-trained model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)




