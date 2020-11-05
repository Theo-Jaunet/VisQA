# coding=utf-8
# Copyright 2019 project LXRT.

import json

import time

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

import os
import pickle
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000
MAX_GQA_LENGTH = 20
MAX_LENGTH = 20

class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number, path):
        if name == 'testdev':
            path = os.path.join(path, "gqa_testdev_obj36.tsv")
        elif name == 'valid':
            path = os.path.join(path, "gqa_val_obj36.tsv")
        else:
            path = os.path.join(path, "vg_gqa_obj36.tsv")
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


