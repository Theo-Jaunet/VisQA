# coding=utf-8
# Copyr, Hmmm, this is just feature loading which could be used anywhere.

import sys
import csv
import base64
import time
import os

import numpy as np
import pickle

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None, split=False, fp16=False):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                if fp16 and item[key].dtype == np.float32:
                    item[key] = item[key].astype(np.float16)    # Save features as half-precision in memory.
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)
            
            if split==True:
                data_write = os.path.join('data/all_data', '%s.pickle' % item['img_id'])
                with open(data_write, 'wb') as handle:
                    pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                data.append(item)
            if topk is not None and len(data) == topk:
                break

    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data

if __name__ == "__main__":
    tsv_files = [#'data/mscoco_imgfeat/test2015_obj36.tsv',
                 'data/mscoco_imgfeat/train2014_obj36.tsv',
                 #'data/vg_gqa_imgfeat/vg_gqa_obj36.tsv',
                 ]
    for tsv in tsv_files:
        _ = load_obj_tsv(tsv, topk=-1, split=True)
    