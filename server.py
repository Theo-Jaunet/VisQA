import binascii
import pickle

import ujson as ujson
import umap.umap_ as umap
import numpy as np
from flask import Flask, render_template, request, session, redirect, logging, jsonify, Response
from flask_caching import Cache
from flask_compress import Compress
import os
import sys
from shutil import copyfile

sys.path.insert(1, 'model/src/tasks')
sys.path.insert(1, 'model/src/')
sys.path.insert(1, 'model')
from model.src.tasks.demo import Demo, empty_mask

# import demo

app = Flask(__name__)

app.secret_key = binascii.hexlify(os.urandom(24))

COMPRESS_MIMETYPES = ['text/html', 'text/css', 'text/csv', 'text/xml', 'application/json',
                      'application/javascript', 'image/jpeg', 'image/png']
COMPRESS_LEVEL = 6
COMPRESS_MIN_SIZE = 500

cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)
Compress(app)

my_demo = Demo()

order = ['lang_0_0', 'lang_0_1', 'lang_0_2', 'lang_0_3', 'lang_0_4', 'lang_0_5', 'lang_0_6', 'lang_0_7', 'lang_0_8',
         'lang_0_9', 'lang_0_10', 'lang_0_11', 'lang_1_0', 'lang_1_1', 'lang_1_2', 'lang_1_3', 'lang_1_4', 'lang_1_5',
         'lang_1_6', 'lang_1_7', 'lang_1_8', 'lang_1_9', 'lang_1_10', 'lang_1_11', 'lang_2_0', 'lang_2_1', 'lang_2_2',
         'lang_2_3', 'lang_2_4', 'lang_2_5', 'lang_2_6', 'lang_2_7', 'lang_2_8', 'lang_2_9', 'lang_2_10', 'lang_2_11',
         'lang_3_0', 'lang_3_1', 'lang_3_2', 'lang_3_3', 'lang_3_4', 'lang_3_5', 'lang_3_6', 'lang_3_7', 'lang_3_8',
         'lang_3_9', 'lang_3_10', 'lang_3_11', 'lang_4_0', 'lang_4_1', 'lang_4_2', 'lang_4_3', 'lang_4_4', 'lang_4_5',
         'lang_4_6', 'lang_4_7', 'lang_4_8', 'lang_4_9', 'lang_4_10', 'lang_4_11', 'lang_5_0', 'lang_5_1', 'lang_5_2',
         'lang_5_3', 'lang_5_4', 'lang_5_5', 'lang_5_6', 'lang_5_7', 'lang_5_8', 'lang_5_9', 'lang_5_10', 'lang_5_11',
         'lang_6_0', 'lang_6_1', 'lang_6_2', 'lang_6_3', 'lang_6_4', 'lang_6_5', 'lang_6_6', 'lang_6_7', 'lang_6_8',
         'lang_6_9', 'lang_6_10', 'lang_6_11', 'lang_7_0', 'lang_7_1', 'lang_7_2', 'lang_7_3', 'lang_7_4', 'lang_7_5',
         'lang_7_6', 'lang_7_7', 'lang_7_8', 'lang_7_9', 'lang_7_10', 'lang_7_11', 'lang_8_0', 'lang_8_1', 'lang_8_2',
         'lang_8_3', 'lang_8_4', 'lang_8_5', 'lang_8_6', 'lang_8_7', 'lang_8_8', 'lang_8_9', 'lang_8_10', 'lang_8_11',
         'vis_0_0', 'vis_0_1', 'vis_0_2', 'vis_0_3', 'vis_0_4', 'vis_0_5', 'vis_0_6', 'vis_0_7', 'vis_0_8', 'vis_0_9',
         'vis_0_10', 'vis_0_11', 'vis_1_0', 'vis_1_1', 'vis_1_2', 'vis_1_3', 'vis_1_4', 'vis_1_5', 'vis_1_6', 'vis_1_7',
         'vis_1_8', 'vis_1_9', 'vis_1_10', 'vis_1_11', 'vis_2_0', 'vis_2_1', 'vis_2_2', 'vis_2_3', 'vis_2_4', 'vis_2_5',
         'vis_2_6', 'vis_2_7', 'vis_2_8', 'vis_2_9', 'vis_2_10', 'vis_2_11', 'vis_3_0', 'vis_3_1', 'vis_3_2', 'vis_3_3',
         'vis_3_4', 'vis_3_5', 'vis_3_6', 'vis_3_7', 'vis_3_8', 'vis_3_9', 'vis_3_10', 'vis_3_11', 'vis_4_0', 'vis_4_1',
         'vis_4_2', 'vis_4_3', 'vis_4_4', 'vis_4_5', 'vis_4_6', 'vis_4_7', 'vis_4_8', 'vis_4_9', 'vis_4_10', 'vis_4_11',
         'vl_0_0', 'vl_0_1', 'vl_0_2', 'vl_0_3', 'vl_0_4', 'vl_0_5', 'vl_0_6', 'vl_0_7', 'vl_0_8', 'vl_0_9', 'vl_0_10',
         'vl_0_11', 'vl_1_0', 'vl_1_1', 'vl_1_2', 'vl_1_3', 'vl_1_4', 'vl_1_5', 'vl_1_6', 'vl_1_7', 'vl_1_8', 'vl_1_9',
         'vl_1_10', 'vl_1_11', 'vl_2_0', 'vl_2_1', 'vl_2_2', 'vl_2_3', 'vl_2_4', 'vl_2_5', 'vl_2_6', 'vl_2_7', 'vl_2_8',
         'vl_2_9', 'vl_2_10', 'vl_2_11', 'vl_3_0', 'vl_3_1', 'vl_3_2', 'vl_3_3', 'vl_3_4', 'vl_3_5', 'vl_3_6', 'vl_3_7',
         'vl_3_8', 'vl_3_9', 'vl_3_10', 'vl_3_11', 'vl_4_0', 'vl_4_1', 'vl_4_2', 'vl_4_3', 'vl_4_4', 'vl_4_5', 'vl_4_6',
         'vl_4_7', 'vl_4_8', 'vl_4_9', 'vl_4_10', 'vl_4_11', 'lv_0_0', 'lv_0_1', 'lv_0_2', 'lv_0_3', 'lv_0_4', 'lv_0_5',
         'lv_0_6', 'lv_0_7', 'lv_0_8', 'lv_0_9', 'lv_0_10', 'lv_0_11', 'lv_1_0', 'lv_1_1', 'lv_1_2', 'lv_1_3', 'lv_1_4',
         'lv_1_5', 'lv_1_6', 'lv_1_7', 'lv_1_8', 'lv_1_9', 'lv_1_10', 'lv_1_11', 'lv_2_0', 'lv_2_1', 'lv_2_2', 'lv_2_3',
         'lv_2_4', 'lv_2_5', 'lv_2_6', 'lv_2_7', 'lv_2_8', 'lv_2_9', 'lv_2_10', 'lv_2_11', 'lv_3_0', 'lv_3_1', 'lv_3_2',
         'lv_3_3', 'lv_3_4', 'lv_3_5', 'lv_3_6', 'lv_3_7', 'lv_3_8', 'lv_3_9', 'lv_3_10', 'lv_3_11', 'lv_4_0', 'lv_4_1',
         'lv_4_2', 'lv_4_3', 'lv_4_4', 'lv_4_5', 'lv_4_6', 'lv_4_7', 'lv_4_8', 'lv_4_9', 'lv_4_10', 'lv_4_11', 'vv_0_0',
         'vv_0_1', 'vv_0_2', 'vv_0_3', 'vv_0_4', 'vv_0_5', 'vv_0_6', 'vv_0_7', 'vv_0_8', 'vv_0_9', 'vv_0_10', 'vv_0_11',
         'vv_1_0', 'vv_1_1', 'vv_1_2', 'vv_1_3', 'vv_1_4', 'vv_1_5', 'vv_1_6', 'vv_1_7', 'vv_1_8', 'vv_1_9', 'vv_1_10',
         'vv_1_11', 'vv_2_0', 'vv_2_1', 'vv_2_2', 'vv_2_3', 'vv_2_4', 'vv_2_5', 'vv_2_6', 'vv_2_7', 'vv_2_8', 'vv_2_9',
         'vv_2_10', 'vv_2_11', 'vv_3_0', 'vv_3_1', 'vv_3_2', 'vv_3_3', 'vv_3_4', 'vv_3_5', 'vv_3_6', 'vv_3_7', 'vv_3_8',
         'vv_3_9', 'vv_3_10', 'vv_3_11', 'vv_4_0', 'vv_4_1', 'vv_4_2', 'vv_4_3', 'vv_4_4', 'vv_4_5', 'vv_4_6', 'vv_4_7',
         'vv_4_8', 'vv_4_9', 'vv_4_10', 'vv_4_11', 'll_0_0', 'll_0_1', 'll_0_2', 'll_0_3', 'll_0_4', 'll_0_5', 'll_0_6',
         'll_0_7', 'll_0_8', 'll_0_9', 'll_0_10', 'll_0_11', 'll_1_0', 'll_1_1', 'll_1_2', 'll_1_3', 'll_1_4', 'll_1_5',
         'll_1_6', 'll_1_7', 'll_1_8', 'll_1_9', 'll_1_10', 'll_1_11', 'll_2_0', 'll_2_1', 'll_2_2', 'll_2_3', 'll_2_4',
         'll_2_5', 'll_2_6', 'll_2_7', 'll_2_8', 'll_2_9', 'll_2_10', 'll_2_11', 'll_3_0', 'll_3_1', 'll_3_2', 'll_3_3',
         'll_3_4', 'll_3_5', 'll_3_6', 'll_3_7', 'll_3_8', 'll_3_9', 'll_3_10', 'll_3_11', 'll_4_0', 'll_4_1', 'll_4_2',
         'll_4_3', 'll_4_4', 'll_4_5', 'll_4_6', 'll_4_7', 'll_4_8', 'll_4_9', 'll_4_10', 'll_4_11']


def load_data(file_path):
    with open(file_path, 'rb') as fpkl:
        data = pickle.load(fpkl)
        return data


def select(data):
    res = []
    for d in data:
        res.append(d["k_dist"])
    return res


dataset = load_data("lxmert_gqaval_reasbias.pickle")

# to_map = select(dataset)
# print(to_map[0])
print('making Umap  ....')

# umaper = umap.UMAP(n_neighbors=20, min_dist=0.3).fit(to_map[:])


def make_proj(data):
    print('making Umap  ....')
    X_umap = umap.UMAP(n_neighbors=20, min_dist=0.3).fit_transform(data).tolist()

    return X_umap


@app.route('/')
def index():
    return render_template("index.html")


def saver():
    coords = make_umap([])
    res = []
    i = 0
    for elem in dataset:
        elem['k_dist'] = coords[i]
        del elem['maps_order']
        res.append(elem)
        i += 1

    with open('%s.json' % "data", 'w') as fjson:
        ujson.dump(res, fjson, ensure_ascii=False, sort_keys=True, indent=4)


def filter(fil, data):
    idxs = toKeepIdx(fil)
    # print(len(order))
    # print(len(idxs))
    # print(len(data))
    for i in range(len(data)):
        data[i] = [data[i][j] for j in idxs]

    return data


def toKeepIdx(fil):
    res = []
    for elem in fil:
        res.append(order.index(elem))
    # print(len(res))
    return np.setdiff1d(range(len(order)), res)


def toFilterIdx(fil):
    res = []
    for elem in fil:
        res.append(order.index(elem))
    return res


def make_umap(fil=None):
    dat = to_map[:]
    if not fil is None and not fil == ['']:
        filter(fil, dat)
    res = make_proj(dat)
    return res


def make_colors():
    dat = to_map[:]
    nb = len(dat)
    res = [0] * len(dat[0])
    for i in range(nb):
        # print(dat[i])
        for w in range(len(dat[i])):
            res[w] += (dat[i][w])

    # print(res)
    res_len = len(res)

    for u in range(res_len):
        # print(u)
        # val = np.median(res[u])
        # print(val)
        val = res[u] / nb
        # if val < 20:
        # print(val)
        res[u] = val

    return res


def median(l):
    half = len(l) // 2
    l.sort()
    if not len(l) % 2:
        return (l[half - 1] + l[half]) / 2.0
    return l[half]


@app.route('/proj', methods=["POST"])
def projector():
    units = request.form['units'].split(",")
    # print(units)
    return ujson.dumps({"proj": make_umap(units)})


def formatK_dist(k_dist):
    res = []

    for k, v in k_dist.items():
        for i in range(len(v)):
            for j in range(len(v[i])):
                # print(v[i][j].tolist())
                res.append(np.median(v[i][j]))
    return res

#
# @app.route('/firstProj', methods=["GET"])
# def fproj():
#     return ujson.dumps({"proj": umaper.transform(to_map[:]).tolist()})


def toSliptDict(data):
    res = {}

    for elem in order:
        temp = elem.split("_")
        res[elem] = round(np.median(data[temp[0]][int(temp[1])][int(temp[2])]))

    # print(res[order[0]])

    return res


def AtttoSliptDict(data):
    res = {}

    for elem in order:
        temp = elem.split("_")

        res[elem] = data[temp[0]][int(temp[1])].squeeze()[int(temp[2])].cpu().numpy().tolist()

    return res


@app.route('/ask', methods=["POST"])
def ask():
    units = request.form['units'].split(",")
    question = request.form['question']
    image = request.form['image']
    head_mask = empty_mask()

    if units is not None and not units == ['']:
        for elem in units:
            temp = elem.split("_")
            # print(temp)
            head_mask[temp[0]][int(temp[1])][int(temp[2])] = 1

    # head_mask['vis'] += 1
    # head_mask['vv'] += 1
    # head_mask['ll'] += 1
    # head_mask['lv'] += 1

    top_prediction, five_predictions, attention_heads, alignment, k_dist, input_labels = my_demo.ask(question, image,
                                                                                                     head_mask)

    print(input_labels)

    k_vals = toSliptDict(k_dist)

    five = {}
    for u in range(len(five_predictions)):
        five[five_predictions[u][0]] = five_predictions[u][1].item()

    print(five)

    for k, v in alignment.items():
        alignment[k]["xywh"] = alignment[k]["xywh"].tolist()

    return ujson.dumps({"pred": top_prediction[0],
                        "confidence": top_prediction[1].item(),
                        "alignment": alignment,
                        "coords": umaper.transform([formatK_dist(k_dist)]).tolist(),
                        "k_dist": k_vals,
                        "five": five,
                        "labels": input_labels,
                        "heatmaps": AtttoSliptDict(attention_heads)
                        })


def getfilext(path):
    files = []
    file = [".pickle"]
    fl = os.listdir(path)
    for f in fl:
        ext = os.path.splitext(f)[1]
        if ext.lower() not in file:
            continue
        files.append(f)

    return files



if __name__ == '__main__':
    # * Display config

    display_k_dist = True
    compact_k_dist = True  # compact=False will display all the k ditribution
    display_alignment = False

    # * /

    my_demo.load_data()
    my_demo.load_model()
    # imgs = list(my_demo.data_loader.img_dst) # FOR The rest

    # imgs = getfilext("model/gqa_testdev_obj36/oracle_data")  # FOR ORACLE
    #
    # imgs = np.random.choice(imgs, 300, replace=False)
    #
    # for i in range(len(imgs)):
    #     imgs[i] = imgs[i].replace(".pickle", "")
    #     copyfile("model/images/" + (imgs[i]) + ".jpg",
    #              "static/assets/images/oracle/" + (imgs[i]) + ".jpg")
    # print(imgs[0])
    #
    # with open('%s.json' % "images_oracle", 'w') as fjson:
    #     ujson.dump({"images": list(imgs)}, fjson, ensure_ascii=False, sort_keys=True, indent=4)

    app.run(host='0.0.0.0', port=5000, debug=True)
    # temp = make_colors()
    # res = {}

    # for i in range(len(temp)):
    #     res[order[i]] = temp[i]
    #
    # with open('%s.json' % "k_median", 'w') as fjson:
    #     ujson.dump(res, fjson, ensure_ascii=False, sort_keys=True, indent=4)

    # saver()
