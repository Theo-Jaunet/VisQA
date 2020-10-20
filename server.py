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

mod = [("lang", 9, 4), ("vis", 5, 4), ("vl", 5, 4), ("lv", 5, 4), ("vv", 5, 4), ("ll", 5, 4)]


def makeOrder(layout):
    res = []

    for block in layout:
        for layer in range(block[1]):
            for head in range(block[2]):
                res.append(block[0] + "_" + str(layer) + "_" + str(head))

    return res


order = makeOrder(mod)


# print(order)

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

to_map = select(dataset)
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


@app.route('/firstProj', methods=["GET"])
def fproj():
    return ujson.dumps({"proj": [[0, 0]]})
    # return ujson.dumps({"proj": umaper.transform(to_map[:]).tolist()})


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
                        # "coords": umaper.transform([formatK_dist(k_dist)]).tolist(),
                        "coords": [0, 0],
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
