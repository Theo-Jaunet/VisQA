import binascii
import pickle

import ujson as ujson
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
my_demo.load_model()

def makeOrder(layout):
    res = []

    for block in layout:
        for layer in range(block[1]):
            for head in range(block[2]):
                res.append(block[0] + "_" + str(layer) + "_" + str(head))

    return res


order = makeOrder(mod)


@app.route('/')
def index():
    return render_template("index.html")


def toSliptDict(data):
    res = {}
    global order
    # print(data['lang'][0][0])
    for elem in order:
        temp = elem.split("_")
        res[elem] = round(np.median(data[temp[0]][int(temp[1])][int(temp[2])]))

    return res


def AtttoSliptDict(data):
    res = {}
    global order
    for elem in order:
        temp = elem.split("_")
        res[elem] = data[temp[0]][int(temp[1])].squeeze()[int(temp[2])].cpu().numpy().tolist()

    return res


def purgeHeats(data, size):
    global order
    for elem in order:
        temp = elem.split("_")
        tres = []
        if temp[0] == "lang" or temp[0] == "ll":
            tres = [x[:size["textual"]] for x in data[elem][:size["textual"]]]
        elif temp[0] == "vis" or temp[0] == "vv":
            tres = [x[:size["visual"]] for x in data[elem][:size["visual"]]]
        elif temp[0] == "vl":
            tres = [x[:size["visual"]] for x in data[elem][:size["textual"]]]
        elif temp[0] == "lv":
            tres = [x[:size["textual"]] for x in data[elem][:size["visual"]]]
        data[elem] = tres
    return data


@app.route('/ask', methods=["POST"])
def ask():
    global my_demo
    units = request.form['units'].split(",")
    question = request.form['question']
    image = request.form['image']
    head_mask = empty_mask()

    if units is not None and not units == ['']:
        for elem in units:
            temp = elem.split("_")
            # print(temp)
            head_mask[temp[0]][int(temp[1])][int(temp[2])] = 1

    top_prediction, five_predictions, attention_heads, alignment, k_dist, input_labels, input_size \
        = my_demo.ask(question, image, head_mask)
    k_vals = toSliptDict(k_dist)

    five = {}
    for u in range(len(five_predictions)):
        five[five_predictions[u][0]] = five_predictions[u][1].item()

    for k, v in alignment.items():
        alignment[k]["xywh"] = alignment[k]["xywh"].tolist()
    return ujson.dumps({
        "k_dist": k_vals,
        "five": five,
        "labels": input_labels,
        "heatmaps": purgeHeats(AtttoSliptDict(attention_heads), input_size)
    })


@app.route('/switchMod', methods=["POST"])
def switchMod():
    global order
    global my_demo

    dataName = request.form['name']
    mod = ujson.loads(request.form['mod'])
    disp = request.form['disp']
    type = request.form['type']
    my_demo.cfg['type'] = type

    print("DISP:", disp)

    if disp in ['lxmert_tiny', 'lxmert_tiny_init_oracle_pretrain', 'lxmert_tiny_init_oracle_scratch']:
        my_demo.cfg["tiny_lxmert"] = 1
        my_demo.cfg["oracle"] = 0
    elif disp in ['lxmert']:
        my_demo.cfg["tiny_lxmert"] = 0
        my_demo.cfg["oracle"] = 0
    elif disp in ['tiny_oracle']:
        my_demo.cfg["tiny_lxmert"] = 1
        my_demo.cfg["oracle"] = 1

    # if mod["head"] == 4:
    #     my_demo.cfg["tiny_lxmert"] = 1
    # else:
    #     my_demo.cfg["tiny_lxmert"] = 0

    if dataName == "oracle":
        # my_demo.cfg["oracle"] = 1
        my_demo.cfg['data_split'] = 'val'
    else:
        # my_demo.cfg["oracle"] = 0
        my_demo.cfg['data_split'] = 'testdev'

    order = makeOrder(
        [("lang", mod["lang"], mod["head"]), ("vis", mod["vis"], mod["head"]), ("vl", mod["cross"], mod["head"]),
         ("lv", mod["cross"], mod["head"]),
         ("vv", mod["cross"], mod["head"]), ("ll", mod["cross"], mod["head"])])

    my_demo.initConf("model/src/pretrain/" + disp)

    #
    if not dataName == "oracle":
        my_demo.load_data()
    # my_demo.load_data()
    my_demo.load_model()

    return 'ok'

if __name__ == '__main__':

    # my_demo.load_data()

    # my_demo.load_model()

    # stackDat()
    app.run(host='0.0.0.0', port=5000, debug=False)


