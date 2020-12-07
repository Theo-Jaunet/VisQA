import ujson as ujson
import numpy as np
from flask import Flask, render_template, request, session, redirect, logging, jsonify, Response

import sys

sys.path.insert(1, 'model/src/tasks')
sys.path.insert(1, 'model/src/')
sys.path.insert(1, 'model')
from model.src.tasks.demo import Demo, empty_mask

# app.secret_key = binascii.hexlify(os.urandom(24))
#
# COMPRESS_MIMETYPES = ['text/html', 'text/css', 'text/csv', 'text/xml', 'application/json',
#                       'application/javascript', 'image/jpeg', 'image/png']
# COMPRESS_LEVEL = 6
# COMPRESS_MIN_SIZE = 500
#
# cache = Cache(config={'CACHE_TYPE': 'simple'})
# cache.init_app(app)
# Compress(app)

my_demo = Demo("tiny_oracle")
print("MODEL 1 LOADED !!!")
my_demo2 = Demo("lxmert_tiny")
print("MODEL 2 LOADED !!!")
my_demo3 = Demo("lxmert_tiny_init_oracle_pretrain")
print("MODEL 3 LOADED !!!")

mod = [("lang", 9, 4), ("vis", 5, 4), ("vl", 5, 4), ("lv", 5, 4), ("vv", 5, 4), ("ll", 5, 4)]
# my_demo.load_model()
app = Flask(__name__)


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
        res[elem] = [
            round(np.min(data[temp[0]][int(temp[1])][int(temp[2])])),
            round(np.median(data[temp[0]][int(temp[1])][int(temp[2])])),
            round(np.max(data[temp[0]][int(temp[1])][int(temp[2])]))
        ]
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
    global my_demo2
    global my_demo3
    units = request.form['units'].split(",")
    question = request.form['question']
    image = request.form['image']
    disp = request.form['disp']
    head_mask = empty_mask()

    if units is not None and not units == ['']:
        for elem in units:
            temp = elem.split("_")
            # print(temp)
            head_mask[temp[0]][int(temp[1])][int(temp[2])] = 1

    if disp == "lxmert_tiny":
        five_predictions, attention_heads, alignment, k_dist, input_labels, input_size = my_demo2.ask(question, image,
                                                                                                      head_mask)

    elif disp == "lxmert_tiny_init_oracle_pretrain":
        five_predictions, attention_heads, alignment, k_dist, input_labels, input_size = my_demo3.ask(question, image,
                                                                                                      head_mask)
    else:
        five_predictions, attention_heads, alignment, k_dist, input_labels, input_size = my_demo.ask(question, image,
                                                                                                     head_mask)

    k_vals = toSliptDict(k_dist)

    five = {}
    for u in range(len(five_predictions)):
        five[five_predictions[u][0]] = five_predictions[u][1].item()

    heats = purgeHeats(AtttoSliptDict(attention_heads), input_size)

    resp = Response(response=ujson.dumps({
        "k_dist": k_vals,
        "five": five,
        "labels": input_labels,
        "heatmaps": heats
    }),
        status=200,
        mimetype="application/json")

    return resp


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


def RCNNStats():
    res = {"total": [[], []]}
    ok = []
    fail = []
    spoon = []
    with open("static/assets/data/info.json", 'r') as fjson:
        imgs = ujson.load(fjson)
        head_mask = empty_mask()
        item = "knife"
        for k, v in imgs.items():

            if hasItem(v["scene"]["objects"], item):
                five_predictions, attention_heads, alignment, k_dist, input_labels, input_size \
                    = my_demo2.ask(v["questions"]["0"]["question"], k, head_mask)

                if item in input_labels["visual"]:
                    ok.append(k)
                else:
                    if "spoon" in input_labels["visual"] or "fork" in input_labels["visual"]:
                        spoon.append(k)
                    else:
                        fail.append(k)

        # TODO CHECK if input labels has item
    print(ok)
    print(spoon)
    print(fail)
    pass


def hasItem(data, item):
    for k, v in data.items():
        if v["name"] == item:
            return True
    return False


def pruneStats():
    head_mask = empty_mask()
    # units = ["lang_4_0", "lang_6_1", "lang_6_0", "lang_5_0", "lang_3_3"]
    # units = ["lang_8_1", "lang_7_1", "lang_7_2", "lang_6_0", "lang_7_3", "lang_6_3", "lang_8_0"]
    # units = ["lv_0_0", "lv_0_1", "lv_0_2", "lv_0_3","lang_8_2","lang_8_3","lang_8_1","lang_8_0"]
    units = ["lang_8_0","lang_6_2","lang_6_3","lang_6_1","lang_6_0"]
    # units = ["vl_2_3", "lang_1_2", "lang_2_1", "lang_3_2", "lang_5_2", "lang_6_3", "lang_6_0", "lang_7_0", "lang_8_3",
    #          "lang_8_2", "lang_7_2", "lang_6_2", "lang_6_1", "lang_5_3", "lang_4_0", "lang_3_3", "lang_2_0", "lang_1_1",
    #          "ll_0_0"]
    head_mask2 = empty_mask()
    stats1 = [0, 0, 0]
    stats2 = [0, 0, 0]
    jf = [0, 0]
    jf2 = [0, 0]
    ref = ["H", "M", "T"]
    if units is not None and not units == ['']:
        for elem in units:
            temp = elem.split("_")
            # print(temp)
            head_mask[temp[0]][int(temp[1])][int(temp[2])] = 1

        with open("static/assets/data/info.json", 'r') as fjson:
            imgs = ujson.load(fjson)
            for k, v in imgs.items():

                for k2, v2 in v["questions"].items():
                    # print(v2)
                    if "and" in v2["operations"]:
                    # if "tail" == v2["ood"]: # "relate" in v2["operations"]:
                        # temp = v2["question"].split(" ")
                        #
                        # rof = temp.index("and")
                        # # print(temp)
                        # bob = temp[rof - 1]
                        #
                        # if "?" in temp[rof + 1]:
                        #     temp[rof - 1] = temp[rof + 1].replace("?","")
                        #     temp[rof + 1] = bob+"?"
                        # else:
                        #     temp[rof - 1] = temp[rof + 1]
                        #     temp[rof + 1] = bob


                        # v2["question"] =" ".join(temp)

                        five_predictions, attention_heads, alignment, k_dist, input_labels, input_size = my_demo3.ask(
                            v2["question"], k, head_mask2)

                        five_predictions2, _, _, _, _, _ = my_demo3.ask(
                            v2["question"], k, head_mask)

                        ood1 = getOod(v2, five_predictions[0][0])
                        ood2 = getOod(v2, five_predictions2[0][0])

                        if v2["answer"] == five_predictions[0][0]:
                            jf[0] += 1
                        else:
                            jf[1] += 1

                        if v2["answer"] == five_predictions2[0][0]:
                            jf2[0] += 1
                        else:
                            jf2[1] += 1

                        stats1[ref.index(ood1)] += 1
                        # if ood2 == "T":
                        #     print(five_predictions2)
                        #     print(k, " --- ", v2["question"], ' --- ', five_predictions2[0][0], " --conf-- ",
                        #           five_predictions2[0][1].item())
                        stats2[ref.index(ood2)] += 1
                    # print(k, "-- ", v2["question"], "-- GT", v2["answer"], "---", five_predictions2[0][0], "--", v2["answer"] == five_predictions2[0][0])

            print(stats1)
            print(stats2)

            print("--*-")
            print(jf)
            print(jf2)

            # five_predictions, attention_heads, alignment, k_dist, input_labels, input_size \
            #     = my_demo2.ask(v["questions"]["0"]["question"], k, head_mask)


def getOod(question, word):
    for elem in question["tail"]:
        if elem["ans"] == word:
            return "T"

    for elem in question["head"]:
        if elem["ans"] == word:
            return "H"

    return "M"


if __name__ == '__main__':
    # RCNNStats()
    # pruneStats()
    app.run(host='0.0.0.0', port=5000, debug=False)
