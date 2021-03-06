import binascii
import pickle
import ujson as ujson
import numpy as np
from flask import Flask, render_template, request
from flask_caching import Cache
from flask_compress import Compress
import os
import sys


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
my_demo = Demo("lxmert_full_12heads_768hdims")
# my_demo = Demo("tiny_oracle")
# my_demo = Demo("tiny_oracle")
# my_demo = Demo("lxmert_tiny")

# mod = [("lang", 9, 4), ("vis", 5, 4), ("vl", 5, 4), ("lv", 5, 4), ("vv", 5, 4), ("ll", 5, 4)]
mod = [("lang", 9, 12), ("vis", 5, 12), ("vl", 5, 12), ("lv", 5, 12), ("vv", 5, 12), ("ll", 5, 12)]

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


# print(dataset[0])


# to_map = select(dataset)
# print(to_map[0])
# print('making Umap  ....')

# umaper = umap.UMAP(n_neighbors=20, min_dist=0.3).fit(to_map[:])


def stater(mod, name, disp):
    global order
    global my_demo

    if mod["head"] == 4:
        my_demo.cfg["tiny_lxmert"] = 1
    else:
        my_demo.cfg["tiny_lxmert"] = 0

    if name == "oracle":
        my_demo.cfg["oracle"] = 1
        my_demo.cfg['data_split'] = 'val'
    else:
        my_demo.cfg["oracle"] = 0
        my_demo.cfg['data_split'] = 'testdev'

    order = makeOrder(
        [("lang", mod["lang"], mod["head"]), ("vis", mod["vis"], mod["head"]), ("vl", mod["cross"], mod["head"]),
         ("lv", mod["cross"], mod["head"]),
         ("vv", mod["cross"], mod["head"]), ("ll", mod["cross"], mod["head"])])

    # print(args)

    my_demo.initConf("model/src/pretrain/" + disp)

    my_demo.load_data()
    my_demo.load_model()

    res = {}

    for elem in order:
        res[elem] = {"functs": {}, "groups": {
        }
                     }

    with open("static/assets/data/images.json", 'r') as fjson:
        imgs = ujson.load(fjson)

        for im in range(len(imgs["default"])):
            five_predictions, attention_heads, alignment, k_dist, input_labels, input_size \
                = my_demo.ask(dataset[im].question, imgs["default"][im], empty_mask(12))
            temp = toSliptD(k_dist)

            for elem in order:
                for func in dataset[im]["operations"]:  # here change
                    if res[elem]["functs"][func] is not None:
                        res[elem]["functs"][func]["median"].append(temp[0])
                        res[elem]["functs"][func]["min"].append(temp[1])
                        res[elem]["functs"][func]["max"].append(temp[2])
                    else:
                        res[elem]["functs"][func]["median"] = [temp[0]]
                        res[elem]["functs"][func]["min"] = [temp[1]]
                        res[elem]["functs"][func]["max"] = [temp[2]]

    with open('%s.json' % "info2", 'w') as wjson:
        ujson.dump(res, wjson, ensure_ascii=False, sort_keys=True, indent=4)


def make_proj(data):
    # print('making Umap  ....')
    X_umap = umap.UMAP(n_neighbors=20, min_dist=0.3).fit_transform(data).tolist()

    return X_umap


@app.route('/')
def index():
    return render_template("index.html")


#
# def saver():
#     coords = make_umap([])
#     res = []
#     i = 0
#     for elem in dataset:
#         elem['k_dist'] = coords[i]
#         del elem['maps_order']
#         res.append(elem)
#         i += 1
#
#     with open('%s.json' % "data", 'w') as fjson:
#         ujson.dump(res, fjson, ensure_ascii=False, sort_keys=True, indent=4)


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


# def make_umap(fil=None):
#     dat = to_map[:]
#     if not fil is None and not fil == ['']:
#         filter(fil, dat)
#     res = make_proj(dat)
#     return res
#
#
# def make_colors():
#     dat = to_map[:]
#     nb = len(dat)
#     res = [0] * len(dat[0])
#     for i in range(nb):
#         # print(dat[i])
#         for w in range(len(dat[i])):
#             res[w] += (dat[i][w])
#
#     # print(res)
#     res_len = len(res)
#
#     for u in range(res_len):
#         # print(u)
#         # val = np.median(res[u])
#         # print(val)
#         val = res[u] / nb
#         # if val < 20:
#         # print(val)
#         res[u] = val
#
#     return res


def median(l):
    half = len(l) // 2
    l.sort()
    if not len(l) % 2:
        return (l[half - 1] + l[half]) / 2.0
    return l[half]


# @app.route('/proj', methods=["POST"])
# def projector():
#     units = request.form['units'].split(",")
#     # print(units)
#     return ujson.dumps({"proj": make_umap(units)})


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


def toSliptD(data):
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
    units = request.form['units'].split(",")
    question = request.form['question']
    image = request.form['image']
    head_mask = empty_mask(12)

    if units is not None and not units == ['']:
        for elem in units:
            temp = elem.split("_")
            # print(temp)
            head_mask[temp[0]][int(temp[1])][int(temp[2])] = 1

    five_predictions, attention_heads, alignment, k_dist, input_labels, input_size \
        = my_demo.ask(question, image, head_mask)

    # print(input_size["textual"])
    k_vals = toSliptDict(k_dist)

    five = {}
    for u in range(len(five_predictions)):
        five[five_predictions[u][0]] = five_predictions[u][1].item()

    # print(five)

    for k, v in alignment.items():
        alignment[k]["xywh"] = alignment[k]["xywh"].tolist()

    # print(input_labels)
    # print(image)
    return ujson.dumps({
        # "pred": top_prediction[0],
        # "confidence": top_prediction[1].item(),
        "alignment": alignment,
        # "coords": umaper.transform([formatK_dist(k_dist)]).tolist(),
        "k_dist": k_vals,
        "five": five,
        "labels": input_labels,
        "heatmaps": purgeHeats(AtttoSliptDict(attention_heads), input_size)
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


def merger():
    res = {}
    with open("static/assets/data/images.json", 'r') as fjson:
        imgs = ujson.load(fjson)

        with open("/home/theo/Downloads/orac/sceneGraphs/val_sceneGraphs.json", 'r') as fjson2:
            scene = ujson.load(fjson2)

            with open("val_all_tail-0.30_head0.20.json", 'r') as fjson3:
                # with open("/home/theo/Downloads/orac/questions1.2/val_balanced_questions.json", 'r') as fjson3:
                quest = ujson.load(fjson3)

                for im in imgs["oracle"]:
                    temp = getQuests(quest, im)
                    if not temp == {}:
                        score, idmin, idmax = makeScore(temp)
                        res[im] = {"questions": temp, "scene": scene[im], "score": score,
                                   "ids": {"min": idmin, "max": idmax}}

                with open('%s.json' % "info", 'w') as wjson:
                    ujson.dump(res, wjson, ensure_ascii=False, sort_keys=True, indent=4)


def makeScore(dat):
    temp = 0
    idmin = ""
    idmax = ""
    ct = 0
    for key, elem in dat.items():
        if elem["ood"] == "tail":
            temp += 1
            if idmax == "":
                idmax = elem["questionId"]
        if idmin == "" and elem["ood"] == "head":
            idmin = elem["questionId"]
        ct += 1
    return temp / ct, idmin, idmax


def merger2():
    res = {}
    with open("static/assets/data/images.json", 'r') as fjson:
        imgs = ujson.load(fjson)

        with open("/home/theo/Downloads/orac/questions1.2/testdev_balanced_questions.json", 'r') as fjson3:
            quest = ujson.load(fjson3)

            # print(quest)
            for im in imgs["default"]:
                print(im)
                res[im] = {"questions": getQuests(quest, im)}
                break
            with open('%s.json' % "info2", 'w') as wjson:
                ujson.dump(res, wjson, ensure_ascii=False, sort_keys=True, indent=4)


def getQuests(data, id):
    res = {}
    i = 0
    for line in data:
        if data[line]['imageId'] == id:
            # print(line)
            # print(data[line])
            # print("---")
            res[i] = formatLine(data[line], line)
            i += 1
    return res


def formatLine(line, id):
    res = {"head": line["ans_head"], "tail": line["ans_tail"], "imageId": line["imageId"], "groups": line["groups"],
           "functions": line["types"]["detailed"], "answer": line["answer"], "question": line["question"]}

    temp = "middle"

    for elem in line["ans_head"]:
        if line["answer"] == elem["ans"]:
            temp = "head"
            break

    for elem in line["ans_tail"]:
        if line["answer"] == elem["ans"]:
            temp = "tail"
            break

    opes = []
    for elem in line['semantic']:
        opes.append(elem['operation'])

    res["ood"] = temp
    res["questionId"] = id
    res["operations"] = opes
    return res


@app.route('/switchMod', methods=["POST"])
def switchMod():
    global order
    global my_demo

    dataName = request.form['name']
    mod = ujson.loads(request.form['mod'])
    disp = request.form['disp']
    type = request.form['type']
    my_demo.cfg['type'] = type

    # print("DISP:", disp)

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


def stackDat():
    with open("static/assets/data/info.json", "r") as datFile:
        data = ujson.load(datFile)
        res = {}
        head_mask = empty_mask(12)
        for elem in order:
            res[elem] = {"functions": {}, "groups": {}, "kmeds": [[], [], []]}
        # skip = [51,54,101,138,151,204,206]
        img = 0
        for k, v in data.items():
            print("doing img ..", img, " with id:", k)
            # if img in skip:
            #     print("SKIPPED")
            #     img += 1
            #     continue
            for k2, v2 in v["questions"].items():
                five_predictions, attention_heads, alignment, k_dist, input_labels, input_size \
                    = my_demo.ask(v2["question"], k, head_mask)

                k_vals = toSliptDict(k_dist)

                for k3, v3 in k_vals.items():
                    for op in v2["operations"]:

                        if not op in res[k3]["functions"]:
                            res[k3]["functions"][op] = [k_split([0, 0, 0, 0], v3[0]),
                                                        k_split([0, 0, 0, 0], v3[1]),
                                                        k_split([0, 0, 0, 0], v3[2])]
                        else:
                            res[k3]["functions"][op][0] = k_split(res[k3]["functions"][op][0], v3[0])
                            res[k3]["functions"][op][1] = k_split(res[k3]["functions"][op][1], v3[1])
                            res[k3]["functions"][op][2] = k_split(res[k3]["functions"][op][2], v3[2])

                    if not v2["groups"]["global"] is None:
                        if not v2["groups"]["global"] in res[k3]["groups"]:
                            res[k3]["groups"][v2["groups"]["global"]] = [k_split([0, 0, 0, 0], v3[0]),
                                                                         k_split([0, 0, 0, 0], v3[1]),
                                                                         k_split([0, 0, 0, 0], v3[2])]
                        else:
                            tab = res[k3]["groups"][v2["groups"]["global"]]
                            tab[0] = k_split(tab[0], v3[0])
                            tab[0] = k_split(tab[1], v3[1])
                            tab[0] = k_split(tab[2], v3[1])

                    res[k3]["kmeds"][0].append(int(v3[0]))
                    res[k3]["kmeds"][1].append(int(v3[1]))
                    res[k3]["kmeds"][2].append(int(v3[2]))

                    # if not v2["functions"] in res[k3]["functions"]:
                    #     res[k3]["functions"][v2["functions"]] = k_split([0, 0, 0, 0], v3)
                    # else:
                    #     res[k3]["functions"][v2["functions"]] = k_split(res[k3]["functions"][v2["functions"]], v3)
                    #
                    # if not v2["groups"]["global"] is None:
                    #     if not v2["groups"]["global"] in res[k3]["groups"]:
                    #         res[k3]["groups"][v2["groups"]["global"]] = k_split([0, 0, 0, 0], v3)
                    #     else:
                    #         res[k3]["groups"][v2["groups"]["global"]] = k_split(
                    #             res[k3]["groups"][v2["groups"]["global"]], v3)
                    # res[k3]["kmeds"].append(int(v3))
            img += 1

        with open('%s.json' % "tiny_oracle_full", 'w') as wjson:
            ujson.dump(res, wjson, ensure_ascii=False, sort_keys=True, indent=4)


def k_split(array, val):
    if val < 12:
        array[0] += 1
        return array
    elif val < 25:
        array[1] += 1
        return array
    elif val < 50:
        array[2] += 1
        return array
    else:
        array[3] += 1
        return array


if __name__ == '__main__':
    # * Display config

    # display_k_dist = True
    # compact_k_dist = True  # compact=False will display all the k ditribution
    # display_alignment = False
    #
    # # * /
    #
    # my_demo.load_data()

    # my_demo.load_model()

    # stackDat()
    app.run(host='0.0.0.0', port=5000, debug=False)

    # with open("/home/theo/Downloads/val_all_tail0.20_head0.20.json", 'r') as fjson:
    #     imgs = ujson.load(fjson)
    #     print(imgs["001002646"])

    # merger()

    # merger2()

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

    # temp = make_colors()

    # res = {}

    # for i in range(len(temp)):
    #     res[order[i]] = temp[i]
    #
    # with open('%s.json' % "k_median", 'w') as fjson:
    #     ujson.dump(res, fjson, ensure_ascii=False, sort_keys=True, indent=4)

    # saver()
