import binascii
import pickle

import ujson as ujson
import umap.umap_ as umap

from flask import Flask, render_template, request, session, redirect, logging, jsonify, Response
from flask_caching import Cache
from flask_compress import Compress
import os

app = Flask(__name__)

app.secret_key = binascii.hexlify(os.urandom(24))

COMPRESS_MIMETYPES = ['text/html', 'text/css', 'text/csv', 'text/xml', 'application/json',
                      'application/javascript', 'image/jpeg', 'image/png']
COMPRESS_LEVEL = 6
COMPRESS_MIN_SIZE = 500

cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(app)
Compress(app)


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


def make_proj(data):
    print('making Umap  ....')
    X_umap = umap.UMAP(n_neighbors=20, min_dist=0.3).fit_transform(data).tolist()

    return X_umap


@app.route('/')
def index():
    return render_template("index.html")


def make_umap(filter=None):
    if not filter is None:
        pass
    res = make_proj(to_map)
    return res


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




@app.route('/proj', methods=["POST"])
def projector():
    units = request.form['units'].split(",")
    print(units)
    return ujson.dumps({"proj": make_umap(units)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    # saver()