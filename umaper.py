import os
import _pickle as pickle
import json
import numpy


def convert_dict_to_json(file_path):
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
        data = pickle.load(fpkl)
        res = []
        print(len(data))
        for i in range(len(data)):
            res.append(formatLine(data[i]))
        print(res[0])
        print(res[1])
        print(res[2])
        print(res[3])
        json.dump(res, fjson, ensure_ascii=False, sort_keys=True, indent=4)


def formatLine(data):
    k_dist = []
    for i in range(len(data["k_dist"])):
        k_dist.append(int(round(data["k_dist"][i])))
    data["k_dist"] = k_dist
    # print(data)
    del data["maps_order"]
    return data


def main(filename):
    if filename and os.path.isfile(filename):
        file_path = filename
        print("Processing %s ..." % file_path)
        convert_dict_to_json(file_path)
    else:
        print("Usage: %s abs_file_path" % (__file__))


if __name__ == '__main__':
    main("lxmert_gqaval_reasbias.pickle")
