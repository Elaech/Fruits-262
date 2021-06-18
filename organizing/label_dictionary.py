import json
import os


def get_all_subfolders(path=None):
    if path:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    return None


def build_direct_and_reversed_dictionary(path):
    labels = [os.path.basename(x) for x in get_all_subfolders(path)]
    direct = {x: counter for counter, x in enumerate(labels)}
    inverse = {counter: x for counter, x in enumerate(labels)}
    return direct, inverse


if __name__ == '__main__':
    direct, inverse = build_direct_and_reversed_dictionary("../ResizedDataset/26x32")
    with open("../Models&Learning/paths/label_to_number_dict.json", "w+") as o_file:
        json.dump(direct, o_file)
    with open("../Models&Learning/paths/number_to_label_dict.json", "w+") as o_file:
        json.dump(inverse, o_file)
