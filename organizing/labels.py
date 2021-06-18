import json
import os


def update_label(name, number=0):
    labels = get_labels()
    labels[name] = number
    put_labels(labels)


def show_labels():
    labels = get_label_list()
    print(len(labels))
    for label in labels:
        print(label, " : ", labels[label])


def show_label_list():
    label_list = get_labels()
    for label in label_list:
        print(label, label_list[label])
    print(len(label_list))


def get_label_list():
    output = []
    labels = get_labels()
    for label in labels:
        output.append(label.lower())
    return sorted(list(set(output)))


def get_labels():
    with open("labels.txt", 'r') as input_file:
        return json.loads(input_file.read())


def put_labels(list_of_labels):
    with open("labels.txt", 'w') as input_file:
        input_file.write(json.dumps(list_of_labels))


def add_labels(list_of_labels):
    list_of_labels = set(sorted(list_of_labels))
    current_labels = get_label_list()
    for el in list_of_labels:
        current_labels[el] = 0
    put_labels(current_labels)


def check_label(label):
    label = label.lower()
    labels = get_label_list()
    if label in labels:
        return True
    return False


def remap_labels(label_list):
    out = dict()
    for label in label_list:
        out[label.lower()] = 0
    put_labels(out)


def get_unchecked_label():
    labels = get_labels()
    for label in labels:
        if labels[label] == 0:
            return label
    return -1


def check_keyword(keyword):
    for file in os.listdir(os.path.join('../Dataset')):
        if os.path.isdir(os.path.join('../Dataset', file)):
            return

        with open(os.path.join('../Dataset', file), 'r') as input_f:
            buff = input_f.read()
            if buff.find(keyword) != -1:
                print(file)


def get_all_label_list(path):
    labels = []
    for file in os.listdir(os.path.join(path)):
        if os.path.isdir(os.path.join(path, file)):
            labels.append(os.path.basename(file))
    return labels


if __name__ == '__main__':
    # update_label("zucchini",9764)
    # print(get_unchecked_label())
    # check_keyword('')
    # show_label_list()
    print(get_all_label_list("../HandFilteredDataset"))