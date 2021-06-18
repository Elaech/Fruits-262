import glob
import random
import os
import json


def build_dataset_from_directory(path, dimesions, base_path, destination):
    all_fruit = get_all_subfolders(path)
    for fruit in all_fruit:
        print(fruit)
        local_training, local_validation, local_testing = get_files_from_dir(fruit, dimesions, base_path)
        for index_dim, dim in enumerate(dimesions):
            with open(destination + f"/{dim[0]}x{dim[1]}/train.json", "a+") as out_file:
                for index_file in range(len(local_training[index_dim])):
                    out_file.write("\"" + local_training[index_dim][index_file] + "\",")
            with open(destination + f"/{dim[0]}x{dim[1]}/validate.json", "a+") as out_file:
                for index_file in range(len(local_validation[index_dim])):
                    out_file.write("\"" + local_validation[index_dim][index_file] + "\",")
            with open(destination + f"/{dim[0]}x{dim[1]}/test.json", "a+") as out_file:
                for index_file in range(len(local_testing[index_dim])):
                    out_file.write("\"" + local_testing[index_dim][index_file] + "\",")


def get_files_from_dir(path, dimesions, base_path):
    image_files = glob.glob(path + "/*jpg")
    label = os.path.basename(path)
    random.shuffle(image_files)
    training = (len(image_files) * 7) // 10
    validation = (len(image_files) - training) * 2 // 3
    testing = len(image_files) - training - validation
    output_training = [list(), list(), list(), list(), list()]
    for image in image_files[:training]:
        for index in range(len(dimesions)):
            output_training[index].append(
                base_path + f'/{dimesions[index][0]}x{dimesions[index][1]}' + '/' + label + '/' + os.path.basename(
                    image))
    output_validation = [list(), list(), list(), list(), list()]
    for image in image_files[training:(validation + training)]:
        for index in range(len(dimesions)):
            output_validation[index].append(
                base_path + f'/{dimesions[index][0]}x{dimesions[index][1]}' + '/' + label + '/' + os.path.basename(
                    image))
    output_testing = [list(), list(), list(), list(), list()]
    for image in image_files[(validation + training):(validation + training + testing)]:
        for index in range(len(dimesions)):
            output_testing[index].append(
                base_path + f'/{dimesions[index][0]}x{dimesions[index][1]}' + '/' + label + '/' + os.path.basename(
                    image))
    return output_training, output_validation, output_testing


def get_all_subfolders(path=None):
    if path:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    return None


def get_label(path_to_image):
    return os.path.basename(os.path.dirname(path_to_image))


def reduce_json_data_same_label_ratio(source, destination, ratio):
    label_dictionary = dict()
    with open(source, "r") as input:
        data = json.load(input)
    for elem in data:
        label = get_label(elem)
        if label in label_dictionary:
            label_dictionary[label].append(elem)
        else:
            label_dictionary[label] = [elem]
    with open(destination, "w+") as output:
        for label in label_dictionary:
            print(label)
            for elem in label_dictionary[label][:int(len(label_dictionary[label]) * ratio)]:
                output.write("\"" + elem + "\",")


if __name__ == '__main__':
    dims = [(13, 16), (26, 32), (52, 64), (104, 128), (208, 256)]
    build_dataset_from_directory("../ResizedDataset/26x32", dims, "../ResizedDataset", "paths")
    # reduce_json_data_same_label_ratio("../Models&Learning/paths/104x128/train.json",
    #                                   "../Models&Learning/paths/104x128/train_reduced.json", 0.5)
