import copy
import json
import random
import threading
import time
import math
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
import PIL.Image as Image
import tensorflow.keras.layers as layers
import matplotlib.pyplot as plt

label_to_number_dict = None
number_to_label_dict = None


def load_dictionaries():
    global label_to_number_dict
    global number_to_label_dict
    with open("../Organizing/labels/label_to_number_dict.json", "r") as f:
        label_to_number_dict = json.load(f)
    with open("../Organizing/labels/number_to_label_dict.json", "r") as f:
        number_to_label_dict = json.load(f)


def load_test_data(train_pack_number, size_string):
    labels = np.load("../DatasetBinaryStorage/" + size_string
                     + "/test/labels" + str(train_pack_number) + ".npz")['arr_0']
    features = np.load("../DatasetBinaryStorage/" + size_string
                       + "/test/features" + str(train_pack_number) + ".npz")['arr_0']
    return features, labels


def mass_predict(model, images, labels, multiply_factor):
    res = np.zeros(shape=(len(images), len(number_to_label_dict)))
    print("Predicting...")

    for _ in range(multiply_factor):
        new_res = model.predict(images)
        res = np.sum([res, new_res], axis=0)
    res /= multiply_factor

    print("Mapping results...")
    res_and_labels = [
        [[labels[index_res], index, x]
         for index, x in enumerate(image_res)]
        for index_res, image_res in enumerate(res)]

    print("Sorting results...")
    for index in range(len(res_and_labels)):
        res_and_labels[index] = sorted(res_and_labels[index], key=lambda x: x[2], reverse=True)
    return res_and_labels


def statistics_for_dictionary(dictionary):
    ok = 0
    failed = 0
    for key, values in dictionary.items():
        ok += values[0]
        failed += values[1]
        print(f"{number_to_label_dict[str(key)]} : {round(values[0] / (values[0] + values[1]) * 100, 2)}%")
    print(f"Global accuracy: {round(ok / (ok + failed) * 100, 2)}%")


def generate_statistics(results):
    # Top 1
    top_one_acc = {label: [0, 0] for label in range(len(number_to_label_dict))}
    # Top 5
    top_five_acc = copy.deepcopy(top_one_acc)
    # Top 10
    top_ten_acc = copy.deepcopy(top_one_acc)

    # Count the wrong and right predicted images for each label for each top category
    for image_prediction in results:
        if image_prediction[0][0] == image_prediction[0][1]:
            top_one_acc[image_prediction[0][0]][0] += 1
            top_five_acc[image_prediction[0][0]][0] += 1
            top_ten_acc[image_prediction[0][0]][0] += 1
        elif image_prediction[0][0] in [image_prediction[i][1] for i in range(1, 5)]:
            top_one_acc[image_prediction[0][0]][1] += 1
            top_five_acc[image_prediction[0][0]][0] += 1
            top_ten_acc[image_prediction[0][0]][0] += 1
        elif image_prediction[0][0] in [image_prediction[i][1] for i in range(5, 10)]:
            top_one_acc[image_prediction[0][0]][1] += 1
            top_five_acc[image_prediction[0][0]][1] += 1
            top_ten_acc[image_prediction[0][0]][0] += 1
        else:
            top_one_acc[image_prediction[0][0]][1] += 1
            top_five_acc[image_prediction[0][0]][1] += 1
            top_ten_acc[image_prediction[0][0]][1] += 1

    # Print Statistics to console for each category
    print("\n\nTop 1 accuracy test:\n")
    statistics_for_dictionary(top_one_acc)

    print("\n\nTop 5 accuracy test:\n")
    statistics_for_dictionary(top_five_acc)

    print("\n\nTop 10 accuracy test:\n")
    statistics_for_dictionary(top_ten_acc)


if __name__ == '__main__':
    load_dictionaries()
    # HYPER PARAMETERS
    MODEL_ID = 19
    SUBJECT_ID = 2
    WIDTH = 52
    HEIGHT = 64
    CHANNELS = 3
    TEST_PACK_NUMBER = 0
    # How many times to put each image through the model
    MULTIPLY_FACTOR = 100

    # Paths and Names
    MODEL_NAME = f"new_gen_models/{WIDTH}x{HEIGHT}_ID{MODEL_ID}"
    size = f"{WIDTH}x{HEIGHT}"
    SUBJECT_NAME = f"subjects/{SUBJECT_ID}.jpg"

    # Load Model
    model = keras.models.load_model(MODEL_NAME)

    # Load Data
    test_features, test_labels = load_test_data(TEST_PACK_NUMBER, size)

    # Get Predictions
    results = mass_predict(model, test_features, test_labels, MULTIPLY_FACTOR)

    # Make Statistics
    generate_statistics(results)
