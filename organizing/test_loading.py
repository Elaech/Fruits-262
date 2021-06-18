import numpy as np
import random
import os
import json
import PIL.Image as Image
import time
import copy
import sys

if __name__ == '__main__':
    all_size = ["13x16", "26x32", "52x64", "104x128", "208x256"]
    for size in all_size:
        x = time.time()
        loaded_training_labels = np.load("../DatasetBinaryStorage/" + size + "/train/labels0.npz")
        loaded_training_features = np.load("../DatasetBinaryStorage/" + size + "/train/features0.npz")
        loaded_validation_labels = np.load("../DatasetBinaryStorage/" + size + "/validate/labels0.npz")
        loaded_validation_features = np.load("../DatasetBinaryStorage/" + size + "/validate/features0.npz")
        loaded_training_features = loaded_training_features['arr_0']
        loaded_training_labels = loaded_training_labels['arr_0']
        loaded_validation_features = loaded_validation_features['arr_0']
        loaded_validation_labels = loaded_validation_labels['arr_0']
        print(size, " - ", time.time() - x)
        time.sleep(5)
