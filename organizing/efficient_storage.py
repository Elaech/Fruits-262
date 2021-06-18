import numpy as np
import os
import json
import PIL.Image as Image
import time
import sys

label_to_number_dict = None
number_to_label_dict = None


def load_dictionaries():
    """
    Loads the dictionaries that map the name of the fruit to a number and vice versa
    """
    global label_to_number_dict
    global number_to_label_dict
    with open("labels/label_to_number_dict.json", "r") as f:
        label_to_number_dict = json.load(f)
    with open("labels/number_to_label_dict.json", "r") as f:
        number_to_label_dict = json.load(f)


def get_label(path_to_image):
    """
    :param path_to_image: path to the selected image
    :return: the number representing the label for the said image
    """
    return label_to_number_dict[os.path.basename(os.path.dirname(path_to_image))]


def build_dataset_parts(json_path, save_path, part_max_size):
    """
    Builds numpy arrays of the images and their labels
    Stores them on the HDD/SSD/etc in multiple parts as npz format
    Stores the images and labels separately but in the same order
    At the end stores the last part of data even it is smaller than the rest
    :param json_path: path to a json file containing
    :param save_path: path to save the npz files
    :param part_max_size: max bytes (on storage device) for each part
    """
    part_index = 0
    with open(json_path, "r") as input:
        image_paths = json.load(input)
    features = []
    labels = []
    current_mem_size = 0
    for index in range(len(image_paths)):
        # Print Progress
        if index % 1000 == 0:
            print(f"Processed images up to index {index}")
            print(current_mem_size, " bytes")

        # Append to the current part
        img = np.array(Image.open(image_paths[index]))
        features.append(img)
        labels.append(get_label(image_paths[index]))
        current_mem_size += sys.getsizeof(img)

        # Save current part to disk and start a new one
        if current_mem_size > part_max_size:
            print(f"Saving to disk part {part_index} of {current_mem_size} bytes")
            features = np.asarray(features)
            labels = np.asarray(labels)
            np.savez_compressed(save_path + '/features' + str(part_index) + ".npz", features)
            np.savez_compressed(save_path + '/labels' + str(part_index) + ".npz", labels)
            part_index += 1
            del features
            del labels
            features = []
            labels = []
            current_mem_size = 0

    # If some data remains unsaved at the end then save it as the last part
    if features[0] is not None:
        np.savez_compressed(save_path + '/features' + str(part_index) + ".npz", features)
        np.savez_compressed(save_path + '/labels' + str(part_index) + ".npz", labels)


if __name__ == '__main__':
    load_dictionaries()
    # Image Dimensions
    WIDTH = 208
    HEIGHT = 256
    # Maximum Storage Bytes per Part
    MAX_BYTES = 2147483648
    x = time.time()
    build_dataset_parts(f"paths/{WIDTH}x{HEIGHT}/train.json",
                        f"../DatasetBinaryStorage/{WIDTH}x{HEIGHT}/train",
                        MAX_BYTES)
    print(time.time() - x)
