import numpy as np
import os
from PIL import Image


def get_all_subfolders(path=None):
    if path:
        return [f.path for f in os.scandir(path) if f.is_dir()]
    return None


def get_number_of_files(path=None):
    if path:
        return len([f for f in os.scandir(path) if f.is_file()])
    return None


def get_dataset_label_numbers(path=None):
    all_label_paths = get_all_subfolders(path)
    result_array = []
    for path in all_label_paths:
        result_array.append(get_number_of_files(path=path))
    return np.array(result_array)


def get_dataset_dimensions(path=None, label_numbers=None):
    all_label_paths = get_all_subfolders(path)
    dimensions = []
    for index, label_path in enumerate(all_label_paths):
        print(label_path)
        label_width = 0.0
        label_height = 0.0
        for image_file in os.scandir(label_path):
            if image_file.is_file():
                specimen_image = Image.open(image_file.path)
                label_width += specimen_image.size[1]
                label_height += specimen_image.size[0]
                specimen_image.close()
        dimensions.append([label_height / label_numbers[index], label_width / label_numbers[index]])
    return np.array(dimensions)


def labels_under_value(labels_numbers, value):
    result = []
    for el in labels_numbers:
        if el < value:
            result.append(el)
    if not result:
        result = [0.0]
    return np.array(result)


def get_statistics_of_dataset(path_to_dataset=None, goal_nr_per_label=None):
    """
    Produces statistics for the dataset (avgs, means, stds etc)
    Bellow are many examples for making statistics on the dataset
    :param path_to_dataset: path to the dataset (must have the same structure as presented)
    :param goal_nr_per_label: goal number of items per label
    """
    nr_specimens_per_fruit = get_dataset_label_numbers(path_to_dataset)
    print(np.sum(nr_specimens_per_fruit))
    label_width_x_height_values = get_dataset_dimensions(path_to_dataset, label_numbers=nr_specimens_per_fruit)
    label_height_values = np.array([x[0] for x in label_width_x_height_values])
    label_width_values = np.array([x[1] for x in label_width_x_height_values])
    labels_under = labels_under_value(nr_specimens_per_fruit, goal_nr_per_label)
    print("Total number of labels: ", len(nr_specimens_per_fruit))
    print("Target image number for label: ", goal_nr_per_label)
    print("Labels Under target value: ", len(labels_under) - 1 * (labels_under[0] == 0.0))
    print("Average missing images: ", np.average(labels_under))
    print("Median missing images: ", np.median(labels_under))
    print("Standard deviation of missing images: ", np.std(labels_under))
    print("Average nr of images per label: ", np.average(nr_specimens_per_fruit))
    print("Median nr of images per label: ", np.median(nr_specimens_per_fruit))
    print("Standard deviation of images per label: ", np.std(nr_specimens_per_fruit))
    print("Average width of image: ", np.average(label_width_values))
    print("Median width of image: ", np.median(label_width_values))
    print("Standard deviation of width of image: ", np.std(label_width_values))
    print("Average height of image: ", np.average(label_height_values))
    print("Median height of image: ", np.median(label_height_values))
    print("Standard deviation of height of image: ", np.std(label_height_values))


if __name__ == '__main__':
    get_statistics_of_dataset(path_to_dataset="HandFilteredDataset", goal_nr_per_label=1000)
