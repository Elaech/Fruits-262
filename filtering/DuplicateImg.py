import glob
import os
from PIL import Image
import hashlib


def get_image_list(path):
    return glob.glob(path + '/**/*.jpg', recursive=True)


def get_dict_image_size(list_of_images):
    img_size_dict = dict()
    for img_path in list_of_images:
        img = Image.open(img_path)
        width, height = img.size
        img.close()
        index = str((height, width))
        if index not in img_size_dict:
            img_size_dict[index] = [img_path]
        else:
            img_size_dict[index].append(img_path)
    return img_size_dict


def duplicates_from(list_of_images):
    output = []
    hashes = []
    for img_path in list_of_images:
        img_hash = hashlib.md5(open(img_path, "rb").read()).digest()
        if img_hash in hashes:
            output.append(img_path)
        else:
            hashes.append(img_hash)
    return output


def delete_duplicates(sized_img_dict):
    for dim_category in sized_img_dict:
        img_list = sized_img_dict[dim_category]
        duplicates = duplicates_from(img_list)
        for duplicate in duplicates:
            os.remove(duplicate)


if __name__ == '__main__':
    # recursively get all images from label
    images = get_image_list('../bottle gourd')
    # create image dictionary from label directory
    img_dict = get_dict_image_size(images)
    # delete duplicates from dictionary from each category
    delete_duplicates(img_dict)
