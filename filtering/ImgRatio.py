import glob
import os
from PIL import Image


def get_image_list(path):
    return glob.glob(path + '/**/*.jpg', recursive=True)


def delete_img_with_low_dim(list_of_img):
    for img_path in list_of_img:
        img = Image.open(img_path)
        width, height = img.size
        img.close()
        if width < height:
            bigger_dim = height
            smaller_dim = width
        else:
            smaller_dim = height
            bigger_dim = width
        if (smaller_dim * 1.0) / bigger_dim < 0.5:
            print(img_path)


if __name__ == '__main__':
    # recursively get all images from label
    images = get_image_list('../Dataset/apple')
    delete_img_with_low_dim(images)
