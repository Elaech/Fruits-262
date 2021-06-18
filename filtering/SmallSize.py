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
        if width < 64 or height < 64:
            os.remove(img_path)
            return


if __name__ == '__main__':
    # recursively get all images from label
    images = get_image_list('../Dataset/data')
    delete_img_with_low_dim(images)