import glob
import os


def get_image_list(path):
    return glob.glob(path + '/**/*.jpg', recursive=True)


def move_images(image_list, destination):
    counter = 0
    for image in image_list:
        os.rename(image, destination + f'\\{counter}.jpg')
        counter += 1
    print(counter)

if __name__ == '__main__':
    move_images(get_image_list('../Dataset/data'), '../CleanedDataset/zucchini')
    # move_images(get_image_list('../CleanedDataset/african cherry orange'), '../Dataset/data/shutter')
