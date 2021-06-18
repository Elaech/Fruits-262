import glob
import PIL
from PIL import Image
from PIL import ImageFile

def get_image_list(path):
    return glob.glob(path + '/**/*.jpg', recursive=True)


def remove_watermark_shutter(list_of_images):
    for img_path in list_of_images:
        try:
            Image.open(img_path).convert('RGB').save(img_path)
            img = Image.open(img_path)
            w,h = img.size
            img = img.crop((0, 0, w, h-20))
            img.save(img_path)
        except PIL.UnidentifiedImageError:
            pass

if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    images = get_image_list("../Dataset/data/shutter")
    remove_watermark_shutter(images)
