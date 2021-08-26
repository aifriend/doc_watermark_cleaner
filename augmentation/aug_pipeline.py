import os.path
import random
import shutil

import albumentations as A
import cv2
from matplotlib import pyplot as plt

from common.ClassFile import ClassFile


def visualize(_image):
    try:
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(_image)
        plt.pause(1)
    except:
        raise IOError()


def load(path, doc=None, ext="jpg"):
    _image_list = list()
    if doc is None:
        image_file_list = ClassFile.list_files_like(path, f".{ext}")
        for image_file in image_file_list:
            _image = cv2.imread(image_file)
            if _image is not None:
                _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
                _image_list.append((image_file, _image))
                print(".", end="")
            else:
                print("X", end="")
    else:
        _image = cv2.imread(os.path.join(path, doc))
        if _image is not None:
            _image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
            _image_list.append((doc, _image))
            print(".", end="")
        else:
            print("X", end="")

    return _image_list


def transform(file, image, count):
    _transform = A.Compose([
        A.HorizontalFlip(p=0.03),
        A.Rotate(limit=-5, p=0.3),
        A.GaussNoise(),
        A.MotionBlur(p=0.3),
        A.MedianBlur(blur_limit=5, p=0.3),
        A.Blur(blur_limit=5, p=0.2),
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
    ])
    random.seed(42)
    augmented_image = list()
    augmented_name = list()
    for _ in range(1, count):
        augmented_name.append(file)
        augmented_image.append(_transform(image=image)['image'])
    return augmented_name, augmented_image


DATA_SOURCE = "../data/invoice2"


def main():
    if os.path.exists(os.path.join(DATA_SOURCE, "../data/train")):
        shutil.rmtree(os.path.join(DATA_SOURCE, "../data/train"))

    f_count = 3
    f_ext = "jpg"
    image_list = load(DATA_SOURCE, ext=f_ext)
    for f_idx, file, image in enumerate(image_list):
        print(f"\nProcessing {f_idx}/{len(image_list)}: {file}\n")

        # original image
        f_path, f_name = os.path.split(file)
        f_path = f_path.replace("MOD_2", "train/MOD_2")
        f_path = f_path.replace("OTROS", "train/OTROS")
        os.makedirs(f_path, exist_ok=True)
        cv2.imwrite(f"{f_path}/{f_name}", image)

        # augmented image
        augmented_name_list, augmented_image_list = \
            transform(file=file, image=image, count=f_count)
        # visualize(augmented_image_list[0])
        print("A", end="")
        for ag_idx, (augmented_name, augmented_image) in \
                enumerate(zip(augmented_name_list, augmented_image_list)):
            f_path, f_name = os.path.split(augmented_name)
            f_path = f_path.replace("MOD_2", "train/MOD_2")
            f_path = f_path.replace("OTROS", "train/OTROS")
            f_name, _ = os.path.splitext(f_name)
            f_name = f"{f_name}_aug{ag_idx}.{f_ext}"
            cv2.imwrite(f"{f_path}/{f_name}", augmented_image)
            print("a", end="")


if __name__ == '__main__':
    main()
