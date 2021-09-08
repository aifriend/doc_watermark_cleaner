import os.path
import random

from tqdm import tqdm

from common.header import *
from common.utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import cv2
import numpy as np


def visualize(_image0, _image1):
    try:
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(_image0, cmap="gray")
        axs[0].axis('off')
        axs[1].imshow(_image1, cmap="gray")
        axs[1].axis('off')
        plt.show()
    except Exception as e:
        print(f"{e}")


def load_data(max_sample):
    wm_image_list = os.listdir(f"{DATASET_PATH}/{DEGRADED_TRAIN_DATA}")
    gt_image_list = os.listdir(f"{DATASET_PATH}/{GT_TRAIN_DATA}")
    print(f"Load {max_sample}/{len(wm_image_list)} ...", end="")

    c_deg_image_list = list()
    c_clean_image_list = list()
    for im in range(len(wm_image_list)):
        if wm_image_list[im] == gt_image_list[im]:
            d_im = f"{DATASET_PATH}/{DEGRADED_TRAIN_DATA}/{wm_image_list[im]}"
            c_deg_image_list.append(d_im)
            c_im = d_im.replace("/wm/", "/gt/")
            c_clean_image_list.append(c_im)

    list_images = list(zip(c_deg_image_list, c_clean_image_list))
    random.shuffle(list_images)
    list_deg, list_clean = zip(*list_images)

    list_deg = list_deg[:max_sample]
    list_deg_im = list()
    for d_im in list_deg:
        # to gray scale for training
        deg_image = image_to_gray(d_im)
        # resize for training
        deg_image = cv2.resize(deg_image, DEFAULT_SHAPE)
        list_deg_im.append(deg_image)
        print("d", end="")

    list_clean = list_clean[:max_sample]
    list_clean_im = list()
    for c_im in list_clean:
        # to gray scale for training
        gt_image = image_to_gray(c_im)
        # resize for training
        gt_image = cv2.resize(gt_image, DEFAULT_SHAPE)
        list_clean_im.append(gt_image)
        print("g", end="")
    print()

    return list_deg_im, list_clean_im


def pipeline():
    try:
        list_deg_images, list_clean_images = load_data(10)

        loop = tqdm(range(len(list_deg_images)), leave=True, position=0)
        for imd in loop:
            deg_image = list_deg_images[imd]
            clean_image = list_clean_images[imd]

            clean_image = clean_image.astype(np.uint8)
            clean_image = cv2.equalizeHist(clean_image)
            image_file = f"_noisy.png"
            clean_image.save('data/train/gt/' + image_file)

            visualize(deg_image, clean_image)

    except Exception as e:
        print(f"Pipeline exception: {e}")


def run():
    for file in ClassFile.list_files(RESULT_PATH):
        os.remove(file)

    pipeline()


if __name__ == '__main__':
    DATASET_PATH = f"./data/data_dn"
    run()
