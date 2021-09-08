import glob
import os
import random

import cv2
import numpy as np
from PIL import Image

from common.header import DEFAULT_SHAPE

if not os.path.exists('data/train/wm'):
    os.makedirs('data/train/wm')
if not os.path.exists('data/train/gt'):
    os.makedirs('data/train/gt')
if not os.path.exists('data/val/wm'):
    os.makedirs('data/val/wm')
if not os.path.exists('data/val/gt'):
    os.makedirs('data/val/gt')

GENERATION_FACTOR = 1
MAX_DATA = 250
DOC_SOURCE = "source"


def blur(image, max_blur):
    img = cv2.resize(image, DEFAULT_SHAPE)
    blur_img = cv2.GaussianBlur(img, (max_blur, max_blur), 0)
    return blur_img


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    threshold = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > threshold:
                output[i][j] = random.randint(128, 255)
            else:
                output[i][j] = image[i][j]
    return output


def noisy(source, epoch, idx, value_set):
    im = Image.open(source)
    im = im.resize(DEFAULT_SHAPE).convert("L")

    clean_image = im.copy()
    background_image = im.copy()

    for aug_idx in range(GENERATION_FACTOR):
        background_image = im.copy()
        background_image_v = np.array(background_image)
        max_noise = random.uniform(0.0001, 0.1)
        image_v = sp_noise(background_image_v, max_noise)
        background_image = Image.fromarray(image_v)
        # background_image.show()
        image_file = f"{epoch}_{idx}_{aug_idx}_noisy.png"
        if not value_set:
            clean_image.save('data/train/gt/' + image_file)
            background_image.save('data/train/wm/' + image_file)
        else:
            clean_image.save('data/val/gt/' + image_file)
            background_image.save('data/val/wm/' + image_file)

        value_set = not value_set

    return value_set, background_image


def noisy_pipe(pipeline):
    background_image = pipeline.copy()

    for aug_idx in range(GENERATION_FACTOR):
        background_image = pipeline.copy()
        background_image_v = np.array(background_image)
        max_noise = random.uniform(0.001, 0.01)
        image_v = sp_noise(background_image_v, max_noise)
        background_image = Image.fromarray(image_v)

    return background_image


def blurring(source, pipeline, epoch, idx, value_set):
    im = Image.open(source)
    im = im.resize(DEFAULT_SHAPE).convert("L")

    clean_image = im.copy()
    background_image = pipeline.copy()

    for aug_idx in range(GENERATION_FACTOR):
        background_image = pipeline.copy()
        background_image_v = np.array(background_image)
        max_blur = random.choice([1, 3, 5])
        image_v = blur(background_image_v, max_blur)
        background_image = Image.fromarray(image_v)
        # background_image.show()
        image_file = f"{epoch}_{idx}_{aug_idx}_noisy_blur.png"
        if not value_set:
            clean_image.save('data/train/gt/' + image_file)
            background_image.save('data/train/wm/' + image_file)
        else:
            clean_image.save('data/val/gt/' + image_file)
            background_image.save('data/val/wm/' + image_file)

        value_set = not value_set

    return value_set, background_image


def run():
    list_source_images = glob.glob(f"{DOC_SOURCE}/*")
    list_images = list_source_images
    random.shuffle(list_images)

    n_value_set = False
    b_value_set = False
    epoch = "max_dn_bl"
    for i, image in enumerate(list_images):
        n_value_set, pipeline = noisy(image, epoch, i, n_value_set)
        b_value_set, pipeline = blurring(image, pipeline, epoch, i, b_value_set)
        print(f"LEFT {i}/{len(list_images)} <= {MAX_DATA}")
        if i >= MAX_DATA:
            break


if __name__ == '__main__':
    run()
