import glob
import os
import random

import numpy as np
from PIL import Image

if not os.path.exists('data/train/wm'):
    os.makedirs('data/train/wm')
if not os.path.exists('data/train/gt'):
    os.makedirs('data/train/gt')
if not os.path.exists('data/val/wm'):
    os.makedirs('data/val/wm')
if not os.path.exists('data/val/gt'):
    os.makedirs('data/val/gt')

GENERATION_FACTOR = 1
MAX_DATA = 50
SOURCE_SOURCE = "source"
DATA_SOURCE = "no_logo"
LOGO_SOURCE = "logo"


def sp_noise(image, prob):
    output = np.zeros(image.shape, np.uint8)
    threshold = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > threshold:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def noisy(source, idx, value_set):
    im = Image.open(source)
    im = im.resize((768, 1024)).convert("L")

    # clean image
    clean_image = im.copy()

    for aug_idx in range(GENERATION_FACTOR):
        noisy_image = im.copy()
        noise_image_v = np.array(noisy_image)
        #Image.fromarray(noise_image_v).show()

        max_noise = random.uniform(0.000001, 0.01)
        noise_image_v = sp_noise(noise_image_v, max_noise)
        noisy_image = Image.fromarray(noise_image_v)
        #noisy_image.show()

        image_file = f"t2_{idx}_{aug_idx}_noisy_gauss.png"
        if not value_set:
            clean_image.save('data/train/gt/' + image_file)
            noisy_image.save('data/train/wm/' + image_file)
        else:
            clean_image.save('data/val/gt/' + image_file)
            noisy_image.save('data/val/wm/' + image_file)

        value_set = not value_set

    return value_set


def run():
    list_source_images = glob.glob(f"{SOURCE_SOURCE}/*")
    list_no_logo_images = glob.glob(f"{DATA_SOURCE}/*")
    list_logo_images = glob.glob(f"{LOGO_SOURCE}/*")[:5]
    list_images = list_source_images + list_logo_images + list_no_logo_images
    random.shuffle(list_images)

    value_set = False
    for i, image in enumerate(list_images):
        value_set = noisy(image, i, value_set)
        print(f"LEFT {i}/{len(list_images)} <= {MAX_DATA}")
        if i >= MAX_DATA:
            break


if __name__ == '__main__':
    run()
