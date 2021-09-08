import glob
import os
import random
from random import randint

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

GENERATION_FACTOR = 2
MAX_DATA = 500
DATA_SOURCE = "source_plus"
LOGO_SOURCE = ""


def watermarking(source, epoch, idx, value_set):
    im = Image.open(source)
    im = im.resize(DEFAULT_SHAPE).convert("RGBA")

    # clean image
    clean_image = im.copy()

    # mark image
    for aug_idx in range(GENERATION_FACTOR):
        background = im.copy()

        image_file = f"{epoch}_{aug_idx}_{idx}_no_logo.png"
        if value_set:
            clean_image.save('data/train/gt/' + image_file)
            background.save('data/train/wm/' + image_file)
        else:
            clean_image.save('data/val/gt/' + image_file)
            background.save('data/val/wm/' + image_file)


def watermarking_logo(source, epoch, idx, value_set):
    im = Image.open(source)
    im = im.resize(DEFAULT_SHAPE).convert("RGBA")

    # clean image
    clean_image = im.copy()

    # mark image
    mk_image_list = dict()
    logo_image_list = os.listdir(LOGO_SOURCE)
    for logo in logo_image_list:
        logo_image = Image.open(f"{LOGO_SOURCE}/{logo}")
        mk_image_list[logo] = logo_image.resize(DEFAULT_SHAPE).convert("RGBA")

    for aug_idx in range(GENERATION_FACTOR):
        mk_label, mk_image = random.choice(list(mk_image_list.items()))
        background = im.copy()
        m_pose_i = randint(-10, 10)
        m_pose_j = randint(-20, 20)
        background.paste(mk_image, (m_pose_i, m_pose_j), mk_image)

        image_file = f"{epoch}_{aug_idx}_{idx}_{mk_label}"
        if value_set:
            clean_image.save('data/train/gt/' + image_file)
            background.save('data/train/wm/' + image_file)
        else:
            clean_image.save('data/val/gt/' + image_file)
            background.save('data/val/wm/' + image_file)


def run():
    list_images = glob.glob(f"{DATA_SOURCE}/*")
    random.shuffle(list_images)

    epoch = "self"
    val_length = len(list_images) * 0.01  # 1% for validation
    for i, image in enumerate(list_images):
        watermarking(image, epoch, i, (i > val_length))
        print(f"LEFT ({'train' if (i > val_length) else 'val'}): "
              f"{i}/{len(list_images)} <= {MAX_DATA}")
        if i >= MAX_DATA:
            break


if __name__ == '__main__':
    run()
