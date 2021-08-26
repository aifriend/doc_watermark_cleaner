import math

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageStat

from common.header import TRAIN_MODEL_PATH


def detect_gray(file, thumb_size=40, mse_cutoff=22, adjust_color_bias=True):
    pil_img = Image.open(file)
    bands = pil_img.getbands()
    if bands == ('R', 'G', 'B') or bands == ('R', 'G', 'B', 'A'):
        thumb = pil_img.resize((thumb_size, thumb_size))
        SSE, bias = 0, [0, 0, 0]
        if adjust_color_bias:
            bias = ImageStat.Stat(thumb).mean[:3]
            bias = [b - sum(bias) / 3 for b in bias]
        for pixel in thumb.getdata():
            mu = sum(pixel) / 3
            SSE += sum((pixel[i] - mu - bias[i]) * (pixel[i] - mu - bias[i]) for i in [0, 1, 2])
        MSE = float(SSE) / (thumb_size * thumb_size)
        if MSE <= mse_cutoff:
            return True
    elif 'L' in bands or 'LA' in bands:
        return True

    return False


def image_to_gray(image_path, dtype=np.float32, show=False):
    # read raw data
    image = plt.imread(image_path)

    # to grayscale
    if not detect_gray(image_path) and len(image.shape) >= 3:
        image = image[:, :, 0] * 0.2989 + image[:, :, 1] * 0.5870 + image[:, :, 2] * 0.1140
    elif len(image.shape) >= 3:
        image = image[:, :, 0]

    # normalize data type
    image = image.astype(dtype)

    if show:
        Image.fromarray(image * 255).show()

    return image


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0

    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def split2(dataset, size, h, w):
    newdataset = []
    nsize1 = 256
    nsize2 = 256
    for i in range(size):
        im = dataset[i]
        for ii in range(0, h, nsize1):  # 2048
            for iii in range(0, w, nsize2):  # 1536
                newdataset.append(im[ii:ii + nsize1, iii:iii + nsize2, :])

    return np.array(newdataset)


def merge_image2(splitted_images, h, w):
    image = np.zeros((h, w, 1))
    nsize1 = 256
    nsize2 = 256
    ind = 0
    for ii in range(0, h, nsize1):
        for iii in range(0, w, nsize2):
            image[ii:ii + nsize1, iii:iii + nsize2, :] = splitted_images[ind]
            ind = ind + 1

    return np.array(image)


def getPatches(watermarked_image, clean_image, my_stride):
    watermarked_patches = []
    h = ((watermarked_image.shape[0] // 256) + 1) * 256
    w = ((watermarked_image.shape[1] // 256) + 1) * 256
    image_padding = np.ones((h, w), dtype=np.float32)
    image_padding[:watermarked_image.shape[0], :watermarked_image.shape[1]] = watermarked_image

    for j in range(0, h - 256, my_stride):  # 128 not 64
        for k in range(0, w - 256, my_stride):
            watermarked_patches.append(image_padding[j:j + 256, k:k + 256])

    clean_patches = []
    h = ((clean_image.shape[0] // 256) + 1) * 256
    w = ((clean_image.shape[1] // 256) + 1) * 256
    image_padding = np.ones((h, w), dtype=np.float32)
    image_padding[:clean_image.shape[0], :clean_image.shape[1]] = clean_image

    for j in range(0, h - 256, my_stride):  # 128 not 64
        for k in range(0, w - 256, my_stride):
            clean_patches.append(image_padding[j:j + 256, k:k + 256])  # 0: text, 1: degradation

    return np.array(watermarked_patches), np.array(clean_patches)


def save_model(generator_model, discriminator_model):
    generator_model.save_weights(TRAIN_MODEL_PATH + '/last_generator_weights.h5')
    discriminator_model.save_weights(TRAIN_MODEL_PATH + '/last_discriminator_weights.h5')
    print("Model saved")


def load_model(generator, discriminator):
    try:
        generator.load_weights(TRAIN_MODEL_PATH + '/last_generator_weights.h5')
        print("Loading generator trained model...")
    except OSError as _:
        print("No generator model loaded!")

    try:
        discriminator.load_weights(TRAIN_MODEL_PATH + '/last_discriminator_weights.h5')
        print("Loading discriminator trained model...")
    except OSError as _:
        print("No discriminator model loaded!")


def load_default(generator):
    try:
        generator.load_weights(TRAIN_MODEL_PATH + '/model/watermark_rem_weights.h5')
        print("Loading default generator trained model...")
    except OSError as _:
        print("No generator model loaded!")
