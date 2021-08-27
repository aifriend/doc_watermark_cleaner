import os.path
import random

from cv2 import cv2
from tqdm import tqdm

from common.header import *
from common.utils import *
from service.Discriminator import Discriminator
from service.GAN import get_gan_network
from service.Generator import Generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def get_patches(deg_image, clean_image, show=False):
    wat_batch, gt_batch = getPatches(deg_image, clean_image, my_stride=128 + 64)
    if show:
        for i, wt in enumerate(wat_batch):
            plt.imshow(wat_batch[i], cmap="gray", vmin=0, vmax=1)
            plt.show()
            plt.imshow(gt_batch[i], cmap="gray", vmin=0, vmax=1)
            plt.show()

    return wat_batch, gt_batch


def _prediction(image_name, generator, epoch):
    watermarked_image_path = f"{DEGRADED_VAL_DATA}/{image_name}"
    watermarked_image = image_to_gray(watermarked_image_path)
    plt.imsave(f"{RESULT_PATH}/{epoch}_original_image_plot.png", watermarked_image, cmap='gray')

    h = ((watermarked_image.shape[0] // 256) + 1) * 256
    w = ((watermarked_image.shape[1] // 256) + 1) * 256
    test_padding = np.zeros((h, w)) + 1
    test_padding[:watermarked_image.shape[0], :watermarked_image.shape[1]] = watermarked_image
    test_image_p = split2(test_padding.reshape(1, h, w, 1), 1, h, w)

    predicted_list = []
    for _l in range(test_image_p.shape[0]):
        predicted_list.append(generator.predict(test_image_p[_l].reshape(1, 256, 256, 1)))
        print(".", end='')

    predicted_image = merge_image2(np.array(predicted_list), h, w)
    predicted_image = predicted_image[:watermarked_image.shape[0], :watermarked_image.shape[1]]
    predicted_image = predicted_image.reshape(predicted_image.shape[0], predicted_image.shape[1])
    predicted_image = predicted_image.astype(np.float32)

    plt.imsave(f"{RESULT_PATH}/{epoch}_predicted_image_plot.png", predicted_image, cmap='gray')

    return predicted_image


def evaluate(generator, epoch):
    gt_image_list = os.listdir(GT_VAL_DATA)
    gt_file_name = random.choice(gt_image_list)

    print(f"Evaluation: {gt_file_name}")
    gt_file_path = f"{GT_VAL_DATA}/{gt_file_name}"
    gt_image = image_to_gray(gt_file_path)
    predicted_image = _prediction(gt_file_name, generator, epoch)
    avg_psnr = psnr(gt_image, predicted_image)
    print(f"\nPSNR: {avg_psnr}")

    return avg_psnr


def load_data(max_sample):
    print("Load data", end="")
    wm_image_list = os.listdir(DEGRADED_TRAIN_DATA)
    gt_image_list = os.listdir(GT_TRAIN_DATA)

    c_deg_image_list = list()
    c_clean_image_list = list()
    for im in range(len(wm_image_list)):
        if wm_image_list[im] in gt_image_list:
            d_im = f"{DEGRADED_TRAIN_DATA}/{wm_image_list[im]}"
            c_deg_image_list.append(d_im)
            c_im = d_im.replace("/wm/", "/gt/")
            c_clean_image_list.append(c_im)

    list_images = list(zip(c_deg_image_list, c_clean_image_list))
    random.shuffle(list_images)
    list_deg, list_clean = zip(*list_images)

    list_deg = list_deg[:max_sample]
    list_deg_im = list()
    for d_im in list_deg:
        deg_image = image_to_gray(d_im)
        deg_image = cv2.resize(deg_image, DEFAULT_SHAPE)
        list_deg_im.append(deg_image)
        print(".", end="")

    list_clean = list_clean[:max_sample]
    list_clean_im = list()
    for c_im in list_clean:
        gt_image = image_to_gray(c_im)
        gt_image = cv2.resize(gt_image, DEFAULT_SHAPE)
        list_clean_im.append(gt_image)
        print(".", end="")
    print()

    return list_deg_im, list_clean_im


def train_gan(model_name,
              generator, discriminator,
              epochs=1, batch_size=10, max_sample=1):
    best_psnr = load_score(model_name)

    gan = get_gan_network(discriminator, generator)

    for e in range(1, epochs + 1):
        print('\nEpoch:', e)

        list_deg_images, list_clean_images = load_data(max_sample)

        loop = tqdm(range(len(list_deg_images)), leave=True, position=0)
        for imd in loop:
            loop.set_description(f"Document [{imd+1}/{max_sample}] - "
                                 f"PSNR [{round(best_psnr, 2)}]")

            # unpack watermarked document dataset
            deg_image = list_deg_images[imd]
            # unpack ground truth clean document dataset
            clean_image = list_clean_images[imd]

            # get image patches
            wat_batch, gt_batch = get_patches(deg_image, clean_image)

            # calculate the number of training iterations
            batch_count = wat_batch.shape[0] // batch_size

            # manually enumerate epochs
            for pat_idx, b in enumerate(range(batch_count)):
                # generate a batch of valid samples
                seed = range(b * batch_size, (b * batch_size) + batch_size)
                b_wat_batch = wat_batch[seed].reshape(batch_size, 256, 256, 1)
                b_gt_batch = gt_batch[seed].reshape(batch_size, 256, 256, 1)
                valid = np.ones((b_wat_batch.shape[0],) + (16, 16, 1))
                fake = np.zeros((b_wat_batch.shape[0],) + (16, 16, 1))

                # generate a batch of fake samples
                generated_images = generator.predict(b_wat_batch)

                # update discriminator for real samples
                discriminator.trainable = True
                d_loss1 = discriminator.train_on_batch([b_gt_batch, b_wat_batch], valid)
                # update discriminator for generated samples
                d_loss2 = discriminator.train_on_batch([generated_images, b_wat_batch], fake)

                # update the generator
                discriminator.trainable = False
                g_loss = gan.train_on_batch([b_wat_batch], [valid, b_gt_batch])

                loop.set_postfix_str(f"Patch: {pat_idx}/{batch_count}")

        # summarize model performance
        psnr = evaluate(generator, e)
        if psnr > best_psnr:
            best_psnr = psnr
            save_default_model(generator, discriminator)
            save_score(model_name, psnr)


def main():
    model_name = "wm"
    for file in ClassFile.list_files(RESULT_PATH):
        os.remove(file)

    generator = Generator(biggest_layer=512)
    discriminator = Discriminator()

    load_model(model_name, generator, discriminator)

    train_gan(model_name, generator, discriminator, epochs=30, batch_size=10, max_sample=150)


if __name__ == '__main__':
    main()
