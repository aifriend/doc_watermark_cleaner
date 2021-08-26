import os.path
import random

from tqdm import tqdm

from common.ClassFile import ClassFile
from common.utils import *
from common.header import *
from service.Discriminator import Discriminator
from service.GAN import get_gan_network
from service.Generator import Generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

MAX_TRAIN_DATA = 200


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


def get_patches(deg_image, clean_image, show=False):
    wat_batch, gt_batch = getPatches(deg_image, clean_image, my_stride=128 + 64)
    if show:
        for i, wt in enumerate(wat_batch):
            plt.imshow(wat_batch[i], cmap="gray", vmin=0, vmax=1)
            plt.show()
            plt.imshow(gt_batch[i], cmap="gray", vmin=0, vmax=1)
            plt.show()

    return wat_batch, gt_batch


def train_gan(generator, discriminator, epochs=1, batch_size=128):
    try:
        best_psnr = float(ClassFile.get_text(TRAIN_PSNR_PATH))
    except:
        best_psnr = 0.0

    print("Loading data...")
    list_deg_images = os.listdir(DEGRADED_TRAIN_DATA)
    list_deg_images = list_deg_images[:MAX_TRAIN_DATA]
    list_clean_images = os.listdir(GT_TRAIN_DATA)
    list_clean_images = list_clean_images[:MAX_TRAIN_DATA]
    list_images = list(zip(list_deg_images, list_clean_images))
    random.shuffle(list_images)
    list_deg_images, list_clean_images = zip(*list_images)

    gan = get_gan_network(discriminator, generator)

    for e in range(1, epochs + 1):
        print('\nEpoch:', e)

        loop = tqdm(enumerate(range(len(list_deg_images))), leave=True, position=0)
        for wm_idx, im in loop:
            loop.set_description(f"Document [{wm_idx+1}/{len(list_deg_images)}] - "
                                 f"PSNR [{round(best_psnr, 2)}]")

            if list_deg_images[im] != list_clean_images[im]:
                print("Training data mismatch!")

            deg_image_path = f"{DEGRADED_TRAIN_DATA}/{list_deg_images[im]}"
            deg_image = image_to_gray(deg_image_path)

            clean_image_path = f"{GT_TRAIN_DATA}/{list_clean_images[im]}"
            clean_image = image_to_gray(clean_image_path)

            wat_batch, gt_batch = get_patches(deg_image, clean_image)

            batch_count = wat_batch.shape[0] // batch_size
            for pat_idx, b in enumerate(range(batch_count)):
                seed = range(b * batch_size, (b * batch_size) + batch_size)
                b_wat_batch = wat_batch[seed].reshape(batch_size, 256, 256, 1)
                b_gt_batch = gt_batch[seed].reshape(batch_size, 256, 256, 1)

                generated_images = generator.predict(b_wat_batch)

                valid = np.ones((b_wat_batch.shape[0],) + (16, 16, 1))
                fake = np.zeros((b_wat_batch.shape[0],) + (16, 16, 1))

                discriminator.trainable = True
                d_loss1 = discriminator.train_on_batch([b_gt_batch, b_wat_batch], valid)
                d_loss2 = discriminator.train_on_batch([generated_images, b_wat_batch], fake)

                discriminator.trainable = False
                g_loss = gan.train_on_batch([b_wat_batch], [valid, b_gt_batch])

                loop.set_postfix_str(f"Patch: {pat_idx}/{batch_count}")

        psnr = evaluate(generator, e)
        if psnr > best_psnr:
            save_model(generator, discriminator)
            best_psnr = psnr
            ClassFile.to_txtfile(str(best_psnr), TRAIN_PSNR_PATH)


def main():
    for file in ClassFile.list_files(RESULT_PATH):
        os.remove(file)

    generator = Generator(biggest_layer=512)
    discriminator = Discriminator()

    load_model(generator, discriminator)
    #load_default(generator)
    #evaluate(generator, 0)
    train_gan(generator, discriminator, epochs=30, batch_size=5)


if __name__ == '__main__':
    main()
