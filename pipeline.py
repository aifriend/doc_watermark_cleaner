import os.path

import cv2

from common.header import *
from common.utils import *
from service.Generator import Generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def prediction(idx, watermarked_image, generator_wm, generator_dn):
    h = ((watermarked_image.shape[0] // 256) + 1) * 256
    w = ((watermarked_image.shape[1] // 256) + 1) * 256
    test_padding = np.zeros((h, w)) + 1
    test_padding[:watermarked_image.shape[0], :watermarked_image.shape[1]] = watermarked_image
    test_image_p = split2(test_padding.reshape(1, h, w, 1), 1, h, w)

    # predicted_wm_image = remove_watermark(generator_wm, test_image_p, watermarked_image, h, w, idx)
    # plt.imsave(f"{RESULT_PATH}/{idx}_predicted_wm_image.png", predicted_wm_image, cmap='gray')

    predicted_dn_image = remove_denoise(generator_dn, test_image_p, watermarked_image, h, w, idx)

    return predicted_dn_image


def remove_watermark(generator_wm, test_image_p, watermarked_image, h, w, idx):
    print("Predicting WM...")
    predicted_list = []
    for _l in range(test_image_p.shape[0]):
        image = test_image_p[_l].reshape(1, 256, 256, 1)
        p_image = generator_wm.predict(image)
        predicted_list.append(p_image)

    predicted_image = merge_image2(np.array(predicted_list), h, w)
    predicted_image = predicted_image[:watermarked_image.shape[0], :watermarked_image.shape[1]]
    predicted_image = predicted_image.reshape(predicted_image.shape[0], predicted_image.shape[1])
    predicted_wm_image = predicted_image.astype(np.float32)

    return predicted_wm_image


def remove_denoise(generator_dn, test_image_p, watermarked_image, h, w, idx):
    print("Predicting DN...")
    predicted_list = []
    for _l in range(test_image_p.shape[0]):
        image = test_image_p[_l].reshape(1, 256, 256, 1)
        p_image = generator_dn.predict(image)
        predicted_list.append(p_image)

    predicted_image = merge_image2(np.array(predicted_list), h, w)
    predicted_image = predicted_image[:watermarked_image.shape[0], :watermarked_image.shape[1]]
    predicted_image = predicted_image.reshape(predicted_image.shape[0], predicted_image.shape[1])
    predicted_dn_image = predicted_image.astype(np.float32)

    return predicted_dn_image


def erode_dilate(image):
    # Kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(image, np.ones((1, 1), np.uint8), iterations=1)
    img_dilation = cv2.dilate(img_erosion, np.ones((1, 1), np.uint8), iterations=1)

    # Image.fromarray(img_dilation).show()

    return img_dilation


def denoise(page):
    # Reading the input image
    result = page.copy()
    cv2.fastNlMeansDenoising(page, result, 30.0, 7, 21)
    return result


def pipeline(generator_wm, generator_dn):
    try:
        wm_image_list = os.listdir(f"{DATA_TEST}")
        for idx, wm_file_name in enumerate(wm_image_list):
            print(f"Pipeline: {wm_file_name}")
            wm_file_path = f"{DATA_TEST}/{wm_file_name}"
            page = cv2.imread(wm_file_path, 0)
            #page = cv2.resize(page, dsize=(0, 0), interpolation=cv2.INTER_CUBIC)

            plt.imsave(f"{RESULT_PATH}/{idx}_original_image.png", page, cmap='gray')

            #page = denoise(page)
            #page = erode_dilate(page)
            #prediction(idx, page, generator_wm, generator_dn)

            plt.imsave(f"{RESULT_PATH}/{idx}_predicted_dn_image.png", page, cmap='gray')

            if idx >= 999:
                break

    except Exception as e:
        print(f"Pipeline exception: {e}")


def load_wm_model(model_name):
    try:
        print(f"Loaded generator trained model")
        model = Generator(biggest_layer=512)
        model.load_weights(os.path.join(TRAIN_MODEL_PATH, model_name))
        return model
    except OSError as _:
        print("No WM generator model loaded!")


def load_dn_model(model_name):
    try:
        print(f"Loaded generator trained model")
        model = Generator(biggest_layer=1024)
        model.load_weights(os.path.join(TRAIN_MODEL_PATH, model_name))
        return model
    except OSError as _:
        print("No DN generator model loaded!")


def run():
    for file in ClassFile.list_files(RESULT_PATH):
        os.remove(file)

    model_wm = load_wm_model("wm_generator.h5")
    model_dn = load_dn_model("dn_generator.h5")
    pipeline(model_wm, model_dn)


if __name__ == '__main__':
    DATA_TEST = f"./data/data_source"
    run()
