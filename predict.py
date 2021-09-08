import os.path
import cv2

from common.header import *
from common.utils import *
from service.Generator import Generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def erode_dilate(image_id, image):
    # Kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.
    img_erosion = cv2.erode(image, np.ones((2, 2), np.uint8), iterations=1)
    img_dilation = cv2.dilate(img_erosion, np.ones((2, 2), np.uint8), iterations=1)

    # Image.fromarray(img_dilation).show()
    plt.imsave(f"{RESULT_PATH}/{image_id}_qlean_image.png", img_dilation, cmap='gray')


def _prediction(image_id, image_name, generator):
    watermarked_image_path = image_name
    watermarked_image = image_to_gray(watermarked_image_path)
    plt.imsave(f"{RESULT_PATH}/{image_id}_original_image.png", watermarked_image, cmap='gray')

    h = ((watermarked_image.shape[0] // 256) + 1) * 256
    w = ((watermarked_image.shape[1] // 256) + 1) * 256
    test_padding = np.zeros((h, w)) + 1
    test_padding[:watermarked_image.shape[0], :watermarked_image.shape[1]] = watermarked_image
    test_image_p = split2(test_padding.reshape(1, h, w, 1), 1, h, w)

    predicted_list = []
    for _l in range(test_image_p.shape[0]):
        image = test_image_p[_l].reshape(1, 256, 256, 1)
        p_image = generator.predict(image)
        predicted_list.append(p_image)

    predicted_image = merge_image2(np.array(predicted_list), h, w)
    predicted_image = predicted_image[:watermarked_image.shape[0], :watermarked_image.shape[1]]
    predicted_image = predicted_image.reshape(predicted_image.shape[0], predicted_image.shape[1])
    predicted_image = predicted_image.astype(np.float32)

    plt.imsave(f"{RESULT_PATH}/{image_id}_predicted_image.png", predicted_image, cmap='gray')

    return predicted_image


def prediction(generator):
    try:
        avg_psnr_list = list()
        wm_image_list = os.listdir(f"{DATA_TEST}")
        for idx, wm_file_name in enumerate(wm_image_list):
            print(f"Evaluation: {wm_file_name}")
            wm_file_path = f"{DATA_TEST}/{wm_file_name}"
            wm_image = image_to_gray(wm_file_path)
            predicted_image = _prediction(idx, wm_file_path, generator)
            erode_dilate(idx, predicted_image)
            avg_psnr_list.append((wm_image, predicted_image))
            if idx == 9999:
                break
    except:
        print("Prediction exception!")


def load_model():
    try:
        print(f"Loaded generator trained model")
        model = Generator(biggest_layer=1024)
        model.load_weights(os.path.join(TRAIN_MODEL_PATH, "dn_generator.h5"))
        return model
    except OSError as _:
        print("No generator model loaded!")


def run():
    for file in ClassFile.list_files(RESULT_PATH):
        os.remove(file)

    model = load_model()
    prediction(model)


if __name__ == '__main__':
    DATA_TEST = f"./data/data_source"
    run()
