import os.path
import random

from common.header import *
from common.utils import *
from service.Generator import Generator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

DATA_TEST = f"./data_wm"


def _prediction(image_name, generator):
    watermarked_image_path = f"{DATA_TEST}/{DEGRADED_VAL_DATA}/{image_name}"
    watermarked_image = image_to_gray(watermarked_image_path)
    plt.imsave(f"{RESULT_PATH}/original_image.png", watermarked_image, cmap='gray')

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

    plt.imsave(f"{RESULT_PATH}/predicted_image.png", predicted_image, cmap='gray')

    return predicted_image


def evaluate(generator):
    try:
        gt_image_list = os.listdir(f"{DATA_TEST}/{GT_VAL_DATA}")
        gt_file_name = random.choice(gt_image_list)

        print(f"Evaluation: {gt_file_name}")
        gt_file_path = f"{DATA_TEST}/{GT_VAL_DATA}/{gt_file_name}"
        gt_image = image_to_gray(gt_file_path)
        predicted_image = _prediction(gt_file_name, generator)
        avg_psnr = psnr(gt_image, predicted_image)
        print(f"PSNR: {avg_psnr}")
    except:
        avg_psnr = .0

    return avg_psnr


def load_model():
    try:
        print(f"Loaded generator trained model")
        model = Generator(biggest_layer=512)
        model.load_weights(TRAIN_MODEL_PATH + f"/wm_generator.h5")
        return model
    except OSError as _:
        print("No generator model loaded!")


def batch_evaluation(generator):
    avg_psnr = .0
    for it in range(10):
        avg_psnr += evaluate(generator)
    avg_psnr = avg_psnr // 10
    print("Mean PSNR: " + str(avg_psnr))

    return avg_psnr


def predict():
    for file in ClassFile.list_files(RESULT_PATH):
        os.remove(file)

    model = load_model()
    batch_evaluation(model)


if __name__ == '__main__':
    predict()
