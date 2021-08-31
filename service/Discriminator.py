import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *


def get_optimizer():
    return


def Discriminator(input_size=(256, 256, 1)):
    def d_layer(layer_input, filters, f_size=4, bn=True):
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    img_A = Input(input_size)
    img_B = Input(input_size)

    df = 64

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = d_layer(combined_imgs, df, bn=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 4)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same', activation='sigmoid')(d4)

    d_model = Model([img_A, img_B], validity, name="critic")

    d_model.compile(
        loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999), metrics=['accuracy'])

    return d_model
