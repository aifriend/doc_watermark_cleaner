import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *


def Gan(generator, discriminator, input_size=(256, 256, 1)):
    gan_input2 = Input(input_size)

    discriminator.trainable = False
    x = generator(gan_input2)
    valid = discriminator([x, gan_input2])

    # input=[wm_img], output=[D([G(wm_img), wm_img]), G(wm_img)]
    model = Model(inputs=[gan_input2], outputs=[valid, x], name="gan")

    model.compile(loss=['mse', 'binary_crossentropy'],
                  loss_weights=[1, 100],
                  optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
                  metrics=['accuracy'])

    return model
