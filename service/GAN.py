import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *


def get_gan_network(discriminator, generator, input_size=(256, 256, 1)):
    gan_input2 = Input(input_size)

    discriminator.trainable = False
    x = generator(gan_input2)
    valid = discriminator([x, gan_input2])

    model = Model(inputs=[gan_input2], outputs=[valid, x])

    model.compile(loss=['mse', 'binary_crossentropy'],
                  loss_weights=[1, 100],
                  optimizer=Adam(learning_rate=1e-4),
                  metrics=['accuracy'])

    # print(model.summary())

    return model
