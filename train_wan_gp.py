import os

from service.Generator import Generator

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

from tensorflow.python.keras.optimizer_v2.adam import Adam
from wgan_gp.TrainingMonitor import TrainingMonitor
from wgan_gp.TrainingService import TrainingService
from service.Discriminator import Discriminator

# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


EPOCHS = 20
noise_dim = 512  # Latent dimensionality
BATCH_SIZE = 10

# Instantiate the customer callback.
cbk = TrainingMonitor(num_img=3, latent_dim=noise_dim)

# Instantiate the WGAN model.
wgan = TrainingService(
    discriminator=Discriminator(),
    generator=Generator(biggest_layer=noise_dim),
    latent_dim=noise_dim,
    discriminator_extra_steps=3,
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# Start training the model.
wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk])
