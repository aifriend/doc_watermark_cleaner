from torch import nn


class GAN(nn.Module):

    # input size (256, 256, 1) -> padding 2

    def __init__(self, generator, discriminator):
        super(GAN, self).__init__()

        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        input_ = x

        X = self.generator(input_)
        self.discriminator.train(False)
        valid = self.discriminator(X, input_)

        return valid, X
