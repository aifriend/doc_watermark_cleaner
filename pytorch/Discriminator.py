import torch
from torch import nn


class Discriminator(nn.Module):

    # input size (256, 256, 1) -> padding 2

    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            *self.decoder(64),
            *self.decoder_bn(128),
            *self.decoder_bn(256),
            *self.decoder_bn(256),
            *self.linear(256, 1)
        )

    @staticmethod
    def decoder(in_):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=in_, kernel_size=(4, 4), stride=(2, 2), padding=2),
            nn.LeakyReLU(negative_slope=0.2),
        )

    @staticmethod
    def decoder_bn(in_):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=in_, kernel_size=(4, 4), stride=(2, 2), padding=2),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm2d(num_features=in_, momentum=0.8)
        )

    @staticmethod
    def linear(in_, out_):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(4, 4), stride=(1, 1), padding=2),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        input_ = (x, y)

        x = torch.add(input_[0], input_[1])
        validity = self.model(x)

        return validity
