import torch
from torch import nn


class Generator(nn.Module):

    # input size (256, 256, 1) -> padding 2

    def __init__(self):
        super(Generator, self).__init__()
        self.en1 = self.encoder(1, 64)
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.en2 = self.encoder(64, 128)
        self.en3 = self.encoder(128, 256)

        self.bottle = self.conv_bottle_neck_layer(256, 512)
        self.neck = self.conv_bottle_neck_layer(512, 1024)

        self.ul1 = self.up_conv(1024, 512)
        self.dc1 = self.decoder(512)
        self.ul2 = self.up_conv(512, 256)
        self.dc2 = self.decoder(256)
        self.ul3 = self.up_conv(256, 128)
        self.dc3 = self.decoder(128)
        self.ul4 = self.up_conv(128, 64)
        self.dc4 = self.decoder(64)

        self.output = self.linear(64)

    @staticmethod
    def encoder(in_, out_):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_, out_channels=out_, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
        )

    @staticmethod
    def conv_bottle_neck_layer(in_, out_):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_, out_channels=out_, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    @staticmethod
    def up_conv(in_, out_):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_, out_channels=in_, kernel_size=(2, 2)),
            nn.Conv2d(in_channels=in_, out_channels=out_, kernel_size=(2, 2), padding=2),
            nn.ReLU(),
        )

    @staticmethod
    def decoder(in_):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=in_, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_, out_channels=in_, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
        )

    @staticmethod
    def linear(in_):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_, out_channels=2, kernel_size=(3, 3), padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(1, 1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        _input = x

        en1 = self.en1(_input)
        mp1 = self.mp(en1)
        en2 = self.en2(mp1)
        mp2 = self.mp(en2)
        en3 = self.en3(mp2)
        mp3 = self.mp(en3)

        btn = self.bottle(mp3)
        btn_mp = self.mp(btn)
        btn_core = self.neck(btn_mp)

        ul1 = self.ul1(btn_core)
        cc1 = torch.add(btn, ul1)
        dc1 = self.dc1(cc1)
        ul2 = self.ul2(dc1)
        cc2 = torch.add(en3, ul2)
        dc2 = self.dc2(cc2)
        ul3 = self.ul3(dc2)
        cc3 = torch.add(en2, ul3)
        dc3 = self.dc3(cc3)
        ul4 = self.ul4(dc3)
        cc4 = torch.add(en1, ul4)
        dc4 = self.dc4(cc4)

        _output = self.output(dc4)

        return _output
