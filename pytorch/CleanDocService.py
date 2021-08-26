import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from common.Configuration import Configuration
from pytorch.Discriminator import Discriminator
from pytorch.GAN import GAN
from pytorch.Generator import Generator

cuda = True if torch.cuda.is_available() else False


class CleanDocService:
    DEVICE_TYPE = 'cpu'
    BATCH_SIZE = 128
    EPOCHS = 100
    LRN_RATE = 0.0001
    WATERMARK_DOC_NAME = "degraded"
    GROUND_TRUTH_DOC_NAME = "clean"
    DEGRADED_SRC_PATH = "./data/src/train/wm"
    GROUND_TRUTH_SRC_PATH = "./data/src/train/gt"
    DEGRADED_TRAIN_PATH = "./data/train/wm"
    GROUND_TRUTH_TRAIN_PATH = "./data/train/gt"

    def __init__(self, conf):
        self.path = conf.root_path
        self.data_loader = dict()
        self.df = None
        self.inputs = None
        self.generator = None
        self.discriminator = None
        self.dis_optimizer = None
        self.dis_scheduler = None
        self.gen_optimizer = None
        self.gen_scheduler = None
        self.fixed_noise = None
        self.real_label = None
        self.fake_label = None

        self.device = torch.device(self.DEVICE_TYPE)
        self.image_size = 256
        self.criterion = nn.MSELoss()
        self.criterion_gan = nn.BCELoss()

        self._make_model()

    def _make_model(self) -> None:
        self.generator = Generator()
        # self.generator.load_state_dict(torch.load(self.path, map_location=self.device))
        self.generator.to(self.device)
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.001)
        self.gen_scheduler = lr_scheduler.StepLR(self.gen_optimizer, step_size=7, gamma=0.1)

        self.discriminator = Discriminator()
        # self.discriminator.load_state_dict(torch.load(self.path, map_location=self.device))
        self.discriminator.to(self.device)
        self.dis_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.001)
        self.dis_scheduler = lr_scheduler.StepLR(self.dis_optimizer, step_size=7, gamma=0.1)

        self.gan = GAN(self.generator, self.discriminator)
        # self.gan.load_state_dict(torch.load(self.path, map_location=self.device))
        self.gan.to(self.device)
        self.gan_optimizer = optim.Adam(self.gan.parameters(), lr=0.001)
        self.gan_scheduler = lr_scheduler.StepLR(self.gan_optimizer, step_size=7, gamma=0.1)

    def load_data(self):
        # Set random seed for reproducibility
        manualSeed = 999
        # manualSeed = random.randint(1, 10000) # use to see new results
        print("Random Seed: ", manualSeed)
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)

        # We can use an image folder dataset the way we have it setup.
        # Create the dataset
        degraded_dataset = dset.ImageFolder(root=self.DEGRADED_TRAIN_PATH,
                                            transform=transforms.Compose([
                                                transforms.Grayscale(3),
                                                transforms.Resize(self.image_size),
                                                transforms.CenterCrop(self.image_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            ]))

        clean_dataset = dset.ImageFolder(root=self.GROUND_TRUTH_TRAIN_PATH,
                                         transform=transforms.Compose([
                                             transforms.Grayscale(3),
                                             transforms.Resize(self.image_size),
                                             transforms.CenterCrop(self.image_size),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))

        # Create the data loader
        self.data_loader[self.WATERMARK_DOC_NAME] = torch.utils.data.DataLoader(
            degraded_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)

        self.data_loader[self.GROUND_TRUTH_DOC_NAME] = torch.utils.data.DataLoader(
            clean_dataset, batch_size=self.BATCH_SIZE, shuffle=False, num_workers=4)

    def pre_process(self):
        list_deg_images = os.listdir(os.path.join(self.path, self.DEGRADED_SRC_PATH))
        list_clean_images = os.listdir(os.path.join(self.path, self.GROUND_TRUTH_SRC_PATH))

        list_deg_images.sort()
        list_clean_images.sort()

        wat_batch_list = list()
        gt_batch_list = list()
        src_degraded = os.path.join(self.path, './data/results/curr_deg_image.png')
        src_clean = os.path.join(self.path, './data/results/curr_clean_image.png')
        for im in tqdm(range(len(list_deg_images))):
            deg_image_path = os.path.join(self.path, self.DEGRADED_SRC_PATH, list_deg_images[im])
            deg_image = Image.open(deg_image_path)  # /255.0
            deg_image = deg_image.convert('L')
            deg_image.save(src_degraded)
            deg_image = plt.imread(src_degraded)

            clean_image_path = os.path.join(self.path, self.GROUND_TRUTH_SRC_PATH, list_clean_images[im])
            clean_image = Image.open(clean_image_path)  # /255.0
            clean_image = clean_image.convert('L')
            clean_image.save(src_clean)
            clean_image = plt.imread(src_clean)

            wat_batch, gt_batch = self._get_patches(deg_image, clean_image, my_stride=128 + 64)
            wat_batch_list.extend(wat_batch)
            gt_batch_list.extend(gt_batch)

        return wat_batch_list, gt_batch_list

    @staticmethod
    def _get_patches(watermarked_image, clean_image, my_stride):
        watermarked_patches = []
        clean_patches = []

        h = ((watermarked_image.shape[0] // 256) + 1) * 256
        w = ((watermarked_image.shape[1] // 256) + 1) * 256
        image_padding = np.ones((h, w))
        image_padding[:watermarked_image.shape[0], :watermarked_image.shape[1]] = watermarked_image

        for j in range(0, h - 256, my_stride):  # 128 not 64
            for k in range(0, w - 256, my_stride):
                watermarked_patches.append(image_padding[j:j + 256, k:k + 256])

        h = ((clean_image.shape[0] // 256) + 1) * 256
        w = ((clean_image.shape[1] // 256) + 1) * 256
        image_padding = np.ones((h, w)) * 255
        image_padding[:clean_image.shape[0], :clean_image.shape[1]] = clean_image

        for j in range(0, h - 256, my_stride):  # 128 not 64
            for k in range(0, w - 256, my_stride):
                clean_patches.append(image_padding[j:j + 256, k:k + 256] / 255)

        return np.array(watermarked_patches), np.array(clean_patches)

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self):
        # Lists to keep track of progress
        G_losses = list()
        D_losses = list()
        iters = 0

        # initialize weights
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        # For each epoch
        for epoch in range(self.EPOCHS):

            # For each batch in the data loader
            loop = tqdm(enumerate(self.data_loader),
                        leave=False, total=len(self.data_loader))
            for i, (_input, _label) in enumerate(loop, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = _input[0].to(self.device)
                b_size = real_cpu.size(0)
                _label = torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, _label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.nz, 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                _label.fill_(self.fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, _label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                _label.fill_(self.real_label)  # fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake batch through D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, _label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                self.optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, self.EPOCHS, i, len(self.data_loader),
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # Save Losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.EPOCHS - 1) and (i == len(self.data_loader) - 1)):
                    with torch.no_grad():
                        fake = self.netG(self.fixed_noise).detach().cpu()
                    self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                iters += 1


def main():
    clean_doc_service = CleanDocService(Configuration())

    clean_doc_service.pre_process()

    clean_doc_service.load_data()

    clean_doc_service.train()


if __name__ == '__main__':
    main()
