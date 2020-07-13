import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor

import numpy as np
import pylab as plt
from tqdm.auto import tqdm


def __test__():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])
    train_ds = torchvision.datasets.MNIST('~/.datasets', download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_ds, 256, shuffle=True, num_workers=4, pin_memory=True)
    gan = ConditionalDCGAN(10, 1, n_classes=10)
    gan.fit(train_loader, n_epochs=10)


class ConditionalDCGAN(nn.Module):
    def __init__(self, n_latent, n_channel, n_classes, n_filters=32):
        super().__init__()
        self.n_latent = n_latent
        self.n_channel = n_channel
        self.n_classes = n_classes

        self.generator = CondGenerator(
            n_latent=n_latent, n_filters=n_filters, n_channels=n_channel, n_classes=n_classes
        )
        self.discriminator = CondDiscriminator(
            n_channels=n_channel, n_filters=n_filters, n_classes=n_classes
        )
        self.reset_parameters()

    def forward(self, X_batch):
        raise NotImplementedError()

    def reset_parameters(self):
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

    def fit(self, train_loader, n_epochs=50, plot_step_size=10, lr=0.0002, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Training on {}.'.format(device))

        self.generator.to(device)
        self.discriminator.to(device)
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), betas=(0.5, .999), lr=lr)
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), betas=(0.5, .999), lr=lr)

        for i_epoch in tqdm(range(1, n_epochs+1), leave=False):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Train Discriminator
                out_real = self.discriminator(X_batch, y_batch)
                loss_1 = F.binary_cross_entropy(out_real, torch.ones_like(out_real))

                noise = torch.randn(X_batch.size(0), self.n_latent, 1, 1, device=device)
                X_fake = self.generator(noise, y_batch)
                out_fake = self.discriminator(X_fake.detach(), y_batch)
                loss_2 = F.binary_cross_entropy(out_fake, torch.zeros_like(out_fake))
                loss_disc = loss_1 + loss_2

                optimizer_disc.zero_grad()
                loss_disc.backward()
                optimizer_disc.step()

                # Train Generator
                X_fake = self.generator(noise, y_batch)
                out = self.discriminator(X_fake, y_batch)
                loss_gen = F.binary_cross_entropy(out, torch.ones_like(out))

                optimizer_gen.zero_grad()
                loss_gen.backward()
                optimizer_gen.step()

            if i_epoch % plot_step_size == 0:
                with torch.no_grad():
                    n_samples = self.n_classes * 2
                    noise = torch.randn(n_samples, self.n_latent, 1, 1, device=device)
                    y_fake = torch.arange(n_samples, device=device) % self.n_classes
                    X_fake = self.generator(noise, y_fake)
                plt.axis('off')
                plt.title('Epoch {}'.format(i_epoch))
                plt.imshow(torchvision.utils.make_grid(X_fake.cpu(), self.n_classes).permute(1, 2, 0))
                plt.show()


class CondDiscriminator(nn.Module):
    def __init__(self, n_channels, n_filters, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.main = nn.Sequential(
            # input size. (nc) x 32 x 32
            nn.Conv2d(n_channels+n_classes, n_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_filters * 2, n_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_filters * 4, n_filters * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(n_filters * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        # Concat input and label
        y_img = torch.zeros(input.size(0), self.n_classes, input.size(2), input.size(3), device=input.device)
        y_img[range(len(y_img)), label] = 1
        concated = torch.cat((input, y_img), dim=1)

        output = self.main(concated)
        return output.view(-1, 1)


class CondGenerator(nn.Module):
    def __init__(self, n_latent, n_filters, n_channels, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(n_latent+n_classes, n_filters * 8,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_filters * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(n_filters * 8, n_filters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(n_filters * 4, n_filters * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.ReLU(True),
            # state size. (n_filters*2) x 16 x 16
            nn.ConvTranspose2d(n_filters * 2, n_channels, kernel_size=4, stride=2, padding=1, bias=False),
            # state size. (nc) x 32 x 32
            nn.Sigmoid(),
            # state size. (nc) x 28 x 28
            # nn.Upsample(size=28),
        )

    def forward(self, input, label):
        # Concat input and label
        y_img = torch.zeros(input.size(0), self.n_classes, input.size(2), input.size(3), device=input.device)
        y_img[range(len(y_img)), label] = 1
        concated = torch.cat((input, y_img), dim=1)

        output = self.main(concated)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    __test__()
