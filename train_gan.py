import os
import argparse

import torch

from utils.gan import ConditionalDCGAN
from utils.datasets import load_mnist_notmnist, load_svhn_cifar10, load_cifar5

file_directory = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='mnist_notmnist')
    parser.add_argument('--n_latent', type=int, default=10)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    train_gan(args)


def train_gan(args):
    if args.dataset == 'mnist_notmnist':
        n_channel = 1
        n_classes = 10
        train_ds, _, _, _, _ = load_mnist_notmnist()
    elif args.dataset == 'svhn_cifar10':
        n_channel = 3
        n_classes = 10
        train_ds, _, _, _, _ = load_svhn_cifar10()
    elif args.dataset == 'cifar5_cifar5':
        n_channel = 3
        n_classes = 5
        train_ds, _, _, _, _ = load_cifar5()

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

    gan = ConditionalDCGAN(n_latent=args.n_latent, n_channel=n_channel, n_classes=n_classes)
    gan.fit(train_loader, n_epochs=args.n_epochs, plot_step_size=1e10, device=args.device)

    # Save model
    save_path = os.path.join(file_directory, 'results', 'ood_state_dict')
    os.makedirs(save_path, exist_ok=True)
    weight_dict = gan.state_dict()
    torch.save(weight_dict, os.path.join(save_path, 'condgan_{}.pth'.format(args.dataset)))

if __name__ == "__main__":
    main()
