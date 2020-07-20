import argparse
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torchvision
import pylab as plt

from torch.utils.data import DataLoader

from utils.ordinary_nn import NNWrapper
from utils.ensembles import EnsembleWrapper
from utils.prior_network import PriorNetWrapper
from utils.edl import EDLWrapper
from utils.xedl import xEDLWrapper
from utils.datasets import load_mnist_notmnist, load_svhn_cifar10, load_cifar5
from utils.models import LeNet5, LeNet5Dropout
from utils.gan import ConditionalDCGAN
from utils.ood_sampling import generate_ood_hypersphere
from utils.dropout import DropoutWrapper
from utils.datasets import TinyImagenet

file_directory = os.path.dirname(os.path.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--n_reps', type=int, default=1)
    parser.add_argument('--random_state', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)

    parser.add_argument('--method_name', type=str, default='dropout')
    parser.add_argument('--dataset', type=str, default='mnist_notmnist')

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--ood_ds', type=int, default=0)

    # Research args
    parser.add_argument('--lmb', type=float, default=0.004)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--dropout_rate', type=float, default=.5)
    parser.add_argument('--ood_factor', type=float, default=5)
    args = parser.parse_args()

    run(args)


def run(args):
    print(vars(args))
    torch.manual_seed(args.random_state)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    if args.dataset == 'mnist_notmnist':
        n_channel = 1
        n_classes = 10
        net_list = [LeNet5(n_channel, n_classes) for _ in range(args.n_reps)]
        if args.method_name == 'dropout':
            net_list = [LeNet5Dropout(n_channel, n_classes, args.dropout_rate) for _ in range(args.n_reps)]
        train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out = load_mnist_notmnist()
    elif args.dataset == 'svhn_cifar10':
        n_channel = 3
        n_classes = 10
        net_list = [LeNet5(n_channel, n_classes) for _ in range(args.n_reps)]
        if args.method_name == 'dropout':
            net_list = [LeNet5Dropout(n_channel, n_classes, args.dropout_rate) for _ in range(args.n_reps)]
        train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out = load_svhn_cifar10()
    elif args.dataset == 'cifar5_cifar5':
        n_channel = 3
        n_classes = 5
        net_list = [LeNet5(n_channel, n_classes) for _ in range(args.n_reps)]
        if args.method_name == 'dropout':
            net_list = [LeNet5Dropout(n_channel, n_classes, args.dropout_rate) for _ in range(args.n_reps)]
        train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out = load_cifar5()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader_in = DataLoader(val_ds_in, batch_size=4*args.batch_size, num_workers=4, pin_memory=True)
    val_loader_out = DataLoader(val_ds_out, batch_size=4*args.batch_size, num_workers=4, pin_memory=True)
    test_loader_in = DataLoader(test_ds_in, batch_size=4*args.batch_size, num_workers=4, pin_memory=True)
    test_loader_out = DataLoader(test_ds_out, batch_size=4*args.batch_size, num_workers=4, pin_memory=True)

    val_aurocs = []
    val_accuracy = []
    val_nll = []
    val_bs = []
    val_ece = []

    test_aurocs = []
    test_accuracy = []
    test_nll = []
    test_bs = []
    test_ece = []
    for _, net_init in enumerate(net_list):
        if args.method_name == 'ordinary':
            net = NNWrapper(net_init)
        elif args.method_name == 'edl':
            net = EDLWrapper(net_init)
        elif args.method_name == 'ensembles':
            net = EnsembleWrapper(net_init)
        elif args.method_name == 'dropout':
            net = DropoutWrapper(net_init, n_mcsamples=100)
        elif args.method_name == 'priornet':
            if args.ood_ds:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ])
                if args.dataset == 'mnist_notmnist':
                    ood_ds = torchvision.datasets.EMNIST(
                        '~/.datasets/', split='letters', transform=transform, download=True)
                elif args.dataset == 'svhn_cifar10':
                    ood_ds = TinyImagenet(transform=transform)
                    transform.transforms.append(
                        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
                    )
                elif args.dataset == 'cifar5_cifar5':
                    ood_ds = TinyImagenet(transform=transform)
                    transform.transforms.append(
                        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
                    )
                ood_ds = torch.utils.data.Subset(ood_ds, torch.randperm(len(train_ds)))
                ood_loader = DataLoader(ood_ds, args.batch_size, shuffle=True,
                                        num_workers=4, pin_memory=True)
                ood_generator = GenOODFromDataloader(ood_loader)
            else:
                # Load ood
                if args.dataset == 'mnist_notmnist':
                    ood_path = os.path.join(file_directory, 'results', 'ood_state_dict', 'condgan_mnist.pth')
                elif args.dataset == 'svhn_cifar10':
                    ood_path = os.path.join(file_directory, 'results', 'ood_state_dict', 'condgan_svhn.pth')
                elif args.dataset == 'cifar5_cifar5':
                    ood_path = os.path.join(file_directory, 'results', 'ood_state_dict', 'condgan_cifar5.pth')
                n_latent = 10
                cgan = ConditionalDCGAN(n_latent, n_channel, n_classes)
                cgan.load_state_dict(torch.load(ood_path))
                ood_generator = OODCGAN(cgan, factor=args.ood_factor)
                # import pylab as plt
                # X_ood = ood_generator(*next(iter(train_loader)))
                # plt.imshow(torchvision.utils.make_grid(X_ood).permute(1, 2, 0))
                # plt.show()
            net = PriorNetWrapper(
                net_init,
                gamma=args.gamma,
                ood_generator=ood_generator,
            )
        elif args.method_name == 'xedl':
            if args.ood_ds:
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((32, 32)),
                    torchvision.transforms.ToTensor(),
                ])
                if args.dataset == 'mnist_notmnist':
                    ood_ds = torchvision.datasets.EMNIST(
                        '~/.datasets/', split='letters', transform=transform, download=True)
                elif args.dataset == 'svhn_cifar10':
                    ood_ds = TinyImagenet(transform=transform)
                    transform.transforms.append(
                        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
                    )
                elif args.dataset == 'cifar5_cifar5':
                    ood_ds = TinyImagenet(transform=transform)
                    transform.transforms.append(
                        torchvision.transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
                    )
                ood_ds = torch.utils.data.Subset(ood_ds, torch.randperm(len(train_ds)))
                ood_loader = DataLoader(ood_ds, args.batch_size, shuffle=True,
                                        num_workers=4, pin_memory=True)
                ood_generator = GenOODFromDataloader(ood_loader)
            else:
                # Load ood
                if args.dataset == 'mnist_notmnist':
                    ood_path = '/home/denis/Documents/projects/2020_xEDL/src/notebooks/ood_models/condgan_mnist.pth'
                elif args.dataset == 'svhn_cifar10':
                    ood_path = '/home/denis/Documents/projects/2020_xEDL/src/notebooks/ood_models/condgan_svhn.pth'
                elif args.dataset == 'cifar5_cifar5':
                    ood_path = '/home/denis/Documents/projects/2020_xEDL/src/notebooks/ood_models/condgan_cifar5.pth'
                n_latent = 10
                cgan = ConditionalDCGAN(n_latent, n_channel, n_classes)
                cgan.load_state_dict(torch.load(ood_path))
                ood_generator = OODCGAN(cgan, factor=args.ood_factor)
                # import pylab as plt
                # X_ood = ood_generator(*next(iter(train_loader)))
                # plt.imshow(torchvision.utils.make_grid(X_ood).permute(1, 2, 0))
                # plt.show()
            net = xEDLWrapper(
                net_init,
                lmb=args.lmb,
                ood_generator=ood_generator,
                evidence_func=torch.exp
            )

        net.fit(
            train_loader,
            val_loader_in,
            val_loader_out,
            weight_decay=args.weight_decay,
            n_epochs=args.n_epochs,
            device=args.device,
            lr=args.lr
        )

        net.load_best_weightdict()
        results_validation = net.score(val_loader_in, val_loader_out)

        results_test = net.score(test_loader_in, test_loader_out)

        # Evaluate Auroc
        val_aurocs.append(results_validation['auroc'])
        val_accuracy.append(results_validation['accuracy'])
        val_nll.append(results_validation['nll'])
        val_bs.append(results_validation['brier_score'])
        val_ece.append(results_validation['ece'])

        test_aurocs.append(results_test['auroc'])
        test_accuracy.append(results_test['accuracy'])
        test_nll.append(results_test['nll'])
        test_bs.append(results_test['brier_score'])
        test_ece.append(results_test['ece'])

    # Log metrics
    val_auroc_mean, val_auroc_std = np.mean(val_aurocs), np.std(val_aurocs)
    val_accuracy_mean, val_accuracy_std = np.mean(val_accuracy), np.std(val_accuracy)
    val_nll_mean, val_nll_std = np.mean(val_nll), np.std(val_nll)
    val_bs_mean, val_bs_std = np.mean(val_bs), np.std(val_bs)
    val_ece_mean, val_ece_std = np.mean(val_ece), np.std(val_ece)

    # Log test metrics
    test_auroc_mean, test_auroc_std = np.mean(test_aurocs), np.std(test_aurocs)
    test_accuracy_mean, test_accuracy_std = np.mean(test_accuracy), np.std(test_accuracy)
    test_nll_mean, test_nll_std = np.mean(test_nll), np.std(test_nll)
    test_bs_mean, test_bs_std = np.mean(test_bs), np.std(test_bs)
    test_ece_mean, test_ece_std = np.mean(test_ece), np.std(test_ece)

    val_results = {
        'val_auroc_mean': val_auroc_mean, 'val_auroc_std': val_auroc_std,
        'val_accuracy_mean': val_accuracy_mean, 'val_accuracy_std': val_accuracy_std,
        'val_nll_mean': val_nll_mean, 'val_nll_std': val_nll_std,
        'val_bs_mean': val_bs_mean, 'val_bs_std': val_bs_std,
        'val_ece_mean': val_ece_mean, 'val_ece_std': val_ece_std,
    }
    test_results = {
        'test_auroc_mean': test_auroc_mean, 'test_auroc_std': test_auroc_std,
        'test_accuracy_mean': test_accuracy_mean, 'test_accuracy_std': test_accuracy_std,
        'test_nll_mean': test_nll_mean, 'test_nll_std': test_nll_std,
        'test_bs_mean': test_bs_mean, 'test_bs_std': test_bs_std,
        'test_ece_mean': test_ece_mean, 'test_ece_std': test_ece_std,
    }

    print('Validation AUROC: {:.3f} +- {:.3f}  | Test AUROC: {:.3f} +- {:.3f}'.format(
        val_auroc_mean, val_auroc_std, test_auroc_mean, test_auroc_std
    ))

    # Save model and results
    save_path = os.path.join(file_directory, 'results')
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        test_results,
        os.path.join(save_path, 'test_results', '{}__{}.pth'.format(args.method_name, args.dataset))
    )
    weight_dict = net.state_dict()
    torch.save(
        weight_dict,
        os.path.join(save_path, 'state_dicts', '{}__{}.pth'.format(args.method_name, args.dataset))
    )


class OODCGAN():
    def __init__(self, cgan, q=.999, factor=1):
        self.cgan = cgan
        self.q = q
        self.factor = factor

        for m in cgan.generator.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
            elif isinstance(m, nn.Dropout2d):
                m.train()

    def __call__(self, X_batch, y_batch):
        device = X_batch.device
        n_samples = X_batch.size(0)
        self.cgan.to(device)
        noise = torch.randn(n_samples, self.cgan.n_latent, device=device)
        z_ood = generate_ood_hypersphere(noise, q=self.q, factor=self.factor)
        z_ood = z_ood[:, :, None, None]
        with torch.no_grad():
            X_ood = self.cgan.generator(z_ood, y_batch)
        return X_ood


class GenOODFromDataloader():
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iter_dataloader = iter(self.dataloader)

    def __call__(self, X_batch, y_batch):
        device = X_batch.device
        X_ood, _ = next(self.iter_dataloader, (None, None))
        if X_ood is None:
            self.iter_dataloader = iter(self.dataloader)
            X_ood, _ = next(self.iter_dataloader, (None, None))

        return X_ood.to(device)


def exp_evidence(logits):
    logits[logits <= -10] = 10
    logits[logits >= 10] = 10
    return torch.exp(logits)


if __name__ == "__main__":
    main()
