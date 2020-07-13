import warnings
import torch
import torch.nn as nn

import numpy as np
import pylab as plt

from torch.utils.data import DataLoader, TensorDataset
from utils import eval, ood_sampling
from tqdm.auto import tqdm
from copy import deepcopy


def __test():
    torch.manual_seed(1)
    X = torch.cat((
        torch.Tensor(50, 2).normal_(-1), torch.Tensor(50, 2).normal_(1)
    )).float()
    y = torch.cat((torch.zeros(50), torch.ones(50))).long()
    X = (X - X.mean(0)) / X.std(0)

    train_loader = DataLoader(TensorDataset(X, y), batch_size=32)

    def gen_ood(X_batch, y_batch, var=2):
        return X_batch + torch.randn_like(X_batch) * var

    # Train edl
    net = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 2))
    net.n_classes = 2
    net = PriorNetWrapper(net, gamma=1, ood_generator=gen_ood)
    net.fit(train_loader, train_loader, train_loader, n_epochs=500)
    net.load_best_weightdict()

    with torch.no_grad():
        xx, yy = np.mgrid[-6:6:0.1, -6:6:0.1]
        zz = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
        uncertainty_dict = net.predict_unc(zz)
        proba = net.predict_proba(zz)[:, 1]
        uncertainty = uncertainty_dict['mutual_information']
        plt.subplot(121)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.contourf(xx, yy, uncertainty.view(xx.shape), alpha=.5)
        plt.subplot(122)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.contourf(xx, yy, proba.view(xx.shape), alpha=.5)
        plt.show()


class PriorNetWrapper(nn.Module):
    def __init__(self, net, gamma, ood_generator, precision=100):
        """Wrapper for PriorNet training.

        Args:
            net (nn.Module): Neural network to train.
            ood_generator (func): Function which generates ood samples.
            gamma (float): Weight parameter for the OOD KL divergence.
            precision (int): Maximal precision to consider when minimizing the KL.
        """
        super().__init__()
        self.net = net
        self.gamma = gamma
        self.ood_generator = ood_generator
        self.precision = precision
        self.n_classes = net.n_classes

        self.history = {}

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        logits = self.net(x)
        alphas = torch.exp(logits)
        proba = alphas / alphas.sum(-1, keepdim=True)
        return proba

    def predict_unc(self, x):
        logits = self.net(x)
        unc_dict = dirichlet_prior_network_uncertainty(logits)
        return unc_dict

    def score(self, dataloader_in, dataloader_out):
        self.eval()
        device = list(self.net.parameters())[0].device

        logits_in, y_in = [], []
        for X_batch, y_batch in dataloader_in:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                logits_in.append(self.net(X_batch))
                y_in.append(y_batch)
        logits_in = torch.cat(logits_in).cpu()
        y_in = torch.cat(y_in).cpu()
        alphas_in = torch.exp(logits_in)
        probas_in = alphas_in / alphas_in.sum(-1, keepdim=True)

        logits_out, y_out = [], []
        for X_batch, y_batch in dataloader_out:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                logits_out.append(self.net(X_batch))
                y_out.append(y_batch)
        logits_out = torch.cat(logits_out).cpu()
        y_out = torch.cat(y_out).cpu()
        alphas_out = torch.exp(logits_out)
        probas_out = alphas_out / alphas_out.sum(-1, keepdim=True)

        uncertainty_in = dirichlet_prior_network_uncertainty(logits_in)
        uncertainty_out = dirichlet_prior_network_uncertainty(logits_out)

        probas_in = probas_in.clamp(1e-8, 1-1e-8)
        probas_out = probas_out.clamp(1e-8, 1-1e-8)

        # Accuracy
        acc = (y_in == probas_in.argmax(-1)).float().mean().item()

        # Calibration Metrics
        criterion_ece = eval.ExpectedCalibrationError()
        criterion_nll = eval.NegativeLogLikelihood()
        criterion_bs = eval.BrierScore()
        criterion_cc = eval.CalibrationCurve()

        ece = criterion_ece(probas_in, y_in)
        nll = criterion_nll(probas_in, y_in)
        brier_score = criterion_bs(probas_in, y_in)
        calibration_curve = criterion_cc(probas_in, y_in)

        # OOD metrics
        unc_in, unc_out = uncertainty_in['mutual_information'], uncertainty_out['mutual_information']
        auroc = eval.get_AUROC_ood(unc_in, unc_out)
        entropy_in = uncertainty_in['entropy_of_expected']
        entropy_out = uncertainty_out['entropy_of_expected']

        self.train()
        results = {
            'accuracy': acc,
            # Calibration
            'ece': ece,
            'nll': nll,
            'brier_score': brier_score,
            'calibration_curve': calibration_curve,
            # OOD
            'auroc': auroc,
            'entropy_in': entropy_in,
            'entropy_out': entropy_out,
            'unc_in': unc_in,
            'unc_out': unc_out,
        }
        return results

    def load_best_weightdict(self):
        self.net.load_state_dict(self.best_weightdict)

    def fit(
            self,
            train_loader,
            val_loader_in,
            val_loader_out,
            n_epochs=50,
            weight_decay=0,
            lr=1e-3,
            optimizer=None,
            verbose=1,
            device=None,
    ):
        """Trains the neural network.

        Args:
            train_loader: Pytorch dataLoader.
            val_loader: Pytorch dataLoader.
            n_epochs (int): Number of epochs to train.
            weight_decay (float)): Weight decay parameter,
            lr (float): Learning rate,
            optimizer: Pytorch optimizer.
            lr_scheduler: Pytorch learning rate scheduler.
            verbose: Whether to print the training progress.
            device: Device to train on.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Training on {}.'.format(device))
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auroc': []}
        self.max_auroc = 0
        self.net.to(device)

        if optimizer is None:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        for i_epoch in tqdm(range(n_epochs)):
            # Train
            self.net.train()
            n_samples, running_loss, running_corrects = 0, 0, 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                X_ood = self.ood_generator(X_batch, y_batch)

                logits_in = self.net(X_batch)
                alphas_in = torch.exp(logits_in)
                target_in = torch.zeros_like(alphas_in).scatter_(1, y_batch[:, None], self.precision-1)+1
                loss_in = torch.mean(dirichlet_reverse_kl_divergence(alphas_in, target_in))

                logits_out = self.net(X_ood)
                alphas_out = torch.exp(logits_out)
                target_out = torch.ones_like(alphas_out)
                loss_out = torch.mean(dirichlet_reverse_kl_divergence(alphas_out, target_out))

                loss = loss_in + self.gamma*loss_out

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    batch_size = X_batch.size(0)
                    n_samples += batch_size
                    running_loss += loss * batch_size
                    running_corrects += (alphas_in.argmax(-1) == y_batch).float().sum()

            train_loss = running_loss / n_samples
            train_acc = running_corrects / n_samples
            self.history['train_loss'].append(train_loss.item())
            self.history['train_acc'].append(train_acc.item())

            self.validation(val_loader_in, val_loader_out)

            if self.history['val_auroc'][-1] >= self.max_auroc:
                self.max_auroc = self.history['val_auroc'][-1]
                self.best_weightdict = deepcopy(self.net.state_dict())

            if verbose:
                print('[Ep {}] Loss={:.3f}/{:.3f} Acc={:.3f}/{:.3f} AUROC={:.3f}'.format(
                    i_epoch,
                    self.history['train_loss'][-1], self.history['val_loss'][-1],
                    self.history['train_acc'][-1], self.history['val_acc'][-1], self.history['val_auroc'][-1]
                ))
        self.cpu()

    def validation(self, val_loader_in, val_loader_out):
        # Evaluation
        device = list(self.net.parameters())[0].device
        self.net.eval()
        n_samples, running_loss, running_corrects = 0, 0, 0
        unc_in, unc_out = [], []
        for (X_batch, y_batch), (X_ood, _) in zip(val_loader_in, val_loader_out):
            X_batch, y_batch, X_ood = X_batch.to(device), y_batch.to(device), X_ood.to(device)

            with torch.no_grad():
                logits_in = self.net(X_batch)
                logits_out = self.net(X_ood)

            alphas_in = torch.exp(logits_in)
            target_in = torch.zeros_like(alphas_in).scatter_(1, y_batch[:, None], self.precision-1)+1
            loss_in = torch.mean(dirichlet_reverse_kl_divergence(alphas_in, target_in))

            alphas_out = torch.exp(logits_out)
            target_out = torch.ones_like(alphas_out)
            loss_out = torch.mean(dirichlet_reverse_kl_divergence(alphas_out, target_out))

            unc_in.append(dirichlet_prior_network_uncertainty(logits_in)['mutual_information'])
            unc_out.append(dirichlet_prior_network_uncertainty(logits_out)['mutual_information'])

            loss = loss_in + self.gamma*loss_out

            batch_size = X_batch.size(0)
            n_samples += batch_size
            running_loss += loss * batch_size
            running_corrects += (alphas_in.argmax(-1) == y_batch).float().sum()
        val_loss = running_loss / n_samples
        val_acc = running_corrects / n_samples
        unc_in = torch.cat(unc_in).cpu()
        unc_out = torch.cat(unc_out).cpu()

        # Logging
        self.history['val_loss'].append(val_loss.item())
        self.history['val_acc'].append(val_acc.item())
        self.history['val_auroc'].append(eval.get_AUROC_ood(unc_in, unc_out))


def dirichlet_kl_divergence(alphas, target_alphas, precision=None, target_precision=None,
                            epsilon=1e-8):
    # based on original code from prior networks, see https://github.com/KaosEngineer/PriorNetworks/

    if not precision:
        precision = torch.sum(alphas, dim=1, keepdim=True)
    if not target_precision:
        target_precision = torch.sum(target_alphas, dim=1, keepdim=True)

    precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)
    assert torch.all(torch.isfinite(precision_term)).item()
    alphas_term = torch.sum(torch.lgamma(alphas + epsilon) - torch.lgamma(target_alphas + epsilon)
                            + (target_alphas - alphas) * (torch.digamma(target_alphas + epsilon)
                                                          - torch.digamma(
                                target_precision + epsilon)), dim=1, keepdim=True)
    assert torch.all(torch.isfinite(alphas_term)).item()

    cost = torch.squeeze(precision_term + alphas_term)
    return cost


def dirichlet_reverse_kl_divergence(alphas, target_alphas, precision=None, target_precision=None, epsilon=1e-8):
    # based on original code from prior networks, see https://github.com/KaosEngineer/PriorNetworks/
    return dirichlet_kl_divergence(alphas=target_alphas, target_alphas=alphas,
                                   precision=target_precision,
                                   target_precision=precision, epsilon=epsilon)


def dirichlet_prior_network_uncertainty(logits, epsilon=1e-10):
    # based on original code from prior networks, see https://github.com/KaosEngineer/PriorNetworks/
    alphas = torch.exp(logits)
    alpha0 = torch.sum(alphas, axis=1, keepdim=True)
    probs = alphas / alpha0

    conf = torch.max(probs, axis=1)[0]

    entropy_of_exp = -torch.sum(probs * torch.log(probs + epsilon), axis=1)
    expected_entropy = -torch.sum(
        (alphas / alpha0) * (torch.digamma(alphas + 1) - torch.digamma(alpha0 + 1.0)), axis=1)
    mutual_info = entropy_of_exp - expected_entropy

    epkl = torch.squeeze((alphas.shape[1] - 1.0) / alpha0)

    dentropy = torch.sum(
        torch.lgamma(alphas) - (alphas - 1.0) * (torch.digamma(alphas) - torch.digamma(alpha0)),
        axis=1, keepdim=True) - torch.lgamma(alpha0)

    uncertainty = {
        'confidence': conf,
        'entropy_of_expected': entropy_of_exp,
        'expected_entropy': expected_entropy,
        'mutual_information': mutual_info,
        'EPKL': epkl,
        'differential_entropy': torch.squeeze(dentropy),
    }

    return uncertainty


if __name__ == "__main__":
    __test()
