import math
from copy import deepcopy

import numpy as np
import pylab as plt
import sklearn.datasets as ds

import torch
import torch.nn as nn

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_

from utils import eval


def __test__():
    X, y = ds.make_blobs(400, random_state=4)
    X, y = torch.from_numpy(X).float(), torch.from_numpy(y).long() % 2
    X = (X - X.mean(0)) / X.std(0)

    # Train edl
    train_loader = DataLoader(TensorDataset(X, y), batch_size=256)

    def gen_ood(X_batch, y_batch, var=2):
        return X_batch + torch.randn_like(X_batch) * var
    inner_act = nn.ReLU()
    net_init = nn.Sequential(nn.Linear(2, 50), inner_act, nn.Linear(50, 50), inner_act, nn.Linear(50, 2))
    net_init.n_classes = 2

    net = xEDLWrapper(
        deepcopy(net_init),
        prior=1,
        lmb=.3,
        ood_generator=gen_ood,
        loss_type_in='categorical',
        loss_type_out='kl',
        evidence_func=torch.exp,
    )
    net.fit(
        train_loader,
        val_loader_in=train_loader,
        val_loader_out=train_loader,
        device='cpu',
        n_epochs=1000,
        weight_decay=1e-3
    )
    print(net.score(train_loader, train_loader))
    __plot_contour(net, X, y)


def __plot_contour(net, X, y):
    with torch.no_grad():
        mgrid_range, r = 5, .1
        torch.manual_seed(1)
        xx, yy = torch.from_numpy(np.mgrid[-mgrid_range:mgrid_range:r, -mgrid_range:mgrid_range:r]).float()
        plt.subplot(121)
        unc = net.predict_epistemic(torch.stack((xx.flatten(), yy.flatten())).T).reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.contourf(xx, yy, unc, alpha=.5, levels=np.linspace(0, 1, 6))

        plt.subplot(122)
        unc = net.predict_aleatoric(torch.stack((xx.flatten(), yy.flatten())).T).reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.contourf(xx, yy, unc, alpha=.5, levels=np.linspace(0, 1, 6))
        plt.show()


class xEDLWrapper(nn.Module):
    def __init__(
            self,
            net,
            lmb,
            ood_generator,
            prior=1,
            evidence_func=torch.exp,
            loss_type_in='categorical',
            loss_type_out='kl',
    ):
        """Wrapper for xEDL training.

        Args:
            net (nn.Module): Neural network to train.
            lmb (float): Influence of out-of-distribution.
            ood_generator: A function returning OOD samples. (Arg is a batch of in-dist samples.)
            prior (float): Prior considered for the Dirichlet.
            evidence_func (func): Output activation function.
            loss_type_in (str): Which distribution to consider for in samples.
            loss_type_out (str): Which distribution to consider for OOD samples.
        """
        super().__init__()
        self.net = net
        self.lmb = lmb
        self.ood_generator = ood_generator
        self.prior = prior
        self.loss_type_in = loss_type_in
        self.loss_type_out = loss_type_out
        self.evidence_func = evidence_func

        supported_loss_in = {
            'categorical',
            'cross_entropy',
            'squared_distance',
        }
        if self.loss_type_in not in supported_loss_in:
            raise ValueError(f'Loss type must be one of {supported_loss_in}.')

        supported_loss_out = {
            'exponential',
            'chi_squared',
            'uncertainty',
            'kl',
            'kl_reverse',
            'kl_symmetric',
            'chernoff_distance',
        }
        if self.loss_type_out not in supported_loss_out:
            raise ValueError(f'Loss type must be one of {supported_loss_out}.')

        self.n_classes = self.net.n_classes

    def forward(self, x):
        """Forward propagation. Returns the logits."""
        return self.net(x)

    def predict_epistemic(self, x):
        """Predicts the uncertainty of a sample. (K / alpha_0)"""
        with torch.no_grad():
            e = self.predict_evidence(x)

        a = e + self.prior
        return self.n_classes * self.prior / a.sum(-1, keepdim=True)

    def predict_aleatoric(self, x):
        """Predicts the uncertainty of a sample. (K / alpha_0)"""
        with torch.no_grad():
            e = self.predict_evidence(x)

        a = e + self.prior
        proba_in = (a / a.sum(-1, keepdim=True)).clamp_(1e-8, 1-1e-8)
        entropy = - torch.sum((proba_in * proba_in.log()), dim=-1)
        normalized_entropy = entropy / math.log(self.n_classes)
        return normalized_entropy

    def get_unc(self, logits):
        e = self.evidence_func(logits)
        a = e + self.prior
        return self.n_classes * self.prior / a.sum(-1, keepdim=True).squeeze()

    def predict_proba(self, x):
        """Predicts the probabilities as the mean of a Dirichlet distribution."""
        e = self.predict_evidence(x)
        a = e + self.prior
        return a / torch.sum(a, dim=-1, keepdim=True)

    def predict_evidence(self, x):
        return self.evidence_func(self(x))

    def score(self, dataloader_in, dataloader_out):
        self.eval()
        device = list(self.net.parameters())[0].device

        logits_in = []
        probas_in = []
        y_in = []
        for X_batch, y_batch in dataloader_in:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                logits_in.append(self.net(X_batch))
            a = self.evidence_func(logits_in[-1]) + self.prior
            proba = a / a.sum(-1, keepdim=True)
            probas_in.append(proba)
            y_in.append(y_batch)
        logits_in = torch.cat(logits_in).cpu()
        probas_in = torch.cat(probas_in).cpu()
        y_in = torch.cat(y_in).cpu()

        logits_out = []
        probas_out = []
        for X_batch, y_batch in dataloader_out:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                logits_out.append(self.net(X_batch))
            a = self.evidence_func(logits_in[-1]) + self.prior
            proba = a / a.sum(-1, keepdim=True)
            probas_out.append(proba)
        logits_out = torch.cat(logits_out).cpu()
        probas_out = torch.cat(probas_out).cpu()

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
        entropy_in = -torch.sum(probas_in * probas_in.log(), dim=-1)
        entropy_out = -torch.sum(probas_out * probas_out.log(), dim=-1)
        unc_in, unc_out = self.get_unc(logits_in), self.get_unc(logits_out)
        auroc = eval.get_AUROC_ood(unc_in, unc_out)

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
        self.train()
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
            lr_scheduler_step=None,
            verbose=1,
            device=None,
    ):
        """Trains the neural network.

        Args:
            train_loader: Pytorch dataLoader.
            val_loader_in: Pytorch dataLoader for in-distribution samples.
            val_loader_out: Pytorch dataLoader for out-of-distribution samples.
            n_epochs (int): Number of epochs to train.
            weight_decay (float)): Weight decay parameter,
            lr (float): Learning rate,
            optimizer: Pytorch optimizer. Weight decay and learning rate will be neglected.
            lr_scheduler_step: When decrease the lr with `gamma=.1`.
            verbose: Whether to print the training progress.
            device: Device to train on.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Training on {}.'.format(device))
        self.history = {'loss_in': [], 'loss_out': [], 'train_acc': [], 'val_acc': [], 'val_auroc': []}
        self.max_auroc = 0
        self.net.to(device)
        self._onehot_encoder = torch.eye(self.n_classes, device=device)
        lr_scheduler = None

        if optimizer is None:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
            if lr_scheduler_step is not None:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=.1)

        for i_epoch in tqdm(range(n_epochs)):
            running_loss_in = 0
            running_loss_out = 0
            running_corrects = 0

            self.net.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                # Compute the Loss of in-distribution samples
                e_in = self.evidence_func(self.net(X_batch)) + self.prior
                loss_in = torch.mean(self._loss_in(e_in, y_batch))

                # Compute the loss on out-of-distribution samples
                X_out = self.ood_generator(X_batch, y_batch)
                e_out = self.evidence_func(self.net(X_out))
                loss_out = torch.mean(self._loss_out(e_out))

                # Optimize Loss
                loss = (1-self.lmb)*loss_in + self.lmb*(loss_out)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), 10.0)
                optimizer.step()

                running_loss_in += loss_in.item() * X_batch.size(0)
                running_loss_out += loss_out.item() * X_batch.size(0)
                running_corrects += (e_in.argmax(-1) == y_batch).float().sum()

            if lr_scheduler is not None:
                lr_scheduler.step()

            self.history['loss_in'].append(running_loss_in / len(train_loader.dataset))
            self.history['loss_out'].append(running_loss_out / len(train_loader.dataset))
            self.history['train_acc'].append(running_corrects.item() / len(train_loader.dataset))

            self.validation(val_loader_in, val_loader_out)

            if self.history['val_auroc'][-1] >= self.max_auroc:
                self.max_auroc = self.history['val_auroc'][-1]
                self.best_weightdict = deepcopy(self.net.state_dict())

            if verbose or i_epoch == n_epochs - 1:
                print('[Ep {:03d}] Loss (In/Out) = {:.3f}/{:.3f} | Acc={:.3f}/{:.3f} | AUROC={:.3f}'.format(
                    i_epoch,
                    self.history['loss_in'][-1], self.history['loss_out'][-1],
                    self.history['train_acc'][-1], self.history['val_acc'][-1],
                    self.history['val_auroc'][-1],
                ))

    def validation(self, val_loader_in, val_loader_out):
        self.net.eval()

        (preds, unc_in), lbls = eval.eval_on_dataloader(
            self.net, val_loader_in, (lambda x: x.argmax(-1), self.get_unc), return_labels=True
        )
        unc_out = eval.eval_on_dataloader(self.net, val_loader_out, self.get_unc)

        self.history['val_acc'].append((preds == lbls).float().mean(0).item())
        self.history['val_auroc'].append(eval.get_AUROC_ood(unc_in, unc_out))

    def _loss_in(self, e_in, y_in):
        """Loss types for in-distribution samples."""
        a_in = e_in + self.prior
        S_in = a_in.sum(-1, keepdim=True)
        if self.loss_type_in == 'categorical':
            # Assume Expectation of a Categorical --> Dirichlet-Categorical
            loss = torch.log(S_in) - torch.log(a_in.gather(1, y_in.view(-1, 1)))

        elif self.loss_type_in == 'cross_entropy':
            loss = torch.digamma(S_in) - torch.digamma(a_in.gather(1, y_in.view(-1, 1)))

        elif self.loss_type_in == 'squared_distance':
            y_onehot = self._onehot_encoder[y_in]
            p_in = a_in / S_in
            loss = torch.sum((y_onehot - p_in) ** 2 + p_in * (1 - p_in) / (S_in + 1), dim=-1)

        return loss

    def _loss_out(self, e_out):
        """Loss types for out-of-distribution samples."""
        prior_out = 1  # self.prior

        if self.loss_type_out == 'exponential':
            loss = torch.sum(e_out, -1, keepdim=True)

        elif self.loss_type_out == 'chi_squared':
            a_out = e_out + prior_out
            df = prior_out + 2
            loss = torch.sum((df / 2 - 1) * torch.log(a_out) + a_out / 2, dim=-1, keepdim=True)

        elif self.loss_type_out == 'uncertainty':
            S_out = (e_out + prior_out).sum(-1, keepdim=True)
            unc = self.n_classes / S_out
            loss = -unc

        elif self.loss_type_out == 'kl':
            a_out = e_out + prior_out
            a_target = torch.full(a_out.size(), prior_out, dtype=float, device=e_out.device)
            kl = kl_dirichlet(a_out, a_target)
            loss = kl

        elif self.loss_type_out == 'kl_reverse':
            a_out = e_out + prior_out
            a_target = torch.full(a_out.size(), prior_out, dtype=float, device=e_out.device)
            kl = kl_dirichlet(a_target, a_out)
            loss = kl

        elif self.loss_type_out == 'kl_symmetric':
            a_out = e_out + prior_out
            a_target = torch.full(a_out.size(), prior_out, dtype=float, device=e_out.device)
            kl = .5*(kl_dirichlet(a_out, a_target) + kl_dirichlet(a_target, a_out))
            loss = kl

        elif self.loss_type_out == 'chernoff_distance':
            a_out = e_out + prior_out
            a_target = torch.full(a_out.size(), prior_out, dtype=float, device=e_out.device)
            dist = chernoff_distance(a_out, a_target)
            loss = dist

        return loss


class Mixup():
    def __init__(self, alpha, device=None):
        self.alpha = alpha
        self.beta_dist = torch.distributions.beta.Beta(alpha, alpha)
        self.device = device

        if alpha < 0:
            raise ValueError('Argument alpha should be greater or equal 0.')

    def __call__(self, X_batch, y_onehot):
        if self.alpha > 0:
            lmbda = np.random.beta(self.alpha, self.alpha, size=(X_batch.size(0), 1))
            lmbda = torch.from_numpy(lmbda).float().to(self.device)

            shuffle_idx = np.random.permutation(len(X_batch))
            X_batch_shuffled = X_batch[shuffle_idx]
            y_batch_shuffled = y_onehot[shuffle_idx]

            lmbda_img = lmbda[:, :, None, None]

            X_mixup = lmbda_img * X_batch + (1 - lmbda_img) * X_batch_shuffled
            y_mixup = lmbda * y_onehot + (1 - lmbda) * y_batch_shuffled
            return X_mixup, y_mixup
        elif self.alpha == 0:
            return X_batch, y_onehot


def chernoff_distance(alpha, beta, lmb=.5):
    """Computes the chernoff_distance. If `lmb=.5` it is equal to Bhattacharyya distances.

    References:
        [1] Rauber, Thomas W., Tim Braun, and Karsten Berns. "Probabilistic distance measures of the Dirichlet
            and Beta distributions." Pattern Recognition 41.2 (2008): 637-645.
    """
    alpha_0 = torch.sum(alpha, dim=-1, keepdim=True)
    beta_0 = torch.sum(beta, dim=-1, keepdim=True)

    t1 = torch.lgamma(torch.sum(lmb*alpha + (1-lmb)*beta, dim=-1, keepdim=True))
    t2 = lmb*torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    t3 = (1-lmb)*torch.sum(torch.lgamma(beta), dim=-1, keepdim=True)
    t4 = torch.sum(torch.lgamma(lmb*alpha + (1-lmb)*beta), dim=-1, keepdim=True)
    t5 = lmb*torch.lgamma(alpha_0)
    t6 = (1-lmb)*torch.lgamma(beta_0)
    return t1 + t2 + t3 - t4 - t5 - t6


def kl_dirichlet(alpha, beta):
    """Computes the KL-Divergence between two dirichlet distributions.

    Args:
        alpha: (N x K)-Tensor where the K-dimension describes the parameters of the dirichlet.
        beta: (N x K)-Tensor where the K-dimension describes the parameters of the dirichlet.
    """
    alpha_0 = torch.sum(alpha, dim=-1, keepdim=True)
    beta_0 = torch.sum(beta, dim=-1, keepdim=True)
    t1 = torch.lgamma(alpha_0) - torch.sum(torch.lgamma(alpha), dim=-1, keepdim=True)
    t2 = torch.lgamma(beta_0) - torch.sum(torch.lgamma(beta), dim=-1, keepdim=True)
    t3 = torch.sum((alpha - beta) * (torch.digamma(alpha) - torch.digamma(alpha_0)), dim=-1, keepdim=True)
    return t1 - t2 + t3


def exp_evidence(logits):
    """Exponential evidence function.

    Args:
        logits: The output of the neural network.
    """
    logits[logits < -10] = -10
    logits[logits > 10] = 10
    return torch.exp(logits)


if __name__ == "__main__":
    __test__()
