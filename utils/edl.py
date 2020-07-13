import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pylab as plt
import pytorch_lightning as pl

from torch.utils.data import DataLoader, TensorDataset
from utils import evaluation
from copy import deepcopy
from tqdm.auto import tqdm


def __test():
    torch.manual_seed(1)
    X = torch.cat((
        torch.Tensor(50, 2).normal_(-1), torch.Tensor(50, 2).normal_(1)
    )).float()
    y = torch.cat((torch.zeros(50), torch.ones(50))).long()
    X = (X - X.mean(0)) / X.std(0)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=256)

    # Train edl
    net = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 2))
    net.n_classes = 2
    net = EDLWrapper(net, prior=1)
    net.fit(train_loader, train_loader, train_loader, n_epochs=500, weight_decay=1e-4)
    print(net.score(train_loader, train_loader))

    with torch.no_grad():
        xx, yy = torch.from_numpy(np.mgrid[-6:6:0.1, -6:6:0.1]).float()
        unc = net.predict_proba(torch.stack((xx.flatten(), yy.flatten())).T)[:, 1].reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.contourf(xx, yy, unc, alpha=.5)
        plt.show()


class EDLWrapper(nn.Module):
    def __init__(self, net, prior=1, experiment=None):
        """Wrapper for EDL training.

        Args:
            net (nn.Module): Neural network to train.
            prior (int): Prior considerd for the Dirichlet.
            experiment: Comet experiment object.
        """
        super().__init__()
        self.net = net
        self.prior = prior
        self.experiment = experiment
        self.history = {}

        self.n_classes = net.n_classes

        # child = list(net.children())[-1]
        # while isinstance(child, nn.Sequential):
        #     child = list(child.children())[-1]
        # self.n_classes = child.out_features

        if self.experiment is not None:
            self.experiment.log_parameters({
                'Prior': self.prior,
            })

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        e = self.predict_evidence(x)
        a = e + self.prior
        return a / a.sum(-1, keepdim=True)

    def predict_unc(self, x):
        a = self.predict_evidence(x) + self.prior
        return self.n_classes / a.sum(-1)

    def predict_evidence(self, x):
        return exp_evidence(self(x))

    def get_unc(self, logits):
        a = exp_evidence(logits) + 1
        return self.n_classes / a.sum(-1)

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
            a = exp_evidence(logits_in[-1]) + 1
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
            a = exp_evidence(logits_in[-1]) + 1
            proba = a / a.sum(-1, keepdim=True)
            probas_out.append(proba)
        logits_out = torch.cat(logits_out).cpu()
        probas_out = torch.cat(probas_out).cpu()

        probas_in = probas_in.clamp(1e-8, 1-1e-8)
        probas_out = probas_out.clamp(1e-8, 1-1e-8)

        # Accuracy
        acc = (y_in == probas_in.argmax(-1)).float().mean().item()

        # Calibration Metrics
        criterion_ece = evaluation.ExpectedCalibrationError()
        criterion_nll = evaluation.NegativeLogLikelihood()
        criterion_bs = evaluation.BrierScore()
        criterion_cc = evaluation.CalibrationCurve()

        ece = criterion_ece(probas_in, y_in)
        nll = criterion_nll(probas_in, y_in)
        brier_score = criterion_bs(probas_in, y_in)
        calibration_curve = criterion_cc(probas_in, y_in)

        # OOD metrics
        unc_in, unc_out = self.get_unc(logits_in), self.get_unc(logits_out)
        auroc = evaluation.get_AUROC_ood(unc_in, unc_out)
        entropy_in = -torch.sum(probas_in * probas_in.log(), dim=-1)
        entropy_out = -torch.sum(probas_out * probas_out.log(), dim=-1)

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
            n_samples = 0
            running_loss = 0
            running_corrects = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                e = exp_evidence(self.net(X_batch))
                loss = self._edl_loss(e, y_batch, i_epoch)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    batch_size = X_batch.size(0)
                    n_samples += batch_size
                    running_loss += loss * batch_size
                    running_corrects += (e.argmax(-1) == y_batch).float().sum()

            train_loss = running_loss / n_samples
            train_acc = running_corrects / n_samples
            self.history['train_loss'].append(train_loss.item())
            self.history['train_acc'].append(train_acc.item())

            # Evaluation
            self.validation(val_loader_in, val_loader_out, i_epoch)

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

    def validation(self, val_loader_in, val_loader_out, i_epoch):
        device = list(self.net.parameters())[0].device
        self.net.eval()
        unc_in, unc_out = [], []
        n_samples, running_loss, running_corrects = 0, 0, 0
        for (X_batch_in, y_batch_in), (X_batch_out, y_batch_out) in zip(val_loader_in, val_loader_out):
            X_batch_in, y_batch_in = X_batch_in.to(device), y_batch_in.to(device)
            X_batch_out, y_batch_out = X_batch_out.to(device), y_batch_out.to(device)

            with torch.no_grad():
                logits_in = self.net(X_batch_in)
                logits_out = self.net(X_batch_out)
            loss = self._edl_loss(exp_evidence(logits_in), y_batch_in, epoch=i_epoch)

            unc_in.append(self.get_unc(logits_in))
            unc_out.append(self.get_unc(logits_out))

            batch_size = X_batch_in.size(0)
            n_samples += batch_size
            running_loss += loss * batch_size
            running_corrects += (logits_in.argmax(-1) == y_batch_in).float().sum()
        val_loss = running_loss / n_samples
        val_acc = running_corrects / n_samples

        unc_in = torch.cat(unc_in).cpu()
        unc_out = torch.cat(unc_out).cpu()

        # Logging
        self.history['val_loss'].append(val_loss.item())
        self.history['val_acc'].append(val_acc.item())
        self.history['val_auroc'].append(evaluation.get_AUROC_ood(unc_in, unc_out))

    def _edl_loss(self, evidence, y, epoch):
        """Implementation of the EDL loss

        Args:
            evidence: Predicted evidence.
            y: Ground truth labels.
            epoch: Current epoch starting with 0.

        Returns:
            float: The loss defined by evidential deep learning.
        """
        device = y.device

        y_one_hot = torch.eye(self.n_classes, device=device)[y]
        alpha = evidence + self.prior
        S = alpha.sum(-1, keepdim=True)
        p_hat = alpha / S

        # comp bayes risk
        bayes_risk = torch.sum((y_one_hot - p_hat)**2 + p_hat * (1 - p_hat) / S, -1)

        # kl-div term
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha  # hadmard first???
        S_alpha_tilde = alpha_tilde.sum(-1, keepdim=True)
        t1 = torch.lgamma(S_alpha_tilde) - math.lgamma(10) - \
            torch.lgamma(alpha_tilde).sum(-1, keepdim=True)
        t2 = torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) -
                                            torch.digamma(S_alpha_tilde)), dim=-1, keepdim=True)
        kl_div = t1 + t2

        lmbda = min((epoch + 1)/10, 1)
        loss = torch.mean(bayes_risk) + lmbda*torch.mean(kl_div)

        return loss


class EDL_lightning(pl.LightningModule):
    def __init__(self, net, weight_decay=0, lr=1e-3):
        super().__init__()
        self.net = net
        self.weight_decay = weight_decay
        self.lr = lr
        self.evidence_func = exp_evidence

        child = list(net.children())[-1]
        while isinstance(child, nn.Sequential):
            child = list(child.children())[-1]
        self.n_classes = child.out_features

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        a = self.evidence_func(self(x)) + 1
        return (a / a.sum(-1, keepdim=True)).squeeze()

    def predict_unc(self, x):
        a = self.evidence_func(self(x)) + 1
        return (self.n_classes / a.sum(-1, keepdim=True)).squeeze()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def training_step(self, batch, _):
        X, y = batch
        y_one_hot = torch.eye(self.n_classes, device=y.device)[y]
        alpha = self.evidence_func(self(X)) + 1
        S = alpha.sum(-1, keepdim=True)
        p_hat = alpha / S

        bayes_risk = torch.sum((y_one_hot - p_hat)**2 + p_hat * (1 - p_hat) / S, -1)
        kl_div = self.kl_div(alpha, y_one_hot)

        lmbda = min((self.current_epoch + 1)/10, 1)
        loss = torch.mean(bayes_risk) + lmbda*torch.mean(kl_div)

        acc = (alpha.argmax(-1) == y).float().mean()
        return {'loss': loss, 'acc': acc}

    def training_epoch_end(self, results):
        train_loss = torch.stack([result['loss'] for result in results]).mean()
        train_acc = torch.stack([result['acc'] for result in results]).mean()
        log = {'train_loss': train_loss, 'train_acc': train_acc}
        return {'log': log}

    def validation_step(self, batch, _):
        X, y = batch
        y_one_hot = torch.eye(self.n_classes, device=y.device)[y]
        alpha = self.evidence_func(self(X)) + 1
        S = alpha.sum(-1, keepdim=True)
        p_hat = alpha / S

        bayes_risk = torch.sum((y_one_hot - p_hat)**2 + p_hat * (1 - p_hat) / S, -1)
        kl_div = self.kl_div(alpha, y_one_hot)

        lmbda = min((self.current_epoch + 1)/10, 1)
        loss = torch.mean(bayes_risk) + lmbda*torch.mean(kl_div)
        acc = (alpha.argmax(-1) == y).float().mean()
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, results):
        train_loss = torch.stack([result['loss'] for result in results]).mean()
        train_acc = torch.stack([result['acc'] for result in results]).mean()
        log = {'val_loss': train_loss, 'val_acc': train_acc}
        return {'log': log}

    def kl_div(self, alpha, y_one_hot):
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha  # hadmard first???
        S_alpha_tilde = alpha_tilde.sum(-1, keepdim=True)
        t1 = torch.lgamma(S_alpha_tilde) - math.lgamma(10) - \
            torch.lgamma(alpha_tilde).sum(dim=-1, keepdim=True)
        t2 = torch.sum(
            (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_alpha_tilde)), dim=-1, keepdim=True
        )
        kl_div = t1 + t2
        return kl_div


def edl_loss(evidence, y, epoch, n_classes, ):
    """Implementation of the EDL loss

    Args:
        evidence: Predicted evidence.
        y: Ground truth labels.
        epoch: Current epoch starting with 0.

    Returns:
        float: The loss defined by evidential deep learning.
    """
    device = y.device

    y_one_hot = torch.eye(n_classes, device=device)[y]
    alpha = evidence + 1
    S = alpha.sum(-1, keepdim=True)
    p_hat = alpha / S

    # comp bayes risk
    bayes_risk = torch.sum((y_one_hot - p_hat)**2 + p_hat * (1 - p_hat) / S, -1)

    # kl-div term
    alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha  # hadmard first???
    S_alpha_tilde = alpha_tilde.sum(-1, keepdim=True)
    t1 = torch.lgamma(S_alpha_tilde) - math.lgamma(10) - torch.lgamma(alpha_tilde).sum(-1, keepdim=True)
    t2 = torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) -
                                        torch.digamma(S_alpha_tilde)), dim=-1, keepdim=True)
    kl_div = t1 + t2

    lmbda = min((epoch + 1)/10, 1)
    loss = torch.mean(bayes_risk) + lmbda*torch.mean(kl_div)

    return loss


def exp_evidence(logits):
    """Exponential evidence function.

    Args:
        logits: The output of the neural network.
    """
    logits[logits < -10] = -10
    logits[logits > 10] = 10
    return torch.exp(logits)


if __name__ == "__main__":
    __test()
