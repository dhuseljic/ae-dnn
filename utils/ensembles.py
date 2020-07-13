import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pylab as plt

from torch.utils.data import DataLoader, TensorDataset
from utils import eval
from tqdm.auto import tqdm
from copy import deepcopy


def __test():
    torch.manual_seed(1)
    X = torch.cat((
        torch.Tensor(50, 2).normal_(-1), torch.Tensor(50, 2).normal_(1)
    )).float()
    y = torch.cat((torch.zeros(50), torch.ones(50))).long()
    X = (X - X.mean(0)) / X.std(0)

    net_init = nn.Sequential(nn.Linear(2, 50), nn.ReLU(), nn.Linear(50, 2))
    train_loader = DataLoader(TensorDataset(X, y), batch_size=256)

    # Train edl
    net = EnsembleWrapper(net=copy.deepcopy(net_init), n_members=10)
    net.fit(train_loader, train_loader, train_loader, n_epochs=500, weight_decay=0, lr=1e-3)
    print(net.score(train_loader, train_loader))

    with torch.no_grad():
        xx, yy = torch.from_numpy(np.mgrid[-10:10:0.1, -10:10:0.1]).float()
        unc = net(torch.stack((xx.flatten(), yy.flatten())).T).std(0).sum(-1).reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=y, zorder=2)
        plt.contourf(xx, yy, unc, alpha=.5, zorder=1)
        plt.show()


class EnsembleWrapper(nn.Module):
    def __init__(self, net, n_members=5):
        super().__init__()
        self.n_members = n_members

        self.nets = []
        for _ in range(self.n_members):
            net_ = copy.deepcopy(net)
            net_ = net_.apply(weight_reset)
            self.nets.append(net_)
        self.nets = nn.ModuleList(self.nets)

    def forward(self, x):
        return torch.stack([net(x) for net in self.nets])

    def predict_proba(self, x):
        preds = torch.stack([net(x) for net in self.nets])
        proba = F.softmax(preds, dim=-1)
        return proba.mean(0)

    def get_unc(self, logits):
        proba = F.softmax(logits, -1)
        return - torch.mean(proba.max(-1)[0], 0)

    def score(self, dataloader_in, dataloader_out):
        self.eval()
        device = list(self.parameters())[0].device

        probas_in = []
        y_in = []
        for X_batch, y_batch in dataloader_in:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                logits = self(X_batch)
            probas_in.append(F.softmax(logits, -1).mean(0))
            y_in.append(y_batch)
        probas_in = torch.cat(probas_in).cpu()
        y_in = torch.cat(y_in).cpu()

        probas_out = []
        for X_batch, y_batch in dataloader_out:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                logits = self(X_batch)
            probas_out.append(F.softmax(logits, -1).mean(0))
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
        unc_in, unc_out = -probas_in.max(1)[0], -probas_out.max(1)[0]
        auroc = eval.get_AUROC_ood(unc_in, unc_out)
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
        for weight_dict, net in zip(self.best_weightdict, self.nets):
            net.load_state_dict(weight_dict)

    def fit(
            self,
            train_loader,
            val_loader_in,
            val_loader_out,
            n_epochs=50,
            weight_decay=0,
            lr=1e-3,
            optimizer=None,
            lr_scheduler=None,
            verbose=1,
            device=None,
    ):
        """Training method.

        Args:
            train_loader: Pytorch dataloader.
            val_loader: Pytorch dataloader.
            n_epochs (int): Number of epochs to train.
            weight_decay (float)): Weight decay parameter,
            lr (float): Learning rate,
            optimizer: Pytorch optimizer.
            lr_scheduler: Pytorch learning rate scheduler.
            verbose: Whether to print the training progress.
            device: Device to train on.
        """
        # Does not use adversarial training, see ...
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Training on {}.'.format(device))
        self.histories = []
        self.best_weightdict = [0 for _ in range(self.n_members)]

        for i_net, net in enumerate(tqdm(self.nets)):
            self.history = {
                'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auroc': []
            }
            self.max_auroc = 0

            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

            # Training
            net.to(device)
            for i_epoch in range(n_epochs):
                net.train()
                n_samples = 0
                running_loss = 0
                running_corrects = 0
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                    out = net(X_batch)
                    loss = F.cross_entropy(out, y_batch)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        batch_size = X_batch.size(0)
                        n_samples += batch_size
                        running_loss += loss * batch_size
                        running_corrects += (out.argmax(-1) == y_batch).float().sum()

                train_loss = running_loss / n_samples
                train_acc = running_corrects / n_samples
                self.history['train_loss'].append(train_loss.item())
                self.history['train_acc'].append(train_acc.item())

                self.validation(net, val_loader_in, val_loader_out)

                if self.history['val_auroc'][-1] >= self.max_auroc:
                    self.max_auroc = self.history['val_auroc'][-1]
                    self.best_weightdict[i_net] = deepcopy(net.state_dict())

                if verbose:
                    print('[Ep {}] Loss={:.3f}/{:.3f} Acc={:.3f}/{:.3f} AUROC={:.3f}'.format(
                        i_epoch,
                        self.history['train_loss'][-1], self.history['val_loss'][-1],
                        self.history['train_acc'][-1], self.history['val_acc'][-1],
                        self.history['val_auroc'][-1],
                    ))
            net.cpu()

    def validation(self, net, val_loader_in, val_loader_out):
        net.eval()
        device = list(net.parameters())[0].device

        unc_in, unc_out = [], []
        n_samples, running_loss, running_corrects = 0, 0, 0
        for (X_batch_in, y_batch_in), (X_batch_out, y_batch_out) in zip(val_loader_in, val_loader_out):
            X_batch_in, y_batch_in = X_batch_in.to(device), y_batch_in.to(device)
            X_batch_out, y_batch_out = X_batch_out.to(device), y_batch_out.to(device)

            with torch.no_grad():
                logits_in = net(X_batch_in)
                logits_out = net(X_batch_out)
                proba_in = F.softmax(logits_in, -1)
                proba_out = F.softmax(logits_out, -1)

            loss = F.cross_entropy(logits_in, y_batch_in)
            unc_in.append(-proba_in.max(-1)[0])
            unc_out.append(-proba_out.max(-1)[0])

            batch_size = X_batch_in.size(0)
            n_samples += batch_size
            running_loss += loss * batch_size
            running_corrects += (logits_in.argmax(-1) == y_batch_in).float().sum()
        unc_in = torch.cat(unc_in).cpu()
        unc_out = torch.cat(unc_out).cpu()

        val_loss = running_loss / n_samples
        val_acc = running_corrects / n_samples

        # results = self.score(val_loader_in, val_loader_out)
        # Logging
        self.history['val_loss'].append(val_loss.item())
        self.history['val_acc'].append(val_acc.item())
        self.history['val_auroc'].append(eval.get_AUROC_ood(unc_in, unc_out))


def weight_reset(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()


if __name__ == "__main__":
    __test()
