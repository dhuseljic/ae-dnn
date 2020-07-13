import numpy as np
import pylab as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import eval
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy


class NN(nn.Module):
    def __init__(self):
        self.n_classes = 2
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(.2), nn.Linear(2, 50), nn.Dropout(.2),
                                 nn.ReLU(), nn.Linear(50, self.n_classes+1))

    def forward(self, x):
        out = self.net(x)
        mean, logvar = torch.split(out, self.n_classes, dim=1)
        return mean, logvar


def __test__():
    torch.manual_seed(1)
    X = torch.cat((
        torch.Tensor(50, 2).normal_(-1), torch.Tensor(50, 2).normal_(1)
    )).float()
    y = torch.cat((torch.zeros(50), torch.ones(50))).long()
    X = (X - X.mean(0)) / X.std(0)
    train_loader = DataLoader(TensorDataset(X, y), batch_size=256)

    # Train edl
    net = NN()
    net.n_classes = 2
    net = DropoutWrapper(net)
    net.fit(train_loader, train_loader, train_loader, n_epochs=500, weight_decay=1e-4)
    print(net.score(train_loader, train_loader))

    with torch.no_grad():
        xx, yy = torch.from_numpy(np.mgrid[-6:6:0.1, -6:6:0.1]).float()
        unc = net(torch.stack((xx.flatten(), yy.flatten())).T)[:, 1].reshape(xx.shape)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.contourf(xx, yy, unc, alpha=.5)
        plt.show()


class DropoutWrapper(nn.Module):
    def __init__(self, net, n_mcsamples=50):
        super().__init__()
        self.net = net
        self.n_mcsamples = n_mcsamples

    def forward(self, x):
        mean = torch.stack([F.softmax(self.net(x)[0], -1) for _ in range(self.n_mcsamples)]).mean(0)
        return mean

    def predic_proba(self, x):
        mean = torch.mean(torch.stack([F.softmax(self.net(x)[0], -1) for _ in range(self.n_mcsamples)]), 0)
        return mean

    def predict_aleatoric_unc(self, x):
        var = torch.mean(torch.stack([torch.exp(self.net(x)[-1]) for _ in range(self.n_mcsamples)]), 0)
        return var

    def get_unc(self, logits):
        proba = logits
        proba = proba.clamp(1e-8, 1-1e-8)
        entropy = - torch.sum(proba * proba.log(), -1)
        return entropy

    def eval(self):
        self.train()

    def score(self, dataloader_in, dataloader_out):
        device = list(self.net.parameters())[0].device

        probas_in = []
        y_in = []
        for X_batch, y_batch in dataloader_in:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                probas_in.append(self(X_batch))
            y_in.append(y_batch)
        probas_in = torch.cat(probas_in).cpu()
        y_in = torch.cat(y_in).cpu()

        probas_out = []
        for X_batch, y_batch in dataloader_out:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            with torch.no_grad():
                probas_out.append(self(X_batch))
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
        # entropy_in = -torch.sum(probas_in * probas_in.log(), dim=-1)
        # entropy_out = -torch.sum(probas_out * probas_out.log(), dim=-1)

        unc_in, unc_out = -probas_in.max(1)[0], -probas_out.max(1)[0]
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
        """Training method.

        Args:
            train_loader: Pytorch dataloader.
            val_loader: Pytorch dataloader.
            n_epochs (int): Number of epochs to train.
            weight_decay (float)): Weight decay parameter,
            lr (float): Learning rate,
            optimizer: Pytorch optimizer.
            verbose: Whether to print the training progress.
            device: Device to train on.
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if optimizer is None:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)

        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auroc': []}
        self.max_auroc = 0
        print('Training on {}.'.format(device))

        # Training
        self.net.to(device)
        n_mcsamples_training = 2

        for i_epoch in tqdm(range(n_epochs)):
            self.net.train()
            n_samples = 0
            running_loss = 0
            running_corrects = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                mean, logvar = self.net(X_batch)
                std = torch.exp(.5*logvar)

                loss = 0
                for _ in range(n_mcsamples_training):
                    x_hat = mean + torch.randn_like(mean) * std  # TODO
                    # Eq. 12
                    loss += torch.exp(
                        x_hat.gather(1, y_batch.view(-1, 1)) -
                        torch.log(torch.sum(torch.exp(x_hat), dim=-1, keepdim=True))
                    )
                loss = torch.log(loss / n_mcsamples_training)
                loss = - torch.sum(loss)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.parameters(), max_norm=10)  # exploding gradient problem
                optimizer.step()

                with torch.no_grad():
                    batch_size = X_batch.size(0)
                    n_samples += batch_size
                    running_loss += loss
                    running_corrects += (mean.argmax(-1) == y_batch).float().sum()

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
        self.net.eval()
        device = list(self.net.parameters())[0].device
        # EvaluatOOion
        n_samples, running_loss, running_corrects = 0, 0, 0
        proba_in, proba_out = [], []
        for (X_batch_in, y_batch_in), (X_batch_out, y_batch_out) in zip(val_loader_in, val_loader_out):
            X_batch_in, y_batch_in = X_batch_in.to(device), y_batch_in.to(device)
            X_batch_out, y_batch_out = X_batch_out.to(device), y_batch_out.to(device)

            with torch.no_grad():
                mean_in, logvar_in = self.net(X_batch_in)

                proba_in.append(self.predic_proba(X_batch_in).clamp(1e-8, 1-1e-8))
                proba_out.append(self.predic_proba(X_batch_out).clamp(1e-8, 1-1e-8))

            std_in = torch.exp(.5*logvar_in)

            loss = 0
            for _ in range(2):
                x_hat = mean_in + torch.randn_like(mean_in) * std_in
                # Eq. 12
                loss += torch.exp(
                    x_hat.gather(1, y_batch_in.view(-1, 1)) -
                    torch.log(torch.sum(torch.exp(x_hat), dim=-1, keepdim=True))
                )
            loss = torch.log(loss / 2)
            loss = - torch.sum(loss)

            batch_size = X_batch_in.size(0)
            n_samples += batch_size
            running_loss += loss
            running_corrects += (mean_in.argmax(-1) == y_batch_in).float().sum()

        proba_in = torch.cat(proba_in).cpu()
        proba_out = torch.cat(proba_out).cpu()
        unc_in = - torch.sum(proba_in * proba_in.log(), -1)
        unc_out = - torch.sum(proba_out * proba_out.log(), -1)

        val_loss = running_loss / n_samples
        val_acc = running_corrects / n_samples

        # Logging
        self.history['val_loss'].append(val_loss.item())
        self.history['val_acc'].append(val_acc.item())
        self.history['val_auroc'].append(eval.get_AUROC_ood(unc_in, unc_out))


if __name__ == "__main__":
    __test__()
