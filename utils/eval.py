import numpy as np
import torch
import math
import warnings
from sklearn.metrics import roc_auc_score


def eval_on_dataloader(net, dataloader, func, return_labels=False):
    """Evaluate the output of a net on a userdefined method on a dataloader.

    Args:
        net (nn.Module): Neural network used for forward propagation.
        dataloader: Pytorch dataloader.
        func (func or array_like): Function that is to be evaluated. Should return (batch_size,D) shaped
            tensor.
        return_labels: Whether to return all labels or not.

    Returns:
        torch.tensor: A tensor containg the values retuned by `func` on all samples of the dataloader. If
                    `return_labels` is true, a tuple of tensors is returned.
    """
    device = next(net.parameters()).device
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise ValueError('Make sure the second argument is a valid pytorch dataloader.')

    is_list = isinstance(func, list) or isinstance(func, tuple)

    if is_list:
        func_vals = []
        [func_vals.append([]) for f in func]
    else:
        func_vals = []

    with torch.no_grad():
        net.to(device)
        labels = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            out = net(X_batch)
            if is_list:
                for i_f, f in enumerate(func):
                    func_vals[i_f].append(f(out))
            else:
                func_vals.append(func(out))
            labels.append(y_batch)
        if return_labels:
            if is_list:
                return [torch.cat(f_vals).cpu() for f_vals in func_vals], torch.cat(labels)
            else:
                return torch.cat(func_vals).cpu(), torch.cat(labels).cpu()
        else:
            if is_list:
                return [torch.cat(f_vals).cpu() for f_vals in func_vals]
            else:
                return torch.cat(func_vals).cpu()


def get_AUROC_ood(unc_in, unc_out):
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC) for out-of-distribution (OOD)
    data. The OOD data is assumed to be the positive class in a binary classification setting.

    Args:
        unc_in: The uncertainty of in-distribution samples.
        unc_out: The uncertainty of out-of-distribution samples

    Returns:
        float: The ROC AUC score.
    """
    with torch.no_grad():
        if len(unc_in) != len(unc_out):
            warnings.warn('Unbalanced in- and out-of-distribution Problem. ROCAUC might be unrepresentative.')
        in_labels = torch.zeros(len(unc_in))
        ood_labels = torch.ones(len(unc_out))
        return roc_auc_score(torch.cat((in_labels, ood_labels)), torch.cat((unc_in, unc_out)))


def eval_accuracy_on_dataloader(net, dataloader):
    """Evaluate the accuracy of a net on a dataloader.

    Args:
        net (nn.Module): Neural network used for forward propagation.
        dataloader: Pytorch dataloader.

    Returns:
        float: The accuracy evaluated on the dataloader.
    """
    device = next(net.parameters()).device
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise ValueError('Make sure the second argument is a pytorch dataloader.')
    with torch.no_grad():
        net.to(device).eval()
        running_corrects = 0
        n_samples = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = net(X_batch)
            preds = out.argmax(-1)
            n_samples += y_batch.size(0)
            running_corrects += (preds == y_batch).float().sum()
        return (running_corrects / n_samples).item()


def eval_entropies_on_dataloader(net, dataloader, proba_func, device=None):
    """Evaluate entropies of a net on a dataloader.

    Args:
        net (nn.Module): Neural network used for forward propagation.
        dataloader: Pytorch dataloader.
        proba_func: A function which transforms the logits to probas.
        device: Device to evaluate on.

    Returns:
        torch.tensor: A tensor containg the entropies for each sample of the dataloader.
    """

    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise ValueError('Make sure the second argument is a pytorch dataloader.')
    with torch.no_grad():
        net.to(device)
        entropies = []
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            out = net(X_batch)
            proba = proba_func(out)
            proba = torch.clamp(proba, 1e-5, 1 - 1e-5)
            entropy = - torch.sum(proba * torch.log(proba), -1)
            entropies.append(entropy.cpu())
        return torch.cat(entropies)


def eval_ece_on_dataloader(net, dataloader, proba_func, device=None):
    """Evaluate the Expected Calibration Error (ECE) of a net on a dataloader.

    Args:
        net (nn.Module): Neural network used for forward propagation.
        dataloader: Pytorch dataloader.
        proba_func: A function which transforms the logits to probas.
        device: Device to evaluate on.

    Returns:
        float: The ECE.
    """
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise ValueError('Make sure the second argument is a pytorch dataloader.')
    criterion = ExpectedCalibrationError()
    with torch.no_grad():
        net.to(device)
        y_proba = []
        y = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = net(X_batch)
            proba = proba_func(logits)
            y_proba.append(proba)
            y.append(y_batch)
        y_proba = torch.cat(y_proba)
        y = torch.cat(y)
        return criterion(y_proba, y)


def get_cc(net, dataloader, proba_func, device=None):
    """Evaluate the Calibration Curve of a net on a dataloader.

    Args:
        net (nn.Module): Neural network used for forward propagation.
        dataloader: Pytorch dataloader.
        proba_func: A function which transforms the logits to probas.
        device: Device to evaluate on.

    Returns:
        tuple: Lists containing `mean_conf`, `bin_acc`, and `bin_count`.
    """
    if not isinstance(dataloader, torch.utils.data.DataLoader):
        raise ValueError('Make sure the second argument is a pytorch dataloader.')
    criterion = CalibrationCurve()
    with torch.no_grad():
        net.to(device)
        y_proba = []
        y = []
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = net(X_batch)
            proba = proba_func(logits)
            y_proba.append(proba)
            y.append(y_batch)
        y_proba = torch.cat(y_proba)
        y = torch.cat(y)
        return criterion(y_proba, y)


class CalibrationMetric:
    def __init__(self, how='dataloader'):
        if how not in {'dataloader', 'tensor', 'dataset'}:
            raise ValueError('The variable how must be one of `dataloader`, `tensor`, `dataset`.')


class BrierScore:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_proba, y):
        device = y_proba.device
        n_classes = y_proba.size(-1)

        y_one_hot = torch.eye(n_classes, device=device)[y.long()]
        return self.brier_score(y_proba, y_one_hot).item()

    def brier_score(self, y_proba, y_one_hot):
        return torch.mean((y_one_hot - y_proba) ** 2)


class NegativeLogLikelihood:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, y_proba, y):
        return self.nll(y_proba, y).item()

    def nll(self, y_proba, y):
        return -torch.mean(torch.log(y_proba[range(len(y)), y.long()] + 1e-20))


class CalibrationCurve():
    """Returns the mean_confidence and accuracy of different bins.

    Args:
        out (tensor): Tensor containing class probabilities or logits. (NxC)
        y (tensor): Tensor containing integers which corresponds to classes. (Cx1)

    Returns:
        mean_conf (list): The confidence mean of each bin.
        acc (list): The accuracy of each bin.
        bin_count (list): Number of samples in the respective bin.

    Code is based on https://github.com/gpleiss/temperature_scaling.
    """

    def __init__(self, n_bins=15, **kwargs):
        super().__init__(**kwargs)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, y_proba, y):
        # device = out.device

        y_conf, y_pred = y_proba.max(-1)

        mean_conf, acc, bin_count = [], [], []

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            idx_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
            n_bin = sum(idx_bin)

            if n_bin > 0:
                acc_bin = torch.mean((y_pred[idx_bin] == y[idx_bin]).float())
                mean_conf_bin = torch.mean(y_conf[idx_bin])

                acc.append(acc_bin.item())
                mean_conf.append(mean_conf_bin.item())
                bin_count.append(n_bin.item())
            else:
                acc.append(np.nan)
                mean_conf.append(np.nan)
                bin_count.append(np.nan)
        return mean_conf, acc, bin_count


class ExpectedCalibrationError:
    """Returns the expected calibration error for a given bin size.

    Args:
        y_proba (tensor): Tensor containing returned class probabilities. (NxC)
        y (tensor): Tensor containing integers which corresponds to classes. (Cx1)

    Returns:
       tensor: The expected calibration error

    Code is based on https://github.com/gpleiss/temperature_scaling.
    """

    def __init__(self, n_bins=15, **kwargs):
        super().__init__(**kwargs)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def __call__(self, y_proba, y):
        # device = y_proba.device
        n_samples = y.size(0)
        y_conf, y_pred = y_proba.max(-1)

        # Eq. 3 from Guo paper
        ece = 0
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            idx_bin = (y_conf > bin_lower) & (y_conf <= bin_upper)
            n_bin = sum(idx_bin).float()

            if n_bin > 0:
                acc_bin = torch.mean((y_pred[idx_bin] == y[idx_bin]).float())
                mean_conf_bin = torch.mean(y_conf[idx_bin])

                ece += n_bin * torch.abs(acc_bin - mean_conf_bin)

        return ece.item()/n_samples


class Accuracy:
    """Computes the accuracy.

    Args:
        output_activation (func): Method that defines the output activation function.
        from_dataloader (bool): Compute the accuracy from a dataloader.
        from_dataset (bool): Compute the accuracy from a dataset.
    """

    def __init__(self, output_activation=lambda x: x, from_dataset=False, from_dataloader=False):
        self.output_activation = output_activation
        self.from_dataloader = from_dataloader
        self.from_dataset = from_dataset

    def __call__(self, **kwargs):
        with torch.no_grad():

            if not self.from_dataloader and not self.from_dataset:
                y_pred = kwargs['y_pred']
                y_true = kwargs['y_true']
                assert y_pred.shape == y_true.shape
                return torch.mean((y_pred == y_true).float()).item()

            if self.from_dataloader:
                net = kwargs['net']
                dataloader = kwargs['dataloader']

                running_corrects = 0
                n_samples = 0
                for X_batch, y_batch in dataloader:
                    out = net(X_batch)
                    y_proba = self.output_activation(out)
                    y_pred = y_proba.argmax(-1)

                    assert y_pred.shape == y_batch.shape
                    running_corrects += sum((y_pred == y_batch).float())
                    n_samples += X_batch.size(0)

                return running_corrects / n_samples

            if self.from_dataset:
                # TODO
                pass


def get_ood_score(entropy_in, entropy_out, n_classes):
    """Deprecated! Use ROCAUC instead."""
    warnings.warn('Deprecated use `get_AUROC_ood` instead.')
    with torch.no_grad():
        # p_uniform = torch.full((n_classes,), 1 / n_classes)
        # max_entropy = -torch.sum(p_uniform * p_uniform.log())
        max_entropy = math.log(10)

        in_score = entropy_in.mean() / max_entropy
        out_score = (max_entropy - entropy_out.mean()) / max_entropy

        return ((in_score + out_score) / 2).item()
