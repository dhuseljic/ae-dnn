import os
import numpy as np
import glob
import torch
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision.datasets.utils import download_url, extract_archive
from PIL import Image


def __test__():
    load_mnist_notmnist()
    load_svhn_cifar10()
    load_cifar5()


def load_mnist_notmnist():
    """Loads the datasets for mnist and notMNIST.

    Returns:
        (train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out): A tuple of 5 pytorch datasets.
    """
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), ])
    train_ds, val_ds_in, test_ds_in = load_mnist(transform=transform)

    val_ds_out, _, _ = load_kmnist(transform=transform)
    np.random.seed(0)
    idx = np.random.permutation(len(val_ds_out))[:len(val_ds_in)]
    val_ds_out = torch.utils.data.Subset(val_ds_out, indices=idx)

    torch.manual_seed(0)
    test_ds_out, _ = load_not_mnist(transform=transform)
    np.random.seed(0)
    idx = np.random.permutation(len(test_ds_out))[:len(test_ds_in)]
    test_ds_out = torch.utils.data.Subset(test_ds_out, indices=idx)

    assert len(val_ds_in) == len(val_ds_out)
    assert len(test_ds_in) == len(test_ds_out)
    return train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out


def load_svhn_cifar10():
    """Loads the datasets for SVHN and CIFAR10.

    Returns:
        (train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out): A tuple of 5 pytorch datasets.
    """
    transform = transforms.Compose([transforms.ToTensor(), ])
    train_ds, val_ds_in, test_ds_in = load_svhn(transform=transform)

    val_ds_out = torchvision.datasets.CIFAR100('~/datasets/', download=True, transform=transform)
    np.random.seed(0)
    idx = np.random.permutation(len(val_ds_out))[:len(val_ds_in)]
    val_ds_out = torch.utils.data.Subset(val_ds_out, indices=idx)

    test_ds_out, _, _ = load_cifar10(transform=transform)
    np.random.seed(0)
    idx = np.random.permutation(len(test_ds_out))[:len(test_ds_in)]
    test_ds_out = torch.utils.data.Subset(test_ds_out, indices=idx)

    assert len(val_ds_in) == len(val_ds_out)
    assert len(test_ds_in) == len(test_ds_out)
    return train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out


def load_cifar5(random_state=42):
    """Loads the datasets for CIFAR5 vs CIFAR5.

        In-Distribution: 'dog', 'frog', 'horse', 'ship', 'truck'
        Out-of-Distribution: 'airplane', 'automobile', 'bird', 'cat', 'deer'

    Args:
        random_state (int): The random state used for the train validation split.

    Returns:
        (train_ds, val_ds, test_ds_in, test_ds_out): A tuple of 4 pytorch datasets.
    """
    transform = transforms.Compose([transforms.ToTensor(), ])
    train_ds = datasets.CIFAR10('~/.datasets', train=True, transform=transform)
    test_ds = datasets.CIFAR10('~/.datasets', train=False, transform=transform)

    (X, y), (_, _) = _split_cifar(train_ds)

    np.random.seed(random_state)
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    n_train_samples = int(X.size(0) * .8)
    X_train = X[:n_train_samples]
    y_train = y[:n_train_samples]
    X_val = X[n_train_samples:]
    y_val = y[n_train_samples:]

    (X_in, y_in), (X_out, y_out) = _split_cifar(test_ds)

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds_in = torch.utils.data.TensorDataset(X_val, y_val)

    val_ds_out = torchvision.datasets.CIFAR100('~/datasets/', download=True, transform=transform)
    np.random.seed(0)
    idx = np.random.permutation(len(val_ds_out))[:len(val_ds_in)]
    val_ds_out = torch.utils.data.Subset(val_ds_out, indices=idx)

    test_ds_in = torch.utils.data.TensorDataset(X_in, y_in)
    test_ds_out = torch.utils.data.TensorDataset(X_out, y_out)

    assert len(val_ds_in) == len(val_ds_out)
    assert len(test_ds_in) == len(test_ds_out)
    return train_ds, val_ds_in, val_ds_out, test_ds_in, test_ds_out


def load_not_mnist(path="~/.datasets", train_test_split=0.8, transform=transforms.ToTensor(), random_state=42):
    ds = NotMNIST(root=path, download=True, transform=transform)
    torch.manual_seed(random_state)
    train_size = int(train_test_split * len(ds))
    test_size = len(ds) - train_size
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
    return train_ds, test_ds


def load_mnist(path="~/.datasets/", train_test_split=0.8, transform=transforms.ToTensor(), random_state=42,):
    """Loads the MNIST dataset.

    Args:
        path (str): Path for the files.
        train_test_split (float):
        transform:
        random_state (int):

    Returns:
        train_dataset, test_dataset, validation_dataset
    """
    torch.manual_seed(random_state)
    train_ds = datasets.MNIST(root=path, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=path, train=False, download=True, transform=transform)

    train_size = int(train_test_split * len(train_ds))
    test_size = len(train_ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, test_size])

    return train_ds, val_ds, test_ds


def load_kmnist(path="~/.datasets/", train_test_split=0.8, transform=transforms.ToTensor(), random_state=42):
    """Loads the Kuzushiji-MNIST dataset.

    Args:
        path (str): Path for the files.

    Returns:
        train_dataset, test_dataset, validation_dataset
    """
    torch.manual_seed(random_state)
    ds = datasets.KMNIST(root=path, train=True, download=True, transform=transform)
    test_ds = datasets.KMNIST(root=path, train=False, download=True, transform=transform)

    train_size = int(train_test_split * len(ds))
    test_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, test_size])
    return train_ds, val_ds, test_ds


def load_svhn(path="~/.datasets/", train_test_split=0.8, transform=transforms.ToTensor(), random_state=42):
    """Loads the SVHN dataset.

    Args:
        path (str): Path for the files.

    Returns:
        train_dataset, test_dataset, validation_dataset
    """
    torch.manual_seed(random_state)
    ds = datasets.SVHN(root=path, split='train', download=True, transform=transform)

    train_size = int(train_test_split * len(ds))
    test_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, test_size])

    test_ds = datasets.SVHN(root=path, split='test', download=True, transform=transform)
    return train_ds, val_ds, test_ds


def load_cifar10(path="~/.datasets/", train_test_split=0.8, transform=transforms.ToTensor(), random_state=42):
    """Loads the SVHN dataset.

    Args:
        path (str): Path for the files.

    Returns:
        train_dataset, test_dataset, validation_dataset
    """
    torch.manual_seed(random_state)
    ds = datasets.CIFAR10(root=path, train=True, download=True, transform=transform)

    train_size = int(train_test_split * len(ds))
    test_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, test_size])
    test_ds = datasets.CIFAR10(root=path, train=False, download=True, transform=transform)

    return train_ds, val_ds, test_ds


class NotMNIST(torch.utils.data.Dataset):
    """NotMnist Dataset.

    Args:
      root (string): Root directory of dataset where directory ``NotMNIST`` exists.
      train (bool, optional): Not used transform (callable, optional): A function/transform that takes in an
            PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the target and transforms it.
      download (bool, optional): If true, downloads the dataset from the internet and puts it in root
            directory.  If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root='~/.datasets', train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.data = []
        self.labels = []
        self.label_dict = {}
        self.bad_samples = [
            'RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png',
            'Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png',
        ]

        if download:
            self.download()

        self.base_folder = os.path.join('notMNIST', 'notMNIST_small')

        for i_class, class_path in enumerate(sorted(glob.glob(os.path.join(self.root, self.base_folder, '*')))):
            self.label_dict[i_class] = os.path.split(class_path)[-1]
            lbl = i_class

            for sample in os.listdir(class_path):
                if sample in self.bad_samples:
                    continue
                self.data.append(os.path.join(class_path, sample))
                self.labels.append(lbl)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download(self):
        path = os.path.join(self.root, 'notMNIST')
        link = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz'

        os.makedirs(path, exist_ok=True)
        download_url(link, path)
        extract_archive(os.path.join(path, 'notMNIST_small.tar.gz'))

    def __len__(self):
        return len(self.data)

    def decode_label(self, lbl: int):
        return self.label_dict[lbl]


class TinyImagenet(torch.utils.data.Dataset):
    """TinyImagenet -- https://tiny-imagenet.herokuapp.com/

    Args:
      root (string): Root directory of dataset where directory ``256_ObjectCategories`` exists.
      train (bool, optional): Not used transform (callable, optional): A function/transform that
                              takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the target and transforms it.
      download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
                                 If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root='~/.datasets', train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        self.data = []
        self.labels = []
        self.label_dict = {}

        if download:
            self.download()

        tmp_label_dict = {}
        with open(os.path.join(self.root, 'tiny-imagenet-200', 'words.txt')) as f:
            for line in f.readlines():
                folder_name, cls = line.split(maxsplit=1)

                tmp_label_dict[folder_name] = cls
        if train:
            self.base_folder = os.path.join('tiny-imagenet-200/train')
        if train is False:
            self.base_folder = os.path.join('tiny-imagenet-200/test')

        for i_class, class_path in enumerate(sorted(glob.glob(os.path.join(self.root, self.base_folder, '*')))):

            self.label_dict[i_class] = tmp_label_dict[os.path.split(class_path)[-1]]
            lbl = i_class

            for sample in os.listdir(os.path.join(class_path, 'images')):
                self.data.append(os.path.join(class_path, 'images', sample))
                self.labels.append(lbl)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)
        img = Image.open(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def download(self):
        path = os.path.join(self.root)
        link = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

        os.makedirs(path, exist_ok=True)
        download_url(link, path)
        extract_archive(os.path.join(path, 'tiny-imagenet-200.zip'))

    def __len__(self):
        return len(self.data)

    def decode_label(self, lbl: int):
        return self.label_dict[lbl]


def _split_cifar(ds):
    train_classes = []
    for lbl_name in ['dog', 'frog', 'horse', 'ship', 'truck']:
        train_classes.append(ds.class_to_idx[lbl_name])

    ood_classes = []
    for lbl_name in ['airplane', 'automobile', 'bird', 'cat', 'deer']:
        ood_classes.append(ds.class_to_idx[lbl_name])

    X_all, y_all = [], []
    for img, lbl in ds:
        X_all.append(img)
        y_all.append(lbl)
    X_all = torch.stack(X_all).float()
    y_all = torch.Tensor(y_all).long()

    X_in, y_in = [], []
    for lbl in train_classes:
        idx = y_all == lbl
        X_in.append(X_all[idx])
        y_in.append(y_all[idx])

    X_in = torch.cat(X_in)
    y_in = torch.cat(y_in) - 5

    X_out, y_out = [], []
    for lbl in ood_classes:
        idx = y_all == lbl
        X_out.append(X_all[idx])
        y_out.append(y_all[idx])

    X_out = torch.cat(X_out)
    y_out = torch.cat(y_out)
    return (X_in, y_in), (X_out, y_out)


if __name__ == '__main__':
    __test__()
