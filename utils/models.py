import torch
import torch.nn as nn


def __test():
    torch.manual_seed(1)
    net = LeNet5(3, 10)
    net(torch.randn(1, 3, 32, 32))

    torch.manual_seed(1)
    net = LeNet5Dropout(3, 10)
    net(torch.randn(1, 3, 32, 32))


class LeNet5(nn.Module):
    """LeNet-5 architecture. Expects an image with shape=(n_channel, 32, 32)"""

    def __init__(self, n_channel, n_classes):
        super().__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.features = nn.Sequential(
            nn.Conv2d(self.n_channel, 6, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(inplace=True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84), nn.ReLU(inplace=True),
            nn.Linear(84, self.n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(len(x), -1)
        x = self.classifier(x)
        return x


class LeNet5Dropout(nn.Module):
    """LeNet-5 architecture used for the wrapper of Kendall."""

    def __init__(self, n_channel, n_classes, dropout_rate=.3):
        super().__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.dropout_rate = dropout_rate
        self.features = nn.Sequential(
            nn.Conv2d(self.n_channel, 6, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2, stride=2), nn.Dropout2d(self.dropout_rate, ),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(2, stride=2), nn.Dropout2d(self.dropout_rate, ),
            nn.Conv2d(16, 120, kernel_size=5), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate, ),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Dropout(self.dropout_rate, ),
            nn.Linear(84, self.n_classes + 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(len(x), -1)
        x = self.classifier(x)
        mean, logvar = torch.split(x, self.n_classes, dim=1)
        return mean, logvar


if __name__ == "__main__":
    __test()
