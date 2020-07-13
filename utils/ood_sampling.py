import math
import torch
import numpy as np
import pylab as plt


def __test():
    z = torch.randn(5000, 2)
    z_ood = generate_ood_hypersphere(z, q=.99999)

    plt.scatter(z[:, 0], z[:, 1])
    plt.scatter(z_ood[:, 0], z_ood[:, 1])
    plt.show()

    z = torch.randn(5000, 2)
    L = torch.cholesky(torch.Tensor([[3., 2.], [2., 3.]]))
    z = torch.matmul(z, L.T)
    z_ood = generate_ood_hypersphere(z)

    plt.scatter(z[:, 0], z[:, 1])
    plt.scatter(z_ood[:, 0], z_ood[:, 1])
    plt.show()

    z = torch.randn(5000, 2)
    L = torch.cholesky(torch.Tensor([[3., 2.], [2., 3.]]))
    z = torch.matmul(z, L.T)

    m = z.mean(0)
    cov = (z - m).T @ (z - m) / (len(z)-1)
    z_ood = generate_ood_hypersphere(z, m, cov)
    z_ood2 = soft_brownian_offset(z, .5, 1)

    plt.scatter(z[:, 0], z[:, 1])
    plt.scatter(z_ood[:, 0], z_ood[:, 1])
    plt.scatter(z_ood2[:, 0], z_ood2[:, 1])
    plt.show()


def uniform_gaussian_hypersphere(n_samples, z_mean, z_cov, q=.999):
    radius = - 2 * math.log(1-q)
    dim = len(z_cov)

    z = torch.randn(n_samples, dim)
    # Eq. from paper
    z_ood = z / torch.sqrt(torch.sum(z**2, dim=-1, keepdim=True))
    z_ood = (z_ood @ torch.cholesky(z_cov * radius).T) + z_mean

    return z_ood


def generate_ood_hypersphere(z, z_mean=None, z_cov=None, q=.999, factor=1):
    if z_mean is None or z_cov is None:
        z_mean = z.mean(0)
        z_cov = (z - z_mean).T@(z - z_mean) / (len(z)-1)

    z_cov_inv = torch.pinverse(z_cov)
    maha = torch.diagonal((z - z_mean) @ z_cov_inv @ (z - z_mean).T, 0)

    # Compute q-quantile
    idx = math.floor(len(maha)*q + 1)
    quantile = maha.sort()[0][idx - 1]

    # Eq. from paper
    z_ood = z / torch.sqrt(torch.sum(z**2, dim=-1, keepdim=True))
    z_ood = (z_ood @ torch.cholesky(z_cov * quantile*factor).T) + z_mean

    return z_ood


def pairwise_distances(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    # dist = torch.sqrt(dist)
    return torch.clamp(dist, 0.0, np.inf)


def soft_brownian_offset(X, d_min, d_off, hs_scale=None, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)
    device = X.device
    n_dim = X.shape[1]

    rnd_idx = np.random.choice(len(X), size=len(X))
    y = X[rnd_idx].clone()  # important to clone

    while True:
        dist = pairwise_distances(y, X)
        idx = dist.min(-1)[0] < d_min

        if torch.any(torch.isnan(dist)):
            raise ValueError('Pairwise distance returns nan.')

        if dist.min() > d_min:
            break

        offset = gaussian_hyperspheric_offset(
            len(y[idx]), n_dim=n_dim, hs_scale=hs_scale, device=device) * d_off
        y[idx] += offset
    return y


def gaussian_hyperspheric_offset(n_samples, mu=4, std=.7, n_dim=3, hs_scale=None, device=None):
    vec = torch.randn((n_samples, n_dim), device=device)
    vec /= torch.norm(vec, dim=1, keepdim=True)
    vec *= std*torch.randn((n_samples, 1), device=device) + mu

    if hs_scale is not None:
        vec *= hs_scale.std(axis=0) + hs_scale.mean(axis=0)
    return vec


if __name__ == "__main__":
    __test()
