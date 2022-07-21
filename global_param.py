from typing import Tuple

import torch

from distributions import NormalInverseWishart, MatrixNormalInverseWishart


def initialize_global_lds_parameters(n, scale=1.0):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    nu = torch.tensor([n + 1])
    Phi = 2 * scale * (n + 1) * torch.eye(n)
    mu_0 = torch.zeros(n)
    kappa = torch.tensor([1 / (2 * scale * n)])

    M = torch.eye(n)
    K = 1 / (2 * scale * n) * torch.eye(n)

    init_state_prior = NormalInverseWishart(torch.zeros_like(nu)).standard_to_natural(
        kappa.unsqueeze(0), mu_0.unsqueeze(0), Phi.unsqueeze(0), nu.unsqueeze(0)
    )
    dynamics_prior = MatrixNormalInverseWishart(
        torch.zeros_like(nu)
    ).standard_to_natural(nu, Phi, M, K)

    dynamics_prior = tuple([d.to(device) for d in dynamics_prior])

    return init_state_prior.to(device), dynamics_prior


def natural_gradient(
    stats, eta_theta, eta_theta_prior, N, num_batches,
):
    """
    Natural gradient for the global variational parameters eta_theta

    Parameters
    ----------
    stats:
       Sufficient statistics.
    eta_theta:
        Posterior natural parameters for global variables.
    eta_theta_prior:
        Prior natural parameters for global variables.
    N:
        Number of data-points.
    num_batches:
        Number of batches in the data.
    """

    def nat_grad(prior, posterior, s) -> torch.Tensor:
        # if isinstance(prior, Tuple):
        #     return [
        #         -1.0 / N * prior[j] - posterior[j] + num_batches * s[j]
        #         for j in range(len(prior))
        #     ]
        return -1.0 / N * (prior - posterior + num_batches * s)

    value = [
        nat_grad(eta_theta_prior[i], eta_theta[i], stats[i])
        for i in range(len(eta_theta))
    ]
    return value
