import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import generate_data
from distributions import NormalInverseWishart, MatrixNormalInverseWishart
from distributions.gaussian import natural_to_info
from global_param import initialize_global_lds_parameters, natural_gradient, gradient_descent
from lds import (
    info_observation_params,
    info_kalman_filter,
    info_kalman_smoothing,
    info_sample_backward,
    sample_forward_messages,
    sample_backward_messages,
)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=np.VisibleDeprecationWarning)

if __name__ == "__main__":
    latents, obs = generate_data(100, noise_scale=1)

    obs = obs.unsqueeze(1)

    niw_prior, mniw_prior = initialize_global_lds_parameters(1)
    niw_param, mniw_param = initialize_global_lds_parameters(1)

    mniw_prior, mniw_param = list(mniw_prior), list(mniw_param)

    D_latent = latents.shape[-1]
    D_obs = 1

    # A = torch.diag(torch.ones(1))
    # Q = torch.diag(torch.ones(1))
    C = torch.diag(torch.ones(1))
    R = torch.diag(torch.ones(1))

    # optimize prior
    n_iter = 1000
    for i in range(n_iter):
        # J11, J12, J22 = info_pair_params(A, Q)
        J22, J12, J11, _ = MatrixNormalInverseWishart(mniw_param).expected_stats()
        J11 *= -2
        J12 *= -1
        J22 *= -2

        y = info_observation_params(obs, C, R)

        # init_params = (torch.inverse(Q), torch.zeros(1))
        init_params = natural_to_info(NormalInverseWishart(niw_param).expected_stats())

        forward_messages = info_kalman_filter(
            init_params=init_params, pair_params=(J11, J12, J22), observations=y
        )
        backward_messages, expected_stats = info_kalman_smoothing(
            forward_messages, pair_params=(J11, J12, J22)
        )
        E_init_stats, E_pair_stats, _ = expected_stats

        forward_samples = sample_forward_messages(forward_messages)
        backward_samples = sample_backward_messages(backward_messages)
        samples = info_sample_backward(forward_messages, pair_params=(J11, J12, J22))

        # plot
        if i % max((n_iter // 10), 1) == 0 or i == 0:
            fig, ax = plt.subplots(1, 2)

            ax1 = ax[0]
            ax1.plot(latents.detach().numpy(), label="true")
            ax1.plot(forward_samples, label="predicted")
            ax1.plot(backward_samples, label="smoothed")
            ax1.legend()
            ax1.set_xlabel("time")
            ax1.set_ylabel("latent x")

            ax2 = ax[1]
            ax2.plot(obs.detach().numpy(), label="observed")
            ax2.plot(samples, label="sampled")
            ax2.legend()
            ax2.set_xlabel("time")
            ax2.set_ylabel("obs y")

            plt.tight_layout()
            plt.show()

        # update global param
        nat_grad_init = natural_gradient(
            E_init_stats, niw_param, niw_prior, len(obs), 1
        )
        niw_param = gradient_descent(
            niw_param, torch.stack(nat_grad_init), step_size=1e-1
        )

        nat_grad_pair = natural_gradient(
            E_pair_stats, mniw_param, mniw_prior, len(obs), 1
        )
        mniw_param = gradient_descent(mniw_param, nat_grad_pair, step_size=1e-2)
