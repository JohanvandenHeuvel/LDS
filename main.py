import torch
import matplotlib.pyplot as plt

from data import generate_data
from lds import (
    info_pair_params,
    info_observation_params,
    info_kalman_filter,
    info_kalman_smoothing,
    info_sample_backward,
    sample_forward_messages,
    sample_backward_messages,
)
from distributions import NormalInverseWishart, MatrixNormalInverseWishart
from distributions.gaussian import natural_to_info
from global_param import initialize_global_lds_parameters

if __name__ == "__main__":
    latents, obs = generate_data(100, noise_scale=1)

    obs = obs.unsqueeze(1)

    niw_param, mniw_param = initialize_global_lds_parameters(1)

    D_latent = latents.shape[-1]
    D_obs = 1

    # A = torch.diag(torch.ones(1))
    # Q = torch.diag(torch.ones(1))
    C = torch.diag(torch.ones(1))
    R = torch.diag(torch.ones(1))

    # optimize prior
    for i in range(10):

        # J11, J12, J22 = info_pair_params(A, Q)
        J11, J12, J22, _ = MatrixNormalInverseWishart(mniw_param).expected_stats()
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

        forward_samples = sample_forward_messages(forward_messages)
        backward_samples = sample_backward_messages(backward_messages)
        samples = info_sample_backward(forward_messages, pair_params=(J11, J12, J22))

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

        plt.show()
