import torch
import matplotlib.pyplot as plt

from data import generate_data
from lds import info_pair_params, info_observation_params, info_kalman_filter

if __name__ == "__main__":
    latents, obs = generate_data(100, noise_scale=1)

    obs = obs.unsqueeze(1)

    D_latent = latents.shape[-1]
    D_obs = 1

    A = torch.diag(torch.ones(1))
    Q = torch.diag(torch.ones(1))
    C = torch.diag(torch.ones(1))
    R = torch.diag(torch.ones(1))

    J11, J12, J22 = info_pair_params(A, Q)

    y = info_observation_params(obs, C, R)

    init_params = (torch.inverse(Q), torch.zeros(1))

    saved = info_kalman_filter(
        init_params=init_params, pair_params=(J11, J12, J22), observations=y
    )

    plt.plot(latents.detach().numpy())
    plt.plot(saved)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
