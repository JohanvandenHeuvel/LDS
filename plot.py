import matplotlib.pyplot as plt
import torch

from distributions import MatrixNormalInverseWishart


def plot_global_parameters(A, Q, mniw_param):
    expected_A, expected_Sigma = MatrixNormalInverseWishart(
        mniw_param
    ).expected_standard_params()
    sampled_A, sampled_Sigma = MatrixNormalInverseWishart(mniw_param).sample()

    """
    Plot global parameters 
    """
    image_A = torch.vstack((A, expected_A, torch.tensor(sampled_A)))
    plt.imshow(image_A.T)
    plt.colorbar()
    plt.title("A")
    plt.show()

    image_Sigma = torch.vstack((Q ** 2, expected_Sigma, torch.tensor(sampled_Sigma)))
    plt.imshow(image_Sigma.T)
    plt.colorbar()
    plt.title("Sigma")
    plt.show()


def plot_observations(obs, latents, samples, title="plot"):
    N = obs.shape[-1]
    fig, ax = plt.subplots(N, 2, figsize=(10, 10))

    for n in range(N):
        ax1 = ax[n, 0]
        ax1.plot(latents.detach().numpy()[:, n], label="true", alpha=0.8)
        ax1.legend()
        ax1.set_xlabel("time")
        ax1.set_ylabel("latent x")

        ax2 = ax[n, 1]
        ax2.plot(obs.squeeze().detach().numpy()[:, n], label="observed", alpha=0.8)
        ax2.plot(
            samples.squeeze().detach().numpy()[:, n],
            linestyle="dashed",
            label="sampled",
            alpha=0.8,
        )
        ax2.legend()
        ax2.set_xlabel("time")
        ax2.set_ylabel("obs y")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
