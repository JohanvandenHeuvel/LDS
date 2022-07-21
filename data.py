import matplotlib.pyplot as plt
import torch

from distributions import Gaussian
from distributions.gaussian import standard_to_natural

A = torch.diag(torch.ones(1))
Q = torch.diag(torch.ones(1))
C = torch.diag(torch.ones(1))
R = torch.diag(torch.ones(1))


def generate_data(n, noise_scale=1e-1):
    init_params = standard_to_natural(
        loc=torch.ones(1).unsqueeze(0) * 4,
        scale=torch.diag(torch.ones(1) * noise_scale).unsqueeze(0),
    )
    x_1 = Gaussian(nat_param=init_params).rsample()

    x = [x_1]
    y = []
    for i in range(n):
        old_x = x[i]

        new_x = A @ old_x + Q @ torch.randn(1) * noise_scale
        new_y = C @ old_x + R @ torch.randn(1) * noise_scale

        x.append(new_x)
        y.append(new_y)

    return torch.stack(x[:100]).squeeze(), torch.stack(y).squeeze()


if __name__ == "__main__":
    x, y = generate_data(100, noise_scale=1e-2)

    plt.plot(x.detach().numpy())
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    plt.plot(y.detach().numpy())
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()
