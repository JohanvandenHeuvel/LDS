import matplotlib.pyplot as plt
import torch


def rot(theta):
    s = torch.sin(theta)
    c = torch.cos(theta)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def generate_data(A, Q, C, R, T):
    P, N = C.shape

    x = [torch.randn(N)]
    for t in range(T - 1):
        old_x = x[t]
        new_x = A @ old_x + Q @ torch.randn(N)
        x.append(new_x)

    x = torch.stack(x)

    y = x @ C.T + torch.randn(T, P) @ R.T

    return x, y


def data_params():
    # size parameters
    N = 2

    A = 0.999 * rot(torch.tensor(2 * torch.pi / 30))
    Q = 0.1 * torch.eye(N)
    C = torch.eye(N)
    R = 0.0001 * torch.eye(N)

    return A, Q, C, R


if __name__ == "__main__":
    T = 900
    A, Q, C, R = data_params()
    x, y = generate_data(A, Q, C, R, T)

    plt.plot(x.detach().numpy())
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    plt.plot(y.detach().numpy())
    plt.xlabel("time")
    plt.ylabel("value")
    plt.show()
