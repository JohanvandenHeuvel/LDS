import torch
import numpy as np

from dense import pack_dense
from distributions import MatrixNormalInverseWishart, NormalInverseWishart

from distributions.gaussian import (
    info_to_standard,
    Gaussian,
    info_to_natural,
    natural_to_info,
)
from global_param import natural_gradient, gradient_descent


def outer_product(x, y):
    """
    Computes xy^T .
    """
    return torch.einsum("i, j -> ij", (x, y))


def info_condition(J, h, J_obs, h_obs):
    """
    Conditions distribution with parameters (J, h) on observation with parameters (J_obs, h_obs).
    """
    return J + J_obs, h + h_obs


def condition(J, h, y, Jxx, Jxy):
    J_cond = J + Jxx
    h_cond = h + (Jxy @ y.T).T
    return J_cond, h_cond


def info_marginalize(J11, J12, J22, h):
    # J11_inv = torch.inverse(J11)
    # temp = J12.T @ J11_inv
    temp = torch.linalg.solve(J11, J12)

    # J_pred = J22 - J12.T @ inv(J11) @ J12
    J_pred = J22 - temp @ J12
    # h_pred = h2 - J12.T @ inv(J11) @ h1
    h_pred = -temp @ h

    return J_pred, h_pred


def info_predict(J, h, J11, J12, J22):
    J_new = J + J11
    return info_marginalize(J_new, J12, J22, h)


def info_kalman_filter(init_params, pair_params, observations):
    J, h = init_params
    J11, J12, J22 = pair_params

    forward_messages = []
    for (J_obs, h_obs) in observations:
        J_cond, h_cond = info_condition(J, h, J_obs, h_obs)
        J, h = info_predict(J_cond, h_cond, J11, J12, J22)
        forward_messages.append(((J_cond, h_cond), (J, h)))

    return forward_messages


def info_rst_smoothing(J, h, cond_msg, pred_msg, pair_params, loc_next):
    J_cond, h_cond = cond_msg
    J_pred, h_pred = pred_msg
    J11, J12, J22 = pair_params

    temp = J12 @ torch.inverse(J - J_pred + J22)
    J_smooth = J_cond + J11 - temp @ J12.T
    # h_smooth = h_cond + h1 - temp @ (h - h_pred + h2)
    h_smooth = h_cond - temp @ (h - h_pred)

    loc, scale = info_to_standard(J_smooth, h_smooth)
    E_xnxT = temp @ scale + outer_product(loc_next, loc)
    E_xxT = scale + outer_product(loc, loc)

    stats = (loc, E_xxT, E_xnxT)

    return J_smooth, h_smooth, stats


def process_expected_stats(expected_stats):
    def make_init_stats(a):
        E_x, E_xxT, _ = a
        return E_xxT, E_x, torch.tensor(1.0), torch.tensor(1.0)

    def make_pair_stats(a, b):
        E_x, E_xxT, E_xnxT = a
        E_xn, E_xnxnT, _ = b
        # return E_xxT, E_xnxT.T, E_xnxnT, 1.0
        return E_xnxnT, E_xnxT.T, E_xxT, 1.0

    def make_node_stats(a):
        E_x, E_xxT, _ = a
        return torch.diag(E_xxT), E_x, 1.0

    E_init_stats = make_init_stats(expected_stats[0])
    E_pair_stats = [
        make_pair_stats(a, b) for a, b in zip(expected_stats[:-1], expected_stats[1:])
    ]
    # same pair for every time step
    E_pair_stats = [sum(stats) for stats in list(zip(*E_pair_stats))]
    E_node_stats = [make_node_stats(a) for a in expected_stats]
    E_node_stats = list(zip(*E_node_stats))

    return E_init_stats, E_pair_stats, E_node_stats


def info_kalman_smoothing(forward_messages, pair_params):
    _, (J_smooth, h_smooth) = forward_messages[-1]
    loc, scale = info_to_standard(J_smooth, h_smooth)
    E_xxT = scale + outer_product(loc, loc)
    E_xnxT = 0.0

    expected_stats = [(loc, E_xxT, E_xnxT)]
    backward_messages = [(J_smooth, h_smooth)]
    for i, (cond_msg, pred_msg) in enumerate(reversed(forward_messages[:-1])):
        loc_next, _, _ = expected_stats[i]
        J_smooth, h_smooth, stats = info_rst_smoothing(
            J_smooth, h_smooth, cond_msg, pred_msg, pair_params, loc_next
        )
        backward_messages.append((J_smooth, h_smooth))
        expected_stats.append(stats)

    expected_stats = process_expected_stats(list(reversed(expected_stats)))

    return list(reversed(backward_messages)), expected_stats


def info_sample_backward(forward_messages, pair_params):
    J11, J12, _ = pair_params

    _, (J_pred, h_pred) = forward_messages[-1]
    next_sample = Gaussian(info_to_natural(J_pred, h_pred)).rsample()

    samples = [next_sample]
    for _, (J_pred, h_pred) in reversed(forward_messages[:-1]):
        J = J_pred + J11
        h = h_pred - next_sample @ J12

        # get the sample
        state = Gaussian(info_to_natural(J, h.squeeze(0)))
        next_sample = state.rsample()
        samples.append(next_sample)

    return torch.stack(list(reversed(samples)))


def info_observation_params(obs, C, R, zero=None):
    """
    Transform observations to information parameter form.
    """
    R_inv = torch.inverse(R)
    R_inv_C = R_inv @ C

    # J_obs = C.T @ inv(R) @ C
    J_obs = C.T @ R_inv_C
    # h_obs = (y - D @ u) @ inv(R) @ C
    h_obs = obs @ R_inv_C

    J_obs = J_obs.unsqueeze(0).repeat(len(obs), 1, 1)

    # zero out a part of the data (for prediction purposes)
    if zero is not None:
        h_obs[zero[0] : zero[1]] = 0
        J_obs[zero[0] : zero[1]] = 0

    return zip(J_obs, h_obs.squeeze())


def info_pair_params(A, Q):
    J22 = torch.inverse(Q)
    J12 = -A.T @ J22
    J11 = A.T @ -J12
    return J11, J12, J22


def sample_forward_messages(messages):
    samples = []
    for _, (J, h) in messages:
        loc, scale = info_to_standard(J, h)
        x = loc + scale @ torch.randn(len(scale))
        samples.append(x.detach().numpy())
    return np.stack(samples)


def sample_backward_messages(messages):
    samples = []
    for (J, h) in messages:
        loc, scale = info_to_standard(J, h)
        x = loc + scale @ torch.randn(len(scale))
        samples.append(x.detach().numpy())
    return np.stack(samples)


def run_iter(y, param, param_prior):
    """
    Run single iteration for main loop.
    """
    niw_prior, mniw_prior = param_prior
    niw_param, mniw_param = param

    """
    Transition prior
    """
    # J11, J12, J22 = info_pair_params(A, Q)
    J22, J12, J11, _ = MatrixNormalInverseWishart(mniw_param).expected_stats()
    J11 = -2 * J11
    J12 = -1 * J12.T
    J22 = -2 * J22

    """
    Initial state prior
    """
    # init_params = (torch.inverse(Q), torch.zeros(1))
    init_params = natural_to_info(NormalInverseWishart(niw_param).expected_stats())

    """
    Kalman filtering and smoothing 
    """
    forward_messages = info_kalman_filter(
        init_params=init_params, pair_params=(J11, J12, J22), observations=y
    )
    backward_messages, expected_stats = info_kalman_smoothing(
        forward_messages, pair_params=(J11, J12, J22)
    )
    E_init_stats, E_pair_stats, _ = expected_stats

    # forward_samples = sample_forward_messages(forward_messages)
    # backward_samples = sample_backward_messages(backward_messages)
    samples = info_sample_backward(forward_messages, pair_params=(J11, J12, J22))

    """
    Update global parameters  
    """
    T = len(samples)
    nat_grad_init = natural_gradient(
        pack_dense(*E_init_stats)[None, ...], niw_param, niw_prior, T, 1
    )
    niw_param = gradient_descent(niw_param, torch.stack(nat_grad_init), step_size=1e-1)

    nat_grad_pair = natural_gradient(E_pair_stats, mniw_param, mniw_prior, T, 1)
    mniw_param = gradient_descent(mniw_param, nat_grad_pair, step_size=1e-2)

    return niw_param, mniw_param, samples
