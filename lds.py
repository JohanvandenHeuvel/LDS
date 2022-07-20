import torch

from distributions.gaussian import info_to_standard, Gaussian, info_to_natural


def outer_product(x, y):
    # computes xyT
    return torch.einsum("i, j -> ij", (x, y))


def info_condition(J, h, J_obs, h_obs):
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
        return E_xxT, E_x, 1.0, 1.0

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

    return list(reversed(samples))


def info_observation_params(obs, C, R):
    R_inv = torch.inverse(R)
    R_inv_C = R_inv @ C

    # J_obs = C.T @ inv(R) @ C
    J_obs = C.T @ R_inv_C
    # h_obs = (y - D @ u) @ inv(R) @ C
    h_obs = obs @ R_inv_C

    J_obs = J_obs.unsqueeze(0).repeat(len(obs), 1, 1)
    return zip(J_obs, h_obs)


def info_pair_params(A, Q):
    J22 = torch.inverse(Q)
    J12 = -A.T @ J22
    J11 = A.T @ -J12
    return J11, J12, J22


def sample_forward_messages(messages):
    samples = []
    for _, (J, h) in messages:
        loc, scale = info_to_standard(J, h)
        x = loc + scale @ torch.randn(1)
        samples.append(x.detach().numpy())
    return samples


def sample_backward_messages(messages):
    samples = []
    for (J, h) in messages:
        loc, scale = info_to_standard(J, h)
        x = loc + scale @ torch.randn(1)
        samples.append(x.detach().numpy())
    return samples
