import torch

from distributions.gaussian import info_to_standard


def info_condition(J, h, J_obs, h_obs):
    return J + J_obs, h + h_obs


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

    saved = []
    for (J_obs, h_obs) in observations:
        J_cond, h_cond = info_condition(J, h, J_obs, h_obs)
        J, h = info_predict(J_cond, h_cond, J11, J12, J22)

        loc, scale = info_to_standard(J, h)
        x = loc + scale @ torch.randn(1)
        saved.append(x.detach().numpy())

    return saved


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
