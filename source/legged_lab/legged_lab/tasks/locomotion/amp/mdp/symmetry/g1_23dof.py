"""Left-right symmetry for Unitree G1 using the standard 23-DoF policy order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states", "switch_joints_left_right"]

JOINT_COUNT = 23
HISTORY_LENGTH = 5
LEFT_INDICES = [0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17]
RIGHT_INDICES = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22]
SIGN_FLIP_INDICES = [1, 2, 5, 7, 8, 11, 12, 14, 15, 17, 19, 20, 22]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return the original and left-right mirrored policy states and actions."""
    del env
    if obs is not None:
        batch_size = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        obs_aug["policy"][:batch_size] = obs["policy"]
        obs_aug["policy"][batch_size:] = _transform_policy_obs_left_right(obs["policy"])
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_aug = torch.empty((batch_size * 2, actions.shape[1]), device=actions.device, dtype=actions.dtype)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = switch_joints_left_right(actions)
    else:
        actions_aug = None
    return obs_aug, actions_aug


def _transform_policy_obs_left_right(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the 405-D policy observation with term-wise history concatenation."""
    mirrored = obs.clone()
    device = mirrored.device
    end_index = 0

    term_specs = (
        (3, torch.tensor([-1.0, 1.0, -1.0], device=device), None),
        (6, torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, 1.0], device=device), None),
        (3, torch.tensor([1.0, -1.0, -1.0], device=device), None),
        (JOINT_COUNT, None, switch_joints_left_right),
        (JOINT_COUNT, None, switch_joints_left_right),
        (JOINT_COUNT, None, switch_joints_left_right),
    )
    for term_dimension, sign, transform in term_specs:
        for _ in range(HISTORY_LENGTH):
            start_index = end_index
            end_index += term_dimension
            if transform is not None:
                mirrored[:, start_index:end_index] = transform(mirrored[:, start_index:end_index])
            else:
                mirrored[:, start_index:end_index] *= sign

    if end_index != mirrored.shape[-1]:
        raise ValueError(f"Expected a {end_index}-D G1 23-DoF policy observation, got {mirrored.shape[-1]}")
    return mirrored


def switch_joints_left_right(joint_data: torch.Tensor) -> torch.Tensor:
    """Mirror joint data in the standard 23-DoF order; applying twice is identity."""
    if joint_data.shape[-1] != JOINT_COUNT:
        raise ValueError(f"Expected {JOINT_COUNT} joint values, got {joint_data.shape[-1]}")
    mirrored = torch.zeros_like(joint_data)
    mirrored[..., 12] = joint_data[..., 12]
    mirrored[..., LEFT_INDICES] = joint_data[..., RIGHT_INDICES]
    mirrored[..., RIGHT_INDICES] = joint_data[..., LEFT_INDICES]
    mirrored[..., SIGN_FLIP_INDICES] *= -1.0
    return mirrored
