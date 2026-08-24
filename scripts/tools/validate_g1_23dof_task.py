"""Create a G1 23-DoF task, inspect dimensions, and run finite-action simulation steps."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeggedLab-Isaac-AMP-G1-23DoF-v0")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import traceback
from tensordict import TensorDict

from isaaclab_tasks.utils import parse_env_cfg

import legged_lab.tasks  # noqa: F401
from legged_lab.assets.unitree import G1_23DOF_JOINT_NAMES
from legged_lab.tasks.locomotion.amp.mdp.symmetry import g1_23dof


DELETED_JOINT_NAMES = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}


def _assert_finite(value, name: str) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains NaN or Inf")
    elif isinstance(value, (dict, TensorDict)):
        for key, child in value.items():
            _assert_finite(child, f"{name}.{key}")


def _shape_tree(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (dict, TensorDict)):
        return {key: _shape_tree(child) for key, child in value.items()}
    return type(value).__name__


def main():
    if args_cli.task not in gym.registry:
        raise RuntimeError(f"Task is not registered: {args_cli.task}")
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped

    try:
        actual_joint_names = list(unwrapped.scene["robot"].joint_names)
        if len(actual_joint_names) != 23 or set(actual_joint_names) != set(G1_23DOF_JOINT_NAMES):
            raise RuntimeError(f"Unexpected articulation joints: {actual_joint_names}")
        if DELETED_JOINT_NAMES & set(actual_joint_names):
            raise RuntimeError("Deleted joints are present in the target articulation")

        action_term = unwrapped.action_manager.get_term("joint_pos")
        action_joint_names = list(action_term._joint_names)
        if action_joint_names != list(G1_23DOF_JOINT_NAMES):
            raise RuntimeError(f"Action joint order is not the standard 23-DoF order: {action_joint_names}")
        if unwrapped.action_manager.total_action_dim != 23:
            raise RuntimeError(f"Expected action dimension 23, got {unwrapped.action_manager.total_action_dim}")

        observations, _ = env.reset()
        _assert_finite(observations, "reset_observations")
        print(f"TASK_REGISTRATION=PASS task={args_cli.task}", flush=True)
        print(f"NUM_ENVS={args_cli.num_envs}", flush=True)
        print(f"ARTICULATION_JOINT_ORDER={actual_joint_names}", flush=True)
        print(f"ACTION_JOINT_ORDER={action_joint_names}", flush=True)
        print(f"ACTION_DIM={unwrapped.action_manager.total_action_dim}", flush=True)
        print(f"OBSERVATION_GROUP_DIMS={unwrapped.observation_manager.group_obs_dim}", flush=True)
        print(f"OBSERVATION_TERM_DIMS={unwrapped.observation_manager.group_obs_term_dim}", flush=True)
        print(f"RESET_OBSERVATION_SHAPES={_shape_tree(observations)}", flush=True)

        if hasattr(unwrapped, "motion_data_manager"):
            motion_term = unwrapped.motion_data_manager.get_term("motion_dataset")
            motion_shapes = {
                "root_pos_w": list(motion_term.root_pos_w.shape),
                "root_quat": list(motion_term.root_quat.shape),
                "root_vel_w": list(motion_term.root_vel_w.shape),
                "root_ang_vel_w": list(motion_term.root_ang_vel_w.shape),
                "dof_pos": list(motion_term.dof_pos.shape),
                "dof_vel": list(motion_term.dof_vel.shape),
                "key_body_pos_w": list(motion_term.key_body_pos_w.shape),
            }
            for name in motion_shapes:
                _assert_finite(getattr(motion_term, name), f"motion_data.{name}")
            if motion_term.num_dofs != 23:
                raise RuntimeError(f"Motion loader expected 23 DoFs, got {motion_term.num_dofs}")
            print(f"MOTION_TENSOR_SHAPES={motion_shapes}", flush=True)

        if "AMP" in args_cli.task:
            sample_actions = torch.randn((4, 23), device=unwrapped.device)
            mirrored_twice = g1_23dof.switch_joints_left_right(
                g1_23dof.switch_joints_left_right(sample_actions)
            )
            if not torch.equal(sample_actions, mirrored_twice):
                raise RuntimeError("Joint symmetry is not an exact involution")
            sample_policy = torch.randn((4, 405), device=unwrapped.device)
            policy_twice = g1_23dof._transform_policy_obs_left_right(
                g1_23dof._transform_policy_obs_left_right(sample_policy)
            )
            if not torch.equal(sample_policy, policy_twice):
                raise RuntimeError("Policy observation symmetry is not an exact involution")
            print("SYMMETRY_VALIDATION=PASS", flush=True)

        actions = torch.zeros((args_cli.num_envs, 23), device=unwrapped.device)
        for step_index in range(args_cli.steps):
            step_result = env.step(actions)
            _assert_finite(step_result[0], f"step_{step_index}.observations")
            _assert_finite(step_result[1], f"step_{step_index}.rewards")
        print(f"ENV_STEP_VALIDATION=PASS steps={args_cli.steps} num_envs={args_cli.num_envs}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        print("G1_23DOF_TASK_VALIDATION=FAIL", flush=True)
        raise
    finally:
        simulation_app.close()
