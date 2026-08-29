"""Evaluate a 23-DoF AMP checkpoint with passive Max-Plus gait measurements."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="LeggedLab-Isaac-AMP-G1-23DoF-MaxPlus-Play-v0")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=32)
parser.add_argument("--steps", "--num-steps", dest="steps", type=int, default=500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--command-speed", type=float, default=1.0)
parser.add_argument("--scheduler-profile", choices=("adaptive", "fixed"), default="adaptive")
parser.add_argument("--output", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import AMPRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

import legged_lab.tasks  # noqa: F401

from legged_lab.tasks.locomotion.amp.mdp.max_plus import DURATION_NAMES, EVENT_NAMES


LEGACY_SWING_TIME_LEFT = 0.467208
LEGACY_SWING_TIME_RIGHT = 0.434058
LEGACY_DOUBLE_SUPPORT_TIME_LEFT = 0.033333
LEGACY_DOUBLE_SUPPORT_TIME_RIGHT = 0.033333
LEGACY_SIGMA_TIME = 0.120862
LEGACY_MISS_TOLERANCE = 0.20


def fixed_reference_settings(settings):
    """Return a speed-independent profile with the legacy recurring event schedule."""
    cycle_time = (
        LEGACY_SWING_TIME_LEFT
        + LEGACY_DOUBLE_SUPPORT_TIME_RIGHT
        + LEGACY_SWING_TIME_RIGHT
        + LEGACY_DOUBLE_SUPPORT_TIME_LEFT
    )
    node_count = len(settings.speed_nodes)
    constant = lambda value: tuple(value for _ in range(node_count))
    return replace(
        settings,
        cycle_time_nodes=constant(cycle_time),
        swing_ratio_left_nodes=constant(LEGACY_SWING_TIME_LEFT / cycle_time),
        swing_ratio_right_nodes=constant(LEGACY_SWING_TIME_RIGHT / cycle_time),
        double_support_ratio_left_nodes=constant(LEGACY_DOUBLE_SUPPORT_TIME_LEFT / cycle_time),
        double_support_ratio_right_nodes=constant(LEGACY_DOUBLE_SUPPORT_TIME_RIGHT / cycle_time),
        standing_speed_threshold=0.0,
        speed_filter_time_constant_s=0.0,
        speed_max_rate_m_s2=0.0,
        sigma_phase=LEGACY_SIGMA_TIME / cycle_time,
        miss_ratio=LEGACY_MISS_TOLERANCE / cycle_time,
    )


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.command_speed, args_cli.command_speed)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    if args_cli.scheduler_profile == "fixed":
        reward_params = env_cfg.rewards.max_plus_gait.params
        reward_params["settings"] = fixed_reference_settings(reward_params["settings"])
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args_cli.seed
    gym_env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = gym_env.unwrapped
    env = RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)
    runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(args_cli.checkpoint.resolve()), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=unwrapped.device)
    policy_nn = runner.alg.policy

    reward_sum = torch.zeros((), device=unwrapped.device)
    velocity_error_sum = torch.zeros((), device=unwrapped.device)
    foot_slide_sum = torch.zeros((), device=unwrapped.device)
    style_reward_sum = torch.zeros((), device=unwrapped.device)
    fall_count = torch.zeros((), device=unwrapped.device)
    done_count = torch.zeros((), device=unwrapped.device)
    completed_length_sum = torch.zeros((), device=unwrapped.device)
    episode_lengths = torch.zeros(args_cli.num_envs, device=unwrapped.device)
    event_phase_error_sum = torch.zeros(4, device=unwrapped.device)
    event_time_error_sum = torch.zeros(4, device=unwrapped.device)
    event_count = torch.zeros(4, device=unwrapped.device)
    event_reward_sum = torch.zeros((), device=unwrapped.device)
    event_reward_count = torch.zeros((), device=unwrapped.device)
    contact_reward_sum = torch.zeros((), device=unwrapped.device)
    violation_sum = torch.zeros((), device=unwrapped.device)
    miss_sum = torch.zeros((), device=unwrapped.device)
    duration_sum = torch.zeros(6, device=unwrapped.device)
    duration_count = torch.zeros(6, device=unwrapped.device)
    cycle_sum = torch.zeros((), device=unwrapped.device)
    cycle_sq_sum = torch.zeros((), device=unwrapped.device)
    cycle_count = torch.zeros((), device=unwrapped.device)
    physical_cycle_sum = torch.zeros((), device=unwrapped.device)
    physical_cycle_sq_sum = torch.zeros((), device=unwrapped.device)
    physical_cycle_count = torch.zeros((), device=unwrapped.device)
    previous_left_liftoff = torch.full((args_cli.num_envs,), -1.0, device=unwrapped.device)
    double_support_start = torch.full((args_cli.num_envs,), -1.0, device=unwrapped.device)
    previous_double_support = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=unwrapped.device)
    physical_double_support_sum = torch.zeros((), device=unwrapped.device)
    physical_double_support_count = torch.zeros((), device=unwrapped.device)
    cycle_time_sum = torch.zeros((), device=unwrapped.device)
    frequency_sum = torch.zeros((), device=unwrapped.device)
    ratio_sum = torch.zeros(4, device=unwrapped.device)

    feet_slide_cfg = unwrapped.reward_manager.get_term_cfg("feet_slide")
    max_plus_term = unwrapped._max_plus_gait_reward
    obs = env.get_observations()
    with torch.inference_mode():
        for step_index in range(args_cli.steps):
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            policy_nn.reset(dones)

            reward_sum += rewards.sum()
            command = unwrapped.command_manager.get_command("base_velocity")[:, :2]
            velocity = unwrapped.scene["robot"].data.root_lin_vel_b[:, :2]
            velocity_error_sum += torch.linalg.vector_norm(command - velocity, dim=1).sum()
            foot_slide_sum += feet_slide_cfg.func(unwrapped, **feet_slide_cfg.params).sum()
            disc_obs = runner.alg.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
            style_reward, _ = runner.alg.amp_discriminator.predict_style_reward(
                disc_obs, dt=runner.alg.amp_cfg["step_dt"]
            )
            style_reward_sum += style_reward.sum()

            episode_lengths += 1.0
            done_count += dones.sum()
            fall_count += unwrapped.reset_terminated.sum()
            completed_length_sum += (episode_lengths * dones).sum()
            episode_lengths[dones.bool()] = 0.0

            step_metrics = max_plus_term.scheduler.last_step_metrics
            step_time = torch.as_tensor((step_index + 1) * unwrapped.step_dt, device=unwrapped.device)
            detected_left_liftoff = step_metrics["detected_events"][:, 0]
            physical_cycle = step_time - previous_left_liftoff
            valid_cycle = detected_left_liftoff & (previous_left_liftoff >= 0.0)
            physical_cycle_sum += torch.where(valid_cycle, physical_cycle, 0.0).sum()
            physical_cycle_sq_sum += torch.where(valid_cycle, physical_cycle.square(), 0.0).sum()
            physical_cycle_count += valid_cycle.sum()
            previous_left_liftoff = torch.where(
                detected_left_liftoff, step_time, previous_left_liftoff
            )

            double_support = step_metrics["contacts"].all(dim=1)
            entering_double_support = double_support & ~previous_double_support
            leaving_double_support = ~double_support & previous_double_support
            double_support_start = torch.where(entering_double_support, step_time, double_support_start)
            double_support_duration = step_time - double_support_start
            valid_double_support = leaving_double_support & (double_support_start >= 0.0)
            physical_double_support_sum += torch.where(
                valid_double_support, double_support_duration, 0.0
            ).sum()
            physical_double_support_count += valid_double_support.sum()
            double_support_start = torch.where(
                leaving_double_support, -torch.ones_like(double_support_start), double_support_start
            )
            previous_double_support.copy_(double_support)

            accepted_count = step_metrics["accepted_events"].sum(dim=1).float()
            event_phase_error_sum += step_metrics["event_abs_errors"].sum(dim=0)
            event_time_error_sum += step_metrics["event_abs_errors_s"].sum(dim=0)
            event_count += step_metrics["accepted_events"].sum(dim=0)
            event_reward_sum += (step_metrics["event_reward"] * accepted_count).sum()
            event_reward_count += accepted_count.sum()
            contact_reward_sum += step_metrics["contact_reward"].sum()
            violation_sum += step_metrics["violation_count"].sum()
            miss_sum += step_metrics["miss_count"].sum()
            duration_sum += step_metrics["duration_values"].sum(dim=0)
            duration_count += step_metrics["duration_counts"].sum(dim=0)
            cycle_sum += step_metrics["cycle_values"].sum()
            cycle_sq_sum += step_metrics["cycle_values"].square().sum()
            cycle_count += step_metrics["cycle_counts"].sum()
            walking = ~step_metrics["standing"]
            cycle_time_sum += torch.where(walking, step_metrics["cycle_time"], 0.0).sum()
            frequency_sum += step_metrics["step_frequency"].sum()
            ratio_sum += (step_metrics["ratios"] * walking.unsqueeze(1)).sum(dim=0)

            done_mask = dones.bool()
            previous_left_liftoff[done_mask] = -1.0
            double_support_start[done_mask] = -1.0
            previous_double_support[done_mask] = False

    sample_count = float(args_cli.steps * args_cli.num_envs)
    result = {
        "task": args_cli.task,
        "checkpoint": str(args_cli.checkpoint.resolve()),
        "num_envs": args_cli.num_envs,
        "steps": args_cli.steps,
        "seed": args_cli.seed,
        "scheduler_profile": args_cli.scheduler_profile,
        "command_speed_m_s": args_cli.command_speed,
        "reward_mean_per_step": float(reward_sum / sample_count),
        "velocity_tracking_error_xy_m_s": float(velocity_error_sum / sample_count),
        "mean_episode_length_steps": float(completed_length_sum / done_count.clamp_min(1.0)),
        "fall_rate_per_completed_episode": float(fall_count / done_count.clamp_min(1.0)),
        "foot_slide_speed_sum_m_s": float(foot_slide_sum / sample_count),
        "amp_style_reward": float(style_reward_sum / sample_count),
        "event_timing_reward": float(event_reward_sum / event_reward_count.clamp_min(1.0)),
        "contact_phase_reward": float(contact_reward_sum / sample_count),
        "constraint_violations_per_env_step": float(violation_sum / sample_count),
        "missed_events_per_env_step": float(miss_sum / sample_count),
    }
    result.update(
        {
            f"{name}_phase_mae": float(
                event_phase_error_sum[index] / event_count[index].clamp_min(1.0)
            )
            for index, name in enumerate(EVENT_NAMES)
        }
    )
    result.update(
        {
            f"{name}_mae_s": float(event_time_error_sum[index] / event_count[index].clamp_min(1.0))
            for index, name in enumerate(EVENT_NAMES)
        }
    )
    result.update(
        {
            f"{name}_mean_s": float(duration_sum[index] / duration_count[index].clamp_min(1.0))
            for index, name in enumerate(DURATION_NAMES)
        }
    )
    accepted_cycle_mean = cycle_sum / cycle_count.clamp_min(1.0)
    result["scheduler_accepted_gait_cycle_mean_s"] = float(accepted_cycle_mean)
    result["scheduler_accepted_gait_cycle_variance_s2"] = float(
        (cycle_sq_sum / cycle_count.clamp_min(1.0) - accepted_cycle_mean.square()).clamp_min(0.0)
    )
    physical_cycle_mean = physical_cycle_sum / physical_cycle_count.clamp_min(1.0)
    result["gait_cycle_mean_s"] = float(physical_cycle_mean)
    result["gait_cycle_variance_s2"] = float(
        (
            physical_cycle_sq_sum / physical_cycle_count.clamp_min(1.0)
            - physical_cycle_mean.square()
        ).clamp_min(0.0)
    )
    result["double_support_mean_s"] = float(
        physical_double_support_sum / physical_double_support_count.clamp_min(1.0)
    )
    walking_sample_count = float(args_cli.steps * args_cli.num_envs)
    result["planned_cycle_time_mean_s"] = float(cycle_time_sum / walking_sample_count)
    result["planned_step_frequency_mean_hz"] = float(frequency_sum / walking_sample_count)
    result["swing_ratio_left"] = float(ratio_sum[0] / walking_sample_count)
    result["swing_ratio_right"] = float(ratio_sum[1] / walking_sample_count)
    result["stance_ratio_left"] = 1.0 - result["swing_ratio_left"]
    result["stance_ratio_right"] = 1.0 - result["swing_ratio_right"]
    result["double_support_ratio_left"] = float(ratio_sum[2] / walking_sample_count)
    result["double_support_ratio_right"] = float(ratio_sum[3] / walking_sample_count)
    if not all(torch.isfinite(torch.tensor(value)) for value in result.values() if isinstance(value, float)):
        raise RuntimeError("Evaluation produced a non-finite metric")
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
