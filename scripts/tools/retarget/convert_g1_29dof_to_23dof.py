"""Convert Legged Lab G1 29-DoF pickle motions to the official G1 23-DoF articulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-dir", type=Path, required=True, help="Root of the G1 29-DoF motion tree.")
parser.add_argument("--output-dir", type=Path, required=True, help="Independent output root for G1 23-DoF motions.")
parser.add_argument(
    "--source-config",
    type=Path,
    default=Path("scripts/tools/retarget/config/g1_29dof.yaml"),
    help="29-DoF retarget configuration used to create the source pickles.",
)
parser.add_argument(
    "--target-config",
    type=Path,
    default=Path("scripts/tools/retarget/config/g1_23dof.yaml"),
    help="23-DoF retarget configuration.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import pickle

import numpy as np
import torch
import yaml

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from legged_lab.assets.unitree import (
    G1_23DOF_JOINT_NAMES,
    UNITREE_G1_23DOF_CFG,
    UNITREE_G1_29DOF_CFG,
)


DELETED_JOINT_NAMES = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}
EXPECTED_SDK_MAPPING = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26]
REQUIRED_FIELDS = ("fps", "root_pos", "root_rot", "dof_pos", "loop_mode", "key_body_pos")


@configclass
class ConversionSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = UNITREE_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _forward_diff(data: np.ndarray, dt: float) -> np.ndarray:
    velocity = np.zeros_like(data)
    velocity[:-1] = (data[1:] - data[:-1]) / dt
    velocity[-1] = velocity[-2]
    return velocity


def _validate_raw_motion(path: Path, motion: dict, source_joint_count: int) -> None:
    missing = [name for name in REQUIRED_FIELDS if name not in motion]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")

    frame_count = np.asarray(motion["dof_pos"]).shape[0]
    expected_shapes = {
        "root_pos": (frame_count, 3),
        "root_rot": (frame_count, 4),
        "dof_pos": (frame_count, source_joint_count),
    }
    for field, expected_shape in expected_shapes.items():
        actual_shape = np.asarray(motion[field]).shape
        if actual_shape != expected_shape:
            raise ValueError(f"{path}: {field} shape {actual_shape} != {expected_shape}")
    if np.asarray(motion["key_body_pos"]).shape[0] != frame_count:
        raise ValueError(f"{path}: key_body_pos frame count does not match dof_pos")
    if frame_count < 2:
        raise ValueError(f"{path}: at least two frames are required")
    if not np.isfinite(float(motion["fps"])) or float(motion["fps"]) <= 0.0:
        raise ValueError(f"{path}: invalid fps {motion['fps']}")
    for field in ("root_pos", "root_rot", "dof_pos", "key_body_pos"):
        if not np.isfinite(np.asarray(motion[field])).all():
            raise ValueError(f"{path}: {field} contains NaN or Inf")

    quaternion_norms = np.linalg.norm(np.asarray(motion["root_rot"]), axis=-1)
    if not np.allclose(quaternion_norms, 1.0, atol=1.0e-5, rtol=0.0):
        max_error = float(np.max(np.abs(quaternion_norms - 1.0)))
        raise ValueError(f"{path}: root quaternions are not normalized; max error={max_error}")


def _validate_joint_orders(source_cfg: dict, target_cfg: dict) -> tuple[list[str], list[str], list[int]]:
    source_sdk_names = list(UNITREE_G1_29DOF_CFG.joint_sdk_names)
    target_sdk_names = list(G1_23DOF_JOINT_NAMES)
    source_gmr_names = list(source_cfg["gmr_dof_names"])
    target_gmr_names = list(target_cfg["gmr_dof_names"])
    if source_gmr_names != source_sdk_names:
        raise ValueError("29-DoF gmr_dof_names does not match UNITREE_G1_29DOF_CFG.joint_sdk_names")
    if target_gmr_names != target_sdk_names:
        raise ValueError("23-DoF gmr_dof_names does not match G1_23DOF_JOINT_NAMES")

    sdk_mapping = [source_sdk_names.index(name) for name in target_sdk_names]
    if sdk_mapping != EXPECTED_SDK_MAPPING:
        raise ValueError(f"Unexpected 29-to-23 SDK mapping: {sdk_mapping}")
    if set(source_sdk_names) - set(target_sdk_names) != DELETED_JOINT_NAMES:
        raise ValueError("The SDK joint-name difference is not exactly the six deleted joints")

    source_lab_names = list(source_cfg["lab_dof_names"])
    target_lab_names = list(target_cfg["lab_dof_names"])
    data_mapping = [source_lab_names.index(name) for name in target_lab_names]
    return source_lab_names, target_lab_names, data_mapping


def _prepare_motions(
    files: list[Path], input_dir: Path, source_lab_names: list[str], target_lab_names: list[str], data_mapping: list[int]
) -> tuple[list[dict], list[dict]]:
    converted_motions = []
    source_motions = []
    for path in files:
        with path.open("rb") as stream:
            source_motion = pickle.load(stream)
        if not isinstance(source_motion, dict):
            raise TypeError(f"{path}: expected a dictionary")
        _validate_raw_motion(path, source_motion, len(source_lab_names))

        target_motion = {
            "fps": source_motion["fps"],
            "root_pos": np.array(source_motion["root_pos"], copy=True),
            "root_rot": np.array(source_motion["root_rot"], copy=True),
            "dof_pos": np.array(source_motion["dof_pos"][:, data_mapping], copy=True),
            "loop_mode": source_motion["loop_mode"],
            "key_body_pos": None,
            "dof_names": list(target_lab_names),
            "source_dof_names": list(source_lab_names),
            "source_to_target_indices": list(data_mapping),
            "source_relative_path": str(path.relative_to(input_dir)),
        }
        source_motions.append(source_motion)
        converted_motions.append(target_motion)
    return source_motions, converted_motions


def _reconstruct_key_bodies(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    motions: list[dict],
    key_body_names: list[str],
) -> None:
    robot: Articulation = scene["robot"]
    key_body_indices = [robot.body_names.index(name) for name in key_body_names]
    frame_counts = [motion["dof_pos"].shape[0] for motion in motions]
    key_body_buffers = [np.empty((count, len(key_body_names), 3), dtype=np.float32) for count in frame_counts]

    root_positions = [torch.as_tensor(motion["root_pos"], device=scene.device, dtype=torch.float32) for motion in motions]
    root_rotations = [torch.as_tensor(motion["root_rot"], device=scene.device, dtype=torch.float32) for motion in motions]
    joint_positions = [torch.as_tensor(motion["dof_pos"], device=scene.device, dtype=torch.float32) for motion in motions]

    for frame_index in range(max(frame_counts)):
        root_state = robot.data.default_root_state.clone()
        joint_position = robot.data.default_joint_pos.clone()
        joint_velocity = torch.zeros_like(robot.data.default_joint_vel)
        for motion_index, frame_count in enumerate(frame_counts):
            index = min(frame_index, frame_count - 1)
            root_state[motion_index, :3] = root_positions[motion_index][index] + scene.env_origins[motion_index]
            root_state[motion_index, 3:7] = root_rotations[motion_index][index]
            root_state[motion_index, 7:13] = 0.0
            joint_position[motion_index] = joint_positions[motion_index][index]

        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(joint_position, joint_velocity)
        sim.render()
        scene.update(sim.get_physics_dt())

        body_positions = robot.data.body_pos_w[:, key_body_indices] - scene.env_origins.unsqueeze(1)
        for motion_index, frame_count in enumerate(frame_counts):
            if frame_index < frame_count:
                key_body_buffers[motion_index][frame_index] = body_positions[motion_index].cpu().numpy()

    for motion, key_body_buffer in zip(motions, key_body_buffers, strict=True):
        motion["key_body_pos"] = key_body_buffer


def _validate_and_write(
    files: list[Path],
    input_dir: Path,
    output_dir: Path,
    source_motions: list[dict],
    converted_motions: list[dict],
    target_lab_names: list[str],
    data_mapping: list[int],
    key_body_names: list[str],
) -> dict:
    per_file = []
    total_frames = 0
    max_quaternion_norm_error = 0.0
    max_common_joint_error = 0.0
    max_adjacent_joint_delta = 0.0
    wrap_boundary_checks = 0

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    for path, source, target in zip(files, source_motions, converted_motions, strict=True):
        frame_count = target["dof_pos"].shape[0]
        dt = 1.0 / float(target["fps"])
        dof_velocity = _forward_diff(target["dof_pos"], dt)
        dof_acceleration = _forward_diff(dof_velocity, dt)
        root_velocity = _forward_diff(target["root_pos"], dt)

        if target["dof_pos"].shape != (frame_count, 23):
            raise ValueError(f"{path}: converted dof_pos shape is {target['dof_pos'].shape}")
        if target["dof_names"] != target_lab_names:
            raise ValueError(f"{path}: converted joint names are not in articulation order")
        if target["key_body_pos"].shape != (frame_count, len(key_body_names), 3):
            raise ValueError(f"{path}: converted key_body_pos shape is {target['key_body_pos'].shape}")
        for field in ("root_pos", "root_rot", "dof_pos", "key_body_pos"):
            if not np.isfinite(target[field]).all():
                raise ValueError(f"{path}: converted {field} contains NaN or Inf")
        for name, value in (
            ("root_velocity", root_velocity),
            ("dof_velocity", dof_velocity),
            ("dof_acceleration", dof_acceleration),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"{path}: derived {name} contains NaN or Inf")

        quaternion_error = float(np.max(np.abs(np.linalg.norm(target["root_rot"], axis=-1) - 1.0)))
        common_joint_error = float(np.max(np.abs(target["dof_pos"] - source["dof_pos"][:, data_mapping])))
        adjacent_joint_delta = float(np.max(np.abs(np.diff(target["dof_pos"], axis=0))))
        if common_joint_error != 0.0:
            raise ValueError(f"{path}: common joint values changed by {common_joint_error}")
        if target["loop_mode"] == 1:
            wrap_boundary_checks += 1
            boundary_delta = float(np.max(np.abs(target["dof_pos"][0] - target["dof_pos"][-1])))
            if boundary_delta > max(5.0 * adjacent_joint_delta, 0.25):
                raise ValueError(f"{path}: abnormal wrap boundary delta {boundary_delta}")

        relative_path = path.relative_to(input_dir)
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as stream:
            pickle.dump(target, stream)

        total_frames += frame_count
        max_quaternion_norm_error = max(max_quaternion_norm_error, quaternion_error)
        max_common_joint_error = max(max_common_joint_error, common_joint_error)
        max_adjacent_joint_delta = max(max_adjacent_joint_delta, adjacent_joint_delta)
        per_file.append(
            {
                "path": str(relative_path),
                "frames": frame_count,
                "fps": float(target["fps"]),
                "dof_pos_shape": list(target["dof_pos"].shape),
                "derived_dof_vel_shape": list(dof_velocity.shape),
                "derived_dof_acc_shape": list(dof_acceleration.shape),
                "key_body_pos_shape": list(target["key_body_pos"].shape),
                "quaternion_norm_max_error": quaternion_error,
                "common_joint_max_error": common_joint_error,
                "max_adjacent_joint_delta": adjacent_joint_delta,
                "loop_mode": int(target["loop_mode"]),
            }
        )

    report = {
        "status": "PASS",
        "source_directory": str(input_dir.resolve()),
        "output_directory": str(output_dir.resolve()),
        "file_count": len(files),
        "total_frames": total_frames,
        "joint_count": 23,
        "joint_names": target_lab_names,
        "key_body_names": key_body_names,
        "source_to_target_articulation_indices": data_mapping,
        "sdk_mapping": EXPECTED_SDK_MAPPING,
        "max_quaternion_norm_error": max_quaternion_norm_error,
        "max_common_joint_error": max_common_joint_error,
        "max_adjacent_joint_delta": max_adjacent_joint_delta,
        "wrap_boundary_checks": wrap_boundary_checks,
        "clamp_boundary_policy": "First/last wrap continuity is not applicable to clamp motions.",
        "source_optional_joint_fields": {
            "dof_vel": "absent; derived by MotionDataManager from converted dof_pos and fps",
            "dof_acc": "absent; validation derives it from converted dof_vel",
            "dof_torque": "absent",
            "target_dof_pos": "absent",
            "reference_action": "dof_pos is the reference joint trajectory",
        },
        "files": per_file,
    }
    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
    return report


def main():
    input_dir = args_cli.input_dir.resolve()
    output_dir = args_cli.output_dir.resolve()
    source_cfg = _load_yaml(args_cli.source_config.resolve())
    target_cfg = _load_yaml(args_cli.target_config.resolve())
    source_lab_names, target_lab_names, data_mapping = _validate_joint_orders(source_cfg, target_cfg)
    key_body_names = list(target_cfg["lab_key_body_names"])

    files = sorted(input_dir.rglob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"No pickle motion files found under {input_dir}")
    source_motions, converted_motions = _prepare_motions(
        files, input_dir, source_lab_names, target_lab_names, data_mapping
    )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    scene = InteractiveScene(
        ConversionSceneCfg(num_envs=len(files), env_spacing=2.5, replicate_physics=True)
    )
    source_robot = Articulation(UNITREE_G1_29DOF_CFG.replace(prim_path="/World/SourceRobot"))
    sim.reset()
    scene.reset()

    target_robot: Articulation = scene["robot"]
    if list(source_robot.joint_names) != source_lab_names:
        raise ValueError(
            "29-DoF source USD order does not match g1_29dof.yaml: "
            f"usd={source_robot.joint_names}, yaml={source_lab_names}"
        )
    if list(target_robot.joint_names) != target_lab_names:
        raise ValueError(
            "23-DoF target USD order does not match g1_23dof.yaml: "
            f"usd={target_robot.joint_names}, yaml={target_lab_names}"
        )
    if target_robot.num_joints != 23 or set(target_robot.joint_names) != set(G1_23DOF_JOINT_NAMES):
        raise ValueError(f"Target articulation is not the required 23-DoF G1: {target_robot.joint_names}")
    if DELETED_JOINT_NAMES & set(target_robot.joint_names):
        raise ValueError("Target articulation contains one or more deleted joints")
    missing_bodies = sorted(set(key_body_names) - set(target_robot.body_names))
    if missing_bodies:
        raise ValueError(f"Target articulation is missing key bodies: {missing_bodies}")

    print(f"SOURCE_ARTICULATION_ORDER={source_lab_names}", flush=True)
    print(f"TARGET_ARTICULATION_ORDER={target_lab_names}", flush=True)
    print(f"SOURCE_TO_TARGET_ARTICULATION_INDICES={data_mapping}", flush=True)
    print(f"SDK_MAPPING={EXPECTED_SDK_MAPPING}", flush=True)
    print(f"KEY_BODY_NAMES={key_body_names}", flush=True)
    _reconstruct_key_bodies(sim, scene, converted_motions, key_body_names)
    report = _validate_and_write(
        files,
        input_dir,
        output_dir,
        source_motions,
        converted_motions,
        target_lab_names,
        data_mapping,
        key_body_names,
    )
    print(
        "G1_29DOF_TO_23DOF_CONVERSION=PASS "
        f"files={report['file_count']} frames={report['total_frames']} output={report['output_directory']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
