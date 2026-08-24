"""Load the G1 23-DoF USD and validate its controllable joints."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from legged_lab.assets.unitree import G1_23DOF_JOINT_NAMES, UNITREE_G1_23DOF_CFG


DELETED_JOINT_NAMES = {
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    prim_utils.create_prim("/World/Origin", "Xform")
    robot = Articulation(UNITREE_G1_23DOF_CFG.replace(prim_path="/World/Origin/Robot"))
    print("MODEL_VALIDATION_STAGE=before_sim_reset", flush=True)
    sim.reset()
    print("MODEL_VALIDATION_STAGE=after_sim_reset", flush=True)

    actual_joint_names = list(robot.joint_names)
    expected_joint_names = list(G1_23DOF_JOINT_NAMES)
    missing = sorted(set(expected_joint_names) - set(actual_joint_names))
    extra = sorted(set(actual_joint_names) - set(expected_joint_names))
    deleted_present = sorted(DELETED_JOINT_NAMES & set(actual_joint_names))

    if robot.num_joints != 23:
        raise RuntimeError(f"Expected 23 controllable joints, found {robot.num_joints}: {actual_joint_names}")
    if missing or extra:
        raise RuntimeError(f"G1 23-DoF joint mismatch: missing={missing}, extra={extra}")
    if deleted_present:
        raise RuntimeError(f"Deleted joints are still present: {deleted_present}")

    standard_to_articulation = [actual_joint_names.index(name) for name in expected_joint_names]
    print("G1_23DOF_MODEL_VALIDATION=PASS", flush=True)
    print(f"USD_PATH={UNITREE_G1_23DOF_CFG.spawn.usd_path}", flush=True)
    print(f"NUM_JOINTS={robot.num_joints}", flush=True)
    print(f"ARTICULATION_JOINT_ORDER={actual_joint_names}", flush=True)
    print(f"STANDARD_JOINT_ORDER={expected_joint_names}", flush=True)
    print(f"STANDARD_TO_ARTICULATION={standard_to_articulation}", flush=True)
    print(f"BODY_NAMES={list(robot.body_names)}", flush=True)
    print(f"SOFT_JOINT_POSITION_LIMITS={robot.data.soft_joint_pos_limits[0].cpu().tolist()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
