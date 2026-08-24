"""G1 23-DoF AMP task configuration."""

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.amp.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.unitree import G1_23DOF_JOINT_NAMES, UNITREE_G1_23DOF_CFG
from legged_lab.tasks.locomotion.amp.config.g1.g1_amp_env_cfg import G1AmpEnvCfg


KEY_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_roll_rubber_hand",
    "right_wrist_roll_rubber_hand",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]


def _standard_joint_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(name="robot", joint_names=list(G1_23DOF_JOINT_NAMES), preserve_order=True)


@configclass
 class G1_23DOFAmpEnvCfg(G1AmpEnvCfg):
    """AMP environment backed by the official G1 23-DoF model and converted dataset."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_23dof", "amp", "walk_and_run"
        )

        self.actions.joint_pos.joint_names = list(G1_23DOF_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True

        self.observations.policy.joint_pos.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.policy.joint_vel.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.critic.joint_pos.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.critic.joint_vel.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.critic.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.disc.joint_pos.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.disc.joint_vel.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.disc_demo.ref_joint_pos.func = mdp.ref_joint_pos_by_name
        self.observations.disc_demo.ref_joint_pos.params["joint_names"] = list(G1_23DOF_JOINT_NAMES)
        self.observations.disc_demo.ref_joint_vel.func = mdp.ref_joint_vel_by_name
        self.observations.disc_demo.ref_joint_vel.params["joint_names"] = list(G1_23DOF_JOINT_NAMES)


@configclass
class G1_23DOFAmpEnvCfg_PLAY(G1_23DOFAmpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 48
        self.scene.env_spacing = 2.5
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 3.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.events.reset_from_ref = None
