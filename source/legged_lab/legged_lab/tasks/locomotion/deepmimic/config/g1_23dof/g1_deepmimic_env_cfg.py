"""G1 23-DoF DeepMimic task configuration."""

import os

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import legged_lab.tasks.locomotion.deepmimic.mdp as mdp
from legged_lab import LEGGED_LAB_ROOT_DIR
from legged_lab.assets.unitree import G1_23DOF_JOINT_NAMES, UNITREE_G1_23DOF_CFG
from legged_lab.tasks.locomotion.deepmimic.config.g1.g1_deepmimic_env_cfg import G1DeepMimicEnvCfg


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
class G1_23DOFDeepMimicEnvCfg(G1DeepMimicEnvCfg):
    """DeepMimic environment backed by the official G1 23-DoF model and dataset."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_G1_23DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.motion_data.motion_dataset.motion_data_dir = os.path.join(
            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_23dof", "deepmimic"
        )

        self.actions.joint_pos.joint_names = list(G1_23DOF_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.observations.policy.joint_pos.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.policy.joint_vel.params = {"asset_cfg": _standard_joint_cfg()}
        self.observations.policy.key_body_pos_b.params = {
            "asset_cfg": SceneEntityCfg(name="robot", body_names=KEY_BODY_NAMES, preserve_order=True)
        }
        self.observations.policy.ref_joint_pos.func = mdp.ref_joint_pos_by_name
        self.observations.policy.ref_joint_pos.params["joint_names"] = list(G1_23DOF_JOINT_NAMES)
        self.observations.policy.ref_key_body_pos_b.params = {"animation": "animation"}

        self.rewards.ref_track_key_body_pos_b_error_exp.params["asset_cfg"] = SceneEntityCfg(
            name="robot", body_names=KEY_BODY_NAMES, preserve_order=True
        )
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "torso_link",
            "pelvis",
            ".*_shoulder_.*_link",
            ".*_elbow_link",
        ]
        self.terminations.deviation_key_body_pos_w.params["asset_cfg"] = SceneEntityCfg(
            name="robot", body_names=KEY_BODY_NAMES, preserve_order=True
        )


@configclass
class G1_23DOFDeepMimicEnvCfg_PLAY(G1_23DOFDeepMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.animation.animation.random_initialize = False
