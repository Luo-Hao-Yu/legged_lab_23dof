from isaaclab.utils import configclass

from legged_lab.tasks.locomotion.deepmimic.config.g1.agents.rsl_rl_ppo_cfg import G1DeepMimicPPORunnerCfg


@configclass
class G1_23DOFDeepMimicPPORunnerCfg(G1DeepMimicPPORunnerCfg):
    experiment_name = "g1_23dof_deepmimic"
