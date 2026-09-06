"""Environment factory, replacing PhysGraph's physgraph_envs.lib.make(...)
(the real call site is in main/rl/train.py's create_isaacgym_env()).
Our version instantiates our own mujoco-warp MyoHandPourEnv instead of
an IsaacGym task, ignoring the IsaacGym-specific kwargs (graphics_device_id,
multi_gpu, has_headless_arg) that don't apply to our engine."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.mjwarp_myohand_env import MyoHandPourEnv


def make(sim_device="cuda:0", rl_device="cuda:0", cfg=None, headless=True, **kwargs):
    num_envs = cfg.get("env", {}).get("numEnvs", 4) if cfg is not None else kwargs.get("num_envs", 4)
    return MyoHandPourEnv(num_envs=num_envs, device=sim_device, headless=headless)
