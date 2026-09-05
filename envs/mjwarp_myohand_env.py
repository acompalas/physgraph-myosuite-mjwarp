"""MyoHand pour-task environment, mujoco-warp backend. Ported from
PhysGraph's dexhandmanip_sh.py (single-hand IsaacGym task), matching
the ComplexObsRLGPUEnv interface (lib/utils/rlgames_utils.py) that
rl_games expects: step/reset/get_number_of_agents/action_space/
observation_space/success_buf/failure_buf/error_buf.

FIRST DRAFT / MINIMAL: proves out the mujoco-warp integration
(batched parallel stepping, warp<->torch interop, rl_games wiring)
with a FIXED trajectory only (no randomized spawn positions yet, per
Runfa's direction 2026-09-04 -- "just prove you can train the first
successful policy for a fixed trajectory"). Reward/success logic here
is a simplified first pass, not yet the full per-finger-weighted
imitation reward from compute_imitation_reward -- that's the next
layer to add once this integration is confirmed working end-to-end.
"""
import os
import pickle

import gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


class MyoHandPourEnv:
    def __init__(self, num_envs, device="cuda:0", max_episode_length=None, headless=True):
        self.num_envs = num_envs
        self.device = device
        self.headless = headless

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        demo_path = os.path.join(repo_root, "assets/retargeted/1292e_training_demo.pkl")
        with open(demo_path, "rb") as f:
            demo = pickle.load(f)
        self.rh_demo = demo["rh"]
        self.lh_demo = demo["lh_stationary"]
        self.pour_frame = demo["pour_frame"]

        self.seq_len = self.rh_demo["opt_dof_pos"].shape[0]
        self.max_episode_length = max_episode_length or self.seq_len

        # build the combined scene (right hand + source mug + stationary
        # destination mug) -- same MjSpec attach pattern as our playback
        # scripts, ONE instance; mjw.make_data(nworld=...) replicates it
        hand_xml = os.path.join(repo_root, "assets/hands/myohand_r_ulnaroot_scene.xml")
        mug_src_xml = os.path.join(repo_root, "assets/objects/O02@0015@00020/O02@0015@00020.xml")
        mug_dst_xml = os.path.join(repo_root, "assets/objects/O02@0010@00003/O02@0010@00003.xml")

        base = mujoco.MjSpec.from_file(hand_xml)
        mug_src = mujoco.MjSpec.from_file(mug_src_xml)
        mug_dst = mujoco.MjSpec.from_file(mug_dst_xml)
        for spec, prefix in [(mug_src, "src_"), (mug_dst, "dst_")]:
            site = base.worldbody.add_site(name=f"attach_{prefix}", pos=[0, 0, 0])
            base.attach(spec, prefix=prefix, site=site)
            base.delete(site)

        self.mj_model = base.compile()
        mj_data = mujoco.MjData(self.mj_model)

        self.model = mjw.put_model(self.mj_model)
        # nconmax/njmax are best-effort starting guesses -- per the mjwarp
        # docs, tune these up if overflow warnings appear once we can
        # actually run this and watch for them
        self.data = mjw.make_data(self.mj_model, nworld=num_envs, nconmax=num_envs * 50, njmax=num_envs * 100)

        self.n_dofs_hand = 23  # MyoHandR dof count, see myohand_def.py
        self.root_adr = self._joint_qpos_adr("root_r")
        self.dof_adrs = np.array([self._joint_qpos_adr(n) for n in self._dof_names()])
        self.src_adr = self._joint_qpos_adr("src_O02@0015@00020_free")
        self.dst_adr = self._joint_qpos_adr("dst_O02@0010@00003_free")

        self.progress_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failure_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.error_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # action space: direct wrist target pose (3 pos + 6 rot6d, per
        # Runfa's usePIDControl-style direct-pose direction) + per-dof
        # joint targets, all in [-1, 1] like PhysGraph's own convention
        act_dim = 9 + self.n_dofs_hand
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        # observation space: placeholder first pass -- qpos/qvel for the
        # hand's own dofs + wrist pose; real per-finger/target-relative
        # obs construction (matching compute_observations) comes next
        obs_dim = self.n_dofs_hand * 2 + 7 + 6
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _joint_qpos_adr(self, name):
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.mj_model.jnt_qposadr[jid])

    def _dof_names(self):
        return [
            "pro_sup_r", "deviation_r", "flexion_r",
            "cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r",
            "mcp2_flexion_r", "mcp2_abduction_r", "pm2_flexion_r", "md2_flexion_r",
            "mcp3_flexion_r", "mcp3_abduction_r", "pm3_flexion_r", "md3_flexion_r",
            "mcp4_flexion_r", "mcp4_abduction_r", "pm4_flexion_r", "md4_flexion_r",
            "mcp5_flexion_r", "mcp5_abduction_r", "pm5_flexion_r", "md5_flexion_r",
        ]

    def get_number_of_agents(self):
        return self.num_envs

    def reset(self):
        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)

        opt_wrist_pos0 = torch.tensor(self.rh_demo["opt_wrist_pos"][0], device=self.device, dtype=torch.float32)
        opt_wrist_rot0 = torch.tensor(self.rh_demo["opt_wrist_rot"][0], device=self.device, dtype=torch.float32)
        opt_dof_pos0 = torch.tensor(self.rh_demo["opt_dof_pos"][0], device=self.device, dtype=torch.float32)
        dst_pose0 = self.lh_demo["obj_trajectory"][0]  # 4x4, static across all frames

        qpos[:, self.root_adr:self.root_adr + 3] = opt_wrist_pos0
        # NOTE: orientation stored as axis-angle in opt_wrist_rot; convert
        # to quat for qpos -- placeholder identity for this first pass,
        # real conversion to follow once the basic loop is confirmed working
        qpos[:, self.root_adr + 3:self.root_adr + 7] = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32)
        qpos[:, self.dof_adrs] = opt_dof_pos0
        qvel.zero_()

        self.progress_buf.zero_()
        self.success_buf_.zero_()
        self.failure_buf_.zero_()
        self.error_buf_.zero_()

        mujoco.mj_forward(self.mj_model, mujoco.MjData(self.mj_model))  # sanity no-op on host model
        return self._compute_obs()

    def _compute_obs(self):
        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)
        dof_pos = qpos[:, self.dof_adrs]
        dof_vel = qvel[:, self.dof_adrs]
        wrist_pose = qpos[:, self.root_adr:self.root_adr + 7]
        obs = torch.cat([dof_pos, dof_vel, wrist_pose, torch.zeros(self.num_envs, 6, device=self.device)], dim=-1)
        return {"obs": {"obs": obs}}

    def step(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        ctrl = wp.to_torch(self.data.ctrl)
        # placeholder direct application -- real PD/target-pose conversion
        # (matching PhysGraph's rh_curr_targets scale+moving-average logic)
        # to follow; this just proves the step loop runs end-to-end
        ctrl[:, : ctrl.shape[1]] = 0.0

        mjw.step(self.model, self.data)

        self.progress_buf += 1
        dones = (self.progress_buf >= self.max_episode_length).float()
        rewards = torch.zeros(self.num_envs, device=self.device)  # placeholder
        obs = self._compute_obs()
        infos = {}
        return obs, rewards, dones, infos

    @property
    def success_buf(self):
        return self.success_buf_

    @property
    def failure_buf(self):
        return self.failure_buf_

    @property
    def error_buf(self):
        return self.error_buf_
