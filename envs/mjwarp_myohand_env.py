"""MyoHand pour-task environment, mujoco-warp backend. Ported from
PhysGraph's dexhandmanip_sh.py (single-hand IsaacGym task), matching
the ComplexObsRLGPUEnv interface (lib/utils/rlgames_utils.py) that
rl_games expects: step/reset/get_number_of_agents/action_space/
observation_space/success_buf/failure_buf/error_buf.

FIXED TRAJECTORY ONLY for now, per Runfa's direction 2026-09-04
("just prove you can train the first successful policy for a fixed
trajectory... we'll consider augmentation later").

Action space: direct wrist target pose (3 pos + 6 rot6d) + per-dof
targets -- matching Runfa's real-hardware preference, implemented via
the same PID-controlled-wrist-pose mechanism PhysGraph itself
documents as an alternate mode (usePIDControl=True in
physgraph_envs/lib/envs/dexhands/base.py), using ArtiMANO's own
reference gains (physgraph_envs/lib/envs/dexhands/artimano_real.py):
Kp_pos=10, Ki_pos=0.003, Kd_pos=0.5, Kp_rot=0.3, Ki_rot=0.01, Kd_rot=0.005
-- NOT yet tuned for MyoHand's own mass/scale.

Real mujoco_warp API confirmed on the pod 2026-09-05 (see project notes
for the full trace) -- mjw.put_model/make_data/step/forward, Data.qpos/
qvel/ctrl/xpos/xquat, wp.to_torch zero-copy interop confirmed working
for both reads and indexed-assignment writes.
"""
import os
import pickle

import gym
import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp


def aa_to_quat(aa):
    """Axis-angle (..., 3) -> quaternion (..., 4), [w, x, y, z] (MuJoCo's
    qpos convention). Self-contained -- avoids depending on PhysGraph's
    own torch_utils, which pulls in IsaacGym via the package __init__
    chain for anything beyond the standalone-loadable files."""
    theta = torch.norm(aa, dim=-1, keepdim=True).clamp(min=1e-8)
    axis = aa / theta
    half = theta / 2
    w = torch.cos(half)
    xyz = axis * torch.sin(half)
    return torch.cat([w, xyz], dim=-1)


def quat_to_aa(q):
    """Inverse of aa_to_quat -- quaternion [w,x,y,z] -> axis-angle (...,3),
    needed to express the PID position/rotation error signal."""
    w = q[..., 0].clamp(-1, 1)
    xyz = q[..., 1:]
    theta = 2 * torch.acos(w)
    sin_half = torch.sqrt((1 - w * w).clamp(min=1e-12))
    axis = xyz / sin_half.unsqueeze(-1)
    return axis * theta.unsqueeze(-1)


def rotmat_to_quat(R):
    """Rotation matrix (..., 3, 3) -> quaternion (..., 4), [w,x,y,z]."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q = torch.zeros(R.shape[:-2] + (4,), device=R.device, dtype=R.dtype)
    q[..., 0] = torch.sqrt(torch.clamp(tr + 1, min=0)) / 2
    q[..., 1] = torch.sign(R[..., 2, 1] - R[..., 1, 2]) * torch.sqrt(
        torch.clamp(1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2], min=0)) / 2
    q[..., 2] = torch.sign(R[..., 0, 2] - R[..., 2, 0]) * torch.sqrt(
        torch.clamp(1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2], min=0)) / 2
    q[..., 3] = torch.sign(R[..., 1, 0] - R[..., 0, 1]) * torch.sqrt(
        torch.clamp(1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2], min=0)) / 2
    return q


def quat_mul(q1, q2):
    """Hamilton product, [w,x,y,z] convention, (..., 4) x (..., 4) -> (..., 4)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def quat_conjugate(q):
    return torch.cat([q[..., 0:1], -q[..., 1:]], dim=-1)


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

        self.demo_opt_wrist_pos = self._to_tensor(self.rh_demo["opt_wrist_pos"])
        self.demo_opt_wrist_rot = self._to_tensor(self.rh_demo["opt_wrist_rot"])
        self.demo_opt_dof_pos = self._to_tensor(self.rh_demo["opt_dof_pos"])
        # real source-mug trajectory (part of the rh demo dict -- that's
        # the object the right hand actually manipulates)
        self.demo_src_obj_traj = self._to_tensor(self.rh_demo["obj_trajectory"])

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

        self.model = mjw.put_model(self.mj_model)
        self.data = mjw.make_data(self.mj_model, nworld=num_envs, nconmax=num_envs * 50, njmax=num_envs * 100)

        self.n_dofs_hand = 23
        self.dof_names = self._dof_names()
        self.body_names = self._body_names()
        self.root_adr = self._joint_qpos_adr("root_r")
        self.root_dof_adr = self._joint_dof_adr("root_r")
        self.dof_adrs = np.array([self._joint_qpos_adr(n) for n in self.dof_names])
        self.dof_veladrs = np.array([self._joint_dof_adr(n) for n in self.dof_names])
        self.body_ids = np.array([mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.body_names])
        self.src_adr = self._joint_qpos_adr("src_O02@0015@00020_free")
        self.dst_adr = self._joint_qpos_adr("dst_O02@0010@00003_free")

        self.progress_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failure_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.error_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # PID gains, ArtiMANO reference values (physgraph_envs/lib/envs/
        # dexhands/artimano_real.py) -- NOT yet tuned for MyoHand
        self.Kp_pos, self.Ki_pos, self.Kd_pos = 10.0, 0.003, 0.5
        self.Kp_rot, self.Ki_rot, self.Kd_rot = 0.3, 0.01, 0.005
        self.dt = self.mj_model.opt.timestep
        self.pos_error_integral = torch.zeros(num_envs, 3, device=device)
        self.prev_pos_error = torch.zeros(num_envs, 3, device=device)
        self.rot_error_integral = torch.zeros(num_envs, 3, device=device)
        self.prev_rot_error = torch.zeros(num_envs, 3, device=device)

        act_dim = 9 + self.n_dofs_hand
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        obs_dim = (23 * 3 + 13) + (23 + 3 + 4) + (3 + 4 + 23)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _to_tensor(self, x):
        return torch.tensor(np.array(x), device=self.device, dtype=torch.float32)

    def _joint_qpos_adr(self, name):
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.mj_model.jnt_qposadr[jid])

    def _joint_dof_adr(self, name):
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self.mj_model.jnt_dofadr[jid])

    def _dof_names(self):
        return [
            "pro_sup_r", "deviation_r", "flexion_r",
            "cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r",
            "mcp2_flexion_r", "mcp2_abduction_r", "pm2_flexion_r", "md2_flexion_r",
            "mcp3_flexion_r", "mcp3_abduction_r", "pm3_flexion_r", "md3_flexion_r",
            "mcp4_flexion_r", "mcp4_abduction_r", "pm4_flexion_r", "md4_flexion_r",
            "mcp5_flexion_r", "mcp5_abduction_r", "pm5_flexion_r", "md5_flexion_r",
        ]

    def _body_names(self):
        return [
            "lunate_r_j1", "firstmc_r_j1", "proximal_thumb_r", "distal_thumb_r", "THtip_r",
            "2proxph_r_j1", "midph2_r", "distph2_r", "IFtip_r",
            "3proxph_r_j1", "midph3_r", "distph3_r", "MFtip_r",
            "4proxph_r_j1", "midph4_r", "distph4_r", "RFtip_r",
            "5proxph_r_j1", "midph5_r", "distph5_r", "LFtip_r",
        ]

    def get_number_of_agents(self):
        return self.num_envs

    def reset(self):
        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)

        opt_wrist_pos0 = self.demo_opt_wrist_pos[0]
        opt_wrist_rot0 = self.demo_opt_wrist_rot[0]
        opt_dof_pos0 = self.demo_opt_dof_pos[0]
        src_pose0 = self.demo_src_obj_traj[0]  # 4x4, real captured frame-0 pose
        dst_pose0 = self._to_tensor(self.lh_demo["obj_trajectory"][0])  # 4x4, static

        qpos[:, self.root_adr:self.root_adr + 3] = opt_wrist_pos0
        qpos[:, self.root_adr + 3:self.root_adr + 7] = aa_to_quat(opt_wrist_rot0[None])[0]
        qpos[:, self.dof_adrs] = opt_dof_pos0

        qpos[:, self.src_adr:self.src_adr + 3] = src_pose0[:3, 3]
        qpos[:, self.src_adr + 3:self.src_adr + 7] = rotmat_to_quat(src_pose0[:3, :3][None])[0]

        qpos[:, self.dst_adr:self.dst_adr + 3] = dst_pose0[:3, 3]
        qpos[:, self.dst_adr + 3:self.dst_adr + 7] = rotmat_to_quat(dst_pose0[:3, :3][None])[0]

        qvel.zero_()
        mjw.forward(self.model, self.data)

        self.progress_buf.zero_()
        self.success_buf_.zero_()
        self.failure_buf_.zero_()
        self.error_buf_.zero_()
        self.pos_error_integral.zero_()
        self.prev_pos_error.zero_()
        self.rot_error_integral.zero_()
        self.prev_rot_error.zero_()

        return self._compute_obs()

    def _compute_obs(self):
        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)

        dof_pos = qpos[:, self.dof_adrs]
        dof_vel = qvel[:, self.dof_veladrs]
        wrist_pos = qpos[:, self.root_adr:self.root_adr + 3]
        wrist_quat = qpos[:, self.root_adr + 3:self.root_adr + 7]
        wrist_linvel = qvel[:, self.root_dof_adr:self.root_dof_adr + 3]
        wrist_angvel = qvel[:, self.root_dof_adr + 3:self.root_dof_adr + 6]

        proprio = torch.cat([
            dof_pos, torch.cos(dof_pos), torch.sin(dof_pos),
            torch.zeros_like(wrist_pos), wrist_quat, wrist_linvel, wrist_angvel,
        ], dim=-1)

        src_pos = qpos[:, self.src_adr:self.src_adr + 3]
        src_quat = qpos[:, self.src_adr + 3:self.src_adr + 7]
        privileged = torch.cat([dof_vel, src_pos - wrist_pos, src_quat], dim=-1)

        next_idx = torch.clamp(self.progress_buf + 1, max=self.seq_len - 1)
        target_wrist_pos = self.demo_opt_wrist_pos[next_idx]
        target_wrist_rot = self.demo_opt_wrist_rot[next_idx]
        target_dof_pos = self.demo_opt_dof_pos[next_idx]
        target_wrist_quat = aa_to_quat(target_wrist_rot)

        delta_wrist_pos = target_wrist_pos - wrist_pos
        delta_wrist_quat = quat_mul(wrist_quat, quat_conjugate(target_wrist_quat))
        delta_dof_pos = target_dof_pos - dof_pos

        future_target = torch.cat([delta_wrist_pos, delta_wrist_quat, delta_dof_pos], dim=-1)

        obs = torch.cat([proprio, privileged, future_target], dim=-1)
        return {"obs": {"obs": obs}}

    def step(self, action):
        action = torch.clamp(action, -1.0, 1.0)

        # direct-pose wrist action -> PID-computed force (matching
        # PhysGraph's usePIDControl mechanism, ArtiMANO reference gains)
        pos_error = action[:, 0:3]  # [-1,1], interpreted as a position error signal
        self.pos_error_integral += pos_error * self.dt
        self.pos_error_integral.clamp_(-1, 1)
        pos_derivative = (pos_error - self.prev_pos_error) / self.dt
        wrist_force = self.Kp_pos * pos_error + self.Ki_pos * self.pos_error_integral + self.Kd_pos * pos_derivative
        self.prev_pos_error = pos_error

        rot_error_6d = action[:, 3:9]
        rot_error = rot_error_6d[:, :3]  # simplified: first 3 components as an axis-angle-style error
        self.rot_error_integral += rot_error * self.dt
        self.rot_error_integral.clamp_(-1, 1)
        rot_derivative = (rot_error - self.prev_rot_error) / self.dt
        wrist_torque = self.Kp_rot * rot_error + self.Ki_rot * self.rot_error_integral + self.Kd_rot * rot_derivative
        self.prev_rot_error = rot_error

        dof_targets = action[:, 9:]  # [-1,1] -- direct per-dof position targets, scaling deferred

        qfrc = wp.to_torch(self.data.qfrc_applied)
        qfrc.zero_()
        qfrc[:, self.root_dof_adr:self.root_dof_adr + 3] = wrist_force
        qfrc[:, self.root_dof_adr + 3:self.root_dof_adr + 6] = wrist_torque

        ctrl = wp.to_torch(self.data.ctrl)
        ctrl[:, :] = 0.0  # finger actuator scaling from dof_targets still TODO

        mjw.step(self.model, self.data)

        self.progress_buf += 1
        dones = (self.progress_buf >= self.max_episode_length).float()
        rewards = torch.zeros(self.num_envs, device=self.device)  # real imitation reward next
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
