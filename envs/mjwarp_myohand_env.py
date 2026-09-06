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


def quat_to_angle(q):
    """Quaternion [w,x,y,z] -> rotation angle (magnitude only, radians).
    Matches PhysGraph's quat_to_angle_axis(...)[0] usage in
    compute_imitation_reward (only the angle component is used there)."""
    w = q[..., 0].clamp(-1, 1)
    return 2 * torch.acos(w.abs())  # abs() avoids the double-cover sign ambiguity


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
        self.src_dof_adr = self._joint_dof_adr("src_O02@0015@00020_free")

        # reuse our own already-built, tested weight_idx (myohand_def.py,
        # matches PhysGraph's DexHand.weight_idx convention exactly)
        import sys
        sys.path.insert(0, os.path.join(repo_root, "scripts/retargeting"))
        from myohand_def import MyoHandR
        self.weight_idx = MyoHandR().weight_idx

        # finite-difference velocity approximation for per-body
        # positions (joints_vel in PhysGraph's reward) -- real MuJoCo
        # per-body velocity (cvel) has a specific 6D com-based
        # convention not yet verified for mujoco_warp, deferred
        self.prev_body_xpos = None

        dexhand = MyoHandR()
        # real target joint positions, in OUR body_names[1:] order
        # (excluding wrist), matching PhysGraph's pack_data flattening
        # logic exactly -- these are the REAL human MANO joint targets,
        # not our own retargeted opt_dof_pos values
        mano_joints = self.rh_demo["mano_joints"]
        target_joints_list = [mano_joints[dexhand.to_hand(b)[0]] for b in self.body_names[1:]]
        self.demo_target_joints_pos = torch.stack(target_joints_list, dim=1).to(device=self.device, dtype=torch.float32)

        self.progress_buf = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failure_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.error_buf_ = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.total_rewards = torch.zeros(num_envs, device=device)

        # PID gains, ArtiMANO reference values (physgraph_envs/lib/envs/
        # dexhands/artimano_real.py) -- NOT yet tuned for MyoHand
        self.Kp_pos, self.Ki_pos, self.Kd_pos = 10.0, 0.003, 0.5
        self.Kp_rot, self.Ki_rot, self.Kd_rot = 0.3, 0.01, 0.005
        self.dt = self.mj_model.opt.timestep
        self.pos_error_integral = torch.zeros(num_envs, 3, device=device)
        self.prev_pos_error = torch.zeros(num_envs, 3, device=device)
        self.rot_error_integral = torch.zeros(num_envs, 3, device=device)
        self.prev_rot_error = torch.zeros(num_envs, 3, device=device)

        self.n_muscles = self.mj_model.nu  # 39, real muscle-tendon actuators
        # (ctrlrange 0-1, dyntype/gaintype/biastype=muscle -- confirmed
        # via mj_model.nu + actuator names, NOT simple position servos)
        act_dim = 9 + self.n_muscles
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
        # bimanual-SHAPED (RH real + LH inert-zero, same width each),
        # matching PhysGraph's own per-modality concatenated-tensor
        # convention -- see _compute_obs() for the full rationale
        rh_proprio_dim = 23 * 3 + 13   # dof_pos/cos/sin + wrist pos/quat/vel/angvel
        rh_privileged_dim = 23 + 3 + 4  # dof_vel + obj_pos_rel + obj_quat
        rh_target_dim = 3 + 4 + 23      # delta_wrist_pos/quat + delta_dof_pos
        self.observation_space = gym.spaces.Dict({
            "proprioception": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(rh_proprio_dim * 2,), dtype=np.float32),
            "privileged": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(rh_privileged_dim * 2,), dtype=np.float32),
            "target": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(rh_target_dim * 2,), dtype=np.float32),
        })  # lib/rl/base.py's play_steps unconditionally does
        # `for k, v in self.obs["obs"].items()`, so the runtime obs dict
        # AND the declared space must both be genuine multi-key dicts --
        # these 3 keys feed lib.nn.features.SimpleFeatureFusion (real,
        # unmodified PhysGraph utility class) via matching Identity
        # extractors in the rl_train config

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

    def reset_idx(self, env_ids):
        """Reset ONLY the specified environments (real PhysGraph pattern --
        dexhandmanip_bih.py's reset_idx/reset_done -- lets other envs keep
        running asynchronously rather than resetting the whole batch)."""
        if len(env_ids) == 0:
            return

        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)

        opt_wrist_pos0 = self.demo_opt_wrist_pos[0]
        opt_wrist_rot0 = self.demo_opt_wrist_rot[0]
        opt_dof_pos0 = self.demo_opt_dof_pos[0]
        src_pose0 = self.demo_src_obj_traj[0]
        dst_pose0 = self._to_tensor(self.lh_demo["obj_trajectory"][0])

        qpos[env_ids, self.root_adr:self.root_adr + 3] = opt_wrist_pos0
        qpos[env_ids, self.root_adr + 3:self.root_adr + 7] = aa_to_quat(opt_wrist_rot0[None])[0]
        qpos[env_ids[:, None], self.dof_adrs] = opt_dof_pos0

        qpos[env_ids, self.src_adr:self.src_adr + 3] = src_pose0[:3, 3]
        qpos[env_ids, self.src_adr + 3:self.src_adr + 7] = rotmat_to_quat(src_pose0[:3, :3][None])[0]

        qpos[env_ids, self.dst_adr:self.dst_adr + 3] = dst_pose0[:3, 3]
        qpos[env_ids, self.dst_adr + 3:self.dst_adr + 7] = rotmat_to_quat(dst_pose0[:3, :3][None])[0]

        qvel[env_ids] = 0.0

        self.progress_buf[env_ids] = 0
        self.success_buf_[env_ids] = False
        self.failure_buf_[env_ids] = False
        self.error_buf_[env_ids] = False
        self.total_rewards[env_ids] = 0.0
        self.pos_error_integral[env_ids] = 0.0
        self.prev_pos_error[env_ids] = 0.0
        self.rot_error_integral[env_ids] = 0.0
        self.prev_rot_error[env_ids] = 0.0

        mjw.forward(self.model, self.data)

    def reset(self):
        """Full reset -- all envs. Used once at the very start of training."""
        all_env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(all_env_ids)
        return self._compute_obs()

    def reset_done(self):
        """Real PhysGraph interface method (dexhandmanip_bih.py:1494),
        called directly by lib/rl/base.py's play_steps -- resets only the
        environments that finished their episode, returns (obs, done_ids)."""
        reset_buf = self.success_buf_ | self.failure_buf_
        done_env_ids = reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
        return self._compute_obs(), done_env_ids

    def _compute_obs(self):
        """Bimanual-SHAPED observation (per stream: [rh_real, lh_inert]
        concatenated), matching PhysGraph's own convention exactly (its
        transformer network internally slices obs['proprioception'] into
        r_prop/l_prop, i.e. it also expects one concatenated tensor per
        modality covering both hands). LH is currently fully inert (no
        left hand rendered/actuated in this env -- see project notes'
        deferred bimanual milestone) -- its stream is a static zero
        tensor of the SAME width RH uses, so the shape is genuinely
        bimanual-ready even though only RH carries real data today.
        Real per-hand streams built with SimpleFeatureFusion/Identity
        (lib/nn/features/), NOT PhysGraph's bimanual transformer network
        (network_builder_transformer_bih_graph_improve_correct.py) --
        that file is hardcoded to a DIFFERENT embodiment's exact dof/body
        counts (22/27, likely ArtiMANO) plus real BPS object-shape
        encoding we don't compute -- adopting it faithfully is a real,
        separate future task once a second hand does real work."""
        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)

        dof_pos = qpos[:, self.dof_adrs]
        dof_vel = qvel[:, self.dof_veladrs]
        wrist_pos = qpos[:, self.root_adr:self.root_adr + 3]
        wrist_quat = qpos[:, self.root_adr + 3:self.root_adr + 7]
        wrist_linvel = qvel[:, self.root_dof_adr:self.root_dof_adr + 3]
        wrist_angvel = qvel[:, self.root_dof_adr + 3:self.root_dof_adr + 6]

        rh_proprio = torch.cat([
            dof_pos, torch.cos(dof_pos), torch.sin(dof_pos),
            torch.zeros_like(wrist_pos), wrist_quat, wrist_linvel, wrist_angvel,
        ], dim=-1)

        src_pos = qpos[:, self.src_adr:self.src_adr + 3]
        src_quat = qpos[:, self.src_adr + 3:self.src_adr + 7]
        rh_privileged = torch.cat([dof_vel, src_pos - wrist_pos, src_quat], dim=-1)

        next_idx = torch.clamp(self.progress_buf + 1, max=self.seq_len - 1)
        target_wrist_pos = self.demo_opt_wrist_pos[next_idx]
        target_wrist_rot = self.demo_opt_wrist_rot[next_idx]
        target_dof_pos = self.demo_opt_dof_pos[next_idx]
        target_wrist_quat = aa_to_quat(target_wrist_rot)

        delta_wrist_pos = target_wrist_pos - wrist_pos
        delta_wrist_quat = quat_mul(wrist_quat, quat_conjugate(target_wrist_quat))
        delta_dof_pos = target_dof_pos - dof_pos

        rh_target = torch.cat([delta_wrist_pos, delta_wrist_quat, delta_dof_pos], dim=-1)

        lh_proprio = torch.zeros_like(rh_proprio)
        lh_privileged = torch.zeros_like(rh_privileged)
        lh_target = torch.zeros_like(rh_target)

        proprioception = torch.cat([rh_proprio, lh_proprio], dim=-1)
        privileged = torch.cat([rh_privileged, lh_privileged], dim=-1)
        target = torch.cat([rh_target, lh_target], dim=-1)

        return {"obs": {"proprioception": proprioception, "privileged": privileged, "target": target}}

    def _compute_reward(self):
        """Faithful (partial) port of PhysGraph's compute_imitation_reward
        (physgraph_envs/lib/envs/tasks/dexhandmanip_sh.py). Same reward
        terms/weights and failed_execute/succeeded logic, EXCEPT:
        - tip_force/tips_distance/tip_contact_state (contact-force reward
          + its failure check) are OMITTED -- needs mujoco_warp's Contact
          structure, not yet explored/verified.
        - power/wrist_power OMITTED -- needs actuator-force fields not yet
          confirmed to exist.
        - joints_vel APPROXIMATED via finite-difference of body xpos
          across steps, not MuJoCo's real cvel (6D com-based convention,
          not yet verified for mujoco_warp).
        - scale_factor fixed at 1.0 (no tightening curriculum yet -- see
          project notes: this is deferred, not a phase-separated
          curriculum PhysGraph itself doesn't have either).
        """
        mjw.forward(self.model, self.data)  # ensure xpos/xquat are fresh

        qpos = wp.to_torch(self.data.qpos)
        qvel = wp.to_torch(self.data.qvel)
        xpos = wp.to_torch(self.data.xpos)  # (nworld, nbody, 3)

        cur_idx = self.progress_buf
        scale_factor = 1.0

        # === current sim state ===
        current_eef_pos = qpos[:, self.root_adr:self.root_adr + 3]
        current_eef_quat = qpos[:, self.root_adr + 3:self.root_adr + 7]
        current_eef_vel = qvel[:, self.root_dof_adr:self.root_dof_adr + 3]
        current_eef_ang_vel = qvel[:, self.root_dof_adr + 3:self.root_dof_adr + 6]

        joints_pos = xpos[:, self.body_ids[1:], :]  # exclude wrist (index 0)
        if self.prev_body_xpos is None:
            joints_vel = torch.zeros_like(joints_pos)
        else:
            joints_vel = (joints_pos - self.prev_body_xpos) / self.dt
        self.prev_body_xpos = joints_pos.clone()

        current_dof_vel = qvel[:, self.dof_veladrs]

        current_obj_pos = qpos[:, self.src_adr:self.src_adr + 3]
        current_obj_quat = qpos[:, self.src_adr + 3:self.src_adr + 7]
        current_obj_vel = qvel[:, self.src_dof_adr:self.src_dof_adr + 3]
        current_obj_ang_vel = qvel[:, self.src_dof_adr + 3:self.src_dof_adr + 6]

        # === target state (current frame, NOT the observation's lookahead) ===
        target_eef_pos = self.demo_opt_wrist_pos[cur_idx]
        target_eef_quat = aa_to_quat(self.demo_opt_wrist_rot[cur_idx])
        target_joints_pos = self.demo_target_joints_pos[cur_idx]
        target_obj_pos = self.demo_src_obj_traj[cur_idx, :3, 3]
        target_obj_quat = rotmat_to_quat(self.demo_src_obj_traj[cur_idx, :3, :3])
        # velocities: finite-difference of the demo trajectory itself
        next_idx = torch.clamp(cur_idx + 1, max=self.seq_len - 1)
        target_eef_vel = (self.demo_opt_wrist_pos[next_idx] - target_eef_pos) / self.dt
        target_eef_ang_vel = torch.zeros_like(target_eef_vel)  # angular vel approx omitted for now
        target_joints_vel = (self.demo_target_joints_pos[next_idx] - target_joints_pos) / self.dt
        target_obj_vel = (self.demo_src_obj_traj[next_idx, :3, 3] - target_obj_pos) / self.dt
        target_obj_ang_vel = torch.zeros_like(target_obj_vel)

        # === diffs ===
        diff_eef_pos_dist = torch.norm(target_eef_pos - current_eef_pos, dim=-1)
        diff_eef_vel = target_eef_vel - current_eef_vel
        diff_eef_ang_vel = target_eef_ang_vel - current_eef_ang_vel

        diff_joints_pos_dist = torch.norm(target_joints_pos - joints_pos, dim=-1)  # (num_envs, 20)

        def tip_dist(key):
            idx = [k - 1 for k in self.weight_idx[key]]
            return diff_joints_pos_dist[:, idx].mean(dim=-1)

        diff_thumb_tip_pos_dist = tip_dist("thumb_tip")
        diff_index_tip_pos_dist = tip_dist("index_tip")
        diff_middle_tip_pos_dist = tip_dist("middle_tip")
        diff_ring_tip_pos_dist = tip_dist("ring_tip")
        diff_pinky_tip_pos_dist = tip_dist("pinky_tip")
        diff_level_1_pos_dist = tip_dist("level_1_joints")
        diff_level_2_pos_dist = tip_dist("level_2_joints")

        diff_joints_vel = target_joints_vel - joints_vel

        reward_eef_pos = torch.exp(-40 * diff_eef_pos_dist)
        reward_thumb_tip_pos = torch.exp(-100 * diff_thumb_tip_pos_dist)
        reward_index_tip_pos = torch.exp(-90 * diff_index_tip_pos_dist)
        reward_middle_tip_pos = torch.exp(-80 * diff_middle_tip_pos_dist)
        reward_pinky_tip_pos = torch.exp(-60 * diff_pinky_tip_pos_dist)
        reward_ring_tip_pos = torch.exp(-60 * diff_ring_tip_pos_dist)
        reward_level_1_pos = torch.exp(-50 * diff_level_1_pos_dist)
        reward_level_2_pos = torch.exp(-40 * diff_level_2_pos_dist)

        reward_eef_vel = torch.exp(-1 * diff_eef_vel.abs().mean(dim=-1))
        reward_eef_ang_vel = torch.exp(-1 * diff_eef_ang_vel.abs().mean(dim=-1))
        reward_joints_vel = torch.exp(-1 * diff_joints_vel.abs().mean(dim=-1).mean(-1))

        diff_eef_rot = quat_mul(target_eef_quat, quat_conjugate(current_eef_quat))
        diff_eef_rot_angle = quat_to_angle(diff_eef_rot)
        reward_eef_rot = torch.exp(-1 * diff_eef_rot_angle.abs())

        diff_obj_pos_dist = torch.norm(target_obj_pos - current_obj_pos, dim=-1)
        reward_obj_pos = torch.exp(-80 * diff_obj_pos_dist)

        diff_obj_rot = quat_mul(target_obj_quat, quat_conjugate(current_obj_quat))
        diff_obj_rot_angle = quat_to_angle(diff_obj_rot)
        reward_obj_rot = torch.exp(-3 * diff_obj_rot_angle.abs())

        diff_obj_vel = target_obj_vel - current_obj_vel
        reward_obj_vel = torch.exp(-1 * diff_obj_vel.abs().mean(dim=-1))
        diff_obj_ang_vel = target_obj_ang_vel - current_obj_ang_vel
        reward_obj_ang_vel = torch.exp(-1 * diff_obj_ang_vel.abs().mean(dim=-1))

        error_buf = (
            (torch.norm(current_eef_vel, dim=-1) > 100)
            | (torch.norm(current_eef_ang_vel, dim=-1) > 200)
            | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
            | (torch.abs(current_dof_vel).mean(-1) > 200)
            | (torch.norm(current_obj_vel, dim=-1) > 100)
            | (torch.norm(current_obj_ang_vel, dim=-1) > 200)
        )
        failed_execute = (
            (
                (diff_obj_pos_dist > 0.02 / 0.343 * scale_factor ** 3)
                | (diff_thumb_tip_pos_dist > 0.04 / 0.7 * scale_factor)
                | (diff_index_tip_pos_dist > 0.045 / 0.7 * scale_factor)
                | (diff_middle_tip_pos_dist > 0.05 / 0.7 * scale_factor)
                | (diff_pinky_tip_pos_dist > 0.06 / 0.7 * scale_factor)
                | (diff_ring_tip_pos_dist > 0.06 / 0.7 * scale_factor)
                | (diff_level_1_pos_dist > 0.07 / 0.7 * scale_factor)
                | (diff_level_2_pos_dist > 0.08 / 0.7 * scale_factor)
                | (diff_obj_rot_angle.abs() / np.pi * 180 > 30 / 0.343 * scale_factor ** 3)
            )
            & (self.progress_buf >= 8)
        ) | error_buf

        reward = (
            0.1 * reward_eef_pos + 0.6 * reward_eef_rot
            + 0.9 * reward_thumb_tip_pos + 0.8 * reward_index_tip_pos
            + 0.75 * reward_middle_tip_pos + 0.6 * reward_pinky_tip_pos
            + 0.6 * reward_ring_tip_pos + 0.5 * reward_level_1_pos
            + 0.3 * reward_level_2_pos + 5.0 * reward_obj_pos
            + 1.0 * reward_obj_rot + 0.1 * reward_eef_vel
            + 0.05 * reward_eef_ang_vel + 0.1 * reward_joints_vel
            + 0.1 * reward_obj_vel + 0.1 * reward_obj_ang_vel
        )

        succeeded = (self.progress_buf + 1 + 3 >= self.max_episode_length) & ~failed_execute
        dones = (succeeded | failed_execute).float()

        self.success_buf_ = succeeded
        self.failure_buf_ = failed_execute
        self.error_buf_ = error_buf
        self.total_rewards += reward

        # infos dict -- lib/rl/base.py's play_steps reads these exact keys
        # (real PhysGraph code, confirmed by tracing what it accesses):
        # diff_metrics (per-frame tracking-error diagnostics), reward_dict
        # (individual reward term breakdown), time_outs (episode ended by
        # reaching the length limit, for PPO value-bootstrapping -- same
        # condition as `succeeded` here, since our fixed-length task makes
        # them coincide exactly), total_rewards/total_steps (cumulative
        # episode trackers), error_masks (same as error_buf).
        # diff_ft (fingertip-force tracking) is a zero placeholder -- the
        # real contact-force reward/diff is a deferred future piece (see
        # project notes), not yet computed.
        infos = {
            "diff_metrics": {
                "diff_obj_pos": diff_obj_pos_dist,
                "diff_obj_rot": diff_obj_rot_angle,
                "diff_joints": diff_joints_pos_dist.mean(dim=-1),
                "diff_ft": torch.zeros_like(diff_obj_pos_dist),
            },
            "reward_dict": {
                "reward_eef_pos": reward_eef_pos,
                "reward_eef_rot": reward_eef_rot,
                "reward_thumb_tip_pos": reward_thumb_tip_pos,
                "reward_index_tip_pos": reward_index_tip_pos,
                "reward_middle_tip_pos": reward_middle_tip_pos,
                "reward_pinky_tip_pos": reward_pinky_tip_pos,
                "reward_ring_tip_pos": reward_ring_tip_pos,
                "reward_level_1_pos": reward_level_1_pos,
                "reward_level_2_pos": reward_level_2_pos,
                "reward_obj_pos": reward_obj_pos,
                "reward_obj_rot": reward_obj_rot,
                "reward_eef_vel": reward_eef_vel,
                "reward_eef_ang_vel": reward_eef_ang_vel,
                "reward_joints_vel": reward_joints_vel,
                "reward_obj_vel": reward_obj_vel,
                "reward_obj_ang_vel": reward_obj_ang_vel,
            },
            "time_outs": succeeded,
            "total_rewards": self.total_rewards,
            "total_steps": self.progress_buf,
            "error_masks": error_buf,
        }

        return reward, dones, infos

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

        # muscle activations must be in [0,1] (real ctrlrange) -- map from the
        # policy's standard [-1,1] action range
        muscle_activations = (action[:, 9:] + 1.0) / 2.0

        qfrc = wp.to_torch(self.data.qfrc_applied)
        qfrc.zero_()
        qfrc[:, self.root_dof_adr:self.root_dof_adr + 3] = wrist_force
        qfrc[:, self.root_dof_adr + 3:self.root_dof_adr + 6] = wrist_torque

        ctrl = wp.to_torch(self.data.ctrl)
        ctrl[:, :] = muscle_activations

        mjw.step(self.model, self.data)

        self.progress_buf += 1
        rewards, dones, infos = self._compute_reward()
        obs = self._compute_obs()
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
