"""Precompute MyoHand's OWN retargeted per-joint positions (all 20
non-wrist body_names entries, not just the 5 fingertips), via real
MuJoCo FK on the training asset driven by opt_wrist_pos/opt_wrist_rot/
opt_dof_pos (the actual retargeted trajectory). Stored RAW/uncorrected,
same convention as mano_joints, so envs/mjwarp_myohand_env.py's
existing self._axis_C correction logic applies unchanged after we
swap the env to read these instead of mano_joints.

Also computes myo_joints_velocity via np.gradient over the frame axis
at the demo's control cadence (1/60), matching PhysGraph's own
compute_velocity() convention (np.gradient on the raw position array)."""
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandR

DEMO_DT = 1.0 / 60.0


def aa_to_quat_np(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta = np.clip(theta, 1e-8, None)
    axis = aa / theta
    half = theta / 2
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)


def main():
    proj_root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(proj_root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
    data = mujoco.MjData(model)

    demo_path = proj_root / "assets/retargeted/1292e_training_demo.pkl"
    with open(demo_path, "rb") as f:
        demo = pickle.load(f)
    rh = demo["rh"]

    opt_wrist_pos = rh["opt_wrist_pos"].cpu().numpy()
    opt_wrist_rot = rh["opt_wrist_rot"].cpu().numpy()
    opt_dof_pos = rh["opt_dof_pos"].cpu().numpy()
    n_frames = opt_wrist_pos.shape[0]
    print(f"loaded {n_frames} frames")
    opt_wrist_quat = aa_to_quat_np(opt_wrist_rot)

    dexhand = MyoHandR()
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")
    root_qpos_adr = model.jnt_qposadr[root_joint_id]

    dof_qpos_adrs = []
    for name in dexhand.dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name} not found")
        dof_qpos_adrs.append(model.jnt_qposadr[jid])
    dof_qpos_adrs = np.array(dof_qpos_adrs)

    # body_names[1:] excludes the wrist (index 0, tracked separately via
    # opt_wrist_pos directly) -- these 20 are what env.py currently pulls
    # from mano_joints via dexhand.to_hand(b)[0]
    target_body_names = dexhand.body_names[1:]
    assert len(target_body_names) == 20, len(target_body_names)

    name_to_objtype_id = {}
    for name in target_body_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            name_to_objtype_id[name] = ("body", bid)
            continue
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid >= 0:
            name_to_objtype_id[name] = ("site", sid)
            continue
        raise ValueError(f"{name} not found as body or site in training asset")

    hand_keys = [dexhand.to_hand(b)[0] for b in target_body_names]
    print("body_name -> hand_key mapping:")
    for b, hk in zip(target_body_names, hand_keys):
        kind, _ = name_to_objtype_id[b]
        print(f"  {b:18s} ({kind:4s}) -> {hk}")

    # (n_frames, 20, 3)
    positions = np.zeros((n_frames, len(target_body_names), 3), dtype=np.float32)

    for frame in range(n_frames):
        data.qpos[root_qpos_adr:root_qpos_adr + 3] = opt_wrist_pos[frame]
        data.qpos[root_qpos_adr + 3:root_qpos_adr + 7] = opt_wrist_quat[frame]
        data.qpos[dof_qpos_adrs] = opt_dof_pos[frame]
        mujoco.mj_forward(model, data)

        for i, name in enumerate(target_body_names):
            kind, oid = name_to_objtype_id[name]
            positions[frame, i] = data.xpos[oid] if kind == "body" else data.site_xpos[oid]

    velocities = np.gradient(positions, DEMO_DT, axis=0).astype(np.float32)

    myo_joints = {}
    myo_joints_velocity = {}
    for i, hk in enumerate(hand_keys):
        myo_joints[hk] = torch.from_numpy(positions[:, i, :].copy())
        myo_joints_velocity[hk] = torch.from_numpy(velocities[:, i, :].copy())

    rh["myo_joints"] = myo_joints
    rh["myo_joints_velocity"] = myo_joints_velocity

    with open(demo_path, "wb") as f:
        pickle.dump(demo, f)

    print(f"\nwrote myo_joints / myo_joints_velocity ({len(myo_joints)} keys each) into {demo_path}")

    # quick sanity: compare against mano_joints to reproduce the diagnostic numbers
    mano_joints = rh["mano_joints"]
    for tip in ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]:
        d = (myo_joints[tip] - mano_joints[tip]).norm(dim=-1)
        print(f"  sanity {tip:12s} mean={d.mean():.4f} max={d.max():.4f}")


if __name__ == "__main__":
    main()
