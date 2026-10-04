"""Diagnostic (no viewer, no physics): measure the per-frame gap between
the RL reward's current fingertip target (raw MANO mano_joints, human
embodiment) and MyoHand's OWN retargeted fingertip position (computed
via real MuJoCo FK on the training asset, from opt_wrist_pos/opt_wrist_rot/
opt_dof_pos -- the same values playback_retargeted.py teleports into qpos).

Both quantities are read from assets/retargeted/1292e_training_demo.pkl's
rh dict so frame indexing is guaranteed aligned (no cross-file pkl
mismatch risk)."""
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandR

TIP_NAME_TO_BODY = {
    "thumb_tip": "THtip_r",
    "index_tip": "IFtip_r",
    "middle_tip": "MFtip_r",
    "ring_tip": "RFtip_r",
    "pinky_tip": "LFtip_r",
}


def aa_to_quat_np(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta = np.clip(theta, 1e-8, None)
    axis = aa / theta
    half = theta / 2
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)  # (N, 4) wxyz


def main():
    proj_root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(proj_root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
    data = mujoco.MjData(model)

    with open(proj_root / "assets/retargeted/1292e_training_demo.pkl", "rb") as f:
        demo = pickle.load(f)
    rh = demo["rh"]

    opt_wrist_pos = rh["opt_wrist_pos"].cpu().numpy()
    opt_wrist_rot = rh["opt_wrist_rot"].cpu().numpy()
    opt_dof_pos = rh["opt_dof_pos"].cpu().numpy()
    mano_joints = rh["mano_joints"]  # dict of (N,3) torch tensors
    tips_distance = rh["tips_distance"].cpu().numpy()  # (N, 5), chamfer-to-mug, for reference

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
            raise ValueError(f"joint {name} not found in training asset")
        dof_qpos_adrs.append(model.jnt_qposadr[jid])
    dof_qpos_adrs = np.array(dof_qpos_adrs)

    tip_site_ids = {}
    for tip_name, site_name in TIP_NAME_TO_BODY.items():
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if sid < 0:
            raise ValueError(f"site {site_name} not found in training asset")
        tip_site_ids[tip_name] = sid

    per_tip_dist = {tip_name: np.zeros(n_frames) for tip_name in TIP_NAME_TO_BODY}

    for frame in range(n_frames):
        data.qpos[root_qpos_adr:root_qpos_adr + 3] = opt_wrist_pos[frame]
        data.qpos[root_qpos_adr + 3:root_qpos_adr + 7] = opt_wrist_quat[frame]
        data.qpos[dof_qpos_adrs] = opt_dof_pos[frame]
        mujoco.mj_forward(model, data)

        for tip_name, sid in tip_site_ids.items():
            myo_tip_pos = data.site_xpos[sid].copy()
            mano_tip_pos = mano_joints[tip_name][frame].cpu().numpy()
            per_tip_dist[tip_name][frame] = np.linalg.norm(myo_tip_pos - mano_tip_pos)

    print()
    print(f"{'tip':14s}  {'mean (m)':>10s}  {'max (m)':>10s}  {'std (m)':>10s}")
    for tip_name, dists in per_tip_dist.items():
        print(f"{tip_name:14s}  {dists.mean():10.4f}  {dists.max():10.4f}  {dists.std():10.4f}")

    all_dists = np.stack(list(per_tip_dist.values()))
    print()
    print(f"overall mean: {all_dists.mean():.4f} m, overall max: {all_dists.max():.4f} m")
    print()
    print("for reference, mug radius ~0.0526 m, mug half-height ~0.0632 m")
    print(f"mean tips_distance (MyoHand-to-mug chamfer, from retargeting's own loss): {tips_distance.mean(axis=0)}")


if __name__ == "__main__":
    main()
