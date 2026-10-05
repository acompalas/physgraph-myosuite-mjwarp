"""Kinematic playback (physics OFF) of BOTH hands + BOTH mugs together,
using raw/retargeted data straight from the OakInk2 demo pkls with NO
manual position offsets -- camera auto-centers on the actual data
cluster rather than assuming proximity to the scene's own pedestal."""
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandR, MyoHandL


def aa_to_rotmat_np(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta_c = np.clip(theta, 1e-8, None)
    axis = aa / theta_c
    x, y, z = axis[...,0], axis[...,1], axis[...,2]
    zero = np.zeros_like(x)
    K = np.stack([np.stack([zero,-z,y],-1), np.stack([z,zero,-x],-1), np.stack([-y,x,zero],-1)], -2)
    I = np.eye(3)[None].repeat(aa.shape[0], axis=0)
    th = theta[...,None]
    return I + np.sin(th)*K + (1-np.cos(th))*(K@K)


def aa_to_quat_np(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta = np.clip(theta, 1e-8, None)
    axis = aa / theta
    half = theta / 2
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)


def axis_correction_matrix():
    # matches fitting()'s own mujoco2gym_transf rotation: Rz(-90) @ Rx(90).
    # We originally skipped this assuming raw data might already be
    # Z-up-native -- wrong assumption; both IsaacGym and MuJoCo are
    # Z-up, so this raw-capture-axis correction is needed regardless
    # of target engine.
    rz = np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=np.float64)   # Rz(-90)
    rx = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)   # Rx(90)
    return rz @ rx


def rotmat_to_quat_np(R):
    tr = np.trace(R, axis1=-2, axis2=-1)
    q = np.zeros(R.shape[:-2] + (4,))
    q[..., 0] = np.sqrt(np.maximum(0, tr + 1)) / 2
    q[..., 1] = np.sign(R[..., 2, 1] - R[..., 1, 2]) * np.sqrt(np.maximum(0, 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2
    q[..., 2] = np.sign(R[..., 0, 2] - R[..., 2, 0]) * np.sqrt(np.maximum(0, 1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2
    q[..., 3] = np.sign(R[..., 1, 0] - R[..., 0, 1]) * np.sqrt(np.maximum(0, 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2
    return q


def main():
    proj_root = Path(__file__).resolve().parents[2]

    base = mujoco.MjSpec.from_file(str(proj_root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
    left = mujoco.MjSpec.from_file(str(proj_root / "assets/hands/myohand_l_ulnaroot.xml"))
    mug_src = mujoco.MjSpec.from_file(str(proj_root / "assets/objects/O02@0015@00020/O02@0015@00020.xml"))
    mug_dst = mujoco.MjSpec.from_file(str(proj_root / "assets/objects/O02@0010@00003/O02@0010@00003.xml"))

    for spec, prefix in [(left, "l2_"), (mug_src, "src_"), (mug_dst, "dst_")]:
        s = base.worldbody.add_site(name=f"attach_{prefix}", pos=[0, 0, 0])
        base.attach(spec, prefix=prefix, site=s)
        base.delete(s)

    model = base.compile()
    data = mujoco.MjData(model)

    with open(proj_root / "assets/retargeted/1292e_rh_myohand.pkl", "rb") as f:
        retgt_r = pickle.load(f)
    with open(proj_root / "assets/retargeted/1292e_lh_myohand.pkl", "rb") as f:
        retgt_l = pickle.load(f)
    with open("/tmp/1292e_demo_data.pkl", "rb") as f:
        raw_r = pickle.load(f)      # obj_trajectory here = SOURCE mug (rh manipulates it)
    with open("/tmp/1292e_demo_data_lh.pkl", "rb") as f:
        raw_l = pickle.load(f)      # obj_trajectory here = DEST mug (lh manipulates it)

    n_frames = min(
        retgt_r["opt_wrist_pos"].shape[0], retgt_l["opt_wrist_pos"].shape[0],
        raw_r["obj_trajectory"].shape[0], raw_l["obj_trajectory"].shape[0],
    )
    print(f"n_frames: {n_frames}")

    C = axis_correction_matrix()

    wrist_pos_r = (C @ retgt_r["opt_wrist_pos"].T).T
    wrist_pos_l = (C @ retgt_l["opt_wrist_pos"].T).T
    quat_r = rotmat_to_quat_np(C[None] @ aa_to_rotmat_np(retgt_r["opt_wrist_rot"]))
    quat_l = rotmat_to_quat_np(C[None] @ aa_to_rotmat_np(retgt_l["opt_wrist_rot"]))

    src_traj = raw_r["obj_trajectory"].numpy()
    dst_traj = raw_l["obj_trajectory"].numpy()
    src_pos = (C @ src_traj[:, :3, 3].T).T
    dst_pos = (C @ dst_traj[:, :3, 3].T).T
    src_quat = rotmat_to_quat_np(C[None] @ src_traj[:, :3, :3])
    dst_quat = rotmat_to_quat_np(C[None] @ dst_traj[:, :3, :3])

    dh_r, dh_l = MyoHandR(), MyoHandL()
    root_r_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")]
    root_l_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l2_root_l")]
    dof_r_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in dh_r.dof_names])
    dof_l_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"l2_{n}")] for n in dh_l.dof_names])

    src_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")
    dst_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dst_O02@0010@00003_free")
    src_adr = model.jnt_qposadr[src_jid]
    dst_adr = model.jnt_qposadr[dst_jid]

    # camera: center on the actual data cluster (mean of all positions
    # across the trajectory), not an assumed pedestal-relative spot
    all_pos = np.concatenate([
        wrist_pos_r[:n_frames], wrist_pos_l[:n_frames],
        src_pos[:n_frames], dst_pos[:n_frames],
    ], axis=0)
    center = all_pos.mean(axis=0)
    print("scene center:", center)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for g in range(6):
            viewer.opt.geomgroup[g] = 1
        viewer.cam.lookat[:] = center
        viewer.cam.distance = 0.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        frame = 0
        while viewer.is_running():
            data.qpos[root_r_adr:root_r_adr + 3] = wrist_pos_r[frame]
            data.qpos[root_r_adr + 3:root_r_adr + 7] = quat_r[frame]
            data.qpos[dof_r_adrs] = retgt_r["opt_dof_pos"][frame]

            data.qpos[root_l_adr:root_l_adr + 3] = wrist_pos_l[frame]
            data.qpos[root_l_adr + 3:root_l_adr + 7] = quat_l[frame]
            data.qpos[dof_l_adrs] = retgt_l["opt_dof_pos"][frame]

            data.qpos[src_adr:src_adr + 3] = src_pos[frame]
            data.qpos[src_adr + 3:src_adr + 7] = src_quat[frame]
            data.qpos[dst_adr:dst_adr + 3] = dst_pos[frame]
            data.qpos[dst_adr + 3:dst_adr + 7] = dst_quat[frame]

            mujoco.mj_forward(model, data)
            viewer.sync()

            frame = (frame + 1) % n_frames
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
