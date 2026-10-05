"""Render frame 0 (initial pose) of both retargeted MyoHand trajectories
+ both mugs, offscreen, saved directly to PNG files -- for a static,
apples-to-apples visual comparison against the ground-truth SMPL-X
frame-0 render."""
import pickle
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandR, MyoHandL

def aa_to_quat_np(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta = np.clip(theta, 1e-8, None)
    axis = aa / theta
    half = theta / 2
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)

def rotmat_to_quat_np(R):
    tr = np.trace(R, axis1=-2, axis2=-1)
    q = np.zeros(R.shape[:-2] + (4,))
    q[..., 0] = np.sqrt(np.maximum(0, tr + 1)) / 2
    q[..., 1] = np.sign(R[..., 2, 1] - R[..., 1, 2]) * np.sqrt(np.maximum(0, 1 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2
    q[..., 2] = np.sign(R[..., 0, 2] - R[..., 2, 0]) * np.sqrt(np.maximum(0, 1 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2
    q[..., 3] = np.sign(R[..., 1, 0] - R[..., 0, 1]) * np.sqrt(np.maximum(0, 1 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2
    return q

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

def axis_correction_matrix():
    rz = np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=np.float64)
    rx = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
    return rz @ rx

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
    raw_r = pickle.load(f)
with open("/tmp/1292e_demo_data_lh.pkl", "rb") as f:
    raw_l = pickle.load(f)

C = axis_correction_matrix()
wrist_pos_r = (C @ retgt_r["opt_wrist_pos"][:1].T).T
wrist_pos_l = (C @ retgt_l["opt_wrist_pos"][:1].T).T
quat_r = rotmat_to_quat_np(C[None] @ aa_to_rotmat_np(retgt_r["opt_wrist_rot"][:1]))
quat_l = rotmat_to_quat_np(C[None] @ aa_to_rotmat_np(retgt_l["opt_wrist_rot"][:1]))

src_traj = raw_r["obj_trajectory"].numpy()[:1]
dst_traj = raw_l["obj_trajectory"].numpy()[:1]
src_pos = (C @ src_traj[:, :3, 3].T).T
dst_pos = (C @ dst_traj[:, :3, 3].T).T
src_quat = rotmat_to_quat_np(C[None] @ src_traj[:, :3, :3])
dst_quat = rotmat_to_quat_np(C[None] @ dst_traj[:, :3, :3])

dh_r, dh_l = MyoHandR(), MyoHandL()
root_r_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")]
root_l_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l2_root_l")]
dof_r_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in dh_r.dof_names])
dof_l_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"l2_{n}")] for n in dh_l.dof_names])
src_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")]
dst_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dst_O02@0010@00003_free")]

data.qpos[root_r_adr:root_r_adr + 3] = wrist_pos_r[0]
data.qpos[root_r_adr + 3:root_r_adr + 7] = quat_r[0]
data.qpos[dof_r_adrs] = retgt_r["opt_dof_pos"][0]
data.qpos[root_l_adr:root_l_adr + 3] = wrist_pos_l[0]
data.qpos[root_l_adr + 3:root_l_adr + 7] = quat_l[0]
data.qpos[dof_l_adrs] = retgt_l["opt_dof_pos"][0]
data.qpos[src_adr:src_adr + 3] = src_pos[0]
data.qpos[src_adr + 3:src_adr + 7] = src_quat[0]
data.qpos[dst_adr:dst_adr + 3] = dst_pos[0]
data.qpos[dst_adr + 3:dst_adr + 7] = dst_quat[0]

mujoco.mj_forward(model, data)

center = np.mean([wrist_pos_r[0], wrist_pos_l[0], src_pos[0], dst_pos[0]], axis=0)

import mujoco.viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    for g in range(6):
        viewer.opt.geomgroup[g] = 1
    viewer.cam.lookat[:] = center
    viewer.cam.distance = 0.6
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    while viewer.is_running():
        viewer.sync()
