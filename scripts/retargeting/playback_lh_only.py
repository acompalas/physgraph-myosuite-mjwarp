"""Right hand + source mug (O02@0010@00003, the one being poured)
only -- isolated view for judging grasp plausibility without the
other hand/mug present."""
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

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

def axis_correction_matrix():
    rz = np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=np.float64)
    rx = np.array([[1,0,0],[0,0,-1],[0,1,0]], dtype=np.float64)
    return rz @ rx

proj_root = Path(__file__).resolve().parents[2]
base = mujoco.MjSpec.from_file(str(proj_root / "assets/hands/myohand_l_ulnaroot_scene.xml"))
mug = mujoco.MjSpec.from_file(str(proj_root / "assets/objects/O02@0010@00003/O02@0010@00003.xml"))

site = base.worldbody.add_site(name="mug_attach", pos=[0, 0, 0])
base.attach(mug, prefix="mug_", site=site)
base.delete(site)

model = base.compile()
data = mujoco.MjData(model)

with open(proj_root / "assets/retargeted/1292e_lh_myohand.pkl", "rb") as f:
    retgt = pickle.load(f)
with open("/tmp/1292e_demo_data_lh.pkl", "rb") as f:
    demo = pickle.load(f)

C = axis_correction_matrix()
wrist_pos = (C @ retgt["opt_wrist_pos"].T).T
obj_traj = demo["obj_trajectory"].numpy()
obj_pos = (C @ obj_traj[:, :3, 3].T).T
obj_quat = rotmat_to_quat_np(C[None] @ obj_traj[:, :3, :3])

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandL

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

wrist_quat = rotmat_to_quat_np(C[None] @ aa_to_rotmat_np(retgt["opt_wrist_rot"]))

dh = MyoHandL()
root_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_l")]
dof_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in dh.dof_names])
mug_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mug_O02@0010@00003_free")
mug_adr = model.jnt_qposadr[mug_jid]

n_frames = min(wrist_pos.shape[0], obj_pos.shape[0])
print(f"n_frames: {n_frames}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for g in range(6):
        viewer.opt.geomgroup[g] = 1
    center = np.concatenate([wrist_pos, obj_pos], axis=0).mean(axis=0)
    viewer.cam.lookat[:] = center
    viewer.cam.distance = 0.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    frame = 0
    while viewer.is_running():
        data.qpos[root_adr:root_adr + 3] = wrist_pos[frame]
        data.qpos[root_adr + 3:root_adr + 7] = wrist_quat[frame]
        data.qpos[dof_adrs] = retgt["opt_dof_pos"][frame]
        data.qpos[mug_adr:mug_adr + 3] = obj_pos[frame]
        data.qpos[mug_adr + 3:mug_adr + 7] = obj_quat[frame]

        mujoco.mj_forward(model, data)
        viewer.sync()
        frame = (frame + 1) % n_frames
        time.sleep(1.0 / 60.0)
