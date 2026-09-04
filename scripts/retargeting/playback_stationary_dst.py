"""Right hand (retargeted) + real source mug trajectory + STATIONARY
destination mug (held fixed at the position/orientation computed from
the real demo's pour-moment XY + frame-0 Z/rotation) -- visual check
for the reduced right-hand-only task (Runfa's direction: skip left
hand, both mugs randomized per RL episode, source mug retargeted
motion + destination mug static as a receiving target)."""
import pickle
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

proj_root = Path(".")
base = mujoco.MjSpec.from_file(str(proj_root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
mug_src = mujoco.MjSpec.from_file(str(proj_root / "assets/objects/O02@0015@00020/O02@0015@00020.xml"))
mug_dst = mujoco.MjSpec.from_file(str(proj_root / "assets/objects/O02@0010@00003/O02@0010@00003.xml"))

for spec, prefix in [(mug_src, "src_"), (mug_dst, "dst_")]:
    s = base.worldbody.add_site(name=f"attach_{prefix}", pos=[0, 0, 0])
    base.attach(spec, prefix=prefix, site=s)
    base.delete(s)

model = base.compile()
data = mujoco.MjData(model)

with open(proj_root / "assets/retargeted/1292e_rh_myohand.pkl", "rb") as f:
    retgt_r = pickle.load(f)
with open("/tmp/1292e_demo_data.pkl", "rb") as f:
    raw_r = pickle.load(f)
with open("/tmp/1292e_demo_data_lh.pkl", "rb") as f:
    raw_l = pickle.load(f)

C = axis_correction_matrix()

wrist_pos = (C @ retgt_r["opt_wrist_pos"].T).T
wrist_quat = rotmat_to_quat_np(C[None] @ aa_to_rotmat_np(retgt_r["opt_wrist_rot"]))

src_traj = raw_r["obj_trajectory"].numpy() if hasattr(raw_r["obj_trajectory"], "numpy") else raw_r["obj_trajectory"].cpu().numpy()
src_pos = (C @ src_traj[:, :3, 3].T).T
src_quat = rotmat_to_quat_np(C[None] @ src_traj[:, :3, :3])

dst_traj = raw_l["obj_trajectory"].numpy() if hasattr(raw_l["obj_trajectory"], "numpy") else raw_l["obj_trajectory"].cpu().numpy()
pour_frame = 373  # CORRECTED max-tilt frame (axis-correction-aware),
print(f"pour moment identified at frame {pour_frame}")

# EXPERIMENT: freeze at the max-tilt frame's FULL position+orientation,
# no mixing with frame 0 at all
# correct EACH frame'''s full position SEPARATELY first, then mix in the
# CORRECTED frame (C is a permutation-style rotation, not per-axis -- mixing
# raw components before correcting silently scrambles which real axis each
# value lands in, a real bug we found and fixed earlier)
dst_pos_pour_corrected = C @ dst_traj[pour_frame, :3, 3]
dst_pos_0_corrected = C @ dst_traj[0, :3, 3]
dst_target_pos = np.array([dst_pos_pour_corrected[0], dst_pos_pour_corrected[1], dst_pos_0_corrected[2]])
dst_target_rotmat_raw = dst_traj[0, :3, :3]
dst_target_quat = rotmat_to_quat_np((C @ dst_target_rotmat_raw)[None])[0]
print("stationary destination mug target position (corrected frame):", dst_target_pos)

dh_root_r = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")
root_r_adr = model.jnt_qposadr[dh_root_r]
import sys
sys.path.insert(0, "scripts/retargeting")
from myohand_def import MyoHandR
dh = MyoHandR()
dof_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in dh.dof_names])

src_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")]
dst_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dst_O02@0010@00003_free")]

n_frames = min(wrist_pos.shape[0], src_pos.shape[0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    for g in range(6):
        viewer.opt.geomgroup[g] = 1
    center = np.mean([wrist_pos.mean(axis=0), src_pos.mean(axis=0), dst_target_pos], axis=0)
    viewer.cam.lookat[:] = center
    viewer.cam.distance = 0.6
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    frame = 0
    while viewer.is_running():
        data.qpos[root_r_adr:root_r_adr + 3] = wrist_pos[frame]
        data.qpos[root_r_adr + 3:root_r_adr + 7] = wrist_quat[frame]
        data.qpos[dof_adrs] = retgt_r["opt_dof_pos"][frame]

        data.qpos[src_adr:src_adr + 3] = src_pos[frame]
        data.qpos[src_adr + 3:src_adr + 7] = src_quat[frame]

        # destination mug: STATIONARY, same target every frame
        data.qpos[dst_adr:dst_adr + 3] = dst_target_pos
        data.qpos[dst_adr + 3:dst_adr + 7] = dst_target_quat

        mujoco.mj_forward(model, data)
        viewer.sync()
        frame = (frame + 1) % n_frames
        import time
        time.sleep(1.0 / 60.0)
