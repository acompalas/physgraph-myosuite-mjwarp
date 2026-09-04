"""JUST the two real mug trajectories, no hands at all -- isolated
sanity check that they're correctly oriented relative to each other
(no reflection/mirroring bug), independent of any hand retargeting."""
import pickle
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

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

# strip the hand actor entirely -- not needed for this check, and its
# default resting pose would just clutter the view
model = base.compile()
data = mujoco.MjData(model)

with open("/tmp/1292e_demo_data.pkl", "rb") as f:
    raw_r = pickle.load(f)
with open("/tmp/1292e_demo_data_lh.pkl", "rb") as f:
    raw_l = pickle.load(f)

C = axis_correction_matrix()

src_traj = raw_r["obj_trajectory"].numpy() if hasattr(raw_r["obj_trajectory"], "numpy") else raw_r["obj_trajectory"].cpu().numpy()
dst_traj = raw_l["obj_trajectory"].numpy() if hasattr(raw_l["obj_trajectory"], "numpy") else raw_l["obj_trajectory"].cpu().numpy()

src_pos = (C @ src_traj[:, :3, 3].T).T
src_quat = rotmat_to_quat_np(C[None] @ src_traj[:, :3, :3])
dst_pos = (C @ dst_traj[:, :3, 3].T).T
dst_quat = rotmat_to_quat_np(C[None] @ dst_traj[:, :3, :3])

src_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")]
dst_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dst_O02@0010@00003_free")]

# hide the (unposed, default rest pose) hand geoms so they don't clutter
# the view -- keep the scene/floor
root_r_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ulna_r")
for i in range(model.ngeom):
    if model.geom_bodyid[i] == root_r_body or model.body_weldid[model.geom_bodyid[i]] == root_r_body:
        model.geom_rgba[i][3] = 0.0  # fully transparent, cheap way to hide without restructuring

n_frames = min(src_pos.shape[0], dst_pos.shape[0])
print(f"n_frames: {n_frames}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    for g in range(6):
        viewer.opt.geomgroup[g] = 1
    center = np.mean([src_pos.mean(axis=0), dst_pos.mean(axis=0)], axis=0)
    viewer.cam.lookat[:] = center
    viewer.cam.distance = 0.5
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20

    frame = 0
    while viewer.is_running():
        data.qpos[src_adr:src_adr + 3] = src_pos[frame]
        data.qpos[src_adr + 3:src_adr + 7] = src_quat[frame]
        data.qpos[dst_adr:dst_adr + 3] = dst_pos[frame]
        data.qpos[dst_adr + 3:dst_adr + 7] = dst_quat[frame]

        mujoco.mj_forward(model, data)
        viewer.sync()
        frame = (frame + 1) % n_frames
        time.sleep(1.0 / 60.0)
