"""Static single-frame check: both mugs at THEIR OWN real captured pose
at one specific frame index, held still, so we can directly judge
whether that frame is a believable pour configuration -- no mixing,
no retargeting, just the raw real data at that instant."""
import pickle
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

FRAME_TO_CHECK = int(sys.argv[1]) if len(sys.argv) > 1 else 567

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

root_r_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ulna_r")
for i in range(model.ngeom):
    if model.geom_bodyid[i] == root_r_body or model.body_weldid[model.geom_bodyid[i]] == root_r_body:
        model.geom_rgba[i][3] = 0.0

with open("/tmp/1292e_demo_data.pkl", "rb") as f:
    raw_r = pickle.load(f)
with open("/tmp/1292e_demo_data_lh.pkl", "rb") as f:
    raw_l = pickle.load(f)

C = axis_correction_matrix()
src_traj = raw_r["obj_trajectory"].numpy() if hasattr(raw_r["obj_trajectory"], "numpy") else raw_r["obj_trajectory"].cpu().numpy()
dst_traj = raw_l["obj_trajectory"].numpy() if hasattr(raw_l["obj_trajectory"], "numpy") else raw_l["obj_trajectory"].cpu().numpy()

src_pos = C @ src_traj[FRAME_TO_CHECK, :3, 3]
src_quat = rotmat_to_quat_np((C @ src_traj[FRAME_TO_CHECK, :3, :3])[None])[0]
dst_pos = C @ dst_traj[FRAME_TO_CHECK, :3, 3]
dst_quat = rotmat_to_quat_np((C @ dst_traj[FRAME_TO_CHECK, :3, :3])[None])[0]

print(f"showing frame {FRAME_TO_CHECK}")
print(f"  source mug pos: {src_pos}")
print(f"  dest mug pos:   {dst_pos}")
print(f"  distance: {np.linalg.norm(src_pos - dst_pos)}")

src_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")]
dst_adr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dst_O02@0010@00003_free")]

data.qpos[src_adr:src_adr + 3] = src_pos
data.qpos[src_adr + 3:src_adr + 7] = src_quat
data.qpos[dst_adr:dst_adr + 3] = dst_pos
data.qpos[dst_adr + 3:dst_adr + 7] = dst_quat
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    for g in range(6):
        viewer.opt.geomgroup[g] = 1
    center = (src_pos + dst_pos) / 2
    viewer.cam.lookat[:] = center
    viewer.cam.distance = 0.4
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    while viewer.is_running():
        viewer.sync()
