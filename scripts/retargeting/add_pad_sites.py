"""Add pad-site fields to a post-processed demo (no re-retargeting).

Pad sites are the analytic palmar-skin points in assets/retargeted/pad_sites_local.json
(distal-body frame). Same FK convention as compute_myo_joints.py: MuJoCo FK on the hand
asset driven by opt_wrist_pos/opt_wrist_rot/opt_dof_pos, RAW/uncorrected frame, velocity by
np.gradient at 1/60. Distance to the mug follows recompute_tips_distance.py (nearest mug vertex).
New rh keys: pad_joints, pad_joints_velocity (dicts of (T,3)), pad_tips_distance (T,5), pad_contact (T,5).
"""
import argparse, copy, json, pickle, sys
from pathlib import Path
import mujoco
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandR

ap = argparse.ArgumentParser()
ap.add_argument("--demo", default="assets/retargeted/1292e_training_demo_smallmug_v2.pkl")
ap.add_argument("--out", default="assets/retargeted/1292e_training_demo_smallmug_v2pad.pkl")
ap.add_argument("--pads", default="assets/retargeted/pad_sites_local.json")
ap.add_argument("--scale_xy", type=float, default=0.879)
ap.add_argument("--scale_z", type=float, default=1.0)
ap.add_argument("--contact_mm", type=float, default=3.0)
args = ap.parse_args()

DEMO_DT = 1.0 / 60.0
FING = ["thumb", "index", "middle", "ring", "pinky"]
BODIES = ["distal_thumb_r", "distph2_r", "distph3_r", "distph4_r", "distph5_r"]
TIPS = ["THtip_r", "IFtip_r", "MFtip_r", "RFtip_r", "LFtip_r"]


def N(x):
    return x.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(x) else np.asarray(x, dtype=np.float64)


def aa_to_quat_np(aa):
    theta = np.clip(np.linalg.norm(aa, axis=-1, keepdims=True), 1e-8, None)
    half = theta / 2
    return np.concatenate([np.cos(half), aa / theta * np.sin(half)], axis=-1)


root_dir = Path(__file__).resolve().parents[2]
model = mujoco.MjModel.from_xml_path(str(root_dir / "assets/hands/myohand_r_ulnaroot_scene.xml"))
data = mujoco.MjData(model)
demo = pickle.load(open(args.demo, "rb"))
rh = demo["rh"]
pads = json.load(open(args.pads))["bodies"]
P, Q, D = N(rh["opt_wrist_pos"]), aa_to_quat_np(N(rh["opt_wrist_rot"])), N(rh["opt_dof_pos"])
T = P.shape[0]
dexhand = MyoHandR()
radr = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")]
dof_adrs = np.array([model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in dexhand.dof_names])
bids = [model.body(b).id for b in BODIES]
sids = [model.site(s).id for s in TIPS]
loc = np.array([pads[b]["pad_local_m"] for b in BODIES])

pad = np.zeros((T, 5, 3)); tip = np.zeros((T, 5, 3))
for t in range(T):
    data.qpos[radr:radr + 3] = P[t]; data.qpos[radr + 3:radr + 7] = Q[t]; data.qpos[dof_adrs] = D[t]
    mujoco.mj_forward(model, data)
    for i in range(5):
        pad[t, i] = data.xpos[bids[i]] + data.xmat[bids[i]].reshape(3, 3) @ loc[i]
        tip[t, i] = data.site_xpos[sids[i]]
vel = np.gradient(pad, DEMO_DT, axis=0)

mj = rh["myo_joints"]
stored = np.stack([N(mj[f + "_tip"]) for f in FING], axis=1)
print("PADFIX FK check vs stored myo_joints tips, max abs diff per finger (mm):", np.round(np.abs(tip - stored).max(axis=(0, 2)) * 1000, 3).tolist())

verts = N(rh["obj_verts"]) * np.array([args.scale_xy, args.scale_xy, args.scale_z])
traj = N(rh["obj_trajectory"])
spacing = cKDTree(verts).query(verts, k=2)[0][:, 1].mean() * 1000
print(f"PADFIX mug vertex spacing mean {spacing:.1f} mm (nearest-vertex distance overestimates surface distance by about half)")
dist = np.zeros((T, 5)); tdist = np.zeros((T, 5))
for t in range(T):
    w = verts @ traj[t, :3, :3].T + traj[t, :3, 3]
    tree = cKDTree(w)
    dist[t] = tree.query(pad[t])[0]
    tdist[t] = tree.query(tip[t])[0]
old_td = N(rh["tips_distance"])
print("PADFIX tips_distance stored vs recomputed, max abs diff (mm):", np.round(np.abs(tdist - old_td).max(0) * 1000, 2).tolist())
sl = slice(40, 140)
for i, f in enumerate(FING):
    d_ = dist[sl, i] * 1000
    print(f"PADFIX {f:7s} pad gap frames 40-140 (mm): min {d_.min():.1f} median {np.median(d_):.1f} | frac <2: {(d_ < 2).mean():.2f} <4: {(d_ < 4).mean():.2f} <6: {(d_ < 6).mean():.2f} <10: {(d_ < 10).mean():.2f} | tip-site gap median {np.median(tdist[sl, i]) * 1000:.1f}")

out = copy.deepcopy(demo)
o = out["rh"]
keys = [f + "_pad" for f in FING]
o["pad_joints"] = {k: torch.from_numpy(pad[:, i].astype(np.float32).copy()) for i, k in enumerate(keys)}
o["pad_joints_velocity"] = {k: torch.from_numpy(vel[:, i].astype(np.float32).copy()) for i, k in enumerate(keys)}
o["pad_tips_distance"] = torch.from_numpy(dist.astype(np.float32))
o["pad_contact"] = torch.from_numpy(dist < args.contact_mm / 1000.0)
pickle.dump(out, open(args.out, "wb"))
print("PADFIX wrote", args.out)
