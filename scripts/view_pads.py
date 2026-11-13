import argparse, json, time
import numpy as np, mujoco, mujoco.viewer
import envs.mjwarp_myohand_env as E
from envs.mjwarp_myohand_env import MyoHandPourEnv

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=200)
ap.add_argument("--speed", type=float, default=0.5)
ap.add_argument("--frame", type=int, default=-1, help="hold on a single frame")
ap.add_argument("--contacts", action="store_true")
ap.add_argument("--tips", action="store_true")
ap.add_argument("--json", default="assets/retargeted/pad_sites_local.json")
a = ap.parse_args()

np_ = lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
env = MyoHandPourEnv(num_envs=1, device="cuda:0")
m = env.mj_model
d = mujoco.MjData(m)
pads = json.load(open(a.json))["bodies"]
T = np_(env.demo_opt_dof_pos).shape[0]
wp_, wq_ = np_(env.demo_opt_wrist_pos), np_(env.demo_opt_wrist_quat)
dof = np_(env.demo_opt_dof_pos)
src_pos = np_(env.demo_src_obj_traj)[:, :3, 3]
src_rot = np_(env.demo_src_obj_traj)[:, :3, :3]
dst_p = np_(env.lh_target_pos).reshape(-1)[:3]
dst_q = np_(env.lh_target_quat).reshape(-1)[:4]
dof_adrs = np_(env.dof_adrs).astype(int)
tip_of = {bn: v["tip_site"] for bn, v in pads.items()}

def set_frame(t):
    t = min(t, T - 1)
    d.qpos[:] = m.qpos0
    d.qpos[env.root_adr:env.root_adr + 3] = wp_[t]
    d.qpos[env.root_adr + 3:env.root_adr + 7] = wq_[t]
    d.qpos[dof_adrs] = dof[t]
    d.qpos[env.src_adr:env.src_adr + 3] = src_pos[t]
    q = E.rotmat_to_quat(E.torch.tensor(src_rot[t][None], dtype=E.torch.float32))
    d.qpos[env.src_adr + 3:env.src_adr + 7] = np_(q).reshape(-1)[:4]
    d.qpos[env.dst_adr:env.dst_adr + 3] = dst_p
    d.qpos[env.dst_adr + 3:env.dst_adr + 7] = dst_q
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)

def pts():
    out = []
    for bn, v in pads.items():
        b = m.body(bn).id
        R = d.xmat[b].reshape(3, 3); p = d.xpos[b]
        out.append((p + R @ np.array(v["pad_local_m"]), 0.004, [0.1, 0.3, 1.0, 1.0]))
        if a.contacts:
            out.append((p + R @ np.array(v["demo_contact_local_m"]), 0.0025, [1.0, 0.55, 0.0, 1.0]))
        if a.tips:
            out.append((d.site_xpos[m.site(tip_of[bn]).id].copy(), 0.0025, [1.0, 0.1, 0.1, 1.0]))
    return out

def draw(v):
    scn = v.user_scn
    scn.ngeom = 0
    for p, r, rgba in pts():
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]), p, np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        scn.ngeom += 1

with mujoco.viewer.launch_passive(m, d) as v:
    v.opt.geomgroup[:] = 1
    set_frame(a.start if a.frame < 0 else a.frame)
    v.cam.lookat[:] = d.xpos[m.body("distal_thumb_r").id]
    v.cam.distance = 0.35
    while v.is_running():
        frames = [a.frame] if a.frame >= 0 else list(range(a.start, min(a.end, T)))
        for t in frames:
            if not v.is_running():
                break
            set_frame(t)
            with v.lock():
                draw(v)
            v.sync()
            time.sleep((1 / 60) / a.speed)
