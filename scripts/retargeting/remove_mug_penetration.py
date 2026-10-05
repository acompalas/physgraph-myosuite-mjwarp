"""Post-process the retargeted trajectory so no hand collision geom penetrates
the mug. Per frame (sequential, warm-started): least-squares correction of
wrist pose + 23 dofs, staying close to the original joint positions."""
import sys, copy, pickle, argparse, time
import numpy as np, torch, mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "."); sys.path.insert(0, "scripts/retargeting")
import envs.mjwarp_myohand_env as E
from myohand_def import MyoHandR

ap = argparse.ArgumentParser()
ap.add_argument("--start", type=int, default=0)
ap.add_argument("--end", type=int, default=None)
ap.add_argument("--write", action="store_true")
args = ap.parse_args()

DEMO = "assets/retargeted/1292e_training_demo.pkl"
OUT = "assets/retargeted/1292e_training_demo_nopen.pkl"
MARGIN, TOUCH = 0.0005, 0.002
dt = 1.0 / 60

with open(DEMO, "rb") as f:
    demo = pickle.load(f)

def find_rh(o, path="demo"):
    if isinstance(o, dict):
        if "opt_dof_pos" in o:
            return o, path
        for k, v in o.items():
            r = find_rh(v, path + "[%r]" % (k,))
            if r is not None:
                return r
    return None

rh, rh_path = find_rh(demo)
print("hand data found at", rh_path)

def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

env = E.MyoHandPourEnv(num_envs=1, device="cuda:0")
m = env.mj_model
d = mujoco.MjData(m)
C = env._axis_C.cpu().numpy().astype(np.float64)
dexhand = MyoHandR()
T = env.demo_opt_wrist_pos.shape[0]
wp0 = env.demo_opt_wrist_pos.cpu().numpy().astype(np.float64)
wq0 = env.demo_opt_wrist_quat.cpu().numpy().astype(np.float64)   # wxyz, axis-corrected
q0 = env.demo_opt_dof_pos.cpu().numpy().astype(np.float64)
mug_p = env.demo_src_obj_traj[:, :3, 3].cpu().numpy().astype(np.float64)
mug_q = E.rotmat_to_quat(env.demo_src_obj_traj[:, :3, :3]).cpu().numpy().astype(np.float64)
human = to_np(rh["tips_distance"]).astype(np.float64)             # (T,5)

# --- sanity: raw <-> corrected conversion must roundtrip on the ORIGINAL data
R_c0 = Rot.from_quat(wq0[:, [1, 2, 3, 0]]).as_matrix()
R_raw0 = np.matmul(C.T[None], R_c0)
e_rot = np.abs(Rot.from_rotvec(to_np(rh["opt_wrist_rot"]).astype(np.float64)).as_matrix() - R_raw0).max()
e_pos = np.abs((C.T @ wp0.T).T - to_np(rh["opt_wrist_pos"])).max()
print("roundtrip check: rot err %.2e, pos err %.2e" % (e_rot, e_pos))
assert e_rot < 1e-4 and e_pos < 1e-4, "axis-correction roundtrip failed; refusing to continue"

def is_col(g):
    return m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0

src_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "src_O02@0015@00020_free")])
root_body = int(m.jnt_bodyid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root_r")])
mug_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == src_body and is_col(g)]
hand_bodies = {}
for g in range(m.ngeom):
    b = int(m.geom_bodyid[g])
    if is_col(g) and m.body_rootid[b] == root_body:
        hand_bodies.setdefault(b, []).append(g)
distal_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in dexhand.contact_body_names]
print("mug geoms %d | hand bodies with collision %d | distal bodies %s" % (len(mug_geoms), len(hand_bodies), distal_ids))
assert all(b in hand_bodies for b in distal_ids)

dof_adrs = np.asarray(env.dof_adrs)
lo, hi = [], []
for adr in dof_adrs:
    jid = int(np.where(m.jnt_qposadr == adr)[0][0])
    l, h = (m.jnt_range[jid] if m.jnt_limited[jid] else (-np.inf, np.inf))
    lo.append(l); hi.append(h)
lo, hi = np.array(lo), np.array(hi)
blo = np.r_[[-0.05] * 3, [-0.3] * 3, lo]
bhi = np.r_[[0.05] * 3, [0.3] * 3, hi]
scale_smooth = np.r_[[0.005] * 3, [0.02] * 3, [0.05] * len(dof_adrs)]

jl = []
for b in dexhand.body_names[1:]:
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, b) if bid < 0 else -1
    assert bid >= 0 or sid >= 0, b
    jl.append((bid, sid))

def joints_xyz():
    return np.array([d.xpos[b] if b >= 0 else d.site_xpos[s] for b, s in jl])

def place(p, quat, q, t):
    d.qpos[:] = m.qpos0
    d.qpos[env.root_adr:env.root_adr + 3] = p
    d.qpos[env.root_adr + 3:env.root_adr + 7] = quat
    d.qpos[dof_adrs] = q
    d.qpos[env.src_adr:env.src_adr + 3] = mug_p[t]
    d.qpos[env.src_adr + 3:env.src_adr + 7] = mug_q[t]
    mujoco.mj_kinematics(m, d)

fromto = np.zeros(6)
def pair_dist(g, h, dmax):
    return mujoco.mj_geomDistance(m, d, g, h, dmax, fromto)

def all_body_min(dmax=0.05):
    out = {}
    for b, gs in hand_bodies.items():
        out[b] = min([dmax] + [pair_dist(g, h, dmax) for g in gs for h in mug_geoms])
    return out

end = T if args.end is None else min(args.end, T)
frames = list(range(args.start, end))
sol_x = {}; sol_joints = {}
before = np.zeros((T, 5)); after = np.zeros((T, 5))
worst_after = np.zeros(T); jdev = np.zeros(T)
cprev = np.zeros(6 + len(dof_adrs))
t_start = time.time()

for t in frames:
    R0 = Rot.from_quat(wq0[t][[1, 2, 3, 0]])
    x0 = np.r_[np.zeros(6), np.clip(q0[t], lo + 1e-9, hi - 1e-9)]
    place(wp0[t], wq0[t], q0[t], t)
    jorig = joints_xyz()
    bm = all_body_min()
    before[t] = [bm[b] for b in distal_ids]
    cand = {}
    for b, gs in hand_bodies.items():
        pr = [(g, h) for g in gs for h in mug_geoms if pair_dist(g, h, 0.03) < 0.03]
        if pr:
            cand[b] = pr
    touch = [i for i in range(5) if human[t, i] < 0.03 and distal_ids[i] in cand]

    def resid(x):
        p = wp0[t] + x[0:3]
        quat = (Rot.from_rotvec(x[3:6]) * R0).as_quat()[[3, 0, 1, 2]]
        place(p, quat, x[6:], t)
        jx = joints_xyz()
        r = [(jx - jorig).ravel() / 0.005,
             (x[6:] - q0[t]) / 0.2,
             x[0:3] / 0.02, x[3:6] / 0.1]
        dmin = {b: min(0.03, min(pair_dist(g, h, 0.03) for g, h in pr)) for b, pr in cand.items()}
        r.append(np.array([max(0.0, MARGIN - v) / MARGIN for v in dmin.values()]))
        r.append(np.array([0.5 * max(0.0, dmin[distal_ids[i]] - TOUCH) / TOUCH for i in touch]))
        r.append(0.5 * ((x - x0) - cprev) / scale_smooth)
        return np.concatenate(r)

    xs = np.r_[[0.01] * 3, [0.05] * 3, [0.1] * len(dof_adrs)]
    ds = np.r_[[2e-3] * 3, [1e-2] * 3, [1e-2] * len(dof_adrs)]
    best = None
    for init in (np.clip(x0 + cprev, blo, bhi), x0):
        r_ = least_squares(resid, init, bounds=(blo, bhi), jac='3-point', diff_step=ds, x_scale=xs,
                           max_nfev=150, xtol=1e-9, ftol=1e-9, gtol=1e-9)
        if best is None or r_.cost < best.cost:
            best = r_
        if best.cost < 80:
            break
    res = best
    x = res.x
    p = wp0[t] + x[0:3]
    quat = (Rot.from_rotvec(x[3:6]) * R0).as_quat()[[3, 0, 1, 2]]
    place(p, quat, x[6:], t)
    bm2 = all_body_min()
    after[t] = [bm2[b] for b in distal_ids]
    worst_after[t] = min(bm2.values())
    jx = joints_xyz()
    jdev[t] = np.linalg.norm(jx - jorig, axis=1).max()
    sol_x[t] = (p.copy(), quat.copy(), x[6:].copy()); sol_joints[t] = jx.copy()
    if (t - args.start) % 10 == 0 or t == frames[-1]:
        mv = np.linalg.norm(jx - jorig, axis=1)
        nlim = int(np.sum(((x[6:] - lo) < 1e-4) | ((hi - x[6:]) < 1e-4)))
        print('     worst-moved joint: %s (%.1f mm) | dofs at limit: %d | wrist shift %.1f mm, rot %.1f deg | cost %.2f'
              % (dexhand.body_names[1:][int(np.argmax(mv))], mv.max() * 1000, nlim,
                 np.linalg.norm(x[0:3]) * 1000, np.degrees(np.linalg.norm(x[3:6])), res.cost))
    cprev = x - x0
    if (t - args.start) % 10 == 0 or t == frames[-1]:
        print("t=%3d  distal min mm before %s -> after %s | worst any-body %.1f mm | max joint move %.1f mm | nfev %d | %.0fs"
              % (t, np.round(before[t] * 1000, 1), np.round(after[t] * 1000, 1), worst_after[t] * 1000,
                 jdev[t] * 1000, res.nfev, time.time() - t_start))

sel = np.array(frames)
labels = ["thumb", "index", "middle", "ring", "pinky"]
print("\n=== summary over %d frames (mm) ===" % len(sel))
print("%-7s %10s %10s %12s %12s" % ("finger", "min before", "min after", "mean before", "mean after"))
for i, l in enumerate(labels):
    print("%-7s %10.1f %10.1f %12.1f %12.1f" % (l, before[sel, i].min() * 1000, after[sel, i].min() * 1000,
                                                before[sel, i].mean() * 1000, after[sel, i].mean() * 1000))
print("worst penetration of ANY hand body after: %.2f mm" % (worst_after[sel].min() * 1000))
print("max joint displacement: %.1f mm (mean of per-frame max %.1f mm)" % (jdev[sel].max() * 1000, jdev[sel].mean() * 1000))

if args.write:
    assert len(sel) == T, "--write needs the full range"
    p_c = np.stack([sol_x[t][0] for t in range(T)])
    R_c = np.stack([Rot.from_quat(sol_x[t][1][[1, 2, 3, 0]]).as_matrix() for t in range(T)])
    q_new = np.stack([sol_x[t][2] for t in range(T)])
    J_c = np.stack([sol_joints[t] for t in range(T)])                 # (T,20,3) corrected frame
    p_raw = (C.T @ p_c.T).T
    R_raw = np.matmul(C.T[None], R_c)
    aa_raw = Rot.from_matrix(R_raw).as_rotvec()
    J_raw = np.einsum("ij,tkj->tki", C.T, J_c)
    w_raw = np.zeros((T, 3))
    for t in range(T):
        a, b = max(t - 1, 0), min(t + 1, T - 1)
        w_raw[t] = Rot.from_matrix(R_raw[b] @ R_raw[a].T).as_rotvec() / ((b - a) * dt)

    new = copy.deepcopy(demo)
    rn, _ = find_rh(new)
    def like(orig, arr):
        if torch.is_tensor(orig):
            return torch.as_tensor(arr, dtype=orig.dtype, device=orig.device)
        return np.asarray(arr, dtype=np.asarray(orig).dtype)
    rn["opt_wrist_pos"] = like(rh["opt_wrist_pos"], p_raw)
    rn["opt_wrist_rot"] = like(rh["opt_wrist_rot"], aa_raw)
    rn["opt_dof_pos"] = like(rh["opt_dof_pos"], q_new)
    rn["opt_wrist_velocity"] = like(rh["opt_wrist_velocity"], np.gradient(p_raw, dt, axis=0))
    rn["opt_wrist_angular_velocity"] = like(rh["opt_wrist_angular_velocity"], w_raw)
    rn["opt_dof_velocity"] = like(rh["opt_dof_velocity"], np.gradient(q_new, dt, axis=0))
    keys = [dexhand.to_hand(b)[0] for b in dexhand.body_names[1:]]
    Jv = np.gradient(J_raw, dt, axis=0)
    rn["myo_joints"] = {k: like(rh["myo_joints"][k], J_raw[:, i]) for i, k in enumerate(keys)}
    rn["myo_joints_velocity"] = {k: like(rh["myo_joints_velocity"][k], Jv[:, i]) for i, k in enumerate(keys)}
    with open(OUT, "wb") as f:
        pickle.dump(new, f)
    print("wrote", OUT)
