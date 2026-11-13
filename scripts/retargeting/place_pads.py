import json, numpy as np, mujoco
from envs.mjwarp_myohand_env import MyoHandPourEnv

FING = {
    "distal_thumb_r": ("THtip_r", "FPL-P11_r", "EPL-P12_r"),
    "distph2_r": ("IFtip_r", "FDP2-P10_r", "EDC2-P9_r"),
    "distph3_r": ("MFtip_r", "FDP3-P10_r", "EDC3-P9_r"),
    "distph4_r": ("RFtip_r", "FDP4-P10_r", "EDC4-P9_r"),
    "distph5_r": ("LFtip_r", "FDP5-P10_r", "EDC5-P9_r"),
}
DEMO_CONTACT_MM = {
    "distal_thumb_r": [3.4, -20.1, -7.2], "distph2_r": [7.6, -17.9, -3.7], "distph3_r": [6.4, -16.6, -7.6],
    "distph4_r": [5.3, -11.9, -6.4], "distph5_r": [3.3, -19.5, -4.9],
}
env = MyoHandPourEnv(num_envs=1, device="cuda:0")
m = env.mj_model
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
out = {"bodies": {}}
for bn, (tn, fn, en) in FING.items():
    b = m.body(bn).id
    tip = m.site(tn).pos.copy(); fl = m.site(fn).pos.copy(); ex = m.site(en).pos.copy()
    a = tip / np.linalg.norm(tip)
    n = fl - ex; n = n - (n @ a) * a; n /= np.linalg.norm(n)
    gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
    nail = [g for g in gs if m.geom_type[g] == 4]
    ang = None
    if nail:
        e = m.geom_pos[nail[0]].copy(); dd = e - (e @ a) * a; dd /= np.linalg.norm(dd)
        ang = round(float(np.degrees(np.arccos(np.clip(n @ (-dd), -1, 1)))), 1)
    col = [g for g in gs if (m.geom_contype[g] or m.geom_conaffinity[g])]
    grp = np.zeros(6, dtype=np.uint8)
    for g in col:
        grp[min(int(m.geom_group[g]), 5)] = 1
    R = d.xmat[b].reshape(3, 3); p = d.xpos[b]
    start_loc = tip + n * 0.015
    gid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(m, d, p + R @ start_loc, -(R @ n), grp, 1, -1, gid)
    ok = dist >= 0 and m.geom_bodyid[gid[0]] == b
    print(f"PLACE {bn}: tip {np.round(tip*1000,1).tolist()} palmar n {np.round(n,2).tolist()} nail-vs-tendon angle {ang} deg | collision geoms (id,type,group) {[(g,int(m.geom_type[g]),int(m.geom_group[g])) for g in col]}", flush=True)
    if not ok:
        print(f"PLACE   WARN ray hit geom {int(gid[0])} body {m.body(m.geom_bodyid[gid[0]]).name if gid[0] >= 0 else None} dist {dist}", flush=True)
        continue
    pad = start_loc - n * dist
    dc = np.array(DEMO_CONTACT_MM[bn]) / 1000.0
    print(f"PLACE   ray hit geom {int(gid[0])} (type {int(m.geom_type[gid[0]])}) | pad site {np.round(pad*1000,1).tolist()} mm | tip->pad {np.linalg.norm(pad-tip)*1000:.1f} mm | pad->demo-contact {np.linalg.norm(pad-dc)*1000:.1f} mm", flush=True)
    out["bodies"][bn] = {"tip_site": tn, "pad_local_m": pad.tolist(), "demo_contact_local_m": dc.tolist()}
json.dump(out, open("assets/retargeted/pad_sites_local.json", "w"), indent=1)
print("PLACE saved assets/retargeted/pad_sites_local.json", flush=True)
