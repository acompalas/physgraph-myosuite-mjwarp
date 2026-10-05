"""Kinematic playback (physics OFF) of BOTH hands' retargeted 1292e@0
trajectories together, skin recolored per-hand for visibility (same
material-level recolor technique used for the earlier Runfa screenshot
-- geom_rgba alone is overridden by the assigned MatSkin material, so
we recolor the material itself, matched by geom-name substring since
MjSpec.attach() prefixes the left hand's copy)."""
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
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


def main():
    proj_root = Path(__file__).resolve().parents[2]

    base = mujoco.MjSpec.from_file(str(proj_root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
    left = mujoco.MjSpec.from_file(str(proj_root / "assets/hands/myohand_l_ulnaroot.xml"))

    right_root = base.body("ulna_r")
    right_root.pos = [right_root.pos[0], right_root.pos[1] - 0.2, right_root.pos[2]]

    site = base.worldbody.add_site(name="left_hand_attach", pos=[0, 0.2, 0], quat=[1, 0, 0, 0])
    base.attach(left, prefix="l2_", site=site)
    base.delete(site)

    model = base.compile()
    data = mujoco.MjData(model)

    # recolor skin: right hand green, left hand blue (material-level, not per-geom)
    for i in range(model.nmat):
        name = model.material(i).name
        if "matskin" in name.lower() and not name.startswith("l2_"):
            model.mat_rgba[i] = [0.0, 1.0, 0.0, 1.0]
        elif "matskin" in name.lower() and name.startswith("l2_"):
            model.mat_rgba[i] = [0.2, 0.4, 1.0, 1.0]

    with open(proj_root / "assets/retargeted/1292e_rh_myohand.pkl", "rb") as f:
        retgt_r = pickle.load(f)
    with open(proj_root / "assets/retargeted/1292e_lh_myohand.pkl", "rb") as f:
        retgt_l = pickle.load(f)

    n_frames = min(retgt_r["opt_wrist_pos"].shape[0], retgt_l["opt_wrist_pos"].shape[0])
    print(f"playing {n_frames} frames")

    quat_r = aa_to_quat_np(retgt_r["opt_wrist_rot"])
    quat_l = aa_to_quat_np(retgt_l["opt_wrist_rot"])

    dh_r, dh_l = MyoHandR(), MyoHandL()

    root_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")
    root_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "l2_root_l")
    root_r_adr = model.jnt_qposadr[root_r_id]
    root_l_adr = model.jnt_qposadr[root_l_id]

    dof_r_adrs = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in dh_r.dof_names
    ])
    dof_l_adrs = np.array([
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"l2_{n}")]
        for n in dh_l.dof_names
    ])

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for g in range(6):
            viewer.opt.geomgroup[g] = 1
        viewer.cam.lookat[:] = [0.28, 0.0, 0.1]
        viewer.cam.distance = 0.9
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        frame = 0
        while viewer.is_running():
            # right hand -- note: right was displayed with a -0.2 y OFFSET
            # for viewing symmetry, so add that offset to the retargeted
            # position too, same as the earlier bimanual display convention
            data.qpos[root_r_adr:root_r_adr + 3] = retgt_r["opt_wrist_pos"][frame] + np.array([0, -0.2, 0])
            data.qpos[root_r_adr + 3:root_r_adr + 7] = quat_r[frame]
            data.qpos[dof_r_adrs] = retgt_r["opt_dof_pos"][frame]

            data.qpos[root_l_adr:root_l_adr + 3] = retgt_l["opt_wrist_pos"][frame] + np.array([0, 0.2, 0])
            data.qpos[root_l_adr + 3:root_l_adr + 7] = quat_l[frame]
            data.qpos[dof_l_adrs] = retgt_l["opt_dof_pos"][frame]

            mujoco.mj_forward(model, data)
            viewer.sync()

            frame = (frame + 1) % n_frames
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
