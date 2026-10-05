"""Kinematic playback (physics OFF, pure qpos teleportation each frame)
of the retargeted MyoHand trajectory, on the REAL training asset
(myohand_r_ulnaroot.xml -- with its freejoint and original unsplit
joint structure), mapping retargeted values back by JOINT NAME rather
than assuming array-order correspondence with the pk-preprocessed
retargeting chain."""
import pickle
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from myohand_def import MyoHandR


def aa_to_quat_np(aa):
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta = np.clip(theta, 1e-8, None)
    axis = aa / theta
    half = theta / 2
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.concatenate([w, xyz], axis=-1)  # (N, 4) wxyz


def main():
    proj_root = Path(__file__).resolve().parents[2]
    model = mujoco.MjModel.from_xml_path(str(proj_root / "assets/hands/myohand_r_ulnaroot_scene.xml"))
    data = mujoco.MjData(model)

    with open(proj_root / "assets/retargeted/1292e_rh_myohand.pkl", "rb") as f:
        retgt = pickle.load(f)

    opt_wrist_pos = retgt["opt_wrist_pos"]      # (N, 3)
    opt_wrist_rot = retgt["opt_wrist_rot"]      # (N, 3) axis-angle
    opt_dof_pos = retgt["opt_dof_pos"]          # (N, 23)
    n_frames = opt_wrist_pos.shape[0]
    print(f"loaded {n_frames} retargeted frames")

    opt_wrist_quat = aa_to_quat_np(opt_wrist_rot)  # (N, 4) wxyz

    dexhand = MyoHandR()

    # map each retargeted DOF (by name) to its real qpos address in this model
    root_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root_r")
    root_qpos_adr = model.jnt_qposadr[root_joint_id]

    dof_qpos_adrs = []
    for name in dexhand.dof_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"joint {name} not found in training asset")
        dof_qpos_adrs.append(model.jnt_qposadr[jid])
    dof_qpos_adrs = np.array(dof_qpos_adrs)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.3, 0.0, 0.15]
        viewer.cam.distance = 0.6
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -20

        frame = 0
        while viewer.is_running():
            data.qpos[root_qpos_adr:root_qpos_adr + 3] = opt_wrist_pos[frame]
            data.qpos[root_qpos_adr + 3:root_qpos_adr + 7] = opt_wrist_quat[frame]
            data.qpos[dof_qpos_adrs] = opt_dof_pos[frame]

            mujoco.mj_forward(model, data)  # kinematics only -- NOT mj_step, no physics
            viewer.sync()

            frame = (frame + 1) % n_frames
            time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
