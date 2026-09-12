"""Comprehensive kinematic verification (physics OFF) using our REAL
environment class's own computed target values directly -- wrist,
source mug, destination mug, using the same coordinate corrections
the actual training environment applies. This is the definitive check
before retraining: if this looks correct, the environment's real
target data is correct."""
import sys
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, ".")
from envs.mjwarp_myohand_env import MyoHandPourEnv


def main():
    env = MyoHandPourEnv(num_envs=1, device="cuda:0")

    # real MjData sharing the exact same real model our env uses
    mj_data = mujoco.MjData(env.mj_model)

    n_frames = env.seq_len
    print(f"n_frames: {n_frames}")
    print(f"src mug pos[0]: {env.demo_src_obj_traj[0, :3, 3].cpu().tolist()}")
    print(f"dst mug pos: {env.demo_dst_obj_traj0[:3, 3].cpu().tolist()}")
    print(f"wrist pos[0]: {env.demo_opt_wrist_pos[0].cpu().tolist()}")

    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        viewer.opt.geomgroup[:] = 1
        center = (
            env.demo_src_obj_traj[0, :3, 3] + env.demo_dst_obj_traj0[:3, 3]
        ) / 2
        viewer.cam.lookat[:] = center.cpu().numpy()
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30

        frame = 0
        while viewer.is_running():
            mj_data.qpos[env.root_adr:env.root_adr + 3] = env.demo_opt_wrist_pos[frame].cpu().numpy()
            mj_data.qpos[env.root_adr + 3:env.root_adr + 7] = env.demo_opt_wrist_quat[frame].cpu().numpy()
            mj_data.qpos[env.dof_adrs] = env.demo_opt_dof_pos[frame].cpu().numpy()
            mj_data.qpos[env.src_adr:env.src_adr + 3] = env.demo_src_obj_traj[frame, :3, 3].cpu().numpy()
            from envs.mjwarp_myohand_env import rotmat_to_quat
            mj_data.qpos[env.src_adr + 3:env.src_adr + 7] = rotmat_to_quat(env.demo_src_obj_traj[frame:frame+1, :3, :3])[0].cpu().numpy()
            mj_data.qpos[env.dst_adr:env.dst_adr + 3] = env.demo_dst_obj_traj0[:3, 3].cpu().numpy()
            mj_data.qpos[env.dst_adr + 3:env.dst_adr + 7] = rotmat_to_quat(env.demo_dst_obj_traj0[None, :3, :3])[0].cpu().numpy()

            mujoco.mj_forward(env.mj_model, mj_data)
            viewer.sync()
            frame += 1
            import time
            if frame == 1:
                _start_time = time.time()
            if frame == n_frames:
                elapsed = time.time() - _start_time
                print(f'REAL measured: {n_frames} frames in {elapsed:.2f}s = {n_frames/elapsed:.2f} actual fps')
            frame = frame % n_frames
            time.sleep(1.0 / 30.0)


if __name__ == "__main__":
    main()
