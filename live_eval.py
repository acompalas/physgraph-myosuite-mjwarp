"""Single-environment live, interactive eval of a trained checkpoint.
Runs the real training physics (mujoco_warp, num_envs=1, matching
training exactly -- no sim-to-sim gap) and our real trained network,
bridging the resulting state into a real mujoco.MjData each step so we
can use the real, interactive mujoco.viewer (camera control/zoom),
which mujoco_warp itself has no equivalent for. Applies the real
observation normalization stats saved in the checkpoint (running_mean_
std) -- the network was trained on normalized inputs, skipping this
would produce meaningless actions.
"""
import sys
import numpy as np
import torch
import mujoco
import mujoco.viewer

sys.path.insert(0, ".")
from envs.mjwarp_myohand_env import MyoHandPourEnv
from lib.rl.network_builder_myohand_transformer import MyoHandTransformerNetwork

CHECKPOINT_PATH = "eval_checkpoints/best_runC_ep_6786.pth"
DEVICE = "cuda:0"


def apply_running_mean_std(x, stats, prefix, eps=1e-5, clip=5.0):
    # rl_games saves RunningMeanStd stats in float64 for numerical
    # stability; our observations are float32 -- cast to match
    mean = stats[f"{prefix}.running_mean"].to(DEVICE).float()
    var = stats[f"{prefix}.running_var"].to(DEVICE).float()
    y = (x - mean) / torch.sqrt(var + eps)
    return torch.clamp(y, -clip, clip)


def main():
    print(f"loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, weights_only=False, map_location=DEVICE)
    model_sd = ckpt["model"]

    print("building env (num_envs=1)...")
    env = MyoHandPourEnv(num_envs=1, device=DEVICE)

    _dd = env.demo_opt_dof_pos.detach().cpu().numpy() * 57.2958
    for _k, _n in enumerate(env.dof_names):
        print(f"[demo range] {_n}: {_dd[:, _k].min():.0f} .. {_dd[:, _k].max():.0f} deg")
    print("building network and loading weights...")
    net = MyoHandTransformerNetwork(params=None, actions_num=48).to(DEVICE)
    net_prefix = "a2c_network."
    net_sd = {k[len(net_prefix):]: v for k, v in model_sd.items() if k.startswith(net_prefix)}
    missing, unexpected = net.load_state_dict(net_sd, strict=False)
    print(f"loaded network weights. missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (first 5):", missing[:5])
    if unexpected:
        print("  unexpected keys (first 5):", unexpected[:5])
    net.eval()

    rms_prefix = "running_mean_std.running_mean_std"

    # real MjData for interactive viewing, sharing the same real model
    # our env already loads -- no separate model needed
    mj_data = mujoco.MjData(env.mj_model)

    obs = env.reset()

    with mujoco.viewer.launch_passive(env.mj_model, mj_data) as viewer:
        # enable all geom visibility groups -- our model spreads visual
        # meshes (skin, mugs) across groups 3/4, which the real viewer
        # does not enable by default (only 0-2)
        viewer.opt.geomgroup[:] = 1
        # explicitly frame the camera to include both mugs -- the real
        # viewer's default camera centers wherever it happens to start
        # (near the hand), which can leave the dst mug (~0.4m away) just
        # outside frame even in a shot that LOOKS wide
        src_pos = env.demo_src_obj_traj[0, :3, 3].cpu().numpy()
        dst_pos = env.demo_dst_obj_traj0[:3, 3].cpu().numpy()
        center = (src_pos + dst_pos) / 2
        viewer.cam.lookat[:] = center
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -30
        print('camera lookat:', viewer.cam.lookat[:], 'distance:', viewer.cam.distance)
        step_count = 0
        while viewer.is_running():
            with torch.no_grad():
                norm_obs = {
                    "proprioception": apply_running_mean_std(obs["obs"]["proprioception"], model_sd, f"{rms_prefix}.proprioception"),
                    "privileged": apply_running_mean_std(obs["obs"]["privileged"], model_sd, f"{rms_prefix}.privileged"),
                    "target": apply_running_mean_std(obs["obs"]["target"], model_sd, f"{rms_prefix}.target"),
                }
                mu, sigma, value, _ = net({"obs": norm_obs})
                # rl_games' real deterministic eval: raw mu, clamped to
                # [-1,1] (rescale_actions is identity here since our
                # action space is already [-1,1]) -- NOT tanh(mu), which
                # would systematically distort the policy's real
                # intended actions (confirmed via direct inspection of
                # rl_games.algos_torch.players.PpoPlayerContinuous.get_action)
                action = torch.clamp(mu, -1.0, 1.0)

            obs, rewards, dones, infos = env.step(action)

            # bridge warp state -> real MjData for rendering
            import warp as wp
            qpos = wp.to_torch(env.data.qpos)[0].cpu().numpy()
            mj_data.qpos[:] = qpos
            mujoco.mj_forward(env.mj_model, mj_data)
            viewer.sync()

            prog = int(env.progress_buf[0].item())
            if prog % 5 == 0 or prog < 5:
                rd = infos["reward_dict"]
                def _e(key, k):
                    return -np.log(max(float(rd[key][0].item()), 1e-9)) / k * 100.0
                tb = [int(i) for i in (env.tip_body_ids.tolist() if hasattr(env.tip_body_ids, "tolist") else list(env.tip_body_ids))]
                px = mj_data.xpos[tb]
                print(f"  [track prog {prog}] tip err cm: thumb {_e('reward_thumb_tip_pos',100):.1f} index {_e('reward_index_tip_pos',90):.1f} "
                      f"middle {_e('reward_middle_tip_pos',80):.1f} ring {_e('reward_ring_tip_pos',60):.1f} pinky {_e('reward_pinky_tip_pos',60):.1f} "
                      f"| wrist {_e('reward_eef_pos',40):.1f} | spans cm: thumb-index {np.linalg.norm(px[0]-px[1])*100:.1f} "
                      f"index-pinky {np.linalg.norm(px[1]-px[4])*100:.1f}")
                if prog <= 60:
                    if prog == 0:
                        print("      hand dofs:", [(i, n) for i, n in enumerate(env.dof_names)])
                    _q = qpos[env.dof_adrs] * 57.2958
                    _d = env.demo_opt_dof_pos[min(prog, env.demo_opt_dof_pos.shape[0] - 1)].detach().cpu().numpy() * 57.2958
                    print(f"      demo deg  {np.round(_d).astype(int).tolist()}")
                    print(f"      policy deg {np.round(_q).astype(int).tolist()}  mean|err| {np.abs(_q - _d).mean():.1f} deg")
                _m = env.mj_model
                _viol, _thumb = [], []
                for _i, _n in enumerate(env.dof_names):
                    _j = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_JOINT, _n)
                    if _j < 0 or not _m.jnt_limited[_j]:
                        continue
                    _lo, _hi = _m.jnt_range[_j]
                    _v = float(qpos[int(env.dof_adrs[_i])])
                    _o = max(_lo - _v, _v - _hi, 0.0)
                    if _o > 0.02:
                        _viol.append(f"{_n} {np.degrees(_v):.0f} (range {np.degrees(_lo):.0f}..{np.degrees(_hi):.0f}, over {np.degrees(_o):.1f})")
                    if _n in ("cmc_flexion_r", "cmc_abduction_r", "mp_flexion_r", "ip_flexion_r"):
                        _thumb.append(f"{_n} {np.degrees(_v):.0f} [{np.degrees(_lo):.0f},{np.degrees(_hi):.0f}]")
                print("      thumb deg:", " | ".join(_thumb))
                if _viol:
                    print("      OVER LIMIT:", "; ".join(_viol))
                if prog <= 6:
                    wf = env._wrist_force[0].detach().cpu().numpy()
                    dw = (env.demo_opt_wrist_pos[min(prog, env.demo_opt_wrist_pos.shape[0] - 1)] - env.demo_opt_wrist_pos[0]).norm().item() * 100
                    print(f"      wrist force N {np.round(wf, 1)} (|F|={np.linalg.norm(wf):.1f}); demo wrist moved {dw:.1f} cm since frame 0")

            if step_count % 10 == 0:
                ids = env.tip_body_ids
                tips = set(int(i) for i in (ids.tolist() if hasattr(ids, "tolist") else list(ids)))
                m = env.mj_model
                f6 = np.zeros(6)
                rows = []
                for ci in range(mj_data.ncon):
                    c = mj_data.contact[ci]
                    b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                    if b1 in tips or b2 in tips:
                        mujoco.mj_contactForce(m, mj_data, ci, f6)
                        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1)
                        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)
                        rows.append((c.dist * 1000, f6[0], n1, n2))
                rows.sort()
                print(f"  [tip contacts, step {step_count}] n={len(rows)} (dist mm, normal N, bodies)")
                for r in rows[:8]:
                    print(f"    {r[0]:+.2f} mm  {r[1]:6.1f} N  {r[2]} <-> {r[3]}")

            step_count += 1
            if step_count % 50 == 0:
                print(f"step {step_count}, reward: {rewards[0].item():.3f}, progress: {env.progress_buf[0].item()}")

            if dones[0]:
                print("episode ended, resetting")
                obs = env.reset()


if __name__ == "__main__":
    main()
