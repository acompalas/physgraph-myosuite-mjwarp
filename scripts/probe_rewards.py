"""Zero-action rollout: reward-term magnitudes, contact-failure fraction and power values per step."""
import sys, torch
sys.path.insert(0, ".")
import envs.mjwarp_myohand_env as E
n = 16
env = E.MyoHandPourEnv(num_envs=n, device="cuda:0")
env.reset()
act = torch.zeros(n, 48, device="cuda:0")
for t in range(14):
    obs, r, d, info = env.step(act)
    rd, dbg = info["reward_dict"], env._debug_diffs
    print("t=%2d done %.2f | reward %6.2f | force_r %.3f | power %8.1f (r %.3f) | wrist_power %8.1f (r %.3f) | contact_fail %.2f | tipF %.1f"
          % (t + 1, d.float().mean(), r.mean(), rd["reward_finger_tip_force"].mean(), dbg["power"].mean(),
             rd["reward_power"].mean(), dbg["wrist_power"].mean(), rd["reward_wrist_power"].mean(),
             dbg["contact_fail"].mean(), dbg["tip_force_sum"].mean()))
print("contact-failure rule enabled:", env.use_contact_failure)
