"""Combine both free-floating MyoHands into one scene for a quick visual check.
Gravity disabled — this is for viewing only, not physics."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/myo_sim"))
from myo_sim.build.compose import load_right_hand_from_arm_spec, load_left_hand_from_arm_spec
import mujoco
import mujoco.viewer

base = mujoco.MjSpec()
base.option.gravity = [0, 0, 0]

site_r = base.worldbody.add_site(name="attach_r", pos=[0.15, 0, 0])
site_l = base.worldbody.add_site(name="attach_l", pos=[-0.15, 0, 0])

hand_r = load_right_hand_from_arm_spec()
hand_r.body("myoarm_r_root").add_freejoint(name="root_r")
base.attach(hand_r, prefix="r_", site=site_r)

hand_l = load_left_hand_from_arm_spec()
hand_l.body("myoarm_l_root").add_freejoint(name="root_l")
base.attach(hand_l, prefix="l_", site=site_l)

model = base.compile()
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)
