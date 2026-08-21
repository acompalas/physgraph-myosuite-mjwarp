"""Export bimanual MyoHand attached to the passive torso scaffold,
with each hand's defaults/bodies prefixed to avoid class-name collisions
(compose.py's own build_both_hands_from_arm_spec doesn't prefix on attach,
which produces duplicate <default class="myoarm"> etc. between the two sides)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/myo_sim"))
from myo_sim.build.compose import (
    load_passive_torso_spec,
    load_right_hand_from_arm_spec,
    load_left_hand_from_arm_spec,
    MODEL_REGISTRY,
    RIGHT_ARM_ATTACH_SITE,
    LEFT_ARM_ATTACH_SITE,
    find_site,
)

registration = MODEL_REGISTRY["myohands"]
torso = load_passive_torso_spec(registration)

hand_r = load_right_hand_from_arm_spec()
torso.attach(hand_r, prefix="r_", site=find_site(torso, RIGHT_ARM_ATTACH_SITE))

hand_l = load_left_hand_from_arm_spec()
torso.attach(hand_l, prefix="l_", site=find_site(torso, LEFT_ARM_ATTACH_SITE))

model = torso.compile()
print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu, "njnt:", model.njnt)

out_path = Path(__file__).resolve().parents[1] / "assets/hands/myohands.xml"
out_path.write_text(torso.to_xml())
print(f"wrote {out_path}")
