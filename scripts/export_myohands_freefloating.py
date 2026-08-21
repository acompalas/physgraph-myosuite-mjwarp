"""Export right and left MyoHand as independent free-floating MJCF models
(no shared torso, no rigid world attachment) — matches PhysGraph's
separate rh/lh tracking rather than myo_sim's default torso-mounted build."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/myo_sim"))
from myo_sim.build.compose import load_right_hand_from_arm_spec, load_left_hand_from_arm_spec

out_dir = Path(__file__).resolve().parents[1] / "assets/hands"
out_dir.mkdir(parents=True, exist_ok=True)

for side, loader, root_name, joint_name in [
    ("r", load_right_hand_from_arm_spec, "myoarm_r_root", "root_r"),
    ("l", load_left_hand_from_arm_spec, "myoarm_l_root", "root_l"),
]:
    spec = loader()
    root = spec.body(root_name)
    root.add_freejoint(name=joint_name)

    model = spec.compile()
    print(f"--- {side} hand ---")
    print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu, "njnt:", model.njnt)

    out_path = out_dir / f"myohand_{side}_free.xml"
    out_path.write_text(spec.to_xml())
    print(f"wrote {out_path}")
