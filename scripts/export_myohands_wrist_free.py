"""Cut MyoHand at the wrist (radius_r/radius_l), discarding the rigid
clavicle/scapula/humerus/ulna chain entirely, and attach the hand
directly to a fresh worldbody with its own freejoint. Preserves the
existing wrist joints (pro_sup, deviation, flexion) and all finger joints."""
import mujoco
from pathlib import Path
from myo_sim.build.compose import load_right_hand_from_arm_spec, load_left_hand_from_arm_spec

out_dir = Path(__file__).resolve().parents[1] / "assets/hands"
out_dir.mkdir(parents=True, exist_ok=True)

for side, loader, cut_body_name in [
    ("r", load_right_hand_from_arm_spec, "radius_r"),
    ("l", load_left_hand_from_arm_spec, "radius_l"),
]:
    arm_spec = loader()
    cut_body = arm_spec.body(cut_body_name)
    frame = cut_body.to_frame()

    new_spec = mujoco.MjSpec()
    new_spec.option.gravity = [0, 0, -9.81]
    attached = new_spec.worldbody.attach_frame(frame)

    # find the reattached body under the new worldbody and give it a freejoint
    new_root = new_spec.body(cut_body_name)
    new_root.add_freejoint(name=f"root_{side}")

    model = new_spec.compile()
    print(f"--- {side} hand (cut at {cut_body_name}) ---")
    print("nq:", model.nq, "nv:", model.nv, "nu:", model.nu, "njnt:", model.njnt)
    print("bodies:", [b for b in [new_spec.body(i).name for i in range(model.nbody)]][:10], "...")

    out_path = out_dir / f"myohand_{side}_wrist_free.xml"
    out_path.write_text(new_spec.to_xml())
    print(f"wrote {out_path}")
