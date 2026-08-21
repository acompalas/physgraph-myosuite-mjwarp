"""Export the composed bimanual MyoHand model to a static MJCF file."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/myo_sim"))
from myo_sim.build.compose import build_spec

spec = build_spec("myohands")

out_path = Path(__file__).resolve().parents[1] / "assets/hands/myohands.xml"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(spec.to_xml())
print(f"wrote {out_path}")

model = spec.compile()
print("nq (position DOF):", model.nq)
print("nv (velocity DOF):", model.nv)
print("nu (actuators):", model.nu)
print("njnt (joints):", model.njnt)
