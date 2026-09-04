import mujoco
from myo_sim.build.compose import load_right_hand_from_arm_spec

def log(msg):
    print(msg, flush=True)

log("1: loading arm spec")
arm_spec = load_right_hand_from_arm_spec()

log("2: finding cut body")
cut_body = arm_spec.body("radius_r")

log("3: converting to frame")
frame = cut_body.to_frame()

log("4: attaching frame back to SAME spec's worldbody")
attached = arm_spec.worldbody.attach_frame(frame)
log(f"   attached: {attached}")

log("5: deleting old root body (should now be an empty ancestor chain)")
old_root = arm_spec.body("myoarm_r_root")
arm_spec.delete(old_root)

log("6: compiling")
model = arm_spec.compile()
log(f"   compiled OK, nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")
