import mujoco
import mujoco.viewer

base = mujoco.MjSpec.from_file('assets/hands/myohand_r_ulnaroot_scene.xml')
left = mujoco.MjSpec.from_file('assets/hands/myohand_l_ulnaroot.xml')

# shift the right hand off-center by -0.2, and attach the left hand at
# +0.2, so the PAIR's midpoint lands on the pedestal's true origin/center
right_root = base.body('ulna_r')
right_root.pos = [0, -0.2, 0]

site = base.worldbody.add_site(name='left_hand_attach', pos=[0, 0.2, 0], quat=[0, 1, 0, 0])
base.attach(left, prefix='l2_', site=site)

model = base.compile()
print(f"combined model: nq={model.nq} nv={model.nv} nu={model.nu} njnt={model.njnt}")

data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.28, 0.0, 0.0]  # y=0 now, since the pair is centered on it
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    while viewer.is_running():
        viewer.sync()
