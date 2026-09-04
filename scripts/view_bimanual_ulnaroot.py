import mujoco
import mujoco.viewer

base = mujoco.MjSpec.from_file('assets/hands/myohand_r_ulnaroot_scene.xml')
left = mujoco.MjSpec.from_file('assets/hands/myohand_l_ulnaroot.xml')

right_root = base.body('ulna_r')
right_root.pos = [right_root.pos[0], right_root.pos[1] - 0.2, right_root.pos[2]]

site = base.worldbody.add_site(name='left_hand_attach', pos=[0, 0.2, 0], quat=[0, 1, 0, 0])
base.attach(left, prefix='l2_', site=site)

# the attach site was only needed to position the left hand -- delete it
# now that attach() has used it, so it doesn't render as a visible sphere
base.delete(site)

model = base.compile()
data = mujoco.MjData(model)
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.28, 0.0, 0.0]
    viewer.cam.distance = 0.8
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -20
    while viewer.is_running():
        viewer.sync()
