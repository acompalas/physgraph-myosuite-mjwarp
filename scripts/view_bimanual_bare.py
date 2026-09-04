import mujoco
import mujoco.viewer

right = mujoco.MjSpec.from_file('assets/hands/myohand_r_final.xml')  # no scene, just the hand
left = mujoco.MjSpec.from_file('assets/hands/myohand_l_final.xml')

right.option.gravity = [0, 0, 0]

site = right.worldbody.add_site(name='left_hand_attach', pos=[0, 0.4, 0], quat=[0, 1, 0, 0])
right.attach(left, prefix='l2_', site=site)

model = right.compile()

for i in range(model.ngeom):
    name = model.geom(i).name
    if 'l2_' in name:
        model.geom_rgba[i] = [0.2, 0.4, 1.0, 1.0]
    elif name.endswith('_r'):
        model.geom_rgba[i] = [0.0, 1.0, 0.0, 1.0]

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("=== DIAGNOSTIC ===")
print("total bodies:", model.nbody)
print("total geoms:", model.ngeom)
for name in ['radius_r', 'l2_radius_l']:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    print(name, '-> world pos:', data.xpos[bid])
print("==================")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.35, 0.2, 0.0]
    viewer.cam.distance = 0.6
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -10
    while viewer.is_running():
        viewer.sync()
