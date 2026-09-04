import mujoco
import mujoco.viewer

base = mujoco.MjSpec.from_file('assets/hands/myohand_r_final_scene.xml')
left = mujoco.MjSpec.from_file('assets/hands/myohand_l_final.xml')

site = base.worldbody.add_site(name='left_hand_attach', pos=[0, 0.4, 0])
base.attach(left, prefix='l2_', site=site)

model = base.compile()

for i in range(model.ngeom):
    name = model.geom(i).name
    if 'l2_' in name:
        model.geom_rgba[i] = [0.2, 0.4, 1.0, 1.0]  # bright blue = left hand
    elif name.endswith('_r'):
        model.geom_rgba[i] = [0.0, 1.0, 0.0, 1.0]  # bright green = right hand

data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("left geoms colored:", sum(1 for i in range(model.ngeom) if 'l2_' in model.geom(i).name))
print("right geoms colored:", sum(1 for i in range(model.ngeom) if model.geom(i).name.endswith('_r')))

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [0.35, 0.2, 0.0]
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 0
    viewer.cam.elevation = -30
    while viewer.is_running():
        viewer.sync()
