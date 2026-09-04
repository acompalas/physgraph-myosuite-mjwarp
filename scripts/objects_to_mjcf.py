"""Convert CoACD-decomposed object parts (separate .obj files, one per
convex piece) into a single-body MJCF asset with a freejoint. Mass/
inertia carried over from the original URDF (already computed at
1000 kg/m^3 density by the original generate_urdfs.py)."""
import re
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_urdf_inertial(urdf_path):
    tree = ET.parse(urdf_path)
    inertial = tree.getroot().find(".//inertial")
    origin = inertial.find("origin")
    xyz = origin.get("xyz") if origin is not None else "0 0 0"
    mass = inertial.find("mass").get("value")
    inertia = inertial.find("inertia").attrib
    return xyz, mass, inertia

def build_object_mjcf(obj_id, base_dir):
    obj_dir = base_dir / obj_id
    parts_dir = obj_dir / "parts"
    parts = sorted(parts_dir.glob("*.obj"))
    xyz, mass, inertia = parse_urdf_inertial(obj_dir / "scan.urdf")

    mesh_assets = "\n".join(
        f'    <mesh name="{obj_id}_part{i}" file="{p.name}"/>'
        for i, p in enumerate(parts)
    )
    geoms = "\n".join(
        f'      <geom type="mesh" mesh="{obj_id}_part{i}" class="collision"/>'
        for i in range(len(parts))
    )

    xml = f'''<mujoco model="{obj_id}">
  <compiler meshdir="parts" angle="radian"/>
  <default>
    <default class="collision">
      <geom group="3" contype="1" conaffinity="1" rgba="0.7 0.5 0.3 1"/>
    </default>
  </default>
  <asset>
{mesh_assets}
  </asset>
  <worldbody>
    <body name="{obj_id}" pos="0 0 0">
      <freejoint name="{obj_id}_free"/>
      <inertial pos="{xyz}" mass="{mass}"
                fullinertia="{inertia['ixx']} {inertia['iyy']} {inertia['izz']} {inertia['ixy']} {inertia['ixz']} {inertia['iyz']}"/>
{geoms}
    </body>
  </worldbody>
</mujoco>
'''
    out_path = obj_dir / f"{obj_id}.xml"
    out_path.write_text(xml)
    print(f"wrote {out_path} ({len(parts)} parts, mass={mass})")
    return out_path

base_dir = Path(__file__).resolve().parents[1] / "assets/objects"
for obj_id in ["O02@0010@00003", "O02@0015@00020"]:
    build_object_mjcf(obj_id, base_dir)
