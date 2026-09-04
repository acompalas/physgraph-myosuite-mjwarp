"""Final MyoHand cut: keep only wrist/finger-actuating geometry.
Removes clavicle/scapula entirely and collapses humerus/ulna's
tendon-referenced sites/geoms onto the new free-floating root via
composed rigid transforms (zero joints between these bodies, so this
is an exact geometric equivalent). Needed sites/geoms are found
automatically by scanning every <tendon><spatial> for every site name
and geom+sidesite pair it actually references, rather than a
hand-maintained list (which kept missing wrap-geom sidesite refs)."""
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from pathlib import Path

REMOVED_BODIES_R = [
    "myoarm_r_root", "clavicle_r", "clavphant_r", "scapula_r",
    "scapphant_r", "humphant_r", "humphant1_r", "humerus_r", "ulna_r",
]
REMOVED_BODIES_L = [b.replace("_r", "_l") for b in REMOVED_BODIES_R]

def get_transform(elem):
    pos = np.array([float(v) for v in elem.get("pos", "0 0 0").split()])
    quat_str = elem.get("quat")
    if quat_str:
        w, x, y, z = [float(v) for v in quat_str.split()]
        R = Rotation.from_quat([x, y, z, w])  # scipy uses xyzw order
    else:
        R = Rotation.identity()
    return R, pos

def compose(R1, p1, R2, p2):
    R = R1 * R2
    p = R1.apply(p2) + p1
    return R, p

def rot_to_mjquat(R):
    x, y, z, w = R.as_quat()  # scipy returns xyzw
    return np.array([w, x, y, z])  # mujoco wants wxyz

for side, removed in [("r", REMOVED_BODIES_R), ("l", REMOVED_BODIES_L)]:
    src = Path(__file__).resolve().parents[1] / f"assets/hands/myohand_{side}_free.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    needed_names = set()
    tendon = root.find("tendon")
    if tendon is not None:
        for spatial in tendon.findall("spatial"):
            for site in spatial.findall("site"):
                needed_names.add(site.get("site"))
            for geom in spatial.findall("geom"):
                needed_names.add(geom.get("geom"))
                if geom.get("sidesite"):
                    needed_names.add(geom.get("sidesite"))

    chain = []
    node = worldbody.find("body")
    while node is not None and node.get("name") != f"radius_{side}":
        chain.append(node)
        node = node.find("body")
    radius_body = node

    R_accum, p_accum = Rotation.identity(), np.zeros(3)
    body_transforms = {}
    for b in chain:
        R_b, p_b = get_transform(b)
        R_accum, p_accum = compose(R_accum, p_accum, R_b, p_b)
        body_transforms[b.get("name")] = (R_accum, p_accum.copy())

    preserved_sites, preserved_geoms = [], []
    for b in chain:
        R_b, p_b = body_transforms[b.get("name")]
        for site in b.findall("site"):
            if site.get("name") in needed_names:
                R_s, p_s = get_transform(site)
                R_new, p_new = compose(R_b, p_b, R_s, p_s)
                new_site = ET.Element("site", dict(site.attrib))
                new_site.set("pos", f"{p_new[0]:.6f} {p_new[1]:.6f} {p_new[2]:.6f}")
                if site.get("quat"):
                    new_site.set("quat", " ".join(f"{v:.6f}" for v in rot_to_mjquat(R_new)))
                preserved_sites.append(new_site)
        for geom in b.findall("geom"):
            if geom.get("name") in needed_names:
                R_g, p_g = get_transform(geom)
                R_new, p_new = compose(R_b, p_b, R_g, p_g)
                new_geom = ET.Element("geom", dict(geom.attrib))
                new_geom.set("pos", f"{p_new[0]:.6f} {p_new[1]:.6f} {p_new[2]:.6f}")
                if geom.get("quat"):
                    new_geom.set("quat", " ".join(f"{v:.6f}" for v in rot_to_mjquat(R_new)))
                preserved_geoms.append(new_geom)

    R_radius, p_radius = get_transform(radius_body)
    R_new_root, p_new_root = compose(R_accum, p_accum, R_radius, p_radius)
    radius_body.set("pos", f"{p_new_root[0]:.6f} {p_new_root[1]:.6f} {p_new_root[2]:.6f}")
    radius_body.set("quat", " ".join(f"{v:.6f}" for v in rot_to_mjquat(R_new_root)))

    worldbody.remove(worldbody.find("body"))
    wrapper = ET.SubElement(worldbody, "body", {"name": f"root_{side}", "pos": "0 0 0"})
    ET.SubElement(wrapper, "freejoint", {"name": f"root_{side}"})
    for s in preserved_sites: wrapper.append(s)
    for g in preserved_geoms: wrapper.append(g)
    wrapper.append(radius_body)

    removed_count = 0
    for parent in root.iter():
        for child in list(parent):
            if any(v in removed for v in child.attrib.values()):
                parent.remove(child)
                removed_count += 1
    print(f"{side}: stripped {removed_count} dangling references")

    out_path = src.parent / f"myohand_{side}_final.xml"
    tree.write(out_path)
    print(f"wrote {out_path}, preserved {len(preserved_sites)} sites, {len(preserved_geoms)} geoms")
