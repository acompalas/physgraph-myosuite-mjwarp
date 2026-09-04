"""Corrected MyoHand cut, matching MUSIC's (arXiv 2604.23886) documented
boundary: free root at ULNA (not radius), with radius kept as ulna's
child via the existing forearm-rotation joint. ulna_r is kept as a REAL
body with its own mass/inertia/geometry -- not discarded for a massless
synthetic wrapper. Only clavicle/scapula/humerus (and phantom bodies)
are removed; their tendon-referenced sites/geoms are preserved via
composed rigid transforms, converted to ulna_body-LOCAL coordinates
(since they become ulna_body's children, not world's), then reattached."""
import numpy as np
from scipy.spatial.transform import Rotation
import xml.etree.ElementTree as ET
from pathlib import Path

REMOVED_BODIES_R = [
    "myoarm_r_root", "clavicle_r", "clavphant_r", "scapula_r",
    "scapphant_r", "humphant_r", "humphant1_r", "humerus_r",
]
REMOVED_BODIES_L = [b.replace("_r", "_l") for b in REMOVED_BODIES_R]

def get_transform(elem):
    pos = np.array([float(v) for v in elem.get("pos", "0 0 0").split()])
    quat_str = elem.get("quat")
    if quat_str:
        vals = [float(v) for v in quat_str.split()]
        R = Rotation.from_quat(vals[1:] + [vals[0]])
    else:
        R = Rotation.identity()
    return R, pos

def compose(R1, p1, R2, p2):
    return R1 * R2, R1.apply(p2) + p1

def rot_to_mjquat(R):
    x, y, z, w = R.as_quat()
    return np.array([w, x, y, z])

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
    while node is not None and node.get("name") != f"ulna_{side}":
        chain.append(node)
        node = node.find("body")
    ulna_body = node

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
                preserved_sites.append((ET.Element("site", dict(site.attrib)), R_new, p_new))
        for geom in b.findall("geom"):
            if geom.get("name") in needed_names:
                R_g, p_g = get_transform(geom)
                R_new, p_new = compose(R_b, p_b, R_g, p_g)
                preserved_geoms.append((ET.Element("geom", dict(geom.attrib)), R_new, p_new))

    # ulna_body's own new world-equivalent transform (this becomes the root)
    R_ulna, p_ulna = get_transform(ulna_body)
    R_root, p_root = compose(R_accum, p_accum, R_ulna, p_ulna)
    ulna_body.set("pos", f"{p_root[0]:.6f} {p_root[1]:.6f} {p_root[2]:.6f}")
    ulna_body.set("quat", " ".join(f"{v:.6f}" for v in rot_to_mjquat(R_root)))

    # convert each preserved element from world-equivalent to ulna_body-LOCAL
    # (they attach as ulna_body's children, so need coords relative to it)
    R_root_inv = R_root.inv()
    for elem, R_world, p_world in preserved_sites + preserved_geoms:
        p_local = R_root_inv.apply(p_world - p_root)
        elem.set("pos", f"{p_local[0]:.6f} {p_local[1]:.6f} {p_local[2]:.6f}")
        if elem.get("quat"):
            R_local = R_root_inv * R_world
            elem.set("quat", " ".join(f"{v:.6f}" for v in rot_to_mjquat(R_local)))
        ulna_body.append(elem)

    ET.SubElement(ulna_body, "freejoint", {"name": f"root_{side}"})

    worldbody.remove(worldbody.find("body"))
    worldbody.append(ulna_body)

    removed_count = 0
    for parent in root.iter():
        for child in list(parent):
            if any(v in removed for v in child.attrib.values()):
                parent.remove(child)
                removed_count += 1
    print(f"{side}: stripped {removed_count} dangling references")

    out_path = src.parent / f"myohand_{side}_ulnaroot.xml"
    tree.write(out_path)
    print(f"wrote {out_path}, preserved {len(preserved_sites)} sites, {len(preserved_geoms)} geoms")
