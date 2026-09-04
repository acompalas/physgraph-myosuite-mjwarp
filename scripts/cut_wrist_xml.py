"""Cut MyoHand at the wrist via direct XML surgery (mujoco's MjSpec
attach_frame/to_frame API segfaults on this model, so avoiding it).
Extracts the radius_r/radius_l subtree, discards the rigid
clavicle/scapula/humerus/ulna chain above it, wraps the extracted
subtree in a new root body carrying a freejoint (radius already has
its own pro_sup joint, and a body can't have both a freejoint and a
regular joint, so the freejoint goes on a new wrapper body instead),
and cleans up any <contact>/<tendon>/<sensor> references and unused
<mesh> assets left dangling by the removed bodies."""
import xml.etree.ElementTree as ET
from pathlib import Path

REMOVED_BODIES_R = {
    "myoarm_r_root", "clavicle_r", "clavphant_r", "scapula_r",
    "scapphant_r", "humphant_r", "humphant1_r", "humerus_r", "ulna_r",
}
REMOVED_BODIES_L = {b.replace("_r", "_l") for b in REMOVED_BODIES_R}

for side, cut_name, removed in [
    ("r", "radius_r", REMOVED_BODIES_R),
    ("l", "radius_l", REMOVED_BODIES_L),
]:
    src = Path(__file__).resolve().parents[1] / f"assets/hands/myohand_{side}_free.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    cut_body = None
    for body in worldbody.iter("body"):
        if body.get("name") == cut_name:
            cut_body = body
            break
    if cut_body is None:
        raise RuntimeError(f"{cut_name} not found")

    old_chain = worldbody.find("body")
    worldbody.remove(old_chain)

    wrapper = ET.SubElement(worldbody, "body", {"name": f"root_{side}", "pos": "0 0 0"})
    ET.SubElement(wrapper, "freejoint", {"name": f"root_{side}"})
    wrapper.append(cut_body)

    # strip any element anywhere in the tree that references a removed body
    # by name (contact excludes/pairs, tendon wraps/sites, sensors, etc.)
    removed_count = 0
    for parent in root.iter():
        for child in list(parent):
            attrs = child.attrib
            if any(v in removed for v in attrs.values()):
                parent.remove(child)
                removed_count += 1
    print(f"{side}: stripped {removed_count} dangling references to removed bodies")

    # drop now-unused mesh assets belonging to removed bodies
    asset = root.find("asset")
    remaining_geom_meshes = {g.get("mesh") for g in root.iter("geom") if g.get("mesh")}
    mesh_removed = 0
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name") in removed and mesh.get("name") not in remaining_geom_meshes:
                asset.remove(mesh)
                mesh_removed += 1
    print(f"{side}: removed {mesh_removed} unused mesh assets")

    out_path = src.parent / f"myohand_{side}_wristcut.xml"
    tree.write(out_path)
    print(f"wrote {out_path}")
