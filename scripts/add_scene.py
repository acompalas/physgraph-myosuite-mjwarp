"""Merge scene elements (floor, background mesh, logo, cameras, light)
from myohands.xml into the finished hand-only asset(s). These elements
reference no bodies we removed, so they can be copied wholesale."""
import xml.etree.ElementTree as ET
from pathlib import Path

SCENE_SOURCE = Path.home() / "physgraph-local/physgraph-myosuite-mjwarp/assets/hands/myohands.xml"
scene_tree = ET.parse(SCENE_SOURCE)
scene_root = scene_tree.getroot()
scene_asset = scene_root.find("asset")
scene_worldbody = scene_root.find("worldbody")

SCENE_ASSET_NAMES = {
    "textscene", "texfloor", "textlogo",
    "matscene", "matfloor", "matlogo",
    "meshscene", "logo",
}

for path in [
    Path.home() / "physgraph-local/physgraph-myosuite-mjwarp/assets/hands/myohand_r_ulnaroot.xml",
    Path.home() / "physgraph-local/physgraph-myosuite-mjwarp/assets/hands/myohand_l_ulnaroot.xml",
]:
    if not path.exists():
        print(f"skipping {path}, not found")
        continue
    tree = ET.parse(path)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    added_assets = 0
    for elem in list(scene_asset):
        if elem.get("name") in SCENE_ASSET_NAMES:
            asset.append(elem)
            added_assets += 1

    added_scene_elems = 0
    for elem in list(scene_worldbody):
        tag = elem.tag
        if tag in ("geom", "camera", "light"):
            worldbody.insert(0, elem)
            added_scene_elems += 1

    out_path = path.parent / (path.stem + "_scene.xml")
    tree.write(out_path)
    print(f"wrote {out_path}: +{added_assets} assets, +{added_scene_elems} worldbody elements")
