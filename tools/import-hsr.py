"""Copy unchanged HSR assets and extract the official link frames for the browser model.

Usage: python3 tools/import-hsr.py /path/to/hsr_description /path/to/hsr_meshes
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

description, meshes = map(Path, sys.argv[1:])
root = Path(__file__).resolve().parents[1]
destination = root / "public/assets/hsr-original-v1"
urdf = description / "robots/hsrb4s.obj.urdf"
robot = ET.parse(urdf).getroot()
names = ["torso_lift_link", "head_pan_link", "head_tilt_link", "arm_lift_link", "arm_flex_link", "arm_roll_link",
         "wrist_flex_link", "wrist_roll_link", "hand_palm_link", "hand_motor_dummy_link"]
for side in ["l", "r"]:
    names += [f"hand_{side}_{part}_link" for part in ["proximal", "spring_proximal", "mimic_distal", "distal"]]

def copy_asset(path):
    relative = path.relative_to(meshes)
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
    if path.suffix == ".obj":
        for line in path.read_text().splitlines():
            if line.startswith("mtllib "):
                copy_asset(path.parent / line[7:].strip())
    if path.suffix == ".mtl":
        for line in path.read_text().splitlines():
            if line.startswith("map_"):
                copy_asset(path.parent / line.split()[-1])

def origin(element):
    value = element.find("origin")
    return {key: [float(v) for v in (value.get(key, "0 0 0") if value is not None else "0 0 0").split()]
            for key in ["xyz", "rpy"]}

def geometry(element, collision=False):
    shape = element.find("geometry")[0]
    result = {"type": shape.tag, **origin(element)}
    if shape.tag == "mesh":
        path = meshes / shape.get("filename").split("package://hsr_meshes/")[1]
        copy_asset(path)
        result["path"] = str(path.relative_to(meshes / "meshes")).removesuffix(".obj")
        if collision:
            result["vertices"] = " ".join(" ".join(line.split()[1:4]) for line in path.read_text().splitlines() if line.startswith("v "))
    else:
        result.update({key: [float(v) for v in value.split()] for key, value in shape.attrib.items()})
    return result

links = []
for name in names:
    link = robot.find(f"link[@name='{name}']")
    joint = next(j for j in robot.findall("joint") if j.find("child").get("link") == name)
    entry = {"name": name, "parent": joint.find("parent").get("link"), "joint": {"name": joint.get("name"), "type": joint.get("type"), **origin(joint)},
             "visuals": [geometry(v) for v in link.findall("visual")], "collisions": [geometry(v, True) for v in link.findall("collision")]}
    for tag in ["axis", "limit", "mimic"]:
        value = joint.find(tag)
        if value is not None:
            entry["joint"][tag] = dict(value.attrib)
    inertial = link.find("inertial")
    if inertial is not None:
        entry["inertial"] = {"mass": float(inertial.find("mass").get("value")), **origin(inertial),
                             "inertia": [float(inertial.find("inertia").get(key)) for key in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]]}
    links.append(entry)
(root / "src/robot/hsr-description.json").write_text(json.dumps({"links": links}, indent=2) + "\n")
shutil.copyfile(urdf, destination / "hsrb4s.obj.urdf")
shutil.copyfile(description / "LICENSE.txt", destination / "DESCRIPTION-LICENSE.txt")
source = json.loads((destination / "sources.json").read_text())
source["description"] = {"repository": "https://github.com/ToyotaResearchInstitute/hsr_description", "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=description, text=True).strip(), "license": "BSD-3-Clause-Clear"}
source["modifications"] = "Original OBJ, MTL and textures copied byte-for-byte. Collision vertices and URDF frames are extracted without mesh edits; runtime rigid transforms assemble the original links."
source["files"] = [{"path": str(p.relative_to(destination)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                   for p in sorted(destination.rglob("*")) if p.is_file() and p.name not in ["sources.json", "ATTRIBUTION.html"]]
(destination / "sources.json").write_text(json.dumps(source, indent=2) + "\n")
print(f"Imported {len(links)} HSR links and {len(source['files'])} original asset files")
