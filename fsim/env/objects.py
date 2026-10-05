"""Furniture parts as robosuite MujocoXMLObjects.

Each part's MJCF is generated next to its visual mesh (robosuite resolves file= relative to the
XML it loads). The object XML uses local names (robosuite prepends "<part>_" itself, also in the
name lists it keeps), so the final names equal the standalone scenes' (fsim.models.names).
robosuite renames the root body "object" to "<part>_main" and adds the free joint "<part>_joint0".
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

from robosuite.models.objects import MujocoXMLObject

from fsim.fileutil import atomic_write_text
from fsim.geometry import fmt
from fsim.models.meshes import build_task_meshes
from fsim.models.parts import build_part_body
from fsim.spec.types import LegPartSpec, PartSpec, TaskSpec


def _half_extents(spec: PartSpec) -> tuple[float, float]:
    """(horizontal radius, half height) of the part in its own frame, for robosuite's sites."""
    if isinstance(spec, LegPartSpec):
        return spec.width / 2 ** 0.5, spec.body_length / 2 + spec.screw_length
    return (spec.size[0] ** 2 + spec.size[1] ** 2) ** 0.5 / 2, spec.size[2] / 2


_NAMED_ATTRS = ("name", "mesh", "material", "texture", "joint", "joint1", "joint2", "body1", "body2", "site")


def _localize(root: ET.Element, part: str) -> None:
    """Strip the "<part>_" prefix from every name/reference (robosuite adds it back)."""
    prefix = f"{part}_"
    for el in root.iter():
        for attr in _NAMED_ATTRS:
            v = el.get(attr)
            if v is not None and v.startswith(prefix):
                el.set(attr, v[len(prefix):])


def write_part_xml(task: TaskSpec, spec: PartSpec, mesh_path: str) -> str:
    """Write <mesh dir>/<part>.xml (robosuite object format) and return its path."""
    root = ET.Element("mujoco", {"model": spec.name})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    asset = ET.SubElement(root, "asset")
    mesh_name = f"{spec.name}_mesh"
    ET.SubElement(asset, "mesh", {"name": mesh_name, "file": os.path.basename(mesh_path)})
    world = ET.SubElement(root, "worldbody")
    outer = ET.SubElement(world, "body")
    outer.append(build_part_body(task, spec, (0, 0, 0), (1, 0, 0, 0), {spec.name: mesh_name},
                                 free_joint=False, body_name="object"))
    r, h = _half_extents(spec)
    for name, pos in (("bottom_site", (0, 0, -h)), ("top_site", (0, 0, h)), ("horizontal_radius_site", (r, 0, 0))):
        ET.SubElement(outer, "site", {"name": name, "pos": fmt(pos), "size": "0.002", "rgba": "0 0 0 0"})
    _localize(root, spec.name)
    ET.indent(root)
    path = os.path.join(os.path.dirname(mesh_path), f"{spec.name}.xml")
    atomic_write_text(path, ET.tostring(root, encoding="unicode"))     # several envs may start at once
    return path


class FurniturePartObject(MujocoXMLObject):
    def __init__(self, xml_path: str, name: str):
        super().__init__(xml_path, name=name, joints="default", obj_type="all", duplicate_collision_geoms=False)


def build_part_objects(task: TaskSpec, root: str | None = None) -> list[FurniturePartObject]:
    meshes = build_task_meshes(task, root)
    return [FurniturePartObject(write_part_xml(task, p, meshes[p.name]), p.name) for p in task.parts]
