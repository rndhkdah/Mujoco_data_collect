"""Standalone MJCF scene (table, fence, cameras, parts, screw constraints) for tests and spikes.

The robosuite arena (fsim/env, Linux) reuses the part/equality builders from fsim.models.parts.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from fsim.geometry import fmt, look_at_xyaxes
from fsim.models import names
from fsim.models.part_meshes import build_task_meshes
from fsim.models.parts import CONSTRAINT_SOLIMP, CONSTRAINT_SOLREF, build_part_body, build_screw_equalities, env_contact_attrs
from fsim.spec.types import Placement, TaskSpec

HAND = "hand"
PIN = "pin"


def grasp_eq(part: str) -> str:
    return f"{HAND}_grasp_{part}"


def pin_eq(part: str) -> str:
    return f"{PIN}_{part}"


def _sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items() if v is not None})


def build_scene_xml(task: TaskSpec, placements: tuple[Placement, ...] | None = None,
                    visual_meshes: bool = True, mocap_hand: bool = False, timestep: float = 0.002) -> str:
    """Return the MJCF string of the task scene.

    placements: initial part poses in W (defaults to the task layout).
    mocap_hand: add a mocap body plus one inactive soft weld per part, used by tests and the
                spike to emulate a firm grasp without a robot, and a second mocap body with one
                inactive stiff weld per part to pin a part in place (Rig.hold).
    """
    scene = task.scene
    W = np.asarray(scene.W_origin)
    root = ET.Element("mujoco", {"model": f"fsim_{task.name}"})
    _sub(root, "compiler", angle="radian", autolimits="true")
    opt = _sub(root, "option", timestep=fmt([timestep]), integrator="implicitfast", cone="elliptic", impratio="10")
    _sub(opt, "flag", multiccd="enable")
    vis = _sub(root, "visual")
    _sub(vis, "global", offwidth="1280", offheight="960")
    _sub(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.5 0.5 0.5")

    asset = _sub(root, "asset")
    _sub(asset, "texture", name="grid", type="2d", builtin="checker", rgb1=".2 .3 .4", rgb2=".1 .15 .2", width="512", height="512")
    _sub(asset, "material", name="grid", texture="grid", texrepeat="8 8", reflectance="0.1")
    mesh_assets = {}
    if visual_meshes:
        for part, path in build_task_meshes(task).items():
            mesh_assets[part] = f"{part}_mesh"
            _sub(asset, "mesh", name=mesh_assets[part], file=path.replace("\\", "/"))

    world = _sub(root, "worldbody")
    _sub(world, "light", pos="0 0 2.0", dir="0 0 -1", directional="true", diffuse="0.8 0.8 0.8")
    _sub(world, "geom", name="floor", type="plane", size="2 2 0.1", material="grid")
    tx, ty, tz = scene.table_size
    tc = scene.table_center
    table = _sub(world, "body", name="table", pos=fmt(tc))
    _sub(table, "geom", name="table_top", type="box", size=fmt((tx / 2, ty / 2, tz / 2)),
         rgba="0.15 0.15 0.15 1", **env_contact_attrs(task.part_friction, scene.table_ref_friction))
    for ob in scene.obstacles:
        _sub(world, "geom", name=ob.name, type="box", pos=fmt(W + np.asarray(ob.center)), size=fmt(ob.half_size),
             rgba="0.55 0.55 0.6 1", **env_contact_attrs(task.part_friction, scene.obstacle_ref_friction))
    for cam in scene.cameras:
        _sub(world, "camera", name=cam.name, pos=fmt(cam.pos), xyaxes=fmt(look_at_xyaxes(cam.pos, cam.target)),
             fovy=fmt([cam.fovy]))
    for pl in placements or task.layout:
        world.append(build_part_body(task, task.part(pl.part), W + np.asarray(pl.pos), pl.quat, mesh_assets))
    if mocap_hand:
        _sub(world, "body", name=HAND, mocap="true", pos="0 0 1.0")
        _sub(world, "body", name=PIN, mocap="true", pos="0 0 1.0")

    eq = _sub(root, "equality")
    for e in build_screw_equalities(task):
        eq.append(e)
    if mocap_hand:
        for p in task.parts:
            _sub(eq, "weld", name=grasp_eq(p.name), body1=HAND, body2=names.body(p.name),
                 relpose="0 0 0 1 0 0 0", active="false", solref="0.02 1", solimp=CONSTRAINT_SOLIMP)
        for p in task.parts:
            _sub(eq, "weld", name=pin_eq(p.name), body1=PIN, body2=names.body(p.name),
                 relpose="0 0 0 1 0 0 0", active="false", solref=CONSTRAINT_SOLREF, solimp=CONSTRAINT_SOLIMP)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def build_model(task: TaskSpec, **kwargs):
    import mujoco
    return mujoco.MjModel.from_xml_string(build_scene_xml(task, **kwargs))
