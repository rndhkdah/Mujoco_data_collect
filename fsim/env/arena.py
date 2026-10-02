"""Furniture arena: robosuite's TableArena sized and placed like the design, plus the U-fence and cameras."""
from __future__ import annotations

import numpy as np
from robosuite.models.arenas import TableArena
from robosuite.utils.mjcf_utils import new_element

from fsim.geometry import fmt, look_at_xyaxes, mat_to_quat
from fsim.models.parts import env_contact_attrs
from fsim.spec.types import TaskSpec


def camera_quat(pos, target) -> np.ndarray:
    """MuJoCo camera quaternion (w, x, y, z) looking from pos at target (cameras look along -z)."""
    ax = look_at_xyaxes(pos, target)
    x, y = ax[:3], ax[3:]
    return mat_to_quat(np.column_stack([x, y, np.cross(x, y)]))


class FurnitureArena(TableArena):
    def __init__(self, task: TaskSpec):
        scene = task.scene
        mu = env_contact_attrs(task.part_friction, scene.table_ref_friction)
        super().__init__(table_full_size=scene.table_size,
                         table_friction=tuple(float(v) for v in mu["friction"].split()),
                         table_offset=(0.0, 0.0, scene.table_surface_z), has_legs=True)
        self.table_collision.set("priority", str(mu["priority"]))
        self.table_visual.set("rgba", "0.15 0.15 0.15 1")
        self.table_visual.attrib.pop("material", None)
        W = np.asarray(scene.W_origin)
        fence = env_contact_attrs(task.part_friction, scene.obstacle_ref_friction)
        for ob in scene.obstacles:
            pos, size = fmt(W + np.asarray(ob.center)), fmt(ob.half_size)
            self.worldbody.append(new_element(
                "geom", name=ob.name, type="box", pos=pos, size=size, group="0", rgba="0.55 0.55 0.6 1",
                friction=fence["friction"], priority=str(fence["priority"])))
            self.worldbody.append(new_element(
                "geom", name=f"{ob.name}_visual", type="box", pos=pos, size=size, group="1",
                contype="0", conaffinity="0", rgba="0.55 0.55 0.6 1"))
        for cam in scene.cameras:
            self.set_camera(cam.name, pos=list(cam.pos), quat=list(camera_quat(cam.pos, cam.target)),
                            camera_attribs={"fovy": f"{cam.fovy:g}"})
