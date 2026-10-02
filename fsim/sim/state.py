"""Read and write part poses (free bodies)."""
from __future__ import annotations

import mujoco
import numpy as np

from fsim.geometry import make_T, mat_to_quat
from fsim.models import names
from fsim.spec.types import TaskSpec


class PartState:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, task: TaskSpec):
        self.model, self.data, self.task = model, data, task
        self.W = np.asarray(task.scene.W_origin)
        self.body_id = {p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names.body(p)) for p in task.part_names}
        missing = [p for p, i in self.body_id.items() if i < 0]
        if missing:
            raise KeyError(f"part bodies not found in model: {missing}")
        self.qpos_adr, self.dof_adr = {}, {}
        for p, b in self.body_id.items():
            j = model.body_jntadr[b]
            if j < 0 or model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"part {p} has no free joint")
            self.qpos_adr[p], self.dof_adr[p] = model.jnt_qposadr[j], model.jnt_dofadr[j]

    def pose(self, part: str) -> np.ndarray:
        """World pose (4x4) from the last forward pass."""
        b = self.body_id[part]
        return make_T(self.data.xpos[b], self.data.xmat[b].reshape(3, 3))

    def pose_W(self, part: str) -> np.ndarray:
        T = self.pose(part)
        T[:3, 3] -= self.W
        return T

    def poses(self) -> dict[str, np.ndarray]:
        return {p: self.pose(p) for p in self.task.part_names}

    def set_pose(self, part: str, T_world: np.ndarray, zero_velocity: bool = True):
        a = self.qpos_adr[part]
        self.data.qpos[a:a + 3] = T_world[:3, 3]
        self.data.qpos[a + 3:a + 7] = mat_to_quat(T_world[:3, :3])
        if zero_velocity:
            v = self.dof_adr[part]
            self.data.qvel[v:v + 6] = 0.0

    def set_pose_W(self, part: str, T_W: np.ndarray, zero_velocity: bool = True):
        T = T_W.copy()
        T[:3, 3] += self.W
        self.set_pose(part, T, zero_velocity)
