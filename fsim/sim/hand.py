"""Scripted mocap "hand" that emulates a firm grasp in standalone scenes (tests, spikes).

The scene must be built with mocap_hand=True: one mocap body plus an inactive soft weld per
part. Grasping snaps the mocap to the part pose first, so activating the weld causes no jump.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from fsim.geometry import make_T, mat_to_quat, quat_to_mat, rot_z
from fsim.models.scene import HAND, grasp_eq


class ScriptedHand:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, task, on_substep=None, control_hz: float = 20.0):
        self.model, self.data, self.task = model, data, task
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, HAND)
        self.mocap = int(model.body_mocapid[body])
        self.eq = {p: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, grasp_eq(p)) for p in task.part_names}
        self.held: str | None = None
        self.on_substep = on_substep          # e.g. screw manager update, called every physics step
        self.substeps = max(1, int(round(1.0 / control_hz / model.opt.timestep)))
        self.steps = 0

    @property
    def T(self) -> np.ndarray:
        return make_T(self.data.mocap_pos[self.mocap], quat_to_mat(self.data.mocap_quat[self.mocap]))

    def set(self, T: np.ndarray):
        self.data.mocap_pos[self.mocap] = T[:3, 3]
        self.data.mocap_quat[self.mocap] = mat_to_quat(T[:3, :3])

    def grasp(self, part: str, T_part: np.ndarray):
        self.set(T_part)
        self.data.eq_active[self.eq[part]] = 1
        self.held = part

    def release(self):
        if self.held is not None:
            self.data.eq_active[self.eq[self.held]] = 0
        self.held = None

    def step(self, n_control: int = 1):
        """Advance n control steps (each = `substeps` physics steps)."""
        for _ in range(n_control):
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
                self.steps += 1
                if self.on_substep is not None:
                    self.on_substep()

    def move_to(self, T_goal: np.ndarray, duration: float, control_hz: float = 20.0):
        """Linear position / slerp-free (small) orientation blend to T_goal over `duration` seconds."""
        T0 = self.T
        n = max(1, int(round(duration * control_hz)))
        q0, q1 = mat_to_quat(T0[:3, :3]), mat_to_quat(T_goal[:3, :3])
        if np.dot(q0, q1) < 0:
            q1 = -q1
        for i in range(1, n + 1):
            a = i / n
            q = (1 - a) * q0 + a * q1
            T = make_T((1 - a) * T0[:3, 3] + a * T_goal[:3, 3], quat_to_mat(q / np.linalg.norm(q)))
            self.set(T)
            self.step()

    def twist_about(self, axis_point, rate_rad_s: float, control_hz: float = 20.0):
        """One control step: rotate the hand about the vertical axis through `axis_point`."""
        T = self.T
        dR = rot_z(rate_rad_s / control_hz)
        p = np.asarray(axis_point)
        T_new = make_T(p + dR @ (T[:3, 3] - p), dR @ T[:3, :3])
        self.set(T_new)
        self.step()
