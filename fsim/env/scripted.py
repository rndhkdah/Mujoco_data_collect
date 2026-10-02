"""Scripted arm motions through the env's OSC_POSE actions (tests, demos, a future one_leg expert).

Actions follow the teammate's SETUP P-law: position action = clip(error * pos_kp, -1, 1) and
rotation action = clip(world rotation-vector error * rot_kp, -1, 1); robosuite's OSC_POSE scales
them by 0.05 m and 0.5 rad per step.
"""
from __future__ import annotations

import numpy as np

from fsim.env.ik import _rot_error


class ScriptedArm:
    def __init__(self, env, pos_kp: float = 12.0, rot_kp: float = 2.5):
        self.env = env
        self.pos_kp, self.rot_kp = pos_kp, rot_kp
        self.gripper = -1.0
        self.last = (None, 0.0, False, {})

    @property
    def eef_pos(self) -> np.ndarray:
        return np.array(self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id])

    @property
    def eef_R(self) -> np.ndarray:
        return np.array(self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id]).reshape(3, 3)

    def action_toward(self, pos, R) -> np.ndarray:
        a = np.zeros(7)
        a[:3] = np.clip((np.asarray(pos) - self.eef_pos) * self.pos_kp, -1.0, 1.0)
        a[3:6] = np.clip(_rot_error(self.eef_R, R) * self.rot_kp, -1.0, 1.0)
        a[6] = self.gripper
        return a

    def step(self, action) -> tuple:
        self.last = self.env.step(action)
        return self.last

    def move_to(self, pos, R, tol: float = 0.003, rot_tol: float = 0.03, max_steps: int = 200) -> bool:
        for _ in range(max_steps):
            if np.linalg.norm(np.asarray(pos) - self.eef_pos) < tol and np.linalg.norm(_rot_error(self.eef_R, R)) < rot_tol:
                return True
            self.step(self.action_toward(pos, R))
        return False

    def hold(self, steps: int, pos=None, R=None):
        pos = self.eef_pos if pos is None else pos
        R = self.eef_R if R is None else R
        for _ in range(steps):
            self.step(self.action_toward(pos, R))

    def grip(self, close: bool, steps: int = 15):
        self.gripper = 1.0 if close else -1.0
        self.hold(steps)
