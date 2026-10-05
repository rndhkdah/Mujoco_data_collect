"""Scripted arm motions through the env's OSC_POSE actions (tests, demos, a future one_leg expert).

Actions follow the teammate's SETUP P-law: position action = clip(error * pos_kp, -1, 1) and
rotation action = clip(world rotation-vector error * rot_kp, -1, 1); robosuite's OSC_POSE scales
them by 0.05 m and 0.5 rad per step.
"""
from __future__ import annotations

import numpy as np

from fsim.env.ik import _rot_error


def axis_angle(axis, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * K @ K


def slerp(R0: np.ndarray, R1: np.ndarray, t: float) -> np.ndarray:
    """Rotation a fraction t of the way from R0 to R1 (about the fixed world axis between them)."""
    v = _rot_error(R0, R1)
    a = float(np.linalg.norm(v))
    return R1.copy() if a < 1e-9 else axis_angle(v / a, a * t) @ R0


class ScriptedArm:
    def __init__(self, env, pos_kp: float = 12.0, rot_kp: float = 2.5, step_fn=None):
        """step_fn(action, target_pos, target_R) -> (obs, reward, done, info) replaces env.step,
        e.g. to record the episode (fsim.collect.Collector.record_scripted)."""
        self.env = env
        self.pos_kp, self.rot_kp = pos_kp, rot_kp
        self.step_fn = step_fn
        self.gripper = -1.0
        self.target = None                    # (pos, R) the last action was computed for
        self.last = (None, 0.0, False, {})

    @property
    def eef_pos(self) -> np.ndarray:
        return np.array(self.env.sim.data.site_xpos[self.env.robots[0].eef_site_id])

    @property
    def eef_R(self) -> np.ndarray:
        return np.array(self.env.sim.data.site_xmat[self.env.robots[0].eef_site_id]).reshape(3, 3)

    def action_toward(self, pos, R) -> np.ndarray:
        self.target = (np.asarray(pos, float), np.asarray(R, float))
        a = np.zeros(7)
        a[:3] = np.clip((np.asarray(pos) - self.eef_pos) * self.pos_kp, -1.0, 1.0)
        a[3:6] = np.clip(_rot_error(self.eef_R, R) * self.rot_kp, -1.0, 1.0)
        a[6] = self.gripper
        return a

    def step(self, action) -> tuple:
        if self.step_fn is None:
            self.last = self.env.step(action)
        else:
            pos, R = self.target if self.target is not None else (self.eef_pos, self.eef_R)
            self.last = self.step_fn(action, pos, R)
        return self.last

    def move_to(self, pos, R, tol: float = 0.003, rot_tol: float = 0.03, max_steps: int = 200, hold: int = 1) -> bool:
        """P-control toward the pose; True once it is within tolerance on `hold` consecutive steps
        (hold > 1 rejects an overshooting pass through the target)."""
        inside = 0
        for _ in range(max_steps):
            ok = np.linalg.norm(np.asarray(pos) - self.eef_pos) < tol and np.linalg.norm(_rot_error(self.eef_R, R)) < rot_tol
            inside = inside + 1 if ok else 0
            if inside >= hold:
                return True
            self.step(self.action_toward(pos, R))
        return False

    def move_linear(self, pos, R, speed: float = 0.15, ang_speed: float = 1.2, tol: float = 0.003,
                    rot_tol: float = 0.03, settle_steps: int = 100) -> bool:
        """Sweep the target along a straight line (and a slerp) at bounded speed, then converge
        (the OSC overshoots by a few mm after a fast move and needs about a second to settle)."""
        p0, R0 = self.eef_pos, self.eef_R
        pos = np.asarray(pos, float)
        span = max(np.linalg.norm(pos - p0) / speed, np.linalg.norm(_rot_error(R0, R)) / ang_speed)
        n = max(1, int(np.ceil(span * self.env.control_freq)))
        for k in range(1, n + 1):
            t = k / n
            self.step(self.action_toward(p0 + (pos - p0) * t, slerp(R0, R, t)))
        return self.move_to(pos, R, tol, rot_tol, settle_steps, hold=3)

    def hold(self, steps: int, pos=None, R=None):
        pos = self.eef_pos if pos is None else pos
        R = self.eef_R if R is None else R
        for _ in range(steps):
            self.step(self.action_toward(pos, R))

    def grip(self, close: bool, steps: int = 15):
        self.gripper = 1.0 if close else -1.0
        self.hold(steps)
