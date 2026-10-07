"""Automatic phase (skill) labels.

A phase completes when its predicate fires; `update` returns True on that step, which is what
the HDF5 `skills` dataset records. Predicates that need the robot (is a part grasped?) use the
`grasped` callback supplied by the env; without it they fall back to motion proxies.

Each task phase is a fsim.spec.types.Phase(kind, part, target, label); the kinds are evaluated
generically here (see Phase for their predicates).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from fsim.geometry import yaw_of
from fsim.sim.screw import ENGAGED, FREE, SEATED
from fsim.spec.types import Phase, TaskSpec

LIFT_HEIGHT = 0.03
MOVE_DIST = 0.01


class _NoInserts:
    def child_state(self, child, parent=None):
        return FREE


class PhaseTracker:
    def __init__(self, task: TaskSpec, grasped: Callable[[str], bool] | None = None):
        self.task = task
        self.grasped = grasped
        self.phase = 0
        self.initial: dict[str, np.ndarray] = {}

    def reset(self, poses_W: dict[str, np.ndarray]):
        self.phase = 0
        self.initial = {k: v.copy() for k, v in poses_W.items()}

    @property
    def done(self) -> bool:
        return self.phase >= len(self.task.phases)

    @property
    def current(self) -> str | None:
        return None if self.done else self.task.phases[self.phase].name

    def _held(self, part: str) -> bool | None:
        return None if self.grasped is None else bool(self.grasped(part))

    def _check(self, ph: Phase, poses_W, screw, assembled: set, inserts) -> bool:
        x, part, target, kind = self.task.extras, ph.part, ph.target, ph.kind
        if kind == "grasp":
            held = self._held(part)
            if held is not None:
                return held
            return np.linalg.norm(poses_W[part][:2, 3] - self.initial[part][:2, 3]) > MOVE_DIST
        if kind == "place_corner":
            T = poses_W[part]
            near = np.linalg.norm(T[:2, 3] - np.asarray(x["corner_xy"])) < x["corner_tol"]
            if ph.yaw_sym_deg is None:
                yaw_ok = True
            elif ph.yaw_sym_deg == 90.0:             # the original expression, kept bit-for-bit
                yaw_ok = math.degrees(abs((yaw_of(T[:3, :3]) + math.pi / 4) % (math.pi / 2) - math.pi / 4)) < x["corner_yaw_tol_deg"]
            else:
                per = math.radians(ph.yaw_sym_deg)
                yaw_ok = math.degrees(abs((yaw_of(T[:3, :3]) + per / 2) % per - per / 2)) < x["corner_yaw_tol_deg"]
            return near and yaw_ok and self._held(part) is not True
        if kind == "pick":
            lifted = poses_W[part][2, 3] - self.initial[part][2, 3] > LIFT_HEIGHT
            return lifted and self._held(part) is not False
        if kind == "place":
            return (target, part) in assembled
        if kind == "place_pose":
            T = poses_W[part]
            near = ph.pos is None or np.linalg.norm(T[:3, 3] - np.asarray(ph.pos)) < ph.pos_tol
            c = float(np.dot(T[:3, :3] @ np.asarray(ph.axis, float), ph.axis_dir) / np.linalg.norm(ph.axis_dir))
            return near and c >= math.cos(math.radians(ph.axis_tol_deg)) and self._held(part) is not True
        screwed = screw.child_state(part)                    # a part is a screw child of one parent at most
        slid = inserts.child_state(part, target)
        if kind == "insert":
            # The design's insert skill ends when the part is released standing in a hole.
            if screwed in (ENGAGED, SEATED) or slid in (ENGAGED, SEATED):
                return True
            return screw.resting_hole(part, target) is not None and self._held(part) is not True
        if kind == "screw":
            return screwed == SEATED or (target, part) in assembled
        if kind == "slide":
            return slid == SEATED or (target, part) in assembled
        raise KeyError(f"no predicate for phase kind {kind!r}")

    def update(self, poses_W: dict[str, np.ndarray], screw, assembled: set, inserts=None) -> bool:
        if self.done:
            return False
        if self._check(self.task.phases[self.phase], poses_W, screw, assembled, inserts or _NoInserts()):
            self.phase += 1
            return True
        return False
