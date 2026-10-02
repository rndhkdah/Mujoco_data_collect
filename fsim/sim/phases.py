"""Automatic phase (skill) labels.

A phase completes when its predicate fires; `update` returns True on that step, which is what
the HDF5 `skills` dataset records. Predicates that need the robot (is a part grasped?) use the
`grasped` callback supplied by the env; without it they fall back to motion proxies.

Phase names: grasp_top, place_top_corner, then per leg pick_*, insert_*, screw_* (one_leg uses
the target leg named in task.extras["target_leg"] for pick_leg / insert_leg / screw_leg).
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from fsim.geometry import yaw_of
from fsim.sim.screw import ENGAGED, SEATED
from fsim.spec.types import TaskSpec

LIFT_HEIGHT = 0.03
MOVE_DIST = 0.01


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
        return None if self.done else self.task.phases[self.phase]

    def _leg_for(self, name: str) -> str:
        if name.endswith("_leg"):
            return self.task.extras["target_leg"]
        suffix = name.rsplit("_", 1)[-1]                      # e.g. "pick_leg4" -> "leg4"
        matches = [p for p in self.task.part_names if p.endswith(suffix)]
        if len(matches) != 1:
            raise KeyError(f"phase {name!r} does not name exactly one part")
        return matches[0]

    def _held(self, part: str) -> bool | None:
        return None if self.grasped is None else bool(self.grasped(part))

    def _check(self, name: str, poses_W, screw, assembled: set) -> bool:
        x = self.task.extras
        top = self.task.screws[0].parent
        if name == "grasp_top":
            held = self._held(top)
            if held is not None:
                return held
            return np.linalg.norm(poses_W[top][:2, 3] - self.initial[top][:2, 3]) > MOVE_DIST
        if name == "place_top_corner":
            T = poses_W[top]
            yaw_err = math.degrees(abs((yaw_of(T[:3, :3]) + math.pi / 4) % (math.pi / 2) - math.pi / 4))
            near = np.linalg.norm(T[:2, 3] - np.asarray(x["corner_xy"])) < x["corner_tol"]
            return near and yaw_err < x["corner_yaw_tol_deg"] and self._held(top) is not True
        kind = name.split("_", 1)[0]
        if kind not in ("pick", "insert", "screw"):
            raise KeyError(f"no predicate for phase {name!r}")
        leg = self._leg_for(name)
        state = screw.child_state(leg)
        if kind == "pick":
            lifted = poses_W[leg][2, 3] - self.initial[leg][2, 3] > LIFT_HEIGHT
            return lifted and self._held(leg) is not False
        if kind == "insert":
            # The design's insert skill ends when the leg is released standing in a hole.
            if state in (ENGAGED, SEATED):
                return True
            return screw.resting_hole(leg) is not None and self._held(leg) is not True
        return state == SEATED or (top, leg) in assembled

    def update(self, poses_W: dict[str, np.ndarray], screw, assembled: set) -> bool:
        if self.done:
            return False
        if self._check(self.task.phases[self.phase], poses_W, screw, assembled):
            self.phase += 1
            return True
        return False
