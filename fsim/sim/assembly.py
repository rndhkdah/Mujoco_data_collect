"""Pose-based assembly success (FurnitureBench design: per-axis position and column-cosine tests)."""
from __future__ import annotations

import numpy as np

from fsim.geometry import inv_T, pose_to_T
from fsim.spec.types import AssemblyTarget, TaskSpec


def matches(rel: np.ndarray, target: AssemblyTarget) -> bool:
    """True if child-in-parent pose `rel` (4x4) matches any of the target's rel poses."""
    for pos, quat in target.rel_poses:
        ref = pose_to_T(pos, quat)
        if np.any(np.abs(rel[:3, 3] - ref[:3, 3]) > np.asarray(target.pos_tol)):
            continue
        if target.position_only:
            return True
        cos = np.einsum("ij,ij->j", rel[:3, :3], ref[:3, :3])   # columns are unit vectors
        if np.all(cos >= target.ori_cos):
            return True
    return False


class AssemblyChecker:
    """Latching assembled-pair bookkeeping; reward = number of newly assembled pairs."""

    def __init__(self, task: TaskSpec):
        self.task = task
        self.assembled: set[tuple[str, str]] = set()

    def reset(self):
        self.assembled.clear()

    def is_assembled(self, poses: dict[str, np.ndarray], target: AssemblyTarget) -> bool:
        if any(req not in self.assembled for req in target.requires):
            return False
        rel = inv_T(poses[target.parent]) @ poses[target.child]
        return matches(rel, target)

    def update(self, poses: dict[str, np.ndarray]) -> int:
        new = 0
        for t in self.task.targets:
            key = (t.parent, t.child)
            if key not in self.assembled and self.is_assembled(poses, t):
                self.assembled.add(key)
                new += 1
        return new

    @property
    def all_assembled(self) -> bool:
        return len(self.assembled) == len(self.task.targets)
