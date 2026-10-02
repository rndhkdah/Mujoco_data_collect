"""Initial layouts: the task layout, randomized (low / med) by rejection sampling, or a high preset."""
from __future__ import annotations

import math

import mujoco
import numpy as np

from fsim.geometry import make_T, mat_to_quat, quat_to_mat, rot_z
from fsim.spec.types import Placement, TaskSpec

LEVELS = ("none", "low", "med", "high")


def rotated_aabb(cx: float, cy: float, lx: float, ly: float, yaw: float) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of an lx-by-ly rectangle centred at (cx, cy) rotated by yaw."""
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    hx, hy = (c * lx + s * ly) / 2.0, (s * lx + c * ly) / 2.0
    return cx - hx, cx + hx, cy - hy, cy + hy


def aabbs_collide(a, b, clearance: float) -> bool:
    """Design rule: two boxes collide unless they are more than `clearance` apart along x or y."""
    if a[0] > b[1] + clearance or a[1] < b[0] - clearance:
        return False
    if a[2] > b[3] + clearance or a[3] < b[2] - clearance:
        return False
    return True


class ResetSampler:
    def __init__(self, task: TaskSpec):
        self.task = task
        self.spec = task.randomization
        self.obstacles = [rotated_aabb(o.center[0], o.center[1], 2 * o.half_size[0], 2 * o.half_size[1], 0.0)
                          for o in self.spec.reset_obstacles]

    def valid(self, placements, yaw_offsets) -> bool:
        s = self.spec
        boxes = []
        for pl, dyaw in zip(placements, yaw_offsets):
            if not (s.bounds_x[0] <= pl.pos[0] <= s.bounds_x[1] and s.bounds_y[0] <= pl.pos[1] <= s.bounds_y[1]):
                return False
            lx, ly = s.footprints[pl.part]
            boxes.append(rotated_aabb(pl.pos[0], pl.pos[1], lx, ly, dyaw))
        for i, a in enumerate(boxes):
            if any(aabbs_collide(a, b, s.clearance) for b in boxes[i + 1:]):
                return False
            if any(aabbs_collide(a, ob, s.clearance) for ob in self.obstacles):
                return False
        return True

    def _valid_batch(self, xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
        """Vectorised `valid` for candidate layouts xy (B, P, 2) and yaw offsets (B, P)."""
        s = self.spec
        ok = ((xy[..., 0] >= s.bounds_x[0]) & (xy[..., 0] <= s.bounds_x[1])
              & (xy[..., 1] >= s.bounds_y[0]) & (xy[..., 1] <= s.bounds_y[1])).all(axis=1)
        lx = np.array([s.footprints[pl.part][0] for pl in self.task.layout])
        ly = np.array([s.footprints[pl.part][1] for pl in self.task.layout])
        c, sn = np.abs(np.cos(yaw)), np.abs(np.sin(yaw))
        hx, hy = (c * lx + sn * ly) / 2.0, (sn * lx + c * ly) / 2.0
        box = np.stack([xy[..., 0] - hx, xy[..., 0] + hx, xy[..., 1] - hy, xy[..., 1] + hy], axis=-1)   # (B, P, 4)

        def collide(a, b):
            sep_x = (a[..., 0] > b[..., 1] + s.clearance) | (a[..., 1] < b[..., 0] - s.clearance)
            sep_y = (a[..., 2] > b[..., 3] + s.clearance) | (a[..., 3] < b[..., 2] - s.clearance)
            return ~(sep_x | sep_y)

        n = box.shape[1]
        for i in range(n):
            for j in range(i + 1, n):
                ok &= ~collide(box[:, i], box[:, j])
            for ob in self.obstacles:
                ok &= ~collide(box[:, i], np.asarray(ob))
        return ok

    def sample(self, rng: np.random.Generator, level: str = "low", high_index: int | None = None,
               max_tries: int = 300_000, batch: int = 4096) -> tuple[Placement, ...]:
        """Joint rejection sampling (the design's distribution), evaluated in numpy batches."""
        if level == "none":
            return self.task.layout
        if level == "high":
            presets = self.spec.high_presets
            idx = int(rng.integers(len(presets))) if high_index is None else high_index
            return presets[idx]
        lv = self.spec.levels[level]
        base = np.array([pl.pos[:2] for pl in self.task.layout])
        n = len(self.task.layout)
        for _ in range(0, max_tries, batch):
            xy = base + rng.uniform(-lv.pos_range, lv.pos_range, size=(batch, n, 2))
            yaw = np.radians(rng.uniform(-lv.yaw_range_deg, lv.yaw_range_deg, size=(batch, n)))
            hits = np.flatnonzero(self._valid_batch(xy, yaw))
            if hits.size:
                k = hits[0]
                return tuple(
                    Placement(pl.part, (float(xy[k, i, 0]), float(xy[k, i, 1]), pl.pos[2]),
                              tuple(mat_to_quat(rot_z(float(yaw[k, i])) @ quat_to_mat(pl.quat))))
                    for i, pl in enumerate(self.task.layout))
        raise RuntimeError(f"no valid {level} layout after {max_tries} tries")

    @staticmethod
    def apply(state, placements) -> None:
        """Write placements (W frame) into a PartState's free joints; call mj_forward afterwards."""
        for pl in placements:
            state.set_pose_W(pl.part, make_T(pl.pos, quat_to_mat(pl.quat)))


def max_penetration(model: mujoco.MjModel, data: mujoco.MjData, bodies) -> float:
    """Deepest contact penetration (m, positive) involving any of `bodies` (after a forward pass)."""
    bodies = set(bodies)
    deepest = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        if model.geom_bodyid[c.geom1] in bodies or model.geom_bodyid[c.geom2] in bodies:
            deepest = max(deepest, -c.dist)
    return deepest


def place_and_settle(model: mujoco.MjModel, data: mujoco.MjData, state, placements, settle_steps: int = 150,
                     tol: float = 0.0005, max_iters: int = 30, after_step=None) -> float:
    """Place parts, push apart anything that interpenetrates, then settle.

    Measured layouts can overlap slightly (with the table, the fence or each other). A part that
    penetrates static geometry (table, fence) is moved out along the contact normal; of two
    overlapping parts the upper one is moved. Returns the remaining max penetration before settling.
    """
    ResetSampler.apply(state, placements)
    mujoco.mj_forward(model, data)
    part_body = {b: p for p, b in state.body_id.items()}
    for _ in range(max_iters):
        shift: dict[str, np.ndarray] = {}
        for i in range(data.ncon):
            c = data.contact[i]
            if -c.dist <= tol:
                continue
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            n = np.asarray(c.frame[:3])                         # normal from geom1 toward geom2
            if b1 in part_body and b2 in part_body:
                b = max((b1, b2), key=lambda bb: data.xpos[bb][2])
            elif b1 in part_body or b2 in part_body:
                b = b1 if b1 in part_body else b2
            else:
                continue
            direction = n if b == b2 else -n
            p = part_body[b]
            step = direction * (-c.dist + tol)
            prev = shift.get(p)
            shift[p] = step if prev is None or np.linalg.norm(step) > np.linalg.norm(prev) else prev
        if not shift:
            break
        for p, dv in shift.items():
            T = state.pose(p)
            T[:3, 3] += dv
            state.set_pose(p, T)
        mujoco.mj_forward(model, data)
    remaining = max_penetration(model, data, state.body_id.values())
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
        if after_step is not None:
            after_step()            # e.g. hold an unactuated robot in place while the parts settle
    return remaining
