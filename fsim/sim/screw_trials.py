"""Scripted screw-in / screw-out trials with the mocap hand (used by tests and spikes/s1)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from fsim.geometry import inv_T, make_T, rot_x, rot_z
from fsim.models.scene import build_scene_xml
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.hand import ScriptedHand
from fsim.sim.screw import SEATED, HelicalScrewManager
from fsim.sim.state import PartState
from fsim.spec.types import TaskSpec


@dataclass
class TrialParams:
    hole: int = 0
    twist0_deg: float = 40.0        # initial twist relative to the seated orientation
    lateral_mm: float = 0.0         # initial offset from the hole axis
    lateral_dir_deg: float = 0.0
    tilt_deg: float = 0.0
    rate_deg_s: float = 120.0       # twist rate of the hand
    press_mm: float = 1.0           # hand target below the leg while screwing (push)
    tighten: bool = True            # False: twist the wrong way (must never engage)
    screw_timeout_s: float = 12.0
    unscrew: bool = True


@dataclass
class TrialResult:
    engaged: bool = False
    seated: bool = False
    assembled: bool = False
    seat_pos_err_mm: float = float("nan")
    seat_cos_min: float = float("nan")
    released_at_start: bool = False
    lifted_out: bool = False
    sim_time: float = 0.0
    wall_time: float = 0.0
    events: list = field(default_factory=list)

    @property
    def rtf(self) -> float:
        return self.sim_time / self.wall_time if self.wall_time > 0 else float("inf")


class Rig:
    """Standalone scene + managers for one task."""

    def __init__(self, task: TaskSpec, visual_meshes: bool = False):
        self.task = task
        self.model = mujoco.MjModel.from_xml_string(build_scene_xml(task, mocap_hand=True, visual_meshes=visual_meshes))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.state = PartState(self.model, self.data, task)
        self.screw = HelicalScrewManager(self.model, self.data, task)
        self.checker = AssemblyChecker(task)
        self.hand = ScriptedHand(self.model, self.data, task, on_substep=self._substep, control_hz=task.control_hz)

    def _substep(self):
        self.screw.update()

    def settle(self, seconds: float = 0.5):
        self.hand.step(int(round(seconds * self.task.control_hz)))


def _twist_in_place(rig: Rig, xy, z: float, dtheta: float):
    """One control step: hand at (xy, z), rotated by dtheta about the vertical axis."""
    Th = rig.hand.T
    Th[:3, :3] = rot_z(dtheta) @ Th[:3, :3]
    Th[:3, 3] = (xy[0], xy[1], z)
    rig.hand.set(Th)
    rig.hand.step()


def run_trial(task: TaskSpec, p: TrialParams, child: str | None = None) -> TrialResult:
    rig = Rig(task)
    child = child or task.extras["target_leg"]
    top = task.screws[0].parent
    screw_spec = task.screws[0]
    res = TrialResult()
    t_wall = time.perf_counter()
    rig.settle(0.3)

    hole = rig.screw.holes[p.hole]
    seat = rig.state.pose(top) @ hole.T_local
    lat = 0.001 * p.lateral_mm * np.array([math.cos(math.radians(p.lateral_dir_deg)), math.sin(math.radians(p.lateral_dir_deg))])
    R0 = rot_z(math.radians(p.twist0_deg)) @ rot_x(math.radians(p.tilt_deg))
    T_pre = seat @ make_T((lat[0], lat[1], screw_spec.travel + 0.004), R0)
    rig.state.set_pose(child, T_pre)
    mujoco.mj_forward(rig.model, rig.data)
    rig.hand.grasp(child, T_pre)
    # Lower onto the catch (4 mm free drop distance + press).
    rig.hand.move_to(T_pre @ make_T((0, 0, -0.004 - 0.001 * p.press_mm)), 0.5, task.control_hz)

    # The hand keeps its commanded xy (hole axis + lateral offset), follows the leg in z with a
    # small push, and twists in place, like an operator holding position while turning the wrist.
    hand_xy = (seat @ make_T((lat[0], lat[1], 0.0)))[:2, 3]
    sign = -1.0 if p.tighten else 1.0
    seated_for = 0
    n = int(p.screw_timeout_s * task.control_hz)
    for _ in range(n):
        _twist_in_place(rig, hand_xy, rig.state.pose(child)[2, 3] - 0.001 * p.press_mm,
                        sign * math.radians(p.rate_deg_s) / task.control_hz)
        if rig.screw.child_state(child) == SEATED:
            seated_for += 1
            if seated_for >= 4:
                break
        else:
            seated_for = 0
    res.engaged = any(e["event"] == "engage" for e in rig.screw.events)
    res.seated = rig.screw.child_state(child) == SEATED

    rig.hand.release()
    rig.settle(0.5)
    rel = inv_T(rig.state.pose(top)) @ rig.state.pose(child)
    if res.engaged:
        ref = np.array([*task.part(top).hole_centers[p.hole], screw_spec.seat_height])
        res.seat_pos_err_mm = float(1000 * np.abs(rel[:3, 3] - ref).max())
        res.seat_cos_min = float(np.diag(rel[:3, :3]).min())
    rig.checker.update(rig.state.poses())
    res.assembled = (top, child) in rig.checker.assembled

    if p.unscrew and res.seated:
        rig.hand.grasp(child, rig.state.pose(child))
        axis_xy = seat[:2, 3]
        for _ in range(n):
            _twist_in_place(rig, axis_xy, rig.state.pose(child)[2, 3] + 0.0003,
                            math.radians(p.rate_deg_s) / task.control_hz)
            if any(e["event"] == "disengage" for e in rig.screw.events):
                break
        res.released_at_start = any(e["event"] == "disengage" for e in rig.screw.events)
        rig.hand.move_to(rig.hand.T @ make_T((0, 0, 0.06)), 1.0, task.control_hz)
        res.lifted_out = rig.screw.measure(child, hole).axial > screw_spec.travel + 0.01

    res.sim_time = float(rig.data.time)
    res.wall_time = time.perf_counter() - t_wall
    res.events = list(rig.screw.events)
    return res
