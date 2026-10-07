"""Scripted screw-in / screw-out trials with the mocap hand (used by tests and spikes/s1), and a
generic drive that seats a child at any screw or insert site (tests/task_checks.py)."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from fsim.geometry import inv_T, make_T, mat_to_quat, rot_x, rot_z
from fsim.models.scene import PIN, build_scene_xml, pin_eq
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.hand import ScriptedHand
from fsim.sim.insert import InsertManager
from fsim.sim.screw import SEATED, HelicalScrewManager
from fsim.sim.state import PartState
from fsim.spec.types import InsertSpec, ScrewSpec, TaskSpec


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
        self.inserts = InsertManager(self.model, self.data, task)
        self.checker = AssemblyChecker(task)
        self.hand = ScriptedHand(self.model, self.data, task, on_substep=self._substep, control_hz=task.control_hz)
        pin = int(self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, PIN)])
        self.data.mocap_pos[pin], self.data.mocap_quat[pin] = 0.0, (1.0, 0.0, 0.0, 0.0)   # the world frame

    def hold(self, part: str, T: np.ndarray):
        """Pin `part` at world pose T (stiff weld to a mocap body at the world origin), e.g. a parent in mid-air."""
        eq = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, pin_eq(part))
        self.model.eq_data[eq, 3:10] = np.r_[T[:3, 3], mat_to_quat(T[:3, :3])]
        self.data.eq_active[eq] = 1
        self.state.set_pose(part, T)
        mujoco.mj_forward(self.model, self.data)

    def _substep(self):
        self.screw.update()
        self.inserts.update()

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
    child = child or task.extras["target_child"]
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


@dataclass
class DriveResult:
    engaged: bool = False
    seated: bool = False
    assembled: bool | None = None       # None: no AssemblyTarget for the pair
    events: list = field(default_factory=list)


def site_holes(rig: Rig, spec: ScrewSpec | InsertSpec) -> list:
    """The manager's holes / slots of one spec, in site order."""
    pool = rig.screw.holes if isinstance(spec, ScrewSpec) else rig.inserts.slots
    return [h for h in pool if h.screw is spec] if isinstance(spec, ScrewSpec) else [s for s in pool if s.spec is spec]


def axis_up_pose(T_local: np.ndarray, seat_world) -> np.ndarray:
    """Parent world pose that puts a site's seated frame at `seat_world` with its axis pointing up."""
    a, up = T_local[:3, 2], np.array([0.0, 0.0, 1.0])
    v, c = np.cross(a, up), float(a @ up)
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else rot_x(math.pi)
    else:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + K + K @ K / (1.0 + c)                     # Rodrigues: rotates a onto up
    T = make_T(np.zeros(3), R)
    T[:3, 3] = np.asarray(seat_world) - (T @ T_local)[:3, 3]
    return T


def drive_site(rig: Rig, spec: ScrewSpec | InsertSpec, site: int = 0, child: str | None = None,
               parent_T: np.ndarray | None = None, twist0_deg: float = 40.0, rate_deg_s: float = 120.0,
               press_mm: float = 1.0, timeout_s: float = 12.0, pin: bool = True) -> DriveResult:
    """Seat `child` (default children[0]) at site `site` of `spec` with the mocap hand, in the site
    frame: screws are lowered onto the catch and twisted (like run_trial), inserts are pushed in
    along the axis. The parent is pinned at `parent_T` (default: 0.35 m above the table, site axis
    up); pin=False leaves it where it is (e.g. already seated in its own parent, for chains)."""
    task, child = rig.task, child or spec.children[0]
    hole = site_holes(rig, spec)[site]
    if not pin:
        mujoco.mj_forward(rig.model, rig.data)
        parent_T = rig.state.pose(spec.parent)
    elif parent_T is None:
        parent_T = axis_up_pose(hole.T_local, np.asarray(task.scene.W_origin) + (0.1, 0.0, 0.35))
    if pin:
        rig.hold(spec.parent, parent_T)
    S = parent_T @ hole.T_local                                  # seated mate frame, world
    inv_mate = np.eye(4) if hole.mate[child] is None else inv_T(hole.mate[child])

    def child_T(z, twist=0.0):
        return S @ make_T((0.0, 0.0, z), rot_z(twist)) @ inv_mate

    hz, res = task.control_hz, DriveResult()
    mgr = rig.screw if isinstance(spec, ScrewSpec) else rig.inserts
    if isinstance(spec, ScrewSpec):
        theta = math.radians(twist0_deg)
        T0 = child_T(spec.travel + 0.004, theta)
        rig.state.set_pose(child, T0)
        mujoco.mj_forward(rig.model, rig.data)
        rig.hand.grasp(child, T0)
        rig.hand.move_to(child_T(spec.travel - 0.001 * press_mm, theta), 0.5, hz)
        seated_for = 0
        for _ in range(int(timeout_s * hz)):
            theta -= math.radians(rate_deg_s) / hz
            rig.hand.set(child_T(rig.screw.measure(child, hole).axial - 0.001 * press_mm, theta))
            rig.hand.step()
            seated_for = seated_for + 1 if mgr.child_state(child) == SEATED else 0
            if seated_for >= 4:
                break
    else:
        T0 = child_T(spec.travel + 0.01)
        rig.state.set_pose(child, T0)
        mujoco.mj_forward(rig.model, rig.data)
        rig.hand.grasp(child, T0)
        rig.hand.move_to(child_T(-0.002), (spec.travel + 0.012) / 0.05, hz)
        for _ in range(int(timeout_s * hz)):
            if mgr.child_state(child) == SEATED:
                break
            rig.hand.step()
    res.engaged = any(e["event"] == "engage" and e["child"] == child for e in mgr.events)
    rig.hand.release()
    rig.settle(0.5)
    res.seated = mgr.child_state(child) == SEATED
    if any((t.parent, t.child) == (spec.parent, child) for t in task.targets):
        rig.checker.update(rig.state.poses())
        res.assembled = (spec.parent, child) in rig.checker.assembled
    res.events = list(mgr.events)
    return res
