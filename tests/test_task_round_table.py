"""round_table: dimensions, assembled poses vs FurnitureBench's (converted), corner phase walk, and the
spinning-top check (a leg screwed into a top that lies free on the table)."""
import math

import mujoco
import numpy as np
import pytest

from fb_pose import get_mat, rot_cos_min, to_ours
from fsim.geometry import make_T, pose_to_T, quat_to_mat, rot_z, yaw_of
from fsim.models.parts import parts_aabb
from fsim.sim.phases import PhaseTracker
from fsim.sim.reset import place_and_settle
from fsim.sim.screw import ENGAGED, FREE, SEATED
from fsim.sim.screw_trials import Rig, drive_site
from fsim.spec.registry import TASKS
from fsim.spec.round_table import (
    BASE, BASE_SCREW, CORNER_XY, FB_T_PART, LEG, LEG_SCREW, ROUND_TABLE, THREAD_FRICTION, TOP,
)
from task_checks import compile_scene

TASK = ROUND_TABLE


def test_registered():
    assert TASKS["round_table"] is ROUND_TABLE


@pytest.mark.parametrize("part,extents,mass", [
    (TOP, (0.2, 0.2, 0.0325), 0.12169),
    (LEG, (0.0425, 0.0425, 0.065), 0.03228),
    (BASE, (0.13, 0.13, 0.0125), 0.03396),
])
def test_dimensions_and_masses(part, extents, mass):
    lo, hi = parts_aabb(TASK.part(part).geoms)
    assert hi - lo == pytest.approx(extents, abs=6e-4)        # polygonal rings stick out < 0.5 mm
    m = compile_scene(TASK, visual_meshes=False)
    assert m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{part}_main")] == pytest.approx(mass)


def test_studs_fill_their_sockets():
    leg, base, top = TASK.part(LEG), TASK.part(BASE), TASK.part(TOP)
    assert leg.feature("stud").length == pytest.approx(top.feature("socket").depth)
    assert base.feature("stud").length == pytest.approx(leg.feature("socket").depth)


@pytest.mark.parametrize("pair,fb", [
    ((TOP, LEG), get_mat((0, 0, 0.044375), (math.pi / 2, 0, math.pi + math.pi / 36))),
    ((LEG, BASE), get_mat((0, 0.053125, 0), (-math.pi / 2, math.pi / 2, 0))),
])
def test_assembled_poses_match_design(pair, fb):
    T = to_ours(fb, FB_T_PART[pair[0]], FB_T_PART[pair[1]])
    target = next(t for t in TASK.targets if (t.parent, t.child) == pair)
    Tr = pose_to_T(*target.rel_poses[0])
    assert np.abs(Tr[:3, 3] - T[:3, 3]).max() < 0.0004           # seat_gap 0.3 mm
    assert rot_cos_min(Tr[:3, :3], T[:3, :3]) > 1 - 1e-9


def test_layout_matches_design_reset_pose():
    from fsim.spec.common import tag_pose_to_placement
    from fsim.geometry import rot_x
    for part, p, R in ((LEG, (0.13, 0.10, -0.02125), np.eye(3)), (BASE, (-0.12, 0.12, -0.02875), rot_x(math.pi))):
        pl = tag_pose_to_placement(part, p, R, fb_T_part=FB_T_PART[part])
        assert TASK.placement(part).pos == pytest.approx(pl.pos, abs=1e-9)
        assert rot_cos_min(quat_to_mat(TASK.placement(part).quat), quat_to_mat(pl.quat)) > 1 - 1e-9
    base = TASK.placement(BASE)                                   # standing on its stud tip
    tip = base.pos[2] + TASK.part(BASE).studs[0].pos[2] - TASK.part(BASE).studs[0].length
    assert tip == pytest.approx(0.0)


def test_corner_and_phase_walk_any_yaw():
    assert CORNER_XY == pytest.approx((0.2715, 0.065))
    poses = {pl.part: make_T(pl.pos, quat_to_mat(pl.quat)) for pl in TASK.layout}
    tr = PhaseTracker(TASK)
    tr.reset(poses)

    class S:
        state, resting = {}, set()

        def child_state(self, c):
            return self.state.get(c, FREE)

        def resting_hole(self, c, parent=None):
            return "h" if c in self.resting else None
    screw = S()
    poses[TOP] = poses[TOP].copy(); poses[TOP][0, 3] += 0.02
    assert tr.update(poses, screw, set()) and tr.current == "place_top_corner"
    poses[TOP] = make_T((*CORNER_XY, 0.01625), rot_z(math.radians(37)))       # round: any yaw
    assert tr.update(poses, screw, set()) and tr.current == "pick_leg"
    for part in (LEG, BASE):
        poses[part] = poses[part].copy(); poses[part][2, 3] += 0.05
        assert tr.update(poses, screw, set())
        screw.state[part] = ENGAGED
        assert tr.update(poses, screw, set())
        assert not tr.update(poses, screw, set())
        screw.state[part] = SEATED
        assert tr.update(poses, screw, set())
    assert tr.done and [p.name for p in TASK.phases][-3:] == ["pick_base", "insert_base", "screw_base"]


def _spin(spec, seat_leg_first: bool):
    """Hand-drive `spec`'s child into a top lying free at its layout pose (not pinned). Returns the drive
    result, the top's yaw change (deg) up to the moment the child seats, and at the end."""
    rig = Rig(TASK)
    place_and_settle(rig.model, rig.data, rig.state, TASK.layout)
    if seat_leg_first:
        rig.screw.force_seat(LEG, TOP, 0)
        mujoco.mj_forward(rig.model, rig.data)
        rig.settle(0.5)
    child = spec.children[0]
    y0 = yaw_of(rig.state.pose(TOP)[:3, :3])
    log, step = [], rig.hand.step

    def logged(n=1):
        step(n)
        log.append((rig.screw.child_state(child), math.degrees(yaw_of(rig.state.pose(TOP)[:3, :3]) - y0)))
    rig.hand.step = logged
    r = drive_site(rig, spec, pin=False)
    at_seat = next(y for s, y in log if s == SEATED)
    return r, at_seat, log[-1][1]


@pytest.mark.parametrize("spec,chain", [(LEG_SCREW, False), (BASE_SCREW, True)], ids=["leg", "base"])
def test_screwing_does_not_spin_a_free_top(spec, chain):
    """Thread friction 0.01 N*m: while the thread runs the top stays put (< 1 deg); 0.02 already lets it
    slip (resisting torque of the top on the table ~0.013-0.015 N*m). The driver then keeps twisting
    24 deg into the seated stop, which turns the whole table 9-13 deg, as overtightening would."""
    assert THREAD_FRICTION == 0.01
    r, at_seat, final = _spin(spec, chain)
    assert r.engaged and r.seated and r.assembled
    assert abs(at_seat) < 1.0
    assert abs(final) < 15.0
