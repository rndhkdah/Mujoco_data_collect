"""chair: task-specific checks (the generic ones run in tests/test_tasks_generic.py)."""
import math

import mujoco
import numpy as np
import pytest

from fsim.geometry import inv_T, make_T, pose_to_T, quat_to_mat, rot_z, yaw_of
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.phases import PhaseTracker
from fsim.sim.screw import SEATED
from fsim.sim.screw_trials import Rig, drive_site
from fsim.spec.chair import (
    BACK, BACK_INSERT, CHAIR, FB_T, LEG_SCREW, LEGS, NUT_SCREW, NUTS, SEAT, SEAT_UPRIGHT, SEAT_YAW_DEG,
)
from task_checks import check_hand_drive

TASK = CHAIR
W = np.asarray(TASK.scene.W_origin)


def _target(child):
    return next(t for t in TASK.targets if t.child == child)


def _fb_rel(child, i=0):
    """Seated rel pose of `child` (site i) in the design's seat and child frames."""
    T = pose_to_T(*_target(child).rel_poses[i])
    return FB_T[SEAT] @ T @ inv_T(FB_T[child])


def test_masses_and_parts():
    masses = {p.name: p.mass for p in TASK.parts}
    assert masses == pytest.approx({SEAT: 0.06187, BACK: 0.12316, LEGS[0]: 0.02244, LEGS[1]: 0.02244,
                                    NUTS[0]: 0.01025, NUTS[1]: 0.01025})
    assert len(TASK.phases) == 17                       # the design's chair has 17 skills


def test_seated_poses_match_the_design():
    """Converted back into the design's frames, our seated poses are its assembled poses (FB's 0.5 mm
    off-axis hole / nut offsets removed, nuts seated on the back = its 0.0795 check pose)."""
    leg = _fb_rel(LEGS[0])
    assert np.allclose(leg[:3, 3], (0.03375, 0.045, -0.01875), atol=0.001)
    assert np.allclose(leg[:3, 1], (0, 1, 0), atol=1e-6)          # leg long axis along the seat's y
    back = _fb_rel(BACK)
    assert np.allclose(back[:3, 3], (0.0, -0.0325, 0.05025), atol=0.001) and np.allclose(back[:3, :3], np.eye(3), atol=1e-6)
    nut = _fb_rel(NUTS[0])
    assert np.allclose(nut[:3, 3], (0.035, 0.0, 0.0795), atol=0.001)
    assert np.allclose(nut[:3, 1], (0, 0, -1), atol=1e-6)         # the design's Rx(-pi/2)


@pytest.mark.parametrize("leg,site", [(LEGS[0], 0), (LEGS[1], 1)])
def test_hand_drive_leg(leg, site):
    check_hand_drive(TASK, LEG_SCREW, site=site, child=leg)


@pytest.mark.parametrize("site", [0, 1], ids=["upright", "turned"])
def test_hand_drive_back_both_orientations(site):
    check_hand_drive(TASK, BACK_INSERT, site=site)


@pytest.mark.parametrize("nut,site", [(NUTS[0], 0), (NUTS[1], 1)])
def test_hand_drive_nut(nut, site):
    check_hand_drive(TASK, NUT_SCREW, site=site, child=nut)


def test_nut_not_assembled_before_the_back():
    rig = Rig(TASK)
    r = drive_site(rig, NUT_SCREW, 0, NUTS[0])
    assert r.seated and r.assembled is False
    checker = AssemblyChecker(TASK)
    checker.update(rig.state.poses())
    assert (SEAT, NUTS[0]) not in checker.assembled


def _upright_T():
    return make_T(W + SEAT_UPRIGHT.pos, quat_to_mat(SEAT_UPRIGHT.quat))


def test_flip_phase_triggers_at_the_design_upright_pose():
    rig = Rig(TASK)
    tr = PhaseTracker(TASK)
    poses = {p: rig.state.pose_W(p) for p in TASK.part_names}
    tr.reset(poses)
    tr.phase = TASK.phase_labels.index("flip_seat")
    flat = dict(poses, **{SEAT: make_T(TASK.placement(SEAT).pos, quat_to_mat(TASK.placement(SEAT).quat))})
    assert not tr.update(flat, rig.screw, set(), rig.inserts)
    upright = dict(poses, **{SEAT: make_T(SEAT_UPRIGHT.pos, quat_to_mat(SEAT_UPRIGHT.quat))})
    assert tr.update(upright, rig.screw, set(), rig.inserts) and tr.current == "pick_back"


def test_corner_phase_checks_the_seat_yaw():
    rig = Rig(TASK)
    tr = PhaseTracker(TASK)
    poses = {p: rig.state.pose_W(p) for p in TASK.part_names}
    xy = TASK.extras["corner_xy"]

    def at(yaw_deg):
        return dict(poses, **{SEAT: make_T((xy[0], xy[1], 0.015), rot_z(math.radians(yaw_deg)))})

    tr.reset(poses)
    tr.phase = TASK.phase_labels.index("place_seat_corner")
    assert not tr.update(at(SEAT_YAW_DEG + 180.0), rig.screw, set(), rig.inserts)    # studs toward the fence
    assert not tr.update(at(0.0), rig.screw, set(), rig.inserts)
    assert tr.update(at(SEAT_YAW_DEG + 10.0), rig.screw, set(), rig.inserts)


def _standing_chair(back: bool = True) -> Rig:
    """Seat standing on its rear face (the design's init-assembled pose), legs (and back) seated."""
    rig = Rig(TASK)
    rig.state.set_pose(SEAT, _upright_T())
    mujoco.mj_forward(rig.model, rig.data)
    for leg in LEGS:
        rig.screw.force_seat(leg, SEAT, next(h.index for h in rig.screw.holes if h.screw is LEG_SCREW and h.child is None))
        mujoco.mj_kinematics(rig.model, rig.data)
    if back:
        rig.inserts.force_seat(BACK, SEAT, 0)
    mujoco.mj_forward(rig.model, rig.data)
    rig.settle(2.0)
    return rig


def _stud_tilt_deg(rig) -> float:
    s = rig.state.pose_W(SEAT)[:3, :3] @ (0.0, -1.0, 0.0)
    return math.degrees(math.acos(min(1.0, s[2])))


def test_standing_chair_rests_on_the_table():
    """Without the back the seat stands studs-up on its rear face and legs. With the back, its centre
    of mass (3 cm toward the top rail) is past the 30 mm rear face: the chair leans back onto the
    top rail, about 33 degrees, and rests there (true of the design's parts too)."""
    rig = _standing_chair(back=False)
    assert _stud_tilt_deg(rig) < 3.0
    rig = _standing_chair()
    tilt = _stud_tilt_deg(rig)
    assert 25.0 < tilt < 40.0, tilt
    rig.settle(1.0)
    assert abs(_stud_tilt_deg(rig) - tilt) < 1.0                    # at rest
    for p in (SEAT, BACK) + LEGS:
        assert rig.state.pose_W(p)[2, 3] > 0.0
    checker = AssemblyChecker(TASK)
    checker.update(rig.state.poses())
    assert {(SEAT, LEGS[0]), (SEAT, LEGS[1]), (SEAT, BACK)} <= checker.assembled


def test_standing_chair_does_not_spin():
    """Twisting a nut on (thread friction 0.004 N*m) must not turn the free-standing chair (leaning on
    its top rail) on the table."""
    rig = _standing_chair()
    T0 = rig.state.pose_W(SEAT)
    r = drive_site(rig, NUT_SCREW, 0, NUTS[0], pin=False)
    T1 = rig.state.pose_W(SEAT)
    dyaw = math.degrees(abs(math.atan2(math.sin(yaw_of(T1[:3, :3]) - yaw_of(T0[:3, :3])),
                                       math.cos(yaw_of(T1[:3, :3]) - yaw_of(T0[:3, :3])))))
    print(f"seat yaw change {dyaw:.2f} deg, moved {1000 * np.linalg.norm(T1[:2, 3] - T0[:2, 3]):.1f} mm")
    assert r.engaged and r.seated and rig.screw.child_state(NUTS[0]) == SEATED
    assert dyaw < 3.0 and np.linalg.norm(T1[:2, 3] - T0[:2, 3]) < 0.005
    assert r.assembled
