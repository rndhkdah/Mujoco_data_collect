import math

import numpy as np

from fsim.geometry import make_T, quat_to_mat, rot_z
from fsim.sim.phases import PhaseTracker
from fsim.sim.screw import ENGAGED, FREE, SEATED
from fsim.spec import get_task
from fsim.spec.square_table import CORNER_XY, LEGS, TOP

TASK = get_task("one_leg")


class FakeScrew:
    def __init__(self):
        self.state = {}
        self.resting = set()

    def child_state(self, child):
        return self.state.get(child, FREE)

    def resting_hole(self, child, parent=None):
        return "hole" if child in self.resting else None


def layout_poses(task=TASK):
    return {pl.part: make_T(pl.pos, quat_to_mat(pl.quat)) for pl in task.layout}


def test_one_leg_phase_progression():
    poses = layout_poses()
    tr = PhaseTracker(TASK)
    tr.reset(poses)
    screw = FakeScrew()
    assert tr.current == "grasp_top" and not tr.update(poses, screw, set())
    poses[TOP] = poses[TOP].copy(); poses[TOP][0, 3] += 0.02                     # top moved
    assert tr.update(poses, screw, set()) and tr.current == "place_top_corner"
    poses[TOP] = make_T((*CORNER_XY, 0.015625), rot_z(math.radians(92)))         # snug in the corner (mod 90 deg)
    assert tr.update(poses, screw, set()) and tr.current == "pick_leg"
    poses[LEGS[3]] = poses[LEGS[3]].copy(); poses[LEGS[3]][2, 3] += 0.05
    assert tr.update(poses, screw, set()) and tr.current == "insert_leg"
    screw.resting.add(LEGS[3])                                                   # standing in a hole, released
    assert tr.update(poses, screw, set()) and tr.current == "screw_leg"
    screw.state[LEGS[3]] = ENGAGED
    assert not tr.update(poses, screw, set())
    screw.state[LEGS[3]] = SEATED
    assert tr.update(poses, screw, set()) and tr.done


def test_insert_waits_for_release_when_grasp_is_known():
    poses = layout_poses()
    held = {LEGS[3]}
    tr = PhaseTracker(TASK, grasped=lambda p: p in held)
    tr.reset(poses)
    tr.phase = TASK.phase_labels.index("insert_leg")
    screw = FakeScrew()
    screw.resting.add(LEGS[3])
    assert not tr.update(poses, screw, set())
    held.clear()
    assert tr.update(poses, screw, set())


def test_grasp_callback_gates_corner_release():
    poses = layout_poses()
    held = {TOP}
    tr = PhaseTracker(TASK, grasped=lambda p: p in held)
    tr.reset(poses)
    assert tr.update(poses, FakeScrew(), set())                                   # grasped top
    poses[TOP] = make_T((*CORNER_XY, 0.015625))
    assert not tr.update(poses, FakeScrew(), set())                               # still held
    held.clear()
    assert tr.update(poses, FakeScrew(), set())


def test_square_table_walks_all_phases():
    task = get_task("square_table")
    poses = layout_poses(task)
    tr = PhaseTracker(task)
    tr.reset(poses)
    screw = FakeScrew()
    poses[TOP] = make_T((*CORNER_XY, 0.015625))
    steps = 0
    while not tr.done and steps < 100:
        steps += 1
        name = tr.current
        if name.startswith("pick"):
            leg = next(p for p in LEGS if p.endswith(name.rsplit("_", 1)[-1]))
            poses[leg] = poses[leg].copy(); poses[leg][2, 3] += 0.05
        elif name.startswith("insert"):
            screw.resting.add(next(p for p in LEGS if p.endswith(name.rsplit("_", 1)[-1])))
        elif name.startswith("screw"):
            screw.state[next(p for p in LEGS if p.endswith(name.rsplit("_", 1)[-1]))] = SEATED
        tr.update(poses, screw, set())
    assert tr.done and steps == len(task.phases)
