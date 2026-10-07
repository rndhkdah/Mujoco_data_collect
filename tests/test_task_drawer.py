"""drawer: task-specific checks (the generic suite in test_tasks_generic.py covers the rest)."""
import mujoco
import numpy as np
import pytest

from fsim.geometry import inv_T, make_T, quat_to_mat
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.screw import ENGAGED, FREE, SEATED
from fsim.sim.screw_trials import Rig, axis_up_pose, site_holes
from fsim.spec.drawer import BOTTOM, BOX, DRAWER, INSERT, SLOT_POSES, TOP
from task_checks import check_hand_drive

TASK = DRAWER


@pytest.mark.parametrize("child", [BOTTOM, TOP])
@pytest.mark.parametrize("site", [0, 1], ids=["lower", "upper"])
def test_each_container_slides_into_each_slot(site, child):
    """Hand-driven from 10 mm beyond the capture point (container back 2 cm outside the opening)."""
    check_hand_drive(TASK, INSERT, site, child)


def test_seat_matches_the_design_poses():
    # Design (FurnitureBench) rel poses in our frames: z 16.6 mm (top mesh) / 12.9 mm (bottom mesh); our seat is
    # the physical front-panel stop and must be within the 5 mm tolerance of both.
    for pos, _ in SLOT_POSES:
        assert abs(pos[2] - 0.0166) < 0.001 and abs(pos[2] - 0.0129) < 0.005
    assert {round(p[1], 6) for p, _ in SLOT_POSES} == {-0.010875, 0.034125}


def test_occupied_slot_rejects_the_other_container():
    rig = Rig(TASK)
    hole = site_holes(rig, INSERT)[0]
    PT = axis_up_pose(hole.T_local, np.asarray(TASK.scene.W_origin) + (0.1, 0.0, 0.35))
    rig.hold(BOX, PT)
    rig.inserts.force_seat(BOTTOM, BOX, hole.index)
    mujoco.mj_forward(rig.model, rig.data)
    # The other container aligned on the same slot's axis, inside the capture zone: nothing happens.
    T = PT @ hole.T_local @ make_T((0, 0, 0.08)) @ inv_T(hole.mate[TOP])
    rig.state.set_pose(TOP, T)
    mujoco.mj_forward(rig.model, rig.data)
    rig.inserts.update()
    assert rig.inserts.child_state(TOP) == FREE and rig.inserts.child_state(BOTTOM) == SEATED


def test_captured_container_stays_on_the_slot_line_when_released():
    """Box upright (slide axis horizontal): a captured, released container hangs on the carriage, no drift."""
    rig = Rig(TASK)
    PT = make_T(np.asarray(TASK.scene.W_origin) + (0.1, 0.0, 0.3), quat_to_mat(TASK.extras["box_corner_quat"]))
    rig.hold(BOX, PT)
    slot = rig.inserts.slots[0]
    T = PT @ slot.T_local @ make_T((0, 0, 0.05)) @ inv_T(slot.mate[BOTTOM])
    rig.state.set_pose(BOTTOM, T)
    mujoco.mj_forward(rig.model, rig.data)
    rig.inserts.update()
    assert rig.inserts.child_state(BOTTOM) == ENGAGED
    rig.settle(1.0)
    assert rig.inserts.child_state(BOTTOM) == ENGAGED
    assert abs(rig.inserts.slide(slot) - 0.05) < 0.003


def test_both_containers_assemble_in_either_order():
    rig = Rig(TASK)
    rig.hold(BOX, make_T(np.asarray(TASK.scene.W_origin) + (0.1, 0.0, 0.3)))
    rig.inserts.force_seat(TOP, BOX, 0)          # the top container in the lower slot
    rig.inserts.force_seat(BOTTOM, BOX, 1)
    mujoco.mj_forward(rig.model, rig.data)
    rig.settle(0.5)
    checker = AssemblyChecker(TASK)
    checker.update(rig.state.poses())
    assert checker.all_assembled
