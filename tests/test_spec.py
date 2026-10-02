"""Design conformance of the square_table / one_leg specs."""
import math

import numpy as np

from fsim.geometry import quat_to_mat, rot_x, rot_y, rot_z
from fsim.sim.reset import ResetSampler
from fsim.spec import get_task
from fsim.spec.square_table import CORNER_XY, HIGH_PRESETS, LEGS, TOP, tag_pose_to_placement


def test_one_leg_scores_only_leg4_in_any_corner():
    task = get_task("one_leg")
    assert len(task.targets) == 1
    t = task.targets[0]
    assert (t.parent, t.child) == (TOP, LEGS[3])
    corners = {(round(p[0], 6), round(p[1], 6), round(p[2], 6)) for p, _ in t.rel_poses}
    assert corners == {(sx * 0.05625, sy * 0.05625, 0.046875) for sx in (1, -1) for sy in (1, -1)}
    assert t.pos_tol == (0.005, 0.005, 0.005) and t.ori_cos == 0.94


def test_square_table_scores_all_legs_and_has_no_extra_phases():
    task = get_task("square_table")
    assert {t.child for t in task.targets} == set(LEGS)
    assert len(task.phases) == 2 + 3 * 4


def test_screw_geometry_is_consistent():
    task = get_task("one_leg")
    s, top, leg = task.screws[0], task.part(TOP), task.part(LEGS[3])
    assert math.isclose(s.pitch, 0.009375) and math.isclose(s.travel, 1.5 * 0.009375)
    assert math.isclose(s.hinge_catch, 3 * math.pi)
    # Seated: the leg shoulder sits exactly seat_gap above the tabletop face.
    assert math.isclose(s.seated_z - leg.body_length / 2, top.top_z + s.seat_gap, abs_tol=1e-12)
    # Seated stud tip stays above the hole floor (the tray plate).
    assert s.seated_z + leg.tip_z > top.plate_top_z
    # Collision peg fits inside the bore; bosses stay inside the rim.
    assert leg.peg_radius < top.bore_radius
    assert 0.05625 + top.boss_outer_radius < top.size[0] / 2 - top.rim_width


def test_tray_volume_matches_design_mass_density():
    """Tray volume (plate + rim + bosses) is far below the bounding block, as in the design."""
    top = get_task("one_leg").part(TOP)
    sx, sy, sz = top.size
    plate = sx * sy * top.plate_thickness
    wall_h = top.top_z - top.plate_top_z
    rim = (2 * sx + 2 * (sy - 2 * top.rim_width)) * top.rim_width * wall_h
    bosses = 4 * math.pi * (top.boss_outer_radius ** 2 - top.bore_radius ** 2) * wall_h
    assert 0.25 < (plate + rim + bosses) / (sx * sy * sz) < 0.45


def test_default_layout_is_valid():
    task = get_task("one_leg")
    assert ResetSampler(task).valid(task.layout, [0.0] * len(task.layout))


def test_corner_target_is_snug_in_built_fence():
    task = get_task("one_leg")
    front, left = task.scene.obstacles[0], task.scene.obstacles[1]
    top = task.part(TOP)
    assert math.isclose(CORNER_XY[0] + top.size[0] / 2, front.center[0] - front.half_size[0])
    assert math.isclose(CORNER_XY[1] + top.size[1] / 2, left.center[1] - left.half_size[1])


def test_tag_frame_conversion_matches_native_layout():
    """The documented tag-frame transform maps FurnitureBench's default leg4 pose onto our layout."""
    task = get_task("one_leg")
    pl = tag_pose_to_placement(LEGS[3], (0.20, 0.07, -0.015), rot_y(-math.pi / 2))
    ours = task.placement(LEGS[3])
    assert np.allclose(pl.pos, ours.pos, atol=1e-9)
    assert np.allclose(quat_to_mat(pl.quat), quat_to_mat(ours.quat), atol=1e-9)
    top = tag_pose_to_placement(TOP, (0.0, 0.24, -0.015625), rot_z(math.pi) @ rot_x(-math.pi / 2))
    assert np.allclose(top.pos, task.placement(TOP).pos, atol=1e-9)
    assert quat_to_mat(top.quat)[2, 2] > 0.999


def test_high_presets_are_sane():
    assert len(HIGH_PRESETS) == 3
    for preset in HIGH_PRESETS:
        assert [pl.part for pl in preset] == [TOP, *LEGS]
        for pl in preset:
            R = quat_to_mat(pl.quat)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
            assert -0.65 < pl.pos[0] < 0.65 and -0.4 < pl.pos[1] < 0.4 and 0.0 < pl.pos[2] < 0.05
        assert quat_to_mat(preset[0].quat)[2, 2] > 0.99      # tabletop holes face up
