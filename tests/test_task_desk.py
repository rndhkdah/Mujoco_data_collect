"""desk: dimensions, assembled poses vs FurnitureBench's (converted), layout, corner phase walk."""
import math

import mujoco
import numpy as np
import pytest

from fb_pose import get_mat, rot_cos_min, to_ours
from fsim.geometry import make_T, pose_to_T, quat_to_mat, rot_z, yaw_of
from fsim.sim.phases import PhaseTracker
from fsim.sim.reset import ResetSampler
from fsim.sim.screw import FREE, SEATED
from fsim.spec.desk import CORNER_XY, DESK, FB_T_PART, LEGS, TOP
from fsim.spec.registry import TASKS
from task_checks import compile_scene

TASK = DESK


def test_registered():
    assert TASKS["desk"] is DESK


def test_dimensions_and_masses():
    top = TASK.part(TOP)
    assert top.size == pytest.approx((0.15, 0.21, 0.03125))
    assert sorted(top.hole_centers) == sorted((sx * 0.05, sy * 0.08) for sx in (1, -1) for sy in (1, -1))
    for leg in LEGS:
        p = TASK.part(leg)
        assert (p.width, p.body_length, p.screw_length, p.mass) == pytest.approx((0.03, 0.1, 0.025, 0.0562))
    m = compile_scene(TASK, visual_meshes=False)
    total = sum(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p}_main")] for p in TASK.part_names)
    assert total == pytest.approx(0.1694 + 4 * 0.0562, rel=1e-6)


def test_assembled_poses_match_design():
    """FurnitureBench: get_mat([+-0.080, 0.065625, +-0.05], 0) in its tabletop frame (y up)."""
    fb = [get_mat((sx * 0.080, 0.065625, sz * 0.05), (0, 0, 0)) for sx in (1, -1) for sz in (1, -1)]
    ours = [to_ours(T, FB_T_PART, FB_T_PART) for T in fb]
    for leg_target in TASK.targets:
        assert len(leg_target.rel_poses) == 4
        for T in ours:
            best = min(leg_target.rel_poses, key=lambda rp: np.linalg.norm(np.asarray(rp[0]) - T[:3, 3]))
            Tr = pose_to_T(*best)
            assert np.abs(Tr[:3, 3] - T[:3, 3]).max() < 1e-9
            assert rot_cos_min(Tr[:3, :3], T[:3, :3]) > 1 - 1e-9


def test_layout_matches_design_reset_pose():
    """Design reset pose 0: top at tag (0, 0.24) yaw-rotated so the 210 mm side runs along W y."""
    T = make_T(TASK.placement(TOP).pos, quat_to_mat(TASK.placement(TOP).quat))
    long_axis = T[:3, :3] @ np.array([0.0, 1.0, 0.0])
    assert abs(long_axis[1]) == pytest.approx(1.0)
    assert TASK.placement(TOP).pos[:2] == pytest.approx((0.2415, 0.0))
    for leg, y in zip(LEGS, (-0.15, -0.08, 0.08, 0.15)):
        pl = TASK.placement(leg)
        assert pl.pos[:2] == pytest.approx((0.0715, y))
        assert (quat_to_mat(pl.quat) @ [0, 0, 1]) == pytest.approx((1.0, 0.0, 0.0))     # stud toward the robot
    from fsim.spec.common import tag_pose_to_placement
    from fsim.geometry import rot_x, rot_y
    for part, p, R in ((TOP, (0, 0.24, -0.015625), rot_x(-math.pi / 2)), (LEGS[0], (-0.15, 0.07, -0.015), rot_y(-math.pi / 2))):
        pl = tag_pose_to_placement(part, p, R, fb_T_part=FB_T_PART)
        assert TASK.placement(part).pos == pytest.approx(pl.pos, abs=1e-9)
        R = quat_to_mat(TASK.placement(part).quat)                # the tabletop is 180 deg symmetric
        assert max(rot_cos_min(R @ rot_z(a), quat_to_mat(pl.quat)) for a in (0, math.pi)) > 1 - 1e-9


def test_top_position_noise_follows_design_override():
    x0, y0 = TASK.placement(TOP).pos[:2]
    s = ResetSampler(TASK)
    for level in ("low", "med"):
        for seed in range(5):
            pl = s.sample(np.random.default_rng(seed), level)[0]
            assert -0.05 - 1e-12 <= pl.pos[0] - x0 <= 0.02 + 1e-12 and -0.05 - 1e-12 <= pl.pos[1] - y0 <= 0.02 + 1e-12


def test_corner_is_snug_and_the_far_hole_matches_square_table():
    assert CORNER_XY == pytest.approx((0.2965, 0.06))
    far = max((CORNER_XY[0] + hx, CORNER_XY[1] + hy) for hx, hy in TASK.part(TOP).hole_centers)
    assert far == pytest.approx((0.3465, 0.14))          # = square_table's farthest hole


class _Screw:
    def __init__(self):
        self.state, self.resting = {}, set()

    def child_state(self, child):
        return self.state.get(child, FREE)

    def resting_hole(self, child, parent=None):
        return "hole" if child in self.resting else None


def test_phase_walk_with_corner_yaw_mod_180():
    poses = {pl.part: make_T(pl.pos, quat_to_mat(pl.quat)) for pl in TASK.layout}
    tr, screw = PhaseTracker(TASK), _Screw()
    tr.reset(poses)
    poses[TOP] = poses[TOP].copy(); poses[TOP][0, 3] += 0.02
    assert tr.update(poses, screw, set()) and tr.current == "place_top_corner"
    poses[TOP] = make_T((*CORNER_XY, 0.015625), rot_z(math.radians(90)))        # rotated: wrong orientation
    assert not tr.update(poses, screw, set())
    poses[TOP] = make_T((*CORNER_XY, 0.015625), rot_z(math.radians(183)))       # 180 deg symmetric
    assert tr.update(poses, screw, set()) and tr.current == "pick_leg1"
    for i, leg in enumerate(LEGS, 1):
        poses[leg] = poses[leg].copy(); poses[leg][2, 3] += 0.05
        assert tr.update(poses, screw, set()) and tr.current == f"insert_leg{i}"
        screw.resting.add(leg)
        assert tr.update(poses, screw, set()) and tr.current == f"screw_leg{i}"
        screw.state[leg] = SEATED
        assert tr.update(poses, screw, set())
    assert tr.done
