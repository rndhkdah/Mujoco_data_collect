import math

import mujoco
import numpy as np
import pytest

from fsim.geometry import quat_to_mat, yaw_of
from fsim.models.scene import build_scene_xml
from fsim.sim.reset import ResetSampler, aabbs_collide, max_penetration, place_and_settle, rotated_aabb
from fsim.sim.state import PartState
from fsim.spec import get_task

TASK = get_task("one_leg")


def test_rotated_aabb():
    assert np.allclose(rotated_aabb(0, 0, 0.2, 0.1, 0.0), (-0.1, 0.1, -0.05, 0.05))
    assert np.allclose(rotated_aabb(0, 0, 0.2, 0.1, math.pi / 2), (-0.05, 0.05, -0.1, 0.1))
    h = (0.2 + 0.1) / 2 * math.cos(math.pi / 4)
    assert np.allclose(rotated_aabb(1, 2, 0.2, 0.1, math.pi / 4), (1 - h, 1 + h, 2 - h, 2 + h))


def test_aabb_collision_rule_uses_clearance_on_either_axis():
    a = (0.0, 0.1, 0.0, 0.1)
    assert aabbs_collide(a, (0.105, 0.2, 0.0, 0.1), 0.01)          # 5 mm gap on x: collides
    assert not aabbs_collide(a, (0.115, 0.2, 0.0, 0.1), 0.01)      # 15 mm gap on x: separated
    assert not aabbs_collide(a, (0.0, 0.1, 0.12, 0.2), 0.01)       # separated on y
    assert aabbs_collide(a, (0.05, 0.15, 0.05, 0.15), 0.01)


@pytest.mark.parametrize("level", ["low", "med"])
def test_random_layouts_respect_rules(level):
    sampler = ResetSampler(TASK)
    lv = TASK.randomization.levels[level]
    for seed in range(25):
        out = sampler.sample(np.random.default_rng(seed), level)
        yaws = []
        for pl, base in zip(out, TASK.layout):
            assert pl.part == base.part
            assert abs(pl.pos[0] - base.pos[0]) <= lv.pos_range + 1e-12
            assert abs(pl.pos[1] - base.pos[1]) <= lv.pos_range + 1e-12
            assert pl.pos[2] == base.pos[2]
            d = yaw_of(quat_to_mat(pl.quat) @ quat_to_mat(base.quat).T)
            assert abs(math.degrees(d)) <= lv.yaw_range_deg + 1e-9
            yaws.append(d)
        assert sampler.valid(out, yaws)


def test_batch_rule_matches_scalar_rule():
    from fsim.spec.types import Placement
    sampler = ResetSampler(TASK)
    rng = np.random.default_rng(3)
    base = np.array([pl.pos[:2] for pl in TASK.layout])
    xy = base + rng.uniform(-0.015, 0.015, size=(2000, len(TASK.layout), 2))
    yaw = np.radians(rng.uniform(-15, 15, size=(2000, len(TASK.layout))))
    batch = sampler._valid_batch(xy, yaw)
    scalar = [sampler.valid([Placement(pl.part, (xy[b, i, 0], xy[b, i, 1], pl.pos[2]), pl.quat)
                             for i, pl in enumerate(TASK.layout)], list(yaw[b])) for b in range(len(xy))]
    assert batch.sum() > 5 and (batch == np.array(scalar)).all()


def test_sampling_is_reproducible():
    sampler = ResetSampler(TASK)
    assert sampler.sample(np.random.default_rng(7), "med") == sampler.sample(np.random.default_rng(7), "med")


def test_high_and_none_levels():
    sampler = ResetSampler(TASK)
    assert sampler.sample(np.random.default_rng(0), "none") == TASK.layout
    assert sampler.sample(np.random.default_rng(0), "high", high_index=2) == TASK.randomization.high_presets[2]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_high_presets_settle_without_explosions(idx):
    """Measured presets overlap slightly (fence, table, each other); placement resolves that."""
    preset = TASK.randomization.high_presets[idx]
    m = mujoco.MjModel.from_xml_string(build_scene_xml(TASK, visual_meshes=False))
    d = mujoco.MjData(m)
    st = PartState(m, d, TASK)
    remaining = place_and_settle(m, d, st, preset, settle_steps=250)
    assert remaining < 0.001
    assert max_penetration(m, d, st.body_id.values()) < 0.001
    for p in TASK.part_names:
        assert np.linalg.norm(d.qvel[st.dof_adr[p]:st.dof_adr[p] + 3]) < 0.05
        assert np.linalg.norm(st.pose_W(p)[:2, 3] - np.asarray(next(pl.pos for pl in preset if pl.part == p))[:2]) < 0.02
