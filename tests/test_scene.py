"""Generated MJCF conforms to the spec and to the helical-constraint requirements."""
import math

import mujoco
import numpy as np
import pytest
import trimesh

from fsim.geometry import make_T
from fsim.models import names
from fsim.models.meshes import build_task_meshes
from fsim.models.scene import build_scene_xml
from fsim.sim.state import PartState
from fsim.spec import get_task
from fsim.spec.square_table import LEGS, TOP

TASK = get_task("one_leg")


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_string(build_scene_xml(TASK, visual_meshes=True))


def jid(m, n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)


def test_xml_uses_radians():
    assert 'angle="radian"' in build_scene_xml(TASK, visual_meshes=False)


def test_hinge_range_and_stiff_thread_friction(model):
    j = jid(model, names.hinge_joint(TOP, 0))
    assert np.allclose(model.jnt_range[j], (0.0, 3 * math.pi))
    dof = model.jnt_dofadr[j]
    assert math.isclose(model.dof_frictionloss[dof], TASK.screws[0].thread_friction)
    assert np.allclose(model.dof_solref[dof], (0.005, 1.0))
    assert model.dof_solimp[dof][0] >= 0.99


def test_coupling_is_hinge_of_slide(model):
    s = TASK.screws[0]
    for h in range(4):
        e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, names.couple_eq(TOP, h))
        assert model.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT
        assert model.eq_obj1id[e] == jid(model, names.hinge_joint(TOP, h))
        assert model.eq_obj2id[e] == jid(model, names.slide_joint(TOP, h))
        assert math.isclose(model.eq_data[e][1], s.rad_per_m, rel_tol=1e-6)
        assert model.eq_active0[e] == 1


def test_welds_start_inactive_with_identity_relpose(model):
    for h in range(4):
        for leg in LEGS:
            e = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, names.weld_eq(TOP, h, leg))
            assert model.eq_active0[e] == 0
            assert np.allclose(model.eq_data[e][3:10], (0, 0, 0, 1, 0, 0, 0))


def test_masses_follow_spec(model):
    for p in TASK.parts:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, names.body(p.name))
        assert math.isclose(model.body_mass[b], p.mass, rel_tol=1e-9)


def _ray_collision(model, d, start):
    """Cast straight down against collision geoms only (mj_ray skips fully transparent geoms)."""
    rgba = model.geom_rgba.copy()
    model.geom_rgba[model.geom_group == 0, 3] = 1.0
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(model, d, np.asarray(start, float), np.array([0.0, 0.0, -1.0]),
                         np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8), 1, -1, geomid)
    model.geom_rgba[:] = rgba
    return dist, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geomid[0])


def test_tray_profile_and_open_holes(model):
    """Down-rays in the top frame: holes end on the catch, the cavity on the plate, the rim on the face."""
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    T_top = PartState(model, d, TASK).pose(TOP)
    s, leg, top = TASK.screws[0], TASK.part(LEGS[3]), TASK.part(TOP)

    def hit_z(x, y):
        dist, name = _ray_collision(model, d, (T_top @ np.array([x, y, 0.05, 1.0]))[:3])
        return 0.05 - dist, name

    for cx, cy in top.hole_centers:
        z, name = hit_z(cx, cy)
        assert math.isclose(z, s.catch_z + leg.tip_z, abs_tol=1e-6) and name.startswith(f"{TOP}_catch")
    for xy in [(0.0, 0.0), (0.03, -0.02), (0.07, 0.0)]:
        assert math.isclose(hit_z(*xy)[0], top.plate_top_z, abs_tol=1e-6)
    assert math.isclose(hit_z(0.079, 0.0)[0], top.top_z, abs_tol=1e-6)                # rim
    assert math.isclose(hit_z(0.05625 + 0.015, 0.05625)[0], top.top_z, abs_tol=1e-6)  # boss ring


def test_environment_friction_is_averaged(model):
    d = mujoco.MjData(model)
    st = PartState(model, d, TASK)
    mujoco.mj_forward(model, d)
    W = np.asarray(TASK.scene.W_origin)
    st.set_pose(TOP, make_T(W + np.array([0.3715 - 0.08125 + 0.0002, 0.0, 0.0155])))   # touching the front fence
    mujoco.mj_forward(model, d)
    fr = {}
    for i in range(d.ncon):
        c = d.contact[i]
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if TOP in (n1 or "") + (n2 or ""):
            other = n1 if not (n1 or "").startswith(TOP) else n2
            fr[other] = c.friction[0]
    assert math.isclose(fr["table_top"], 0.125, rel_tol=1e-6)
    assert math.isclose(fr["fence_front"], 0.575, rel_tol=1e-6)


def test_parts_settle_on_table(model):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    st = PartState(model, d, TASK)
    for _ in range(500):
        mujoco.mj_step(model, d)
    for p in TASK.part_names:
        assert np.linalg.norm(d.qvel[st.dof_adr[p]:st.dof_adr[p] + 6]) < 1e-3
        assert abs(st.pose_W(p)[2, 3] - TASK.placement(p).pos[2]) < 0.001


def test_generated_meshes_are_watertight(tmp_path):
    for part, path in build_task_meshes(TASK, root=str(tmp_path), force=True).items():
        mesh = trimesh.load(path)
        assert mesh.is_watertight and mesh.volume > 0, part
