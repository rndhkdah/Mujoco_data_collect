"""The generic mechanisms end to end on a test-only task (tests/synthetic_task.py)."""
import math

import mujoco
import numpy as np
import pytest

from fsim.geometry import make_T
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.insert import InsertManager
from fsim.sim.screw import ENGAGED, FREE, SEATED
from fsim.sim.screw_trials import Rig, drive_site
from synthetic_task import KNOB_SCREW, NUT_SCREW, PEG_SCREW, SLEEVE_INSERT, SYNTHETIC
from task_checks import check_force_seat, check_hand_drive, check_reset, check_spec, compile_scene

TASK = SYNTHETIC
LEVEL_POSE = make_T(np.asarray(TASK.scene.W_origin) + (0.1, 0.0, 0.3))      # base flat, in mid-air


def test_spec_sanity():
    check_spec(TASK)


def test_scene_compiles_with_csg_meshes(tmp_path, monkeypatch):
    monkeypatch.setenv("FSIM_GENERATED", str(tmp_path))
    from fsim.models.part_meshes import build_task_meshes
    files = build_task_meshes(TASK)
    assert set(files) == {"base", "peg", "knob", "nut", "cap"}          # sleeve: visual="primitive"
    m = compile_scene(TASK)
    for p in TASK.parts:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{p.name}_main")
        assert math.isclose(m.body_mass[b], p.mass, rel_tol=1e-9) and (m.body_inertia[b] > 0).all()


@pytest.mark.parametrize("level", ["low", "med"])
def test_reset_settles(level):
    check_reset(TASK, level)


def test_part_pos_range_override():
    from fsim.sim.reset import ResetSampler
    base = TASK.placement("base").pos
    for seed in range(10):
        pl = ResetSampler(TASK).sample(np.random.default_rng(seed), "med")[0]
        assert -0.05 - 1e-12 <= pl.pos[0] - base[0] <= 0.02 + 1e-12 and abs(pl.pos[1] - base[1]) <= 0.03 + 1e-12


def test_force_seat_assembles_everything():
    check_force_seat(TASK)


@pytest.mark.parametrize("spec", [PEG_SCREW, NUT_SCREW, SLEEVE_INSERT], ids=["tilted", "inverted", "insert"])
def test_hand_drive_axis_up(spec):
    check_hand_drive(TASK, spec)


@pytest.mark.parametrize("spec", [PEG_SCREW, NUT_SCREW, SLEEVE_INSERT], ids=["tilted", "inverted", "insert"])
def test_hand_drive_with_level_base(spec):
    """The base lies flat, so the tilted socket's axis is 25 degrees off vertical in the world."""
    check_hand_drive(TASK, spec, parent_T=LEVEL_POSE)


def test_chained_screw_into_seated_peg():
    rig = Rig(TASK)
    rig.hold("base", LEVEL_POSE)
    rig.screw.force_seat("peg", "base", 0)
    mujoco.mj_forward(rig.model, rig.data)
    r = drive_site(rig, KNOB_SCREW, pin=False)
    assert r.engaged and r.seated and r.assembled
    assert rig.screw.child_state("peg") == SEATED
    checker = AssemblyChecker(TASK)
    checker.update(rig.state.poses())
    assert {("base", "peg"), ("peg", "knob")} <= checker.assembled


def test_wrong_direction_never_engages():
    rig = Rig(TASK)
    r = drive_site(rig, PEG_SCREW, rate_deg_s=-120.0, timeout_s=5.0)
    assert not r.engaged


def test_insert_pulls_out_before_seating_and_locks_when_seated():
    rig = Rig(TASK)
    rig.hold("base", LEVEL_POSE)
    slot = rig.inserts.slots[0]
    S = LEVEL_POSE @ slot.T_local
    mate_inv = np.linalg.inv(slot.mate["sleeve"])
    T = S @ make_T((0, 0, 0.02)) @ mate_inv                      # halfway down the pole
    rig.state.set_pose("sleeve", T)
    mujoco.mj_forward(rig.model, rig.data)
    rig.inserts.update()
    assert rig.inserts.child_state("sleeve") == ENGAGED
    rig.hand.grasp("sleeve", T)
    rig.hand.move_to(S @ make_T((0, 0, 0.08)) @ mate_inv, 1.0)   # pull it off
    assert rig.inserts.child_state("sleeve") == FREE
    assert any(e["event"] == "disengage" for e in rig.inserts.events)
    rig.hand.release()


def test_insert_snapshot_restore_and_reset():
    rig = Rig(TASK)
    rig.inserts.force_seat("sleeve", "base", 0)
    snap = rig.inserts.snapshot()
    rig.inserts.reset()
    assert rig.inserts.child_state("sleeve") == FREE
    rig.inserts.restore(snap)
    assert rig.inserts.child_state("sleeve") == SEATED
    rig.inserts.restore({})                                     # files recorded before inserts existed
    assert rig.inserts.child_state("sleeve") == FREE
    fresh = InsertManager(rig.model, rig.data, TASK)
    assert fresh.child_state("sleeve") == FREE


def test_phases_walk_with_real_managers():
    """Every phase kind completes, in order, from the seated scene (no grasp callback: motion proxies)."""
    from fsim.sim.phases import PhaseTracker
    rig = Rig(TASK)
    tr = PhaseTracker(TASK)
    tr.reset({p: rig.state.pose_W(p) for p in TASK.part_names})
    assert tr.current == "grasp_base" and not tr.update(tr.initial, rig.screw, set(), rig.inserts)
    W = np.asarray(TASK.scene.W_origin)
    rig.hold("base", make_T(W + (*TASK.extras["corner_xy"], 0.2)))
    for spec in (PEG_SCREW, KNOB_SCREW, NUT_SCREW):
        rig.screw.force_seat(spec.children[0], spec.parent, next(h.index for h in rig.screw.holes if h.screw is spec))
        mujoco.mj_kinematics(rig.model, rig.data)
    rig.inserts.force_seat("sleeve", "base", 0)
    mujoco.mj_forward(rig.model, rig.data)
    poses = {p: rig.state.pose_W(p) for p in TASK.part_names}
    poses["cap"][2, 3] += 0.05                                    # lifted
    done = []
    for _ in TASK.phases:
        name = tr.current
        if tr.update(poses, rig.screw, {("base", "cap")}, rig.inserts):
            done.append(name)
    assert tr.done and done == list(TASK.phase_labels)


def test_robosuite_env(tmp_path, monkeypatch):
    pytest.importorskip("robosuite")
    from task_checks import check_env
    monkeypatch.setenv("FSIM_GENERATED", str(tmp_path))
    check_env(TASK)


def test_many_geoms_sphere_and_inertia_override():
    """A lamp-hood-like cone of 24 staves plus a sphere: CSG mesh, compile, and the inertia override."""
    import xml.etree.ElementTree as ET
    from dataclasses import replace

    from fsim.geometry import mat_to_quat, rot_y, rot_z
    from fsim.models.part_meshes import primitive_mesh
    from fsim.models.parts import build_part_body, part_inertial
    from fsim.spec.types import Geom, PrimitivePartSpec

    staves = tuple(Geom("box", (0.004, 0.012, 0.03), tuple(rot_z(a) @ (0.04, 0, 0)),
                        tuple(mat_to_quat(rot_z(a) @ rot_y(0.4)))) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False))
    hood = PrimitivePartSpec("hood", 0.05, staves + (Geom("sphere", (0.02,)),) + tuple(
        Geom("cylinder", (0.003, 0.01), (0.0, 0.0, 0.03 + 0.002 * i)) for i in range(5)))
    assert len(hood.geoms) == 30
    mesh = primitive_mesh(hood)
    assert len(mesh.faces) > 100 and mesh.volume > 0
    com, mass, I = part_inertial(hood)
    assert len(I) == 6 and mass == 0.05
    over = replace(hood, com=(0.0, 0.0, 0.01), inertia=(1e-5, 1e-5, 2e-5))
    com2, _, I2 = part_inertial(over)
    assert np.allclose(com2, (0, 0, 0.01)) and len(I2) == 3
    for spec in (hood, over):
        root = ET.fromstring('<mujoco><compiler angle="radian"/><worldbody/></mujoco>')
        root.find("worldbody").append(build_part_body(TASK, spec, (0, 0, 0.5), (1, 0, 0, 0)))
        m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        assert m.ngeom == 60 and math.isclose(m.body_mass[1], 0.05)
