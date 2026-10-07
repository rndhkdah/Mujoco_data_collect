"""Reusable checks that any registered task (or a test-only TaskSpec) must pass.

Used by tests/test_tasks_generic.py (every task in fsim.spec.registry.TASKS) and
tests/test_synthetic_task.py. Each helper raises AssertionError with a readable message.
"""
from __future__ import annotations

import mujoco
import numpy as np

from fsim.geometry import make_T, pose_to_T, quat_to_mat
from fsim.models.parts import mechanism_sites, parts_aabb
from fsim.models.scene import build_scene_xml
from fsim.sim.assembly import AssemblyChecker
from fsim.sim.reset import ResetSampler, max_penetration, place_and_settle
from fsim.sim.screw_trials import Rig, drive_site, site_holes
from fsim.sim.state import PartState
from fsim.spec.types import PHASE_KINDS, PrimitivePartSpec, ScrewSpec, TaskSpec

AIR_Z = 0.3            # roots are pinned this high above the table while seating children


def mechanisms(task: TaskSpec) -> list:
    return list(task.screws) + list(task.inserts)


def check_spec(task: TaskSpec):
    names = set(task.part_names)
    assert len(names) == len(task.parts), "duplicate part names"
    r = task.randomization
    for p in task.part_names:
        assert p in r.footprints, f"{p}: no footprint"
    assert sorted(pl.part for pl in task.layout) == sorted(names), "layout must place every part once"
    assert {"low", "med"} <= set(r.levels), "low and med levels required"
    for preset in r.high_presets:
        assert [pl.part for pl in preset] == [pl.part for pl in task.layout], "high preset parts must follow the layout"
    for spec in mechanisms(task):
        assert spec.parent in names and set(spec.children) <= names, f"unknown part in {spec}"
        assert spec.parent not in spec.children
    sites = mechanism_sites(task)                   # resolves features / mates, raises on bad specs
    for spec in mechanisms(task):
        assert any(s.spec is spec for s in sites), f"{spec.parent}: no sites"
    for t in task.targets:
        assert t.parent in names and t.child in names, f"target {t.parent}->{t.child}: unknown part"
        has_mech = any(s.parent == t.parent and t.child in s.children for s in mechanisms(task))
        assert has_mech or t.position_only, f"target {t.parent}->{t.child}: no screw / insert and not position_only"
        for req in t.requires:
            assert req in {(u.parent, u.child) for u in task.targets}, f"target requires unknown pair {req}"
    labels = [ph.name for ph in task.phases]
    assert len(set(labels)) == len(labels), "duplicate phase labels"
    for ph in task.phases:
        assert ph.kind in PHASE_KINDS, f"phase {ph.name}: unknown kind {ph.kind}"
        assert ph.part in names and (ph.target is None or ph.target in names), f"phase {ph.name}: unknown part"
        if ph.kind in ("insert", "screw", "slide", "place"):
            assert ph.target is not None, f"phase {ph.name} needs a target"
        if ph.kind == "place_corner":
            assert {"corner_xy", "corner_tol", "corner_yaw_tol_deg"} <= set(task.extras)
    for p in task.parts:
        if isinstance(p, PrimitivePartSpec):
            lo, hi = parts_aabb(p.geoms)
            assert np.abs(lo + hi).max() / 2 < 1e-3, f"{p.name}: origin must be the bounding-box center of its geoms"


def compile_scene(task: TaskSpec, **kw) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_scene_xml(task, **kw))


def check_reset(task: TaskSpec, level: str, seeds=(0, 1)):
    m = compile_scene(task, visual_meshes=False)
    d = mujoco.MjData(m)
    st = PartState(m, d, task)
    sampler = ResetSampler(task)
    layouts = list(task.randomization.high_presets) if level == "high" else \
        [sampler.sample(np.random.default_rng(s), level) for s in seeds]
    for layout in layouts:
        mujoco.mj_resetData(m, d)
        remaining = place_and_settle(m, d, st, layout, settle_steps=250)
        assert remaining < 0.001, f"{level}: parts still overlap by {1000 * remaining:.1f} mm after placement"
        pen = max_penetration(m, d, st.body_id.values())
        assert pen < 0.001, f"{level}: {1000 * pen:.1f} mm penetration after settling"
        for p in task.part_names:
            v = np.linalg.norm(d.qvel[st.dof_adr[p]:st.dof_adr[p] + 3])
            assert v < 0.05, f"{level}: {p} still moving ({v:.3f} m/s)"


def _depth(task: TaskSpec, part: str, seen=()) -> int:
    for spec in mechanisms(task):
        if part in spec.children and part not in seen:
            return 1 + _depth(task, spec.parent, seen + (part,))
    return 0


def pin_roots(rig: Rig):
    """Pin every mechanism root parent in mid-air (layout orientation), spread along y."""
    task, W = rig.task, np.asarray(rig.task.scene.W_origin)
    roots = sorted({s.parent for s in mechanisms(task) if _depth(task, s.parent) == 0}
                   | {t.parent for t in task.targets if _depth(task, t.parent) == 0})
    for i, root in enumerate(roots):
        q = task.placement(root).quat
        rig.hold(root, make_T(W + (0.1, 0.35 * (i - (len(roots) - 1) / 2), AIR_Z), quat_to_mat(q)))


def seat_all(rig: Rig) -> set:
    """Force-seat children at free sites (parents before children; target children first) and put
    position-only targets at their first rel pose. Returns the seated (parent, child) pairs."""
    task, seated, done = rig.task, set(), set()
    targeted = {t.child for t in task.targets}
    for spec in sorted(mechanisms(task), key=lambda s: _depth(task, s.parent)):
        mgr = rig.screw if isinstance(spec, ScrewSpec) else rig.inserts
        for child in sorted(spec.children, key=lambda c: c not in targeted):
            free = [h for h in site_holes(rig, spec) if h.child is None]
            if child in done or not free:
                continue
            mgr.force_seat(child, spec.parent, free[0].index)
            seated.add((spec.parent, child))
            done.add(child)
    mujoco.mj_kinematics(rig.model, rig.data)
    for t in task.targets:
        if t.position_only and t.child not in done:
            rig.state.set_pose(t.child, rig.state.pose(t.parent) @ pose_to_T(*t.rel_poses[0]))
            mujoco.mj_kinematics(rig.model, rig.data)
    mujoco.mj_forward(rig.model, rig.data)
    return seated


def check_force_seat(task: TaskSpec, settle_s: float = 0.5):
    rig = Rig(task)
    pin_roots(rig)
    seated = seat_all(rig)
    checker = AssemblyChecker(task)
    for _ in range(len(task.targets)):           # `requires` chains latch over repeated updates
        checker.update(rig.state.poses())
    assert checker.all_assembled, f"not assembled after force_seat: {set((t.parent, t.child) for t in task.targets) - checker.assembled}"
    rig.settle(settle_s)
    fresh = AssemblyChecker(task)
    for _ in range(len(task.targets)):
        fresh.update(rig.state.poses())
    lost = {(t.parent, t.child) for t in task.targets if (t.parent, t.child) in seated} - fresh.assembled
    assert not lost, f"seated pairs fell apart while settling: {lost}"


def check_hand_drive(task: TaskSpec, spec, site: int = 0, child: str | None = None, **kw):
    """Hand-drive one child into one site; it must engage, seat, and (if targeted) be assembled."""
    rig = Rig(task)
    r = drive_site(rig, spec, site, child, **kw)
    name = f"{spec.parent}<-{child or spec.children[0]} site {site}"
    assert r.engaged, f"{name}: never engaged ({r.events})"
    assert r.seated, f"{name}: not seated ({r.events})"
    assert r.assembled in (None, True), f"{name}: seated but not assembled"
    return r


def check_env(task: TaskSpec, steps: int = 5):
    """robosuite env: builds, resets (low), steps, records / restores its snapshot, relocates its own
    model XML (the replay path), and force-seats every mechanism child without errors."""
    import json

    from fsim.env.furniture_env import make_env

    env = make_env(task=task, randomness="low", seed=0, has_offscreen_renderer=False, use_camera_obs=False)
    try:
        env.reset()
        for _ in range(steps):
            _, _, _, info = env.step(np.zeros(7))
        assert info["phase"] == 0 and isinstance(info["screw_events"], list)
        snap = json.loads(json.dumps(env.fsim_snapshot()))
        assert "inserts" in snap
        env.restore_snapshot({k: v for k, v in snap.items() if k != "inserts"})     # old recordings
        env.restore_snapshot(snap)
        ET_xml = env.edit_model_xml(env.sim.model.get_xml())
        assert "<mujoco" in ET_xml
        for spec in mechanisms(task):
            mgr = env.screw if isinstance(spec, ScrewSpec) else env.inserts
            holes = site_holes(type("R", (), {"screw": env.screw, "inserts": env.inserts})(), spec)
            mgr.force_seat(spec.children[0], spec.parent, holes[0].index)
            mujoco.mj_forward(env.sim.model._model, env.sim.data._data)
        env.step(np.zeros(7))
    finally:
        env.close()
