"""Visual meshes for every part type, and the per-task mesh directory.

Tray and leg meshes come from fsim/models/meshes.py, which also owns the directory hash. That file
is kept unchanged so existing tasks keep their hash (recorded demos reference the mesh paths); a
task with primitive parts also hashes this file. Primitive parts get a CSG mesh: geoms minus
sockets plus stud cores, with helical thread ridges. If CSG fails the part keeps primitive visuals.
"""
from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import trimesh

from fsim.fileutil import atomic_replace, unique_tmp
from fsim.geometry import make_T, quat_to_mat
from fsim.models import meshes
from fsim.models.meshes import CIRCLE_SEGMENTS, generated_dir, helical_ridge  # noqa: F401  (generated_dir re-exported)
from fsim.spec.types import Geom, PrimitivePartSpec, TaskSpec, TrayPartSpec

_EPS = 0.001


def _T34(pos, quat) -> np.ndarray:
    return make_T(pos, quat_to_mat(quat))[:3, :]


def _solid(g: Geom):
    from manifold3d import Manifold

    if g.type == "box":
        m = Manifold.cube(tuple(2.0 * np.asarray(g.size)), center=True)
    elif g.type == "sphere":
        m = Manifold.sphere(g.size[0], CIRCLE_SEGMENTS)
    elif g.type in ("cylinder", "capsule"):
        r, h = g.size
        m = Manifold.cylinder(2.0 * h, r, r, CIRCLE_SEGMENTS, center=True)
        if g.type == "capsule":
            for z in (-h, h):
                m = m + Manifold.sphere(r, CIRCLE_SEGMENTS).translate((0.0, 0.0, z))
    else:
        raise ValueError(g.type)
    return m.transform(_T34(g.pos, g.quat))


def primitive_mesh(spec: PrimitivePartSpec) -> trimesh.Trimesh:
    """Union of the geoms, minus sockets, plus stud cores; thread ridges appended (render only)."""
    from manifold3d import Manifold, OpType

    solid = Manifold.batch_boolean([_solid(g) for g in spec.geoms], OpType.Add)
    ridges = []
    for so in spec.sockets:
        T = _T34(so.pos, so.quat)
        if so.half_size is not None:
            cut = Manifold.cube((2 * so.half_size[0], 2 * so.half_size[1], so.depth + _EPS))
            cut = cut.translate((-so.half_size[0], -so.half_size[1], -so.depth))
        else:
            r = so.thread.major_radius if so.thread else so.radius
            cut = Manifold.cylinder(so.depth + _EPS, r, r, CIRCLE_SEGMENTS).translate((0.0, 0.0, -so.depth))
        solid = solid - cut.transform(T)
        if so.thread and so.half_size is None:
            t = so.thread
            ridges.append((helical_ridge(t.major_radius + 0.0002, t.minor_radius, t.pitch, -so.depth, -0.0005,
                                         t.right_handed), so.pos, so.quat))
    for st in spec.studs:
        r = st.thread.minor_radius if st.thread else st.radius
        core = Manifold.cylinder(st.length + 0.0005, r, r, CIRCLE_SEGMENTS).translate((0.0, 0.0, -0.0005))
        solid = solid + core.transform(_T34(st.pos, st.quat))
        if st.thread:
            t = st.thread
            ridges.append((helical_ridge(t.minor_radius - 0.0002, t.major_radius, t.pitch, 0.0, st.length,
                                         t.right_handed), st.pos, st.quat))
    out = solid.to_mesh()
    parts = [trimesh.Trimesh(vertices=np.asarray(out.vert_properties)[:, :3], faces=np.asarray(out.tri_verts))]
    for ridge, pos, quat in ridges:
        ridge.apply_transform(make_T(pos, quat_to_mat(quat)))
        parts.append(ridge)
    mesh = trimesh.util.concatenate(parts)
    if not len(mesh.faces) or not math.isfinite(mesh.volume):
        raise ValueError("empty mesh")
    return mesh


def spec_hash(task: TaskSpec) -> str:
    """meshes.py's hash (unchanged for tray / leg tasks), extended by this file for primitive parts."""
    h = meshes._spec_hash(task)
    if not any(isinstance(p, PrimitivePartSpec) for p in task.parts):
        return h
    with open(__file__, "rb") as f:
        source = f.read().replace(b"\r\n", b"\n")
    return hashlib.sha1(h.encode() + b"|" + hashlib.sha1(source).digest()).hexdigest()[:10]


def task_mesh_dir(task: TaskSpec, root: str | None = None) -> str:
    return os.path.join(generated_dir(root), f"{task.name}-{spec_hash(task)}")


def build_task_meshes(task: TaskSpec, root: str | None = None, force: bool = False) -> dict[str, str]:
    """Write the visual meshes of every part of `task`; return {part name: STL path}. Primitive parts
    whose CSG fails (or with visual="primitive") are left out and render their primitives.

    Files land in <root>/<task>-<spec hash>/ so changed parameters never reuse stale meshes.
    """
    out_dir = task_mesh_dir(task, root)
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for part in task.parts:
        if isinstance(part, PrimitivePartSpec) and part.visual != "mesh":
            continue
        path = os.path.join(out_dir, f"{part.name}.stl")
        if force or not os.path.isfile(path):
            try:
                if isinstance(part, PrimitivePartSpec):
                    mesh = primitive_mesh(part)
                else:
                    mesh = meshes.tray_mesh(part) if isinstance(part, TrayPartSpec) else meshes.leg_stud_mesh(part)
            except Exception as e:  # noqa: BLE001  (CSG failures fall back to primitive visuals)
                if not isinstance(part, PrimitivePartSpec):
                    raise
                print(f"[meshes] {part.name}: CSG failed ({e}); using primitive visuals", flush=True)
                continue
            tmp = unique_tmp(path)
            mesh.export(tmp, file_type="stl")
            atomic_replace(tmp, path)
        files[part.name] = path
    return files
