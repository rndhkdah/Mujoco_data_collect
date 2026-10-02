"""Procedural visual meshes: helical threads, threaded studs and trays with threaded bosses.

Collision never uses these meshes (MuJoCo would convex-hull them); they are render-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict

import numpy as np
import trimesh

from fsim.spec.types import LegPartSpec, TaskSpec, TrayPartSpec

SEGMENTS_PER_TURN = 48
CIRCLE_SEGMENTS = 48


def helical_ridge(r_root: float, r_crest: float, pitch: float, z0: float, z1: float,
                  right_handed: bool = True, root_frac: float = 0.62, crest_frac: float = 0.18,
                  segments_per_turn: int = SEGMENTS_PER_TURN) -> trimesh.Trimesh:
    """Closed trapezoid-profile ridge swept along a helix between z0 and z1 (z0 < z1).

    For a male thread r_root < r_crest (ridge points outward); for a female thread
    r_root > r_crest (ridge points inward from the bore wall).
    """
    w_root, w_crest = root_frac * pitch, crest_frac * pitch
    span = (z1 - z0) - w_root
    if span <= 0:
        raise ValueError("thread section shorter than one profile width")
    theta_end = 2.0 * math.pi * span / pitch
    n = max(2, int(math.ceil(theta_end / (2.0 * math.pi) * segments_per_turn)) + 1)
    thetas = np.linspace(0.0, theta_end, n)
    handed = 1.0 if right_handed else -1.0
    profile = [(r_root, -w_root / 2), (r_crest, -w_crest / 2), (r_crest, w_crest / 2), (r_root, w_root / 2)]
    verts = []
    for th in thetas:
        zc = z0 + w_root / 2 + pitch * th / (2.0 * math.pi)
        c, s = math.cos(handed * th), math.sin(handed * th)
        for r, dz in profile:
            verts.append((r * c, r * s, zc + dz))
    verts = np.asarray(verts)
    faces = []
    for i in range(n - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            faces.append((a + k, b + k, b + k2))
            faces.append((a + k, b + k2, a + k2))
    faces.append((0, 1, 2)); faces.append((0, 2, 3))
    e = 4 * (n - 1)
    faces.append((e, e + 2, e + 1)); faces.append((e, e + 3, e + 2))
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    if mesh.volume < 0:          # female (inward) or left-handed sweeps come out inside-out
        mesh.invert()
    return mesh


def _cylinder(radius: float, z0: float, z1: float, segments: int = CIRCLE_SEGMENTS) -> trimesh.Trimesh:
    cyl = trimesh.creation.cylinder(radius=radius, height=z1 - z0, sections=segments)
    cyl.apply_translation((0.0, 0.0, (z0 + z1) / 2.0))
    return cyl


def leg_stud_mesh(spec: LegPartSpec) -> trimesh.Trimesh:
    """Threaded stud in the leg frame: core cylinder plus external helical ridge."""
    z0, z1 = spec.tip_z, -spec.body_length / 2.0
    t = spec.thread
    core = _cylinder(t.minor_radius, z0, z1 + 0.0005)
    ridge = helical_ridge(t.minor_radius - 0.0002, t.major_radius, t.pitch, z0, z1, t.right_handed)
    return trimesh.util.concatenate([core, ridge])


def tray_mesh(spec: TrayPartSpec) -> trimesh.Trimesh:
    """Open tray (plate + rim) with a boss around each blind hole and internal thread ridges."""
    from manifold3d import Manifold

    sx, sy, sz = spec.size
    rim, z_lo, z_hi = spec.rim_width, spec.plate_top_z, spec.top_z
    t = spec.thread
    tray = Manifold.cube((sx, sy, sz), center=True)
    cavity = Manifold.cube((sx - 2 * rim, sy - 2 * rim, z_hi - z_lo + 0.002), center=False)
    tray = tray - cavity.translate((-(sx - 2 * rim) / 2, -(sy - 2 * rim) / 2, z_lo))
    for cx, cy in spec.hole_centers:
        boss = Manifold.cylinder(z_hi - z_lo, spec.boss_outer_radius, spec.boss_outer_radius, CIRCLE_SEGMENTS)
        tray = tray + boss.translate((cx, cy, z_lo))
    for cx, cy in spec.hole_centers:
        hole = Manifold.cylinder(z_hi - z_lo + 0.002, t.major_radius, t.major_radius, CIRCLE_SEGMENTS)
        tray = tray - hole.translate((cx, cy, z_lo))
    out = tray.to_mesh()
    parts = [trimesh.Trimesh(vertices=np.asarray(out.vert_properties)[:, :3], faces=np.asarray(out.tri_verts))]
    for cx, cy in spec.hole_centers:
        ridge = helical_ridge(t.major_radius + 0.0002, t.minor_radius, t.pitch, z_lo, z_hi - 0.0005, t.right_handed)
        ridge.apply_translation((cx, cy, 0.0))
        parts.append(ridge)
    return trimesh.util.concatenate(parts)


def _spec_hash(task: TaskSpec) -> str:
    """Hash of the part specs and of this generator's source (line endings normalised), so a
    change to either never reuses stale cached meshes."""
    with open(__file__, "rb") as f:
        source = f.read().replace(b"\r\n", b"\n")
    payload = json.dumps([asdict(p) for p in task.parts], sort_keys=True, default=str).encode()
    return hashlib.sha1(payload + b"|" + hashlib.sha1(source).digest()).hexdigest()[:10]


def generated_dir(root: str | None = None) -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.abspath(root or os.environ.get("FSIM_GENERATED", os.path.join(here, "assets", "generated")))


def build_task_meshes(task: TaskSpec, root: str | None = None, force: bool = False) -> dict[str, str]:
    """Write the visual meshes of every part of `task`; return {part name: STL path}.

    Files land in <root>/<task>-<spec hash>/ so changed parameters never reuse stale meshes.
    """
    out_dir = os.path.join(generated_dir(root), f"{task.name}-{_spec_hash(task)}")
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    for part in task.parts:
        path = os.path.join(out_dir, f"{part.name}.stl")
        if force or not os.path.isfile(path):
            mesh = tray_mesh(part) if isinstance(part, TrayPartSpec) else leg_stud_mesh(part)
            tmp = path + ".tmp"
            mesh.export(tmp, file_type="stl")
            os.replace(tmp, path)
        files[part.name] = path
    return files
