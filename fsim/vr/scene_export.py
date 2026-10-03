"""Export the visual scene of a compiled MuJoCo model as GLB (one node per body) + a manifest,
and pack body poses into compact binary frames for the 3D WebXR client.

Coordinates stay MuJoCo's (z up, metres); the client rotates them into WebXR's y-up frame.
Binary pose frame (little endian):
  uint8 type=2, 3 pad bytes, uint32 seq, float32 sim_time,
  float32[7*N] body poses (pos xyz, quat wxyz) in manifest order,
  float32[7] commanded grip-site target (pos, quat wxyz; zeros when none), float32[4] status flags
"""
from __future__ import annotations

import json
import re
import struct

import mujoco
import numpy as np
import trimesh

from fsim.geometry import make_T, quat_to_mat

MSG_POSES = 2
MSG_WRIST_JPEG = 3
STATUS_FLAGS = ("clutched", "precision", "stale", "gripper")


def _geom_mesh(model: mujoco.MjModel, g: int) -> trimesh.Trimesh | None:
    t, size = model.geom_type[g], model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        return trimesh.Trimesh(vertices=model.mesh_vert[v0:v0 + nv].copy(), faces=model.mesh_face[f0:f0 + nf].copy(),
                               process=False)
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return trimesh.creation.box(extents=2 * size)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return trimesh.creation.icosphere(subdivisions=2, radius=size[0])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return trimesh.creation.capsule(height=2 * size[1], radius=size[0], count=[16, 16])
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        m = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        m.apply_scale(size)
        return m
    if t == mujoco.mjtGeom.mjGEOM_PLANE:
        sx = size[0] if size[0] > 0 else 2.0
        sy = size[1] if size[1] > 0 else 2.0
        m = trimesh.creation.box(extents=(2 * sx, 2 * sy, 0.002))
        m.apply_translation((0, 0, -0.001))
        return m
    return None


def _geom_rgba(model: mujoco.MjModel, g: int) -> np.ndarray:
    rgba = np.array(model.geom_rgba[g], float)
    mat = model.geom_matid[g]
    if mat >= 0 and np.allclose(rgba, (0.5, 0.5, 0.5, 1.0)):
        rgba = np.array(model.mat_rgba[mat], float)
    return rgba


def export_scene(model: mujoco.MjModel, groups=(1,), exclude: str | None = None,
                 crease_deg: float = 30.0) -> tuple[bytes, dict]:
    """GLB bytes and manifest {"bodies": [{"id", "name", "node"}], ...} for the visible geoms in `groups`.

    exclude: regex on geom names to leave out (e.g. robosuite's floor and walls).
    crease_deg: vertex normals are split at edges sharper than this, so curved links shade smoothly
    and boxes keep crisp edges.
    """
    skip = re.compile(exclude) if exclude else None
    per_body: dict[int, list] = {}
    for g in range(model.ngeom):
        if int(model.geom_group[g]) not in groups:
            continue
        if skip is not None and skip.search(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""):
            continue
        rgba = _geom_rgba(model, g)
        if rgba[3] <= 0.0:
            continue
        mesh = _geom_mesh(model, g)
        if mesh is None:
            continue
        mesh.apply_transform(make_T(model.geom_pos[g], quat_to_mat(model.geom_quat[g])))
        mesh = trimesh.graph.smooth_shade(mesh, angle=np.radians(crease_deg))
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile((np.clip(rgba, 0, 1) * 255).astype(np.uint8),
                                                                             (len(mesh.vertices), 1)))
        per_body.setdefault(int(model.geom_bodyid[g]), []).append(mesh)
    scene = trimesh.Scene()
    bodies = []
    for b in sorted(per_body):
        node = f"b{b}"
        scene.add_geometry(trimesh.util.concatenate(per_body[b]), node_name=node, geom_name=f"g{b}")
        bodies.append({"id": b, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}", "node": node})
    glb = scene.export(file_type="glb", include_normals=True)
    manifest = {"bodies": bodies, "flags": list(STATUS_FLAGS), "pose_floats": 7, "up": "z"}
    return glb, manifest


class PosePacker:
    """Packs the manifest bodies' poses into the binary frame described above."""

    def __init__(self, manifest: dict):
        self.ids = np.array([b["id"] for b in manifest["bodies"]], int)
        self.seq = 0

    def pack(self, data: mujoco.MjData, target=None, flags=None) -> bytes:
        poses = np.concatenate([data.xpos[self.ids], data.xquat[self.ids]], axis=1).astype(np.float32).ravel()
        tgt = np.zeros(7, np.float32) if target is None else np.asarray(target, np.float32)
        fl = np.zeros(len(STATUS_FLAGS), np.float32) if flags is None else np.asarray(flags, np.float32)
        self.seq += 1
        return struct.pack("<B3xIf", MSG_POSES, self.seq & 0xFFFFFFFF, float(data.time)) + poses.tobytes() \
            + tgt.tobytes() + fl.tobytes()


def unpack_poses(buf: bytes, n_bodies: int) -> dict:
    """Inverse of PosePacker.pack (tests and debugging)."""
    kind, seq, t = struct.unpack_from("<B3xIf", buf, 0)
    floats = np.frombuffer(buf, np.float32, offset=12)
    poses = floats[:7 * n_bodies].reshape(n_bodies, 7)
    target = floats[7 * n_bodies:7 * n_bodies + 7]
    flags = floats[7 * n_bodies + 7:]
    return {"type": kind, "seq": seq, "time": t, "poses": poses, "target": target, "flags": flags}


def manifest_json(manifest: dict) -> str:
    return json.dumps(manifest)
