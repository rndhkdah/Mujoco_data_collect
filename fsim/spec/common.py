"""Scene pieces shared by every FurnitureBench-style task: table, fence, robot base, cameras, reset
rectangles, randomization levels, the AprilTag-frame conversion for measured layouts, and small
geometry helpers for part specs.

Design source: FurnitureBench (Heo et al., RSS 2023; MIT License), re-expressed in this project's
frames (see fsim/spec/types.py).
"""
from __future__ import annotations

import math

import numpy as np

from fsim.geometry import inv_T, make_T, mat_to_quat, orthonormalize, pose_to_T, quat_to_mat, rot_x, rot_z
from fsim.spec.types import (
    BoxObstacle, CameraSpec, Geom, InsertSpec, LegPartSpec, PartSpec, Placement, PrimitivePartSpec, Quat, RandomLevel,
    SceneSpec, ScrewSpec, Stud, Vec3,
)

# --- Measured layouts ------------------------------------------------------------------------
# FurnitureBench records its "high" layouts in its AprilTag base-tag frame and its part frames:
#   tag frame -> W:   p_W = (0.0015 + p_y, p_x, -p_z),  R_W = M @ R_tag
#   their part frame -> ours: our (x, y, z) axes = their (x, -z, y)  ->  R_ours = R_W @ C
_M = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_C = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


FB_AXES = make_T(R=_C)       # default: our part frame = theirs with axes remapped, same origin


def tag_pose_to_placement(part: str, pos_tag, rot_tag, fb_T_part: np.ndarray | None = None) -> Placement:
    """Convert a FurnitureBench base-tag-frame part pose into a W-frame Placement.

    fb_T_part: our part frame expressed in their part frame (4x4: axis map and origin offset, e.g.
    their frame sits at a tag, not at the bbox centre); default FB_AXES."""
    p = np.asarray(pos_tag, dtype=float)
    if fb_T_part is None:
        R = orthonormalize(_M @ np.asarray(rot_tag, dtype=float) @ _C)
        return Placement(part, (0.0015 + p[1], p[0], -p[2]), tuple(mat_to_quat(R)))
    R_fb = _M @ np.asarray(rot_tag, dtype=float)
    pos = np.array([0.0015 + p[1], p[0], -p[2]]) + R_fb @ np.asarray(fb_T_part)[:3, 3]
    R = orthonormalize(R_fb @ np.asarray(fb_T_part)[:3, :3])
    return Placement(part, tuple(float(v) for v in pos), tuple(mat_to_quat(R)))


# --- Scene -----------------------------------------------------------------------------------
# Reset rectangles of the design (they sit 2 cm closer to the parts than the built fence).
RESET_OBSTACLES = (
    BoxObstacle("reset_front", (0.3615, 0.0, 0.0), (0.01, 0.175, 0.015)),
    BoxObstacle("reset_left", (0.3065, 0.175, 0.0), (0.085, 0.01, 0.015)),
    BoxObstacle("reset_right", (0.3065, -0.175, 0.0), (0.085, 0.01, 0.015)),
)
LOW = RandomLevel(0.015, 15.0)
MED = RandomLevel(0.05, 45.0)
LEVELS = {"low": LOW, "med": MED}

FENCE_FRONT = BoxObstacle("fence_front", (0.3815, 0.0, 0.015), (0.01, 0.175, 0.015))
FENCE_LEFT = BoxObstacle("fence_left", (0.3065, 0.175, 0.015), (0.085, 0.01, 0.015))
FENCE_RIGHT = BoxObstacle("fence_right", (0.3065, -0.175, 0.015), (0.085, 0.01, 0.015))
ROBOT_BASE = (-0.3, 0.0, 0.43)
OPERATOR_CAMERA = "operator"
_REAR_POS = (ROBOT_BASE[0] + 0.08, 0.0, ROBOT_BASE[2] + 0.2)
SCENE = SceneSpec(
    robot_base=ROBOT_BASE,
    obstacles=(FENCE_FRONT, FENCE_LEFT, FENCE_RIGHT),
    cameras=(
        CameraSpec("front", (0.90, 0.0, 0.65), (-1.0, 0.0, 0.30), fovy=40.0),
        CameraSpec("rear", _REAR_POS,
                   (_REAR_POS[0] + math.cos(math.radians(35)), 0.0, _REAR_POS[2] - math.sin(math.radians(35))),
                   fovy=40.0),
        # At the VR operator's mean head pose over the one_leg demos (robot's left, looking down across the table).
        CameraSpec("side", (0.27, 0.59, 1.04), (0.25, -0.05, 0.45), fovy=60.0),
        # Above the table's far corner on the robot's left, looking down across the workspace at the arm.
        CameraSpec("corner", (0.70, 0.45, 1.15), (0.05, 0.0, 0.55), fovy=60.0),
        # The VR operator's view (--record-operator): the collector moves it to the headset pose each
        # step. This default is the page's starting viewpoint, at the robot's left-rear.
        CameraSpec(OPERATOR_CAMERA, (-0.62, 0.475, 1.015), (0.2, 0.0, 0.415), fovy=90.0),
    ),
)


def corner_extras(size_x: float, size_y: float, tol: float = 0.02, yaw_tol_deg: float = 15.0) -> dict:
    """Task extras for a place_corner phase: a size_x-by-size_y footprint snug in the +y corner of the fence."""
    xy = (FENCE_FRONT.center[0] - FENCE_FRONT.half_size[0] - size_x / 2,
          FENCE_LEFT.center[1] - FENCE_LEFT.half_size[1] - size_y / 2)
    return {"corner_xy": xy, "corner_tol": tol, "corner_yaw_tol_deg": yaw_tol_deg}


# --- Part geometry helpers -------------------------------------------------------------------
def ring_geoms(pos: Vec3, quat: Quat, r_in: float, r_out: float, z0: float, z1: float,
               segments: int = 8) -> tuple[Geom, ...]:
    """Polygonal tube of boxes around a bore (collision for sockets and nuts): inscribed radius
    r_in, outer apothem r_out, spanning z0..z1 of the frame (pos, quat) in the part frame. The bore
    stays clear; boxes only overlap outside it."""
    T = make_T(pos, quat_to_mat(quat))
    half_t = r_out * math.tan(math.pi / segments)
    r_mid = (r_in + r_out) / 2.0
    out = []
    for k in range(segments):
        a = 2.0 * math.pi * k / segments
        L = T @ make_T((r_mid * math.cos(a), r_mid * math.sin(a), (z0 + z1) / 2.0), rot_z(a))
        out.append(Geom("box", ((r_out - r_in) / 2.0, half_t, (z1 - z0) / 2.0), tuple(L[:3, 3]), tuple(mat_to_quat(L[:3, :3]))))
    return tuple(out)


# --- Mating ----------------------------------------------------------------------------------
def child_mate(spec: ScrewSpec | InsertSpec, child: PartSpec) -> tuple[np.ndarray, float]:
    """(mate frame in the child frame, how far the child protrudes beyond it along the mate's -z):
    spec.child_mate, else the child's `child_feature` (or only stud / socket) flipped about x."""
    if spec.child_mate is not None:
        return pose_to_T(*spec.child_mate), 0.0
    if isinstance(child, LegPartSpec):
        return make_T((0.0, 0.0, -child.body_length / 2.0)), child.screw_length
    if isinstance(child, PrimitivePartSpec):
        feats = child.studs + child.sockets
        if spec.child_feature:
            f = child.feature(spec.child_feature)
        elif len(feats) == 1:
            f = feats[0]
        else:
            raise ValueError(f"{child.name}: set child_feature (features: {[x.name for x in feats]})")
        T = make_T(f.pos, quat_to_mat(f.quat) @ rot_x(math.pi))
        return T, f.length if isinstance(f, Stud) else 0.0
    raise ValueError(f"{child.name}: no mate feature; set child_mate")


def seated_rel_pose(spec: ScrewSpec | InsertSpec, site: int, child: PartSpec) -> tuple[Vec3, Quat]:
    """Child pose in the parent frame when seated at spec.sites[site] (seat_gap included), for
    AssemblyTarget.rel_poses."""
    s = spec.sites[site]
    T = make_T(s.pos, quat_to_mat(s.quat)) @ make_T((0.0, 0.0, spec.seat_gap)) @ inv_T(child_mate(spec, child)[0])
    return tuple(float(v) for v in T[:3, 3]), tuple(float(v) for v in mat_to_quat(T[:3, :3]))
