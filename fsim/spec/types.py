"""Data types for furniture assembly tasks.

Frames used throughout:
  * world: z up; the table surface sits at `SceneSpec.table_surface_z`.
  * W (table frame): origin on the table surface at the table center, axes as world,
    +x pointing away from the robot. Layouts and footprints are expressed in W.
  * part frames: origin at the part's bounding-box center, local z = assembly axis
    (tabletop: threaded holes open on its +z face; leg: long axis, screw on the -z end).
    PrimitivePartSpec: origin at the center of the bounding box of its `geoms`.
  * feature frames (Stud, Socket, Site): origin on the part surface (stud shoulder, socket mouth),
    +z pointing out of the part toward the mating part. Mated: the child feature's origin sits on
    the parent site's, z anti-parallel, x axes equal (the child "mate" frame is the child feature
    flipped about its x axis, so its z points into the child, along the site z).
Quaternions are (w, x, y, z).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Union

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ThreadSpec:
    """Visual thread geometry (collision does not use threads; see ScrewSpec)."""
    pitch: float           # m per turn
    minor_radius: float
    major_radius: float
    right_handed: bool = True


@dataclass(frozen=True)
class TrayPartSpec:
    """An open tray (e.g. a tabletop seen from below): a base plate, a rim around the edge and a
    threaded boss around each blind hole. Holes open on the +z face; their floor is the plate."""
    name: str
    size: Vec3                     # full extents (x, y, height)
    mass: float                    # kg
    hole_centers: tuple[Vec2, ...]  # (x, y) of each hole axis in the part frame
    plate_thickness: float
    rim_width: float
    boss_outer_radius: float
    bore_radius: float             # inner radius of the collision bore (inside the boss)
    thread: ThreadSpec             # female thread (visual)
    rgba: tuple[float, float, float, float] = (0.80, 0.68, 0.52, 1.0)

    @property
    def plate_top_z(self) -> float:
        return -self.size[2] / 2.0 + self.plate_thickness

    @property
    def top_z(self) -> float:
        return self.size[2] / 2.0


@dataclass(frozen=True)
class LegPartSpec:
    """A square-section leg with a threaded stud on its -z end."""
    name: str
    width: float                   # square body side
    body_length: float             # body spans z in [-body_length/2, +body_length/2]
    screw_length: float            # stud spans z in [-body_length/2 - screw_length, -body_length/2]
    mass: float
    peg_radius: float              # collision cylinder radius of the stud
    thread: ThreadSpec             # male thread (visual)
    rgba: tuple[float, float, float, float] = (0.85, 0.74, 0.58, 1.0)

    @property
    def tip_z(self) -> float:
        return -self.body_length / 2.0 - self.screw_length


@dataclass(frozen=True)
class Geom:
    """Collision (and fallback visual) primitive in the part frame."""
    type: str                      # "box" | "cylinder" | "capsule" | "sphere"
    size: tuple[float, ...]        # box: half extents (3); cylinder / capsule: (radius, half length along z); sphere: (radius,)
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = IDENTITY_QUAT


@dataclass(frozen=True)
class Stud:
    """External thread or peg; spans z in [0, length] of its frame (see feature frames above).
    Adds a collision cylinder of `radius`; `thread` (visual) wraps a core of thread.minor_radius."""
    pos: Vec3
    quat: Quat
    radius: float
    length: float
    thread: ThreadSpec | None = None
    name: str = ""


@dataclass(frozen=True)
class Socket:
    """Hole (radius) or box slot (half_size x, y) spanning z in [-depth, 0] of its frame. Visual
    only: the collision `geoms` must leave the bore clear (see fsim.spec.common.ring_geoms)."""
    pos: Vec3
    quat: Quat
    depth: float
    radius: float = 0.0
    half_size: Vec2 | None = None
    thread: ThreadSpec | None = None
    name: str = ""


@dataclass(frozen=True)
class PrimitivePartSpec:
    """A part built from primitive geoms plus stud / socket features. Inertia assumes uniform
    density over geoms and studs (overlaps are counted twice)."""
    name: str
    mass: float
    geoms: tuple[Geom, ...]
    studs: tuple[Stud, ...] = ()
    sockets: tuple[Socket, ...] = ()
    rgba: tuple[float, float, float, float] = (0.85, 0.74, 0.58, 1.0)
    visual: str = "mesh"           # "mesh": CSG mesh (primitives if CSG fails); "primitive": primitives only
    com: Vec3 | None = None        # overrides of the computed inertial (part frame)
    inertia: tuple[float, ...] | None = None   # diag (3) or full (6: xx yy zz xy xz yz), about com

    def feature(self, name: str) -> Stud | Socket:
        for f in self.studs + self.sockets:
            if f.name == name:
                return f
        raise KeyError(f"{self.name} has no feature {name!r}")


PartSpec = Union[TrayPartSpec, LegPartSpec, PrimitivePartSpec]


@dataclass(frozen=True)
class Site:
    """Mechanism site on the parent (feature frame convention): the seated child mate frame sits at
    this frame (+ seat_gap along z), and the child moves in along -z. `depth`: how far the parent's
    socket goes below the face (bottom of the screw catch; 0 for a parent stud)."""
    pos: Vec3
    quat: Quat = IDENTITY_QUAT
    depth: float = 0.0


def site_at(feature: Stud | Socket) -> Site:
    """Site on a parent stud or socket."""
    return Site(feature.pos, feature.quat, getattr(feature, "depth", 0.0))


@dataclass(frozen=True)
class ScrewSpec:
    """Helical screw joints between the threaded holes of `parent` and candidate `children`.

    Without `sites` (tray parents): the child is seated when its origin sits at (hole_x, hole_y,
    seat_height + seat_gap) in the parent frame with identity relative rotation. With `sites`
    (any parent, any axis): the child's mate frame (`child_mate`, else the flipped `child_feature`,
    else its only stud / socket) sits at the site frame + seat_gap along z. The thread "catches"
    `travel_turns` turns before that, i.e. `travel` further out along the axis. A parent stud with
    a child socket (nut) works the same way; its catch is a collar of `catch_radius` around the stud.
    """
    parent: str
    children: tuple[str, ...]
    pitch: float                         # m per turn (right-handed)
    seat_height: float = 0.0             # tray holes: child origin above parent origin when seated
    sites: tuple[Site, ...] = ()
    child_feature: str = ""
    child_mate: tuple[Vec3, Quat] | None = None   # mate frame in the child frame (z into the child)
    travel_turns: float = 1.5            # turns from thread catch to seated (540 deg)
    seat_gap: float = 0.0003             # keeps the child shoulder just off the parent face
    # Thread friction (hinge frictionloss, N*m). An axial load F back-drives the thread only when
    # F * pitch / (2*pi) exceeds it, so 0.01 N*m holds ~6.7 N (the whole table weighs ~2.2 N).
    thread_friction: float = 0.01
    catch_radius: float = 0.011          # radius of the catch cylinder the stud tip rests on
    carriage_mass: float = 1e-3
    slide_armature: float = 0.01         # kg; see helical-constraint notes in fsim/sim/screw.py
    hinge_armature: float = 1e-5         # kg*m^2
    engage_axial_tol: float = 0.0025     # |axial - travel| allowed when the thread catches
    engage_radial_tol: float = 0.0025
    engage_tilt_deg: float = 8.0
    arm_margin_deg: float = 3.0          # twist must be this far before the catch phase to arm
    # Hinge-stop torque (N*m) at the thread start that pops the child out: an untightening
    # torque of this size, or an axial pull of release_torque * 2*pi / pitch (~3.4 N).
    release_torque: float = 0.005
    seated_tol_deg: float = 1.0

    @property
    def travel(self) -> float:
        return self.pitch * self.travel_turns

    @property
    def hinge_catch(self) -> float:
        """Hinge angle (rad) at the thread catch; 0 is seated."""
        return 2.0 * math.pi * self.travel_turns

    @property
    def rad_per_m(self) -> float:
        """Hinge radians per metre of slide (coupling coefficient, hinge = f(slide))."""
        return 2.0 * math.pi / self.pitch

    @property
    def seated_z(self) -> float:
        return self.seat_height + self.seat_gap

    @property
    def catch_z(self) -> float:
        return self.seated_z + self.travel


@dataclass(frozen=True)
class InsertSpec:
    """Slide-only joints (drawers, doors on poles, a back pressed onto studs): carriage + weld like
    ScrewSpec but without the hinge. Sites, child mate and seat_gap as in ScrewSpec. The child
    engages when its mate frame is aligned within the tolerances and at most `travel` (+ tol) out
    along the site axis, then slides; at seat_tol from the end it seats and is locked there.
    Pulling it out past `travel` with more than `release_force` disengages it (unless seated)."""
    parent: str
    children: tuple[str, ...]
    sites: tuple[Site, ...]
    travel: float
    child_feature: str = ""
    child_mate: tuple[Vec3, Quat] | None = None
    seat_gap: float = 0.0
    engage_axial_tol: float = 0.0        # may engage this far before the entry (the weld snaps it in)
    engage_radial_tol: float = 0.003
    engage_tilt_deg: float = 8.0
    engage_twist_deg: float = 10.0
    twist_period_deg: float = 360.0      # symmetry of the child about the axis
    seat_tol: float = 0.001
    release_force: float = 3.0           # N, not within release_grace_s of engaging (no chatter at the entry)
    release_grace_s: float = 0.25
    carriage_mass: float = 1e-3
    slide_armature: float = 0.01


@dataclass(frozen=True)
class Phase:
    """A phase label and its completion predicate (fsim/sim/phases.py):
      grasp         `part` grasped (without a grasp callback: moved)
      pick          `part` lifted while not known released
      place_corner  `part` released in the corner extras["corner_xy"], yaw within
                    extras["corner_yaw_tol_deg"] modulo `yaw_sym_deg` (None: no yaw check)
      insert        `part` engaged at a site of `target`, or resting on a screw catch released
      screw / slide `part` seated in `target`'s screw / insert, or (target, part) assembled
      place         (target, part) assembled (position-only targets)
      place_pose    `part` released with its origin within pos_tol of `pos` (W; None: any) and its
                    local `axis` within axis_tol_deg of the W direction `axis_dir` (e.g. flipped upright)
    """
    kind: str
    part: str
    target: str | None = None
    label: str | None = None
    yaw_sym_deg: float | None = 90.0       # place_corner
    pos: Vec3 | None = None                # place_pose
    pos_tol: float = 0.02
    axis: Vec3 = (0.0, 0.0, 1.0)
    axis_dir: Vec3 = (0.0, 0.0, 1.0)
    axis_tol_deg: float = 15.0

    @property
    def name(self) -> str:
        return self.label or f"{self.kind}_{self.part}"


PHASE_KINDS = ("grasp", "place_corner", "pick", "insert", "screw", "slide", "place", "place_pose")


@dataclass(frozen=True)
class AssemblyTarget:
    """`child` counts as assembled to `parent` when inv(T_parent) @ T_child matches any
    rel pose within `pos_tol` (per axis, parent frame) and every rotation column has
    cosine >= `ori_cos` (skipped if `position_only`)."""
    parent: str
    child: str
    rel_poses: tuple[tuple[Vec3, Quat], ...]
    pos_tol: Vec3 = (0.005, 0.005, 0.005)
    ori_cos: float = 0.94
    position_only: bool = False
    requires: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Placement:
    part: str
    pos: Vec3          # W frame
    quat: Quat         # W frame


@dataclass(frozen=True)
class RandomLevel:
    pos_range: float       # +/- metres on x and y
    yaw_range_deg: float   # +/- degrees about W z


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    center: Vec3       # W frame
    half_size: Vec3


@dataclass(frozen=True)
class RandomizationSpec:
    """Layout randomization. Acceptance follows the design's rule: each footprint is rotated by
    its yaw noise and replaced by its axis-aligned bounding box; two boxes collide unless they are
    more than `clearance` apart along x or along y. Footprints are checked against each other and
    against `reset_obstacles` (the design's reset rectangles, not the built fence)."""
    levels: dict[str, RandomLevel]
    bounds_x: Vec2                     # allowed part-origin range in W
    bounds_y: Vec2
    footprints: dict[str, Vec2]        # (length along W x, width along W y) at the layout yaw
    clearance: float = 0.01
    reset_obstacles: tuple[BoxObstacle, ...] = ()
    high_presets: tuple[tuple[Placement, ...], ...] = ()
    # Per level, per part position noise ((xmin, xmax), (ymin, ymax)) in metres, replacing that level's
    # +/- pos_range (e.g. {"low": {top: ((-0.05, 0.02), (-0.05, 0.02))}, "med": {...}}).
    part_pos_range: dict[str, dict[str, tuple[Vec2, Vec2]]] = field(default_factory=dict)
    max_tries: int = 300_000              # rejection-sampling budget per reset


@dataclass(frozen=True)
class CameraSpec:
    name: str
    pos: Vec3          # world
    target: Vec3       # world point the camera looks at
    fovy: float = 40.0
    resolution: tuple[int, int] = (224, 224)


@dataclass(frozen=True)
class SceneSpec:
    table_size: Vec3 = (1.3, 0.8, 0.03)
    table_surface_z: float = 0.415
    # Reference frictions of the environment surfaces. Contacts with parts use the average of the
    # part friction and these values (the design's simulator averages; MuJoCo would take the max).
    table_ref_friction: float = 0.10
    obstacle_ref_friction: float = 1.0
    robot_base: Vec3 = (-0.3, 0.0, 0.43)   # on a 15 mm bench clamp above the table surface
    obstacles: tuple[BoxObstacle, ...] = ()
    cameras: tuple[CameraSpec, ...] = ()

    @property
    def table_center(self) -> Vec3:
        return (0.0, 0.0, self.table_surface_z - self.table_size[2] / 2.0)

    @property
    def W_origin(self) -> Vec3:
        return (0.0, 0.0, self.table_surface_z)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    parts: tuple[PartSpec, ...]
    screws: tuple[ScrewSpec, ...]
    targets: tuple[AssemblyTarget, ...]
    layout: tuple[Placement, ...]
    randomization: RandomizationSpec
    scene: SceneSpec
    phases: tuple[Phase, ...]
    inserts: tuple[InsertSpec, ...] = ()
    part_friction: float = 0.15
    max_steps: int = 3000
    control_hz: float = 20.0
    extras: dict = field(default_factory=dict)

    def part(self, name: str) -> PartSpec:
        for p in self.parts:
            if p.name == name:
                return p
        raise KeyError(name)

    def placement(self, name: str) -> Placement:
        for p in self.layout:
            if p.part == name:
                return p
        raise KeyError(name)

    @property
    def part_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.parts)

    @property
    def phase_labels(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.phases)
