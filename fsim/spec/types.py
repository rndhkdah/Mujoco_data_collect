"""Data types for furniture assembly tasks.

Frames used throughout:
  * world: z up; the table surface sits at `SceneSpec.table_surface_z`.
  * W (table frame): origin on the table surface at the table center, axes as world,
    +x pointing away from the robot. Layouts and footprints are expressed in W.
  * part frames: origin at the part's bounding-box center, local z = assembly axis
    (tabletop: threaded holes open on its +z face; leg: long axis, screw on the -z end).
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


PartSpec = Union[TrayPartSpec, LegPartSpec]


@dataclass(frozen=True)
class ScrewSpec:
    """Helical screw joints between the threaded holes of `parent` and candidate `children`.

    The child is seated when its origin sits at (hole_x, hole_y, seat_height + seat_gap) in
    the parent frame with identity relative rotation. The thread "catches" `travel_turns`
    turns before that, i.e. `travel` higher along the hole axis.
    """
    parent: str
    children: tuple[str, ...]
    pitch: float                         # m per turn (right-handed)
    seat_height: float                   # child origin above parent origin when seated (design value)
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
    phases: tuple[str, ...]
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
