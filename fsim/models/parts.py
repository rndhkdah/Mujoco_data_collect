"""Parametric part bodies (MJCF elements) built from the specs.

Collision uses primitives only (MuJoCo convex-hulls meshes, which would fill holes).
Visuals use the generated meshes from fsim/models/meshes.py when available, else primitives.
All attributes are explicit (no <default> classes) because robosuite does not merge defaults.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from fsim.geometry import fmt, inv_T, make_T, mat_to_quat, quat_to_mat, rot_z
from fsim.models import names
from fsim.spec.common import child_mate
from fsim.spec.types import (
    Geom, InsertSpec, LegPartSpec, PartSpec, PrimitivePartSpec, ScrewSpec, Stud, TaskSpec, TrayPartSpec,
)

PART_SOLREF = "0.005 1"
PART_SOLIMP = "0.95 0.99 0.001"
CONSTRAINT_SOLREF = "0.005 1"
CONSTRAINT_SOLIMP = "0.95 0.99 0.001"
# Stiff friction for the thread hinge. MuJoCo's default friction impedance lets frictionloss
# creep like a viscous damper, which would let hanging legs slowly unscrew.
THREAD_FRICTION_SOLREF = "0.005 1"
THREAD_FRICTION_SOLIMP = "0.99 0.999 0.001"
# Catch geoms are switched off while a child is engaged; these are their "on" values. The screw
# manager uses these constants, never the live model (which a previous episode may have modified).
CATCH_CONTYPE = 1
CATCH_CONAFFINITY = 1


@dataclass(frozen=True)
class BoxGeom:
    pos: tuple[float, float, float]
    half: tuple[float, float, float]
    yaw: float = 0.0


def tray_boxes(spec: TrayPartSpec) -> list[BoxGeom]:
    """Plate plus rim of an open tray."""
    sx, sy, sz = spec.size
    rim, pt = spec.rim_width, spec.plate_thickness
    z_lo, z_hi = spec.plate_top_z, spec.top_z
    zc, hz = (z_lo + z_hi) / 2.0, (z_hi - z_lo) / 2.0
    return [
        BoxGeom((0.0, 0.0, -sz / 2 + pt / 2), (sx / 2, sy / 2, pt / 2)),
        BoxGeom((sx / 2 - rim / 2, 0.0, zc), (rim / 2, sy / 2, hz)),
        BoxGeom((-sx / 2 + rim / 2, 0.0, zc), (rim / 2, sy / 2, hz)),
        BoxGeom((0.0, sy / 2 - rim / 2, zc), (sx / 2 - rim, rim / 2, hz)),
        BoxGeom((0.0, -sy / 2 + rim / 2, zc), (sx / 2 - rim, rim / 2, hz)),
    ]


def boss_ring_boxes(spec: TrayPartSpec, hole: int, segments: int = 8) -> list[BoxGeom]:
    """Polygonal boss around a hole: inscribed bore radius `bore_radius`, outer apothem
    `boss_outer_radius`, from the plate to the top face. The tangential half-length closes the
    outer corners; boxes only overlap outside the bore, so the bore stays clear."""
    cx, cy = spec.hole_centers[hole]
    r_in, r_out = spec.bore_radius, spec.boss_outer_radius
    z_lo, z_hi = spec.plate_top_z, spec.top_z
    half_t = r_out * math.tan(math.pi / segments)
    out = []
    for k in range(segments):
        a = 2.0 * math.pi * k / segments
        r_mid = (r_in + r_out) / 2.0
        out.append(BoxGeom((cx + r_mid * math.cos(a), cy + r_mid * math.sin(a), (z_lo + z_hi) / 2.0),
                           ((r_out - r_in) / 2.0, half_t, (z_hi - z_lo) / 2.0), a))
    return out


def catch_top_z(screw: ScrewSpec, child: LegPartSpec) -> float:
    """Height (parent frame) of the catch surface the child's stud tip rests on at thread catch."""
    return screw.catch_z + child.tip_z


def _box_inertia(m, hx, hy, hz):
    return m / 3.0 * np.array([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])


def _composite(pieces):
    """pieces: (volume, center(3), diag inertia per unit mass about own center (3)). Returns
    (com, diag inertia about com) for unit total mass (principal axes assumed aligned)."""
    vol = sum(p[0] for p in pieces)
    com = sum(p[0] * np.asarray(p[1]) for p in pieces) / vol
    inertia = np.zeros(3)
    for v, c, i_unit in pieces:
        w = v / vol
        d = np.asarray(c) - com
        inertia += w * (np.asarray(i_unit) + np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2]))
    return com, inertia


def stud_geom(st: Stud) -> Geom:
    """Collision cylinder of a stud."""
    T = make_T(st.pos, quat_to_mat(st.quat)) @ make_T((0.0, 0.0, st.length / 2.0))
    return Geom("cylinder", (st.radius, st.length / 2.0), tuple(T[:3, 3]), st.quat)


def _geom_unit(g: Geom) -> tuple[float, np.ndarray]:
    """(volume, diag inertia per unit mass about the geom center, in the geom frame)."""
    if g.type == "box":
        return 8.0 * g.size[0] * g.size[1] * g.size[2], _box_inertia(1.0, *g.size)
    if g.type == "sphere":
        return 4.0 / 3.0 * math.pi * g.size[0] ** 3, np.full(3, 0.4 * g.size[0] ** 2)
    r, L = g.size[0], 2.0 * g.size[1]
    vc = math.pi * r * r * L
    if g.type == "cylinder":
        return vc, np.array([(3 * r * r + L * L) / 12.0] * 2 + [r * r / 2.0])
    if g.type == "capsule":
        vs = 4.0 / 3.0 * math.pi * r ** 3
        ixx = vc * (L * L / 12.0 + r * r / 4.0) + vs * (0.4 * r * r + L * L / 4.0 + 3.0 * L * r / 8.0)
        return vc + vs, np.array([ixx, ixx, vc * r * r / 2.0 + vs * 0.4 * r * r]) / (vc + vs)
    raise ValueError(f"unknown geom type {g.type!r}")


def primitive_geoms(spec: PrimitivePartSpec) -> tuple[Geom, ...]:
    """Collision geoms: the spec geoms plus one cylinder per stud."""
    return spec.geoms + tuple(stud_geom(st) for st in spec.studs)


def geom_aabb(g: Geom) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds (lo, hi) of a geom in the part frame (cylinders as boxes)."""
    if g.type == "box":
        h = np.asarray(g.size, float)
    elif g.type == "sphere":
        h = np.full(3, g.size[0])
    else:
        h = np.array([g.size[0], g.size[0], g.size[1] + (g.size[0] if g.type == "capsule" else 0.0)])
    ext = np.abs(quat_to_mat(g.quat)) @ h
    return np.asarray(g.pos) - ext, np.asarray(g.pos) + ext


def parts_aabb(geoms) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = zip(*(geom_aabb(g) for g in geoms))
    return np.min(lo, axis=0), np.max(hi, axis=0)


def part_inertial(spec: PartSpec) -> tuple[np.ndarray, float, np.ndarray]:
    """(com, mass, inertia) in the part frame; inertia is diagonal (3) or full (6: xx yy zz xy xz yz)."""
    if isinstance(spec, PrimitivePartSpec):
        pieces = [(*_geom_unit(g), np.asarray(g.pos, float), quat_to_mat(g.quat)) for g in primitive_geoms(spec)]
        vol = sum(p[0] for p in pieces)
        com = sum(v * c for v, _, c, _ in pieces) / vol
        I = np.zeros((3, 3))
        for v, i_unit, c, R in pieces:
            d = c - com
            I += v / vol * (R @ np.diag(i_unit) @ R.T + (d @ d) * np.eye(3) - np.outer(d, d))
        I *= spec.mass
        com = com if spec.com is None else np.asarray(spec.com, float)
        if spec.inertia is not None:
            return com, spec.mass, np.asarray(spec.inertia, float)
        return com, spec.mass, np.array([I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]])
    if isinstance(spec, TrayPartSpec):
        pieces = [(8 * b.half[0] * b.half[1] * b.half[2], b.pos, _box_inertia(1.0, *b.half)) for b in tray_boxes(spec)]
        r_in, r_out = spec.bore_radius, spec.boss_outer_radius
        h = spec.top_z - spec.plate_top_z
        v_boss = math.pi * (r_out ** 2 - r_in ** 2) * h
        i_boss = np.array([(3 * (r_in ** 2 + r_out ** 2) + h * h) / 12.0] * 2 + [(r_in ** 2 + r_out ** 2) / 2.0])
        for cx, cy in spec.hole_centers:
            pieces.append((v_boss, (cx, cy, (spec.plate_top_z + spec.top_z) / 2.0), i_boss))
        com, inertia = _composite(pieces)
        return com, spec.mass, spec.mass * inertia
    # Leg: solid square body plus a stud approximated by a cylinder at the mean thread radius.
    w, lb, ls = spec.width, spec.body_length, spec.screw_length
    r = 0.5 * (spec.thread.minor_radius + spec.thread.major_radius)
    com, inertia = _composite([
        (w * w * lb, (0.0, 0.0, 0.0), _box_inertia(1.0, w / 2, w / 2, lb / 2)),
        (math.pi * r * r * ls, (0.0, 0.0, -lb / 2.0 - ls / 2.0), np.array([(3 * r * r + ls * ls) / 12.0] * 2 + [r * r / 2.0])),
    ])
    return com, spec.mass, spec.mass * inertia


def env_contact_attrs(part_friction: float, ref_friction: float) -> dict:
    """Friction/priority for environment geoms (table, fences) so that part contacts use the
    average of the part and reference frictions, as the design's simulator does."""
    return dict(friction=f"{(part_friction + ref_friction) / 2.0:g} 0.005 0.0001", priority=1)


def _sub(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items() if v is not None})


def _collision_attrs(friction: float) -> dict:
    return dict(group=0, contype=1, conaffinity=1, friction=f"{friction} 0.005 0.0001",
                solref=PART_SOLREF, solimp=PART_SOLIMP, rgba="0.5 0.5 0.5 0.0")


def _visual_attrs(rgba) -> dict:
    return dict(group=1, contype=0, conaffinity=0, rgba=fmt(rgba, 4))


def _add_box(body, name, box: BoxGeom, attrs):
    q = mat_to_quat(rot_z(box.yaw)) if box.yaw else None
    _sub(body, "geom", name=name, type="box", pos=fmt(box.pos), size=fmt(box.half),
         quat=fmt(q) if q is not None else None, **attrs)


@dataclass(frozen=True)
class MechSite:
    """One screw hole or insert slot, resolved from the specs (shared by the builders and managers)."""
    spec: ScrewSpec | InsertSpec
    index: int                       # per parent: hole{index} (screws) or slot{index} (inserts)
    pos: tuple                       # carriage frame at qpos 0 (= seated child mate frame), parent frame
    quat: tuple | None               # None: identity (tray holes)
    catch: tuple | None              # screws: catch cylinder fromto (6), parent frame
    mates: dict                      # child -> mate frame (4x4) in the child frame; None = child origin

    @property
    def parent(self) -> str:
        return self.spec.parent

    @property
    def is_screw(self) -> bool:
        return isinstance(self.spec, ScrewSpec)


def mechanism_sites(task: TaskSpec) -> list[MechSite]:
    """Every screw hole (task.screws order) then every insert slot (task.inserts order)."""
    out, count = [], {}
    for spec in task.screws + task.inserts:
        key = (spec.parent, isinstance(spec, ScrewSpec))
        if isinstance(spec, ScrewSpec) and not spec.sites:            # tray holes (square_table)
            parent = task.part(spec.parent)
            assert isinstance(parent, TrayPartSpec), f"{spec.parent}: screws without sites need a tray parent"
            ztop = catch_top_z(spec, task.part(spec.children[0]))
            for cx, cy in parent.hole_centers:
                i = count[key] = count.get(key, -1) + 1
                out.append(MechSite(spec, i, (cx, cy, spec.seated_z), None,
                                    (cx, cy, parent.plate_top_z, cx, cy, ztop), {c: None for c in spec.children}))
            continue
        gap = spec.seat_gap
        mates = {c: child_mate(spec, task.part(c)) for c in spec.children}
        protrude = {round(p, 9) for _, p in mates.values()}
        if len(protrude) > 1:
            raise ValueError(f"{spec.parent}: children of one spec must protrude equally ({protrude})")
        prot = protrude.pop()
        for site in spec.sites:
            i = count[key] = count.get(key, -1) + 1
            T_site = make_T(site.pos, quat_to_mat(site.quat))
            T = T_site @ make_T((0.0, 0.0, gap))
            catch = None
            if isinstance(spec, ScrewSpec):
                z0, z1 = -site.depth, gap + spec.travel - prot
                if z1 < z0 + 1e-4:
                    raise ValueError(f"{spec.parent} site {i}: the catch would sit below the socket floor")
                catch = tuple((T_site @ [0, 0, z0, 1])[:3]) + tuple((T_site @ [0, 0, z1, 1])[:3])
            out.append(MechSite(spec, i, tuple(T[:3, 3]), tuple(mat_to_quat(T[:3, :3])), catch,
                                {c: m for c, (m, _) in mates.items()}))
    return out


def _add_mechanisms(body: ET.Element, task: TaskSpec, part: str, col: dict):
    """Catch geoms and carriage bodies of every hole / slot on `part`."""
    for ms in mechanism_sites(task):
        if ms.parent != part:
            continue
        if ms.is_screw:
            catch = dict(col, contype=CATCH_CONTYPE, conaffinity=CATCH_CONAFFINITY)
            _sub(body, "geom", name=names.catch_geom(part, ms.index), type="cylinder",
                 fromto=fmt(ms.catch), size=fmt([ms.spec.catch_radius]), **catch)
            body.append(_carriage_body(part, ms.index, ms, ms.spec))
        else:
            body.append(_slot_body(part, ms))


def _add_geom(body, name, g: Geom, attrs):
    q = None if tuple(g.quat) == (1.0, 0.0, 0.0, 0.0) else fmt(g.quat)
    _sub(body, "geom", name=name, type=g.type, pos=fmt(g.pos), size=fmt(g.size), quat=q, **attrs)


def build_part_body(task: TaskSpec, spec: PartSpec, pos, quat, mesh_assets: dict[str, str] | None = None,
                    free_joint: bool = True, body_name: str | None = None) -> ET.Element:
    """MJCF <body> for one part, including hole carriages for screw parents.

    body_name overrides the root body name (robosuite object XMLs need "object", which robosuite
    renames to "<part>_main" = names.body(part)); all other names are already globally prefixed.
    """
    body = ET.Element("body", {"name": body_name or names.body(spec.name), "pos": fmt(pos), "quat": fmt(quat)})
    if free_joint:
        _sub(body, "freejoint", name=names.free_joint(spec.name))
    com, mass, inertia = part_inertial(spec)
    _sub(body, "inertial", pos=fmt(com), mass=fmt(mass),
         **({"fullinertia": fmt(inertia)} if len(inertia) == 6 else {"diaginertia": fmt(inertia)}))
    col = _collision_attrs(task.part_friction)
    vis = _visual_attrs(spec.rgba)
    mesh_name = (mesh_assets or {}).get(spec.name)
    n = spec.name
    if isinstance(spec, TrayPartSpec):
        boxes = tray_boxes(spec)
        for h in range(len(spec.hole_centers)):
            boxes += boss_ring_boxes(spec, h)
        for i, b in enumerate(boxes):
            _add_box(body, names.elem(n, f"col{i}"), b, col)
        _add_mechanisms(body, task, n, col)
        if mesh_name:
            _sub(body, "geom", name=names.elem(n, "vis0"), type="mesh", mesh=mesh_name, **vis)
        else:
            for i, b in enumerate(boxes):
                _add_box(body, names.elem(n, f"vis{i}"), b, vis)
    elif isinstance(spec, LegPartSpec):
        hw, hl = spec.width / 2, spec.body_length / 2
        _add_box(body, names.elem(n, "col0"), BoxGeom((0, 0, 0), (hw, hw, hl)), col)
        _sub(body, "geom", name=names.elem(n, "col1"), type="cylinder",
             fromto=fmt((0, 0, spec.tip_z, 0, 0, -hl)), size=fmt([spec.peg_radius]), **col)
        _add_box(body, names.elem(n, "vis0"), BoxGeom((0, 0, 0), (hw, hw, hl)), vis)
        if mesh_name:
            _sub(body, "geom", name=names.elem(n, "vis1"), type="mesh", mesh=mesh_name, **_visual_attrs((0.7, 0.6, 0.45, 1.0)))
        else:
            _sub(body, "geom", name=names.elem(n, "vis1"), type="cylinder",
                 fromto=fmt((0, 0, spec.tip_z, 0, 0, -hl)), size=fmt([spec.thread.major_radius]), **vis)
        _add_mechanisms(body, task, n, col)
    else:
        geoms = primitive_geoms(spec)
        for i, g in enumerate(geoms):
            _add_geom(body, names.elem(n, f"col{i}"), g, col)
        _add_mechanisms(body, task, n, col)
        if mesh_name:
            _sub(body, "geom", name=names.elem(n, "vis0"), type="mesh", mesh=mesh_name, **vis)
        else:
            for i, g in enumerate(geoms):
                _add_geom(body, names.elem(n, f"vis{i}"), g, vis)
    return body


def _carriage_body(parent: str, hole: int, site: MechSite, screw: ScrewSpec) -> ET.Element:
    """Virtual carriage at the seated child pose; its slide+hinge carry the helical DOF (along local z).

    The hinge (not the slide) owns the stops and the thread friction: a constraint written as
    slide = K*hinge with K ~ 1.5 mm/rad barely loads the hinge and lets it spin freely.
    """
    attrs = {"name": names.carriage(parent, hole), "pos": fmt(site.pos)}
    if site.quat is not None:
        attrs["quat"] = fmt(site.quat)
    b = ET.Element("body", dict(attrs, gravcomp="1"))
    _sub(b, "inertial", pos="0 0 0", mass=fmt(screw.carriage_mass), diaginertia="1e-7 1e-7 1e-7")
    _sub(b, "joint", name=names.slide_joint(parent, hole), type="slide", axis="0 0 1",
         armature=fmt(screw.slide_armature), limited="false")
    _sub(b, "joint", name=names.hinge_joint(parent, hole), type="hinge", axis="0 0 1",
         limited="true", range=fmt((0.0, screw.hinge_catch)), armature=fmt(screw.hinge_armature),
         frictionloss=fmt(screw.thread_friction), damping="0.0001",
         solreflimit=CONSTRAINT_SOLREF, solimplimit=CONSTRAINT_SOLIMP,
         solreffriction=THREAD_FRICTION_SOLREF, solimpfriction=THREAD_FRICTION_SOLIMP)
    return b


def _slot_body(parent: str, site: MechSite) -> ET.Element:
    """Insert carriage at the seated child mate frame: one slide along local z, range [0, travel]."""
    ins: InsertSpec = site.spec
    b = ET.Element("body", {"name": names.slot_carriage(parent, site.index), "pos": fmt(site.pos),
                            "quat": fmt(site.quat), "gravcomp": "1"})
    _sub(b, "inertial", pos="0 0 0", mass=fmt(ins.carriage_mass), diaginertia="1e-7 1e-7 1e-7")
    _sub(b, "joint", name=names.slot_slide(parent, site.index), type="slide", axis="0 0 1",
         limited="true", range=fmt((0.0, ins.travel)), armature=fmt(ins.slide_armature), damping="0.0001",
         solreflimit=CONSTRAINT_SOLREF, solimplimit=CONSTRAINT_SOLIMP)
    return b


def _weld(name: str, body1: str, child: str, mate) -> ET.Element:
    relpose = "0 0 0 1 0 0 0" if mate is None else fmt(np.r_[inv_T(mate)[:3, 3], mat_to_quat(inv_T(mate)[:3, :3])])
    return ET.Element("weld", {"name": name, "body1": body1, "body2": names.body(child), "relpose": relpose,
                               "active": "false", "solref": CONSTRAINT_SOLREF, "solimp": CONSTRAINT_SOLIMP})


def build_screw_equalities(task: TaskSpec) -> list[ET.Element]:
    """Screws: always-on hinge/slide couplings plus one inactive weld per (hole, candidate child).
    Inserts: an inactive slide lock plus one inactive weld per (slot, candidate child)."""
    out = []
    for ms in mechanism_sites(task):
        p, h = ms.parent, ms.index
        if ms.is_screw:
            out.append(ET.Element("joint", {
                "name": names.couple_eq(p, h),
                "joint1": names.hinge_joint(p, h), "joint2": names.slide_joint(p, h),
                "polycoef": fmt((0.0, ms.spec.rad_per_m, 0.0, 0.0, 0.0)),
                "solref": CONSTRAINT_SOLREF, "solimp": CONSTRAINT_SOLIMP}))
            for child in ms.spec.children:
                out.append(_weld(names.weld_eq(p, h, child), names.carriage(p, h), child, ms.mates[child]))
        else:
            out.append(ET.Element("joint", {
                "name": names.slot_lock_eq(p, h), "joint1": names.slot_slide(p, h),
                "polycoef": "0 0 0 0 0", "active": "false", "solref": CONSTRAINT_SOLREF, "solimp": CONSTRAINT_SOLIMP}))
            for child in ms.spec.children:
                out.append(_weld(names.slot_weld_eq(p, h, child), names.slot_carriage(p, h), child, ms.mates[child]))
    return out
