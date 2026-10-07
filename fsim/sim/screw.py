"""Helical screw joints for threaded part connections.

Each threaded hole (tray hole or ScrewSpec site) owns a virtual carriage (child body of the parent
part, placed at the seated child mate frame) with a slide and a hinge along the hole axis. An always-on joint equality couples
them, hinge = (2*pi/pitch) * slide, so the carriage can only move along the thread helix. When a
child's stud rests on the hole's catch and its twist crosses the thread-start phase while
tightening, the carriage is snapped to the child's pose, the hole's catch geom is switched off
and a weld (carriage <-> child) is activated: from then on the child can only move helically.

The hinge range [0, hinge_catch] provides the seated stop and the thread start. At the thread
start, a hinge-stop torque above `release_torque` pops the child back out. That torque comes
either from untightening past the start or from pulling the child axially (a pull F becomes a
hinge torque F * pitch / (2*pi)). Elsewhere along the thread a pull is resisted by the coupling,
and an axial load back-drives the thread only above thread_friction * 2*pi / pitch.

Lessons from the MuJoCo 3.3.7 spike that this module (and fsim.models.parts) relies on:
  * the coupling must be written hinge = f(slide); slide = K*hinge with K ~ 1.5 mm/rad barely
    constrains the hinge, which then spins freely;
  * MJCF angles default to degrees: models must use <compiler angle="radian"/> (as robosuite does);
  * the carriage needs real armature, otherwise every constraint acting on it is limp;
  * the thread friction needs stiff friction solver parameters, otherwise it creeps.

data.eq_active is the source of truth for which child is engaged, so the manager stays
consistent across mj_resetData or state restores; reset() must still be called after a data
reset to re-enable catches (model fields) and clear the FSM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from fsim.geometry import inv_T, make_T, mat_to_quat, quat_to_mat, wrap_angle
from fsim.models import names
from fsim.models.parts import CATCH_CONAFFINITY, CATCH_CONTYPE, mechanism_sites
from fsim.spec.types import ScrewSpec, TaskSpec

FREE, ENGAGED, SEATED = "free", "engaged", "seated"


@dataclass
class Hole:
    screw: ScrewSpec
    parent: str
    index: int
    parent_body: int
    T_local: np.ndarray            # carriage frame at qpos 0 (= seated child pose) in the parent frame
    hinge_jnt: int
    hinge_q: int
    hinge_v: int
    slide_q: int
    slide_v: int
    catch_geom: int
    weld: dict[str, int]
    mate: dict[str, np.ndarray | None]   # child mate frame in the child frame (None: child origin)
    child: str | None = None

    @property
    def key(self) -> str:
        return f"{self.parent}/hole{self.index}"


@dataclass
class Measurement:
    axial: float      # child origin height above the seated pose, along the hole axis (m)
    radial: float     # distance of the child origin from the hole axis (m)
    tilt_deg: float   # angle between child z and hole axis
    twist: float      # child rotation about the hole axis relative to the seated pose, in (-pi, pi]


@dataclass
class _PairTrack:
    armed: bool = False
    prev_err: float | None = None


class HelicalScrewManager:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, task: TaskSpec):
        self.model, self.data, self.task = model, data, task
        self.holes: list[Hole] = []
        self.body_id: dict[str, int] = {}
        self._tracks: dict[tuple[str, str], _PairTrack] = {}
        self.events: list[dict] = []
        self._limit_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)

        def oid(kind, name):
            i = mujoco.mj_name2id(model, kind, name)
            if i < 0:
                raise KeyError(f"{name} not found in model")
            return i

        for site in mechanism_sites(task):
            if not site.is_screw:
                continue
            screw, h = site.spec, site.index
            for part in (screw.parent,) + screw.children:
                self.body_id[part] = oid(mujoco.mjtObj.mjOBJ_BODY, names.body(part))
            cb = oid(mujoco.mjtObj.mjOBJ_BODY, names.carriage(screw.parent, h))
            hj = oid(mujoco.mjtObj.mjOBJ_JOINT, names.hinge_joint(screw.parent, h))
            sj = oid(mujoco.mjtObj.mjOBJ_JOINT, names.slide_joint(screw.parent, h))
            self.holes.append(Hole(
                screw=screw, parent=screw.parent, index=h, parent_body=self.body_id[screw.parent],
                T_local=make_T(model.body_pos[cb], quat_to_mat(model.body_quat[cb])),
                hinge_jnt=hj, hinge_q=model.jnt_qposadr[hj], hinge_v=model.jnt_dofadr[hj],
                slide_q=model.jnt_qposadr[sj], slide_v=model.jnt_dofadr[sj],
                catch_geom=oid(mujoco.mjtObj.mjOBJ_GEOM, names.catch_geom(screw.parent, h)),
                weld={c: oid(mujoco.mjtObj.mjOBJ_EQUALITY, names.weld_eq(screw.parent, h, c)) for c in screw.children},
                mate=site.mates,
            ))
        # Holes at the same spot of one parent (e.g. a second site rotated 180 degrees for a symmetric
        # child) share one physical catch: it is off while any of them holds a child.
        self._twins = {id(h): [t for t in self.holes if t.parent == h.parent
                               and np.linalg.norm(t.T_local[:3, 3] - h.T_local[:3, 3]) < 1e-4] for h in self.holes}
        self._sync_from_eq_active()

    # ---------------------------------------------------------------- state sync
    def _set_catch(self, hole: Hole, enabled: bool):
        for h in self._twins.get(id(hole), [hole]):
            on = enabled and all(t.child is None for t in self._twins.get(id(h), [h]) if t is not hole)
            if h is not hole:
                on = on and h.child is None
            self.model.geom_contype[h.catch_geom] = CATCH_CONTYPE if on else 0
            self.model.geom_conaffinity[h.catch_geom] = CATCH_CONAFFINITY if on else 0

    def _sync_from_eq_active(self):
        """Make hole.child and the catch switches agree with data.eq_active (source of truth)."""
        for hole in self.holes:
            active = [c for c, eq in hole.weld.items() if self.data.eq_active[eq]]
            child = active[0] if active else None
            if child != hole.child:
                hole.child = child
            self._set_catch(hole, child is None)

    # ---------------------------------------------------------------- queries
    def _body_T(self, body: int) -> np.ndarray:
        return make_T(self.data.xpos[body], self.data.xmat[body].reshape(3, 3))

    def _mate_T(self, child: str, hole: Hole) -> np.ndarray:
        """World pose of the child's mate frame for this hole."""
        T, mate = self._body_T(self.body_id[child]), hole.mate[child]
        return T if mate is None else T @ mate

    def measure(self, child: str, hole: Hole) -> Measurement:
        """Child mate frame relative to the hole's seated frame (valid after a forward pass)."""
        rel = inv_T(self._body_T(hole.parent_body) @ hole.T_local) @ self._mate_T(child, hole)
        R = rel[:3, :3]
        return Measurement(
            axial=float(rel[2, 3]),
            radial=float(math.hypot(rel[0, 3], rel[1, 3])),
            tilt_deg=math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))),
            twist=math.atan2(R[1, 0], R[0, 0]),
        )

    def in_zone(self, m: Measurement, s: ScrewSpec) -> bool:
        """Stud resting on the catch, aligned with the hole."""
        return (abs(m.axial - s.travel) < s.engage_axial_tol and m.radial < s.engage_radial_tol
                and m.tilt_deg < s.engage_tilt_deg)

    def resting_hole(self, child: str, parent: str | None = None) -> Hole | None:
        """Free hole (of `parent`, if given) on whose catch `child` rests aligned (inserted, not yet threaded)."""
        for hole in self.holes:
            if parent is not None and hole.parent != parent:
                continue
            if hole.child is None and child in hole.weld and self.in_zone(self.measure(child, hole), hole.screw):
                return hole
        return None

    def hinge(self, hole: Hole) -> float:
        return float(self.data.qpos[hole.hinge_q])

    def hole_state(self, hole: Hole) -> str:
        self._sync_from_eq_active()
        if hole.child is None:
            return FREE
        return SEATED if self.hinge(hole) <= math.radians(hole.screw.seated_tol_deg) else ENGAGED

    def child_hole(self, child: str) -> Hole | None:
        self._sync_from_eq_active()
        for h in self.holes:
            if h.child == child:
                return h
        return None

    def child_state(self, child: str) -> str:
        h = self.child_hole(child)
        return FREE if h is None else self.hole_state(h)

    def stop_force(self, hole: Hole) -> float:
        """Total joint-limit constraint torque on the hole's hinge from the last step."""
        d = self.data
        f = 0.0
        for i in range(d.nefc):
            if d.efc_type[i] == self._limit_type and d.efc_id[i] == hole.hinge_jnt:
                f += float(d.efc_force[i])
        return f

    # ---------------------------------------------------------------- transitions
    def _log(self, event: str, hole: Hole, child: str, **extra):
        rec = dict(event=event, time=float(self.data.time), hole=hole.key, child=child,
                   hinge=self.hinge(hole), slide=float(self.data.qpos[hole.slide_q]), **extra)
        self.events.append(rec)
        return rec

    def _engage(self, child: str, hole: Hole, hinge: float, reason: str, **extra) -> dict:
        s = hole.screw
        hinge = float(np.clip(hinge, 0.0, s.hinge_catch))
        self.data.qpos[hole.hinge_q] = hinge
        self.data.qpos[hole.slide_q] = hinge / s.rad_per_m
        self.data.qvel[hole.hinge_v] = 0.0
        self.data.qvel[hole.slide_v] = 0.0
        self._set_catch(hole, False)
        self.data.eq_active[hole.weld[child]] = 1
        hole.child = child
        return self._log("engage", hole, child, reason=reason, **extra)

    def _disengage(self, hole: Hole, reason: str) -> dict:
        child = hole.child
        self.data.eq_active[hole.weld[child]] = 0
        self._set_catch(hole, True)
        hole.child = None
        self._tracks[(child, hole.key)] = _PairTrack()
        return self._log("disengage", hole, child, reason=reason)

    def reset(self):
        """Release every child, re-enable catches and zero the carriages (call after a data reset)."""
        for hole in self.holes:
            for eq in hole.weld.values():
                self.data.eq_active[eq] = 0
            self._set_catch(hole, True)
            self.data.qpos[hole.hinge_q] = 0.0
            self.data.qpos[hole.slide_q] = 0.0
            self.data.qvel[hole.hinge_v] = 0.0
            self.data.qvel[hole.slide_v] = 0.0
            hole.child = None
        self._tracks.clear()
        self.events.clear()

    def update(self) -> list[dict]:
        """Run the FSM once (after a forward pass / step). Returns the events it produced."""
        self._sync_from_eq_active()
        new = []
        busy = {h.child for h in self.holes if h.child is not None}
        for hole in self.holes:
            s = hole.screw
            if hole.child is not None:
                at_start = self.hinge(hole) >= s.hinge_catch - math.radians(0.5)
                if at_start and self.stop_force(hole) > s.release_torque:
                    new.append(self._disengage(hole, "thread_start"))
                continue
            phase = wrap_angle(s.hinge_catch)
            seat_world = self._body_T(hole.parent_body) @ hole.T_local
            for child in s.children:
                if child in busy:
                    continue
                track = self._tracks.setdefault((child, hole.key), _PairTrack())
                pos = self.data.xpos[self.body_id[child]] if hole.mate[child] is None else self._mate_T(child, hole)[:3, 3]
                if np.linalg.norm(pos - seat_world[:3, 3]) > s.travel + 0.03:
                    track.armed, track.prev_err = False, None      # cheap reject: far from this hole
                    continue
                m = self.measure(child, hole)
                if not self.in_zone(m, s):
                    track.armed, track.prev_err = False, None
                    continue
                err = wrap_angle(m.twist - phase)
                if err > math.radians(s.arm_margin_deg):
                    track.armed = True
                crossed = (track.prev_err is not None and track.prev_err > 0.0 >= err
                           and track.prev_err - err < math.pi)
                if track.armed and crossed:
                    new.append(self._engage(child, hole, s.hinge_catch + err, "thread_catch",
                                            err_deg=math.degrees(err)))
                    busy.add(child)
                    break
                track.prev_err = err
        return new

    def force_seat(self, child: str, parent: str, hole_index: int) -> dict:
        """Screw `child` fully into a hole (assembled spawns, tests). The parent pose is taken from
        the current qpos; call mj_forward afterwards."""
        self._sync_from_eq_active()
        hole = next(h for h in self.holes if h.parent == parent and h.index == hole_index)
        if hole.child is not None:
            raise ValueError(f"{hole.key} is already occupied by {hole.child}")
        if self.child_hole(child) is not None:
            raise ValueError(f"{child} is already engaged in {self.child_hole(child).key}")
        mujoco.mj_kinematics(self.model, self.data)
        T = self._body_T(hole.parent_body) @ hole.T_local
        if hole.mate[child] is not None:
            T = T @ inv_T(hole.mate[child])
        b = self.body_id[child]
        j = self.model.body_jntadr[b]
        qa, va = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        self.data.qpos[qa:qa + 3] = T[:3, 3]
        self.data.qpos[qa + 3:qa + 7] = mat_to_quat(T[:3, :3])
        self.data.qvel[va:va + 6] = 0.0
        return self._engage(child, hole, 0.0, "force_seat")

    # ---------------------------------------------------------------- recording / replay
    def snapshot(self) -> dict:
        """Engaged pairs plus FSM arming state (everything not already in the MuJoCo state)."""
        return {
            "children": {h.key: h.child for h in self.holes},
            "tracks": {f"{c}|{k}": (t.armed, t.prev_err) for (c, k), t in self._tracks.items()},
        }

    def restore(self, snapshot: dict):
        """Re-apply a snapshot after restoring the matching MuJoCo state (qpos incl. carriages)."""
        children = snapshot.get("children", snapshot)
        for hole in self.holes:
            for eq in hole.weld.values():
                self.data.eq_active[eq] = 0
            child = children.get(hole.key)
            hole.child = child
            self._set_catch(hole, child is None)
            if child is not None:
                self.data.eq_active[hole.weld[child]] = 1
        self._tracks = {}
        for key, (armed, prev) in snapshot.get("tracks", {}).items():
            c, k = key.split("|", 1)
            self._tracks[(c, k)] = _PairTrack(armed=bool(armed), prev_err=prev)
