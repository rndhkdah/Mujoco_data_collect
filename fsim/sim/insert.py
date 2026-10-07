"""Slide-only insert joints (drawers, doors on poles, a chair back pressed onto studs).

Same idea as the helical screw (fsim/sim/screw.py) without the hinge: each InsertSpec site owns a
carriage at the seated child mate frame with one slide along the site axis, range [0, travel].
When a child's mate frame is aligned with the axis (radial, tilt, twist) and at most `travel` out
(+ engage_axial_tol), the carriage is snapped to its axial position and an inactive weld
carriage <-> child is activated: from then on the child can only slide. At `seat_tol` from the
end it seats: a joint equality locks the slide at 0 for good. Pulled out against the travel stop
with more than `release_force` (while not seated), it disengages.

data.eq_active (welds and locks) is the source of truth; call reset() after a data reset.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from fsim.geometry import inv_T, make_T, mat_to_quat, quat_to_mat
from fsim.models import names
from fsim.models.parts import mechanism_sites
from fsim.sim.screw import ENGAGED, FREE, SEATED, Measurement
from fsim.spec.types import InsertSpec, TaskSpec


@dataclass
class Slot:
    spec: InsertSpec
    parent: str
    index: int
    parent_body: int
    T_local: np.ndarray            # carriage frame at slide 0 (= seated child mate frame) in the parent frame
    slide_jnt: int
    slide_q: int
    slide_v: int
    lock: int                      # equality id of the slide lock
    weld: dict[str, int]
    mate: dict[str, np.ndarray | None]
    child: str | None = None
    since: float = -1e9            # data.time of the last engagement

    @property
    def key(self) -> str:
        return f"{self.parent}/slot{self.index}"


class InsertManager:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, task: TaskSpec):
        self.model, self.data, self.task = model, data, task
        self.slots: list[Slot] = []
        self.body_id: dict[str, int] = {}
        self.events: list[dict] = []
        self._limit_type = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)

        def oid(kind, name):
            i = mujoco.mj_name2id(model, kind, name)
            if i < 0:
                raise KeyError(f"{name} not found in model")
            return i

        for site in mechanism_sites(task):
            if site.is_screw:
                continue
            ins, i = site.spec, site.index
            for part in (ins.parent,) + ins.children:
                self.body_id[part] = oid(mujoco.mjtObj.mjOBJ_BODY, names.body(part))
            cb = oid(mujoco.mjtObj.mjOBJ_BODY, names.slot_carriage(ins.parent, i))
            sj = oid(mujoco.mjtObj.mjOBJ_JOINT, names.slot_slide(ins.parent, i))
            self.slots.append(Slot(
                spec=ins, parent=ins.parent, index=i, parent_body=self.body_id[ins.parent],
                T_local=make_T(model.body_pos[cb], quat_to_mat(model.body_quat[cb])),
                slide_jnt=sj, slide_q=model.jnt_qposadr[sj], slide_v=model.jnt_dofadr[sj],
                lock=oid(mujoco.mjtObj.mjOBJ_EQUALITY, names.slot_lock_eq(ins.parent, i)),
                weld={c: oid(mujoco.mjtObj.mjOBJ_EQUALITY, names.slot_weld_eq(ins.parent, i, c)) for c in ins.children},
                mate=site.mates,
            ))
        self._sync_from_eq_active()

    # ---------------------------------------------------------------- state
    def _sync_from_eq_active(self):
        for slot in self.slots:
            active = [c for c, eq in slot.weld.items() if self.data.eq_active[eq]]
            slot.child = active[0] if active else None

    def _body_T(self, body: int) -> np.ndarray:
        return make_T(self.data.xpos[body], self.data.xmat[body].reshape(3, 3))

    def _mate_T(self, child: str, slot: Slot) -> np.ndarray:
        T, mate = self._body_T(self.body_id[child]), slot.mate[child]
        return T if mate is None else T @ mate

    def measure(self, child: str, slot: Slot) -> Measurement:
        """Child mate frame relative to the slot's seated frame (valid after a forward pass)."""
        rel = inv_T(self._body_T(slot.parent_body) @ slot.T_local) @ self._mate_T(child, slot)
        R = rel[:3, :3]
        return Measurement(axial=float(rel[2, 3]), radial=float(math.hypot(rel[0, 3], rel[1, 3])),
                           tilt_deg=math.degrees(math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))),
                           twist=math.atan2(R[1, 0], R[0, 0]))

    def in_zone(self, m: Measurement, s: InsertSpec) -> bool:
        """Aligned with the slot and inside the entry (up to engage_axial_tol before it)."""
        period = math.radians(s.twist_period_deg)
        twist = (m.twist + period / 2) % period - period / 2
        return (-s.seat_tol <= m.axial <= s.travel + s.engage_axial_tol and m.radial < s.engage_radial_tol
                and m.tilt_deg < s.engage_tilt_deg and abs(math.degrees(twist)) < s.engage_twist_deg)

    def slide(self, slot: Slot) -> float:
        return float(self.data.qpos[slot.slide_q])

    def slot_state(self, slot: Slot) -> str:
        self._sync_from_eq_active()
        if slot.child is None:
            return FREE
        return SEATED if self.data.eq_active[slot.lock] else ENGAGED

    def child_slot(self, child: str, parent: str | None = None) -> Slot | None:
        self._sync_from_eq_active()
        return next((s for s in self.slots if s.child == child and parent in (None, s.parent)), None)

    def child_state(self, child: str, parent: str | None = None) -> str:
        s = self.child_slot(child, parent)
        return FREE if s is None else self.slot_state(s)

    def stop_force(self, slot: Slot) -> float:
        d = self.data
        return float(sum(d.efc_force[i] for i in range(d.nefc)
                         if d.efc_type[i] == self._limit_type and d.efc_id[i] == slot.slide_jnt))

    # ---------------------------------------------------------------- transitions
    def _log(self, event: str, slot: Slot, child: str, **extra) -> dict:
        rec = dict(event=event, time=float(self.data.time), hole=slot.key, child=child, slide=self.slide(slot), **extra)
        self.events.append(rec)
        return rec

    def _engage(self, child: str, slot: Slot, axial: float, reason: str) -> dict:
        self.data.qpos[slot.slide_q] = float(np.clip(axial, 0.0, slot.spec.travel))
        self.data.qvel[slot.slide_v] = 0.0
        self.data.eq_active[slot.weld[child]] = 1
        slot.child, slot.since = child, float(self.data.time)
        return self._log("engage", slot, child, reason=reason)

    def _seat(self, slot: Slot) -> dict:
        self.data.eq_active[slot.lock] = 1
        return self._log("seat", slot, slot.child)

    def reset(self):
        """Release every child and zero the carriages (call after a data reset)."""
        for slot in self.slots:
            for eq in slot.weld.values():
                self.data.eq_active[eq] = 0
            self.data.eq_active[slot.lock] = 0
            self.data.qpos[slot.slide_q] = 0.0
            self.data.qvel[slot.slide_v] = 0.0
            slot.child, slot.since = None, -1e9
        self.events.clear()

    def update(self) -> list[dict]:
        """Run the FSM once (after a forward pass / step). Returns the events it produced."""
        self._sync_from_eq_active()
        new = []
        busy = {s.child for s in self.slots if s.child is not None}
        for slot in self.slots:
            s = slot.spec
            if slot.child is not None:
                if self.data.eq_active[slot.lock]:
                    continue
                if self.slide(slot) <= s.seat_tol:
                    new.append(self._seat(slot))
                elif (self.slide(slot) >= s.travel - 0.0005 and self.data.time - slot.since > s.release_grace_s
                      and self.stop_force(slot) > s.release_force):
                    child = slot.child
                    self.data.eq_active[slot.weld[child]] = 0
                    slot.child = None
                    new.append(self._log("disengage", slot, child, reason="pulled_out"))
                continue
            seat = (self._body_T(slot.parent_body) @ slot.T_local)[:3, 3]
            for child in s.children:
                if child in busy:
                    continue
                if np.linalg.norm(self._mate_T(child, slot)[:3, 3] - seat) > s.travel + 0.03:
                    continue
                m = self.measure(child, slot)
                if self.in_zone(m, s):
                    new.append(self._engage(child, slot, m.axial, "entry"))
                    busy.add(child)
                    break
        return new

    def force_seat(self, child: str, parent: str, slot_index: int) -> dict:
        """Put `child` fully in a slot and lock it (assembled spawns, tests); call mj_forward afterwards."""
        self._sync_from_eq_active()
        slot = next(s for s in self.slots if s.parent == parent and s.index == slot_index)
        if slot.child is not None:
            raise ValueError(f"{slot.key} is already occupied by {slot.child}")
        if self.child_slot(child) is not None:
            raise ValueError(f"{child} is already engaged in {self.child_slot(child).key}")
        mujoco.mj_kinematics(self.model, self.data)
        T = self._body_T(slot.parent_body) @ slot.T_local
        if slot.mate[child] is not None:
            T = T @ inv_T(slot.mate[child])
        j = self.model.body_jntadr[self.body_id[child]]
        qa, va = self.model.jnt_qposadr[j], self.model.jnt_dofadr[j]
        self.data.qpos[qa:qa + 3] = T[:3, 3]
        self.data.qpos[qa + 3:qa + 7] = mat_to_quat(T[:3, :3])
        self.data.qvel[va:va + 6] = 0.0
        ev = self._engage(child, slot, 0.0, "force_seat")
        self._seat(slot)
        return ev

    # ---------------------------------------------------------------- recording / replay
    def snapshot(self) -> dict:
        return {"children": {s.key: s.child for s in self.slots},
                "seated": {s.key: bool(self.data.eq_active[s.lock]) for s in self.slots},
                "since": {s.key: s.since for s in self.slots}}

    def restore(self, snapshot: dict):
        """Re-apply a snapshot after restoring the matching MuJoCo state (missing keys: all free)."""
        children, seated = snapshot.get("children", {}), snapshot.get("seated", {})
        for slot in self.slots:
            for eq in slot.weld.values():
                self.data.eq_active[eq] = 0
            slot.child = children.get(slot.key)
            slot.since = float(snapshot.get("since", {}).get(slot.key, -1e9))
            if slot.child is not None:
                self.data.eq_active[slot.weld[slot.child]] = 1
            self.data.eq_active[slot.lock] = int(bool(seated.get(slot.key, False)) and slot.child is not None)
