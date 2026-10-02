# furniture-collect-sim

FurnitureBench-style furniture assembly tasks built natively in MuJoCo, for collecting VR (Meta Quest 3)
teleoperated demonstrations with a Kinova Gen3.

The tasks follow the design of [FurnitureBench](https://clvrai.com/furniture-bench/) (Heo et al., RSS 2023,
MIT License): part dimensions, masses, scene layout, randomization levels and acceptance rule, assembled poses
and tolerances, and phases. No FurnitureBench code or assets are used. Parts are parametric (collision
primitives plus generated visual meshes), and all frames are our own (z up, quaternions w-x-y-z, part z =
assembly axis).

## Status

| Milestone | State |
|---|---|
| M1 native core: specs, parametric parts, scene, helical screw constraint, assembly checks, reset sampling, phase labels | done; 66 tests and spike S1 pass on Windows |
| M0 Linux setup and robot spikes; M2 robosuite Kinova env; M3 server and HDF5; M4 3D WebXR client; M5 collection | next, on the Linux collection box |

## Layout

```
fsim/geometry.py          SE(3) helpers
fsim/spec/                task specs (square_table, one_leg) and their types
fsim/models/              parametric MJCF parts, generated meshes, standalone scene
fsim/sim/                 screw manager, assembly checker, reset sampler, phase tracker, scripted hand
spikes/s1_helical_screw.py
tests/
```

## Run (development)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-dev.txt   # Linux: .venv/bin/python
.venv/Scripts/python -m pytest
.venv/Scripts/python spikes/s1_helical_screw.py --trials 40
.venv/Scripts/python -m fsim.models.build                    # pre-generate visual meshes
```

Generated meshes go to `fsim/assets/generated/<task>-<hash>/`, which is gitignored. The hash covers the part
specs and the generator source. Set `FSIM_GENERATED` to use a different directory.

## square_table / one_leg design

- **Tabletop:** an open tray, 162.5 x 162.5 x 31.25 mm, 151 g. It has a 6.25 mm plate, a 6.25 mm rim, and a
  threaded boss (bore radius 12.5 mm, outer radius 16.25 mm) around each of the 4 blind holes at
  (+-56.25, +-56.25) mm. The hole floor is the plate.
- **Legs:** 30 x 30 x 62.5 mm body plus a 25 mm stud, 23.1 g each. Threads are right-handed with a 9.375 mm pitch.
- **Assembly:** a leg counts as assembled at (+-56.25, +-56.25, 46.875) mm in the tabletop frame with identity
  rotation. The tolerances are 5 mm per axis and rotation-column cosine >= 0.94, so the final screw angle
  matters. one_leg scores leg4 in any corner.
- **Randomization:**
  - low: +-1.5 cm and +-15 degrees.
  - med: +-5 cm and +-45 degrees.
  - high: the design's 3 measured layouts.
  - Acceptance follows the design's rule: footprints are rotated and replaced by their axis-aligned bounding
    boxes, which must be more than 1 cm apart on x or y and clear the design's reset rectangles. med accepts
    only about 1 in 50,000 draws, so sampling is vectorised (about 15 ms).
  - Measured layouts can overlap the fence slightly. `place_and_settle` pushes parts out before settling.
- **Environment friction:** contacts with the table use 0.125 and contacts with the fence use 0.575. These are
  averages of the part friction 0.15 with 0.10 and 1.0, which reproduces the averaging of the design's
  simulator.

## Screwing: the helical constraint

FurnitureSim screws parts together through thread-on-thread SDF contact in PhysX. MuJoCo would convex-hull
those meshes, so here each threaded hole gets a virtual carriage at the seated child pose:

- **Joints:** the carriage has a slide and a hinge along the hole axis, coupled by an always-on joint equality
  `hinge = (2*pi/pitch) * slide`.
- **Catch:** before the thread catches, the stud rests on a catch cylinder inside the bore.
- **Engage:** when the stud is aligned in the bore and its twist crosses the thread-start phase while
  tightening, the manager (`fsim/sim/screw.py`) snaps the carriage to the child, disables the catch, and
  activates a weld carriage <-> child. From then on the child can only move along the helix.
- **Stops:** the hinge range `[0, 3*pi]` gives the seated stop (1.5 turns, 540 degrees, matching the assembled
  pose) and the thread start.
- **Release:** a stop torque above 0.005 N*m at the thread start pops the child back out. That comes from
  unscrewing past the start, or from pulling with more than about 3.4 N. Mid-thread, a pull back-drives the
  thread only above about 6.7 N (`thread_friction` 0.01 N*m with stiff friction solver parameters). A seated
  leg hanging upside down creeps under 1 degree in 20 s.
- **Source of truth:** `data.eq_active` is authoritative for which child is engaged. Call `reset()` after a
  data reset. `snapshot()` / `restore()` also carry the arming state for replay.

Four things are required or the mechanism silently fails:

- Write the coupling as **hinge = f(slide)**. `slide = K * hinge` with K about 1.5 mm/rad lets the hinge spin
  freely.
- Models must compile with **`<compiler angle="radian"/>`**. MJCF defaults to degrees.
- The carriage needs **armature**. A near-massless carriage makes every constraint on it limp.
- The hinge needs **stiff friction parameters** (`solreffriction`, `solimpfriction`). Default frictionloss
  creeps like a viscous damper.

Spike S1 (MuJoCo 3.3.7) results:

- 40/40 randomized screw-in/screw-out trials succeed (100/100 on another seed before the tray change).
- Seated pose error is at most 0.27 mm.
- Engagement never happens off the hole, when tilted 15 degrees, or when twisting the wrong way.
- The standalone scene runs at about 15-19x real time with the manager updating every physics step.

## Notes for the Linux collection box (M0/M2)

- robosuite 1.4.0 cannot run on the Windows dev laptop: Windows Application Control blocks numba's DLL, and
  robosuite requires numba. Develop the robosuite layer on Linux.
- robosuite 1.4.0 calls `mujoco.mj_fullM(model, M, data.qM)`. MuJoCo 3.10 changed that signature and 3.11
  removed `qM`, so keep `mujoco<3.10` (see requirements.txt).
- robosuite does not merge `<default>` classes, so every generated element carries explicit attributes.
  Equalities must be appended to the task model after `ManipulationTask` is built.
- Call `HelicalScrewManager.update()` every physics substep (from `_pre_action`) and `reset()` in
  `_reset_internal` (robosuite uses `mj_resetData` when `hard_reset=False`).
- Generated mesh paths are absolute. robosuite's `edit_model_xml` (used by its playback, and by robomimic's
  `reset_to`) only relocates robosuite assets. M2 must override it to remap
  `<task>-<hash>/<part>.stl`, regenerating from the stored spec hash if needed.
- Robot base: (-0.3, 0, 0.43), on a 15 mm clamp. The rear camera is derived from it.
