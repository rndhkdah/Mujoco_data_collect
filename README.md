# furniture-collect-sim

FurnitureBench-style furniture assembly tasks built natively in MuJoCo, for collecting VR (Meta Quest 3)
teleoperated demonstrations with a Kinova Gen3.

The tasks follow the design of [FurnitureBench](https://clvrai.com/furniture-bench/) (Heo et al., RSS 2023,
MIT License): part dimensions, masses, scene layout, randomization levels and acceptance rule, assembled poses
and tolerances, and phases. No FurnitureBench code or assets are used. Parts are parametric (collision
primitives plus generated visual meshes), and all frames are our own (z up, quaternions w-x-y-z, part z =
assembly axis).

## Status

Verified on the Windows dev laptop (pure MuJoCo parts) and on WSL2 Ubuntu as a stand-in for the Linux
collection box (robosuite 1.4.0, MuJoCo 3.3.7, software EGL).

| Milestone | State |
|---|---|
| M1 native core: specs, parametric parts, scene, helical screw constraint, assembly checks, reset sampling, phase labels | done; spike S1 passes |
| M0 robosuite setup and robot spikes | done: every one_leg pose is reachable, the grip holds leg4 while twisting, the arm screws leg4 in (success, reward 1) |
| M2 robosuite Kinova env (arena, cameras, home IK, grasp predicate) and scripted one_leg expert | done; the expert completed 40/40 low and 100/100 med layouts (target: 90% / 70%) |
| M3 server, teleop mapping, per-episode HDF5, merge, replay | done; a full recorded expert episode (1141 steps) replays to success from states and from open-loop actions |
| M4 3D WebXR client | done and checked in a desktop browser against the live server; the headset checks (72 Hz, pose lag under 120 ms, an operator completing one_leg) need the Quest |
| M5 pilot and collection | needs the Linux box, the Quest and an operator |

## Layout

```
fsim/geometry.py          SE(3) helpers
fsim/spec/                task specs (square_table, one_leg) and their types
fsim/models/              parametric MJCF parts, generated meshes, standalone scene
fsim/sim/                 screw manager, assembly checker, reset sampler, phase tracker, scripted hand
fsim/env/                 robosuite layer: KinovaFurnitureEnv, arena, Robotiq 2F-85, IK, scripted one_leg expert
fsim/teleop/mapping.py    Quest controller -> grip-site target -> OSC_POSE action
fsim/collect.py           episode loop (READY -> RECORD -> SAVE / DISCARD), headless-testable
fsim/server.py            collection server: collect thread + aiohttp page/WebSocket
fsim/vr/scene_export.py   scene GLB + manifest, binary pose frames for the 3D view
fsim/static/              vr3d.html (3D view), panel.html (camera panel), vendored three.js
fsim/data/                HDF5 writer, merge, replay, inspect
tools/                    reach check, gripper calibration
spikes/s1_helical_screw.py
stack.sh _env.sh kill.sh  Linux box: setup, check, start, status, merge, replay
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

On Windows the robosuite, collector and server tests skip (see the Linux notes below). Run them on Linux or
WSL2 with `MUJOCO_GL=egl`. One test records a full expert episode with cameras, which takes minutes on
software rendering; `-m "not slow"` leaves it out.

## Collecting demos (Linux box + Quest 3)

```bash
bash stack.sh setup     # venv ../fsim_venv, requirements, Robotiq gripper asset, part meshes
bash stack.sh check     # tests + reach check (no Quest needed)
bash stack.sh start --autopilot                        # watch the scripted expert in the headset first
bash stack.sh start --task one_leg --randomness low --num-demos 50 --operator alice
```

Connect the Quest by USB (developer mode). `stack.sh start` sets up `adb reverse tcp:8003 tcp:8003` and runs
the server. In the Quest browser, open `http://localhost:8003` and press **Enter VR**, or **Passthrough** to
keep the room visible.

**Episode flow:**
1. READY: the arm holds. B re-rolls the layout.
2. The first R grip press starts RECORD.
3. On success with the gripper open, a short settle is recorded and the episode is saved.
4. B discards the episode. Hitting the step limit (3000 steps, 150 s) times it out.

Episodes go to `$FSIM_DATA_DIR/<task>/<randomness>/*.hdf5` (default `~/fsim_dataset`).

`--auto-screw` (one_leg) hands the screwing to the scripted expert: once the leg stands in a hole with the
gripper open for half a second, the expert regrasps it from above and screws it in, then control returns to
you. Those steps are recorded with the episode and listed in the `auto_screw` attr. B discards.

`--autopilot` lets the scripted expert drive the arm in real time and saves nothing. Use it to check the
headset view (frame rate, smoothness, depth, scene placement with the left hand) before learning the
controls.

### Controls

Every teleop motion (hand following and both stick rates) is scaled by `TeleopConfig.speed`, 0.7 by default.

| Input | Action |
|---|---|
| R grip (hold) | clutch: the gripper follows your hand in position and rotation |
| R trigger (hold) | close the gripper |
| A (hold) | precision mode: the gripper moves 0.2x your hand in position and 0.3x in rotation |
| R stick left/right | twist about the gripper axis, for screwing (works without the clutch) |
| R stick up/down | push / pull along the gripper axis |
| B | discard the episode (in READY: new layout) |
| X | manual phase mark |
| Y | recenter the scene in front of you |
| L grip (hold) + move | drag the scene |
| L stick | left/right turns the scene, up/down zooms (ignored while the R grip is held) |

### The 3D view

`--view 3d` (the default) renders the simulation in the headset with three.js, in stereo and at the
headset's frame rate. The workspace sits 1:1 in front of you.

- **What you see:**
  - By default you look from the robot's left-rear, with the scene turned 30 degrees. The arm then comes in
    from your right, like your own right arm, and the fence corner where the tabletop goes is straight ahead.
    Use `?yaw=0` for a view straight from behind, or `?yaw=-30` for the right-rear.
  - The part to assemble (leg4 in one_leg) is tinted.
  - A ghost gripper shows the commanded grip-site target. It is brighter while clutched, its fingers show the
    open/close command, and they turn cyan in precision mode.
  - A HUD shows state, phase, steps, saved count and flags, next to the live wrist camera.
- **How it works:**
  - At startup the server exports the compiled model's visual geoms to `/scene.glb`, one node per body (about
    6 MB, sent gzipped at about 2.8 MB), plus `/manifest.json`.
  - Every control step it sends a binary pose frame: a 12-byte header, then 7 floats per body, the commanded
    target and 4 flags. Every second step it also sends a wrist JPEG (`fsim/vr/scene_export.py`).
  - The page draws one control step behind (about 50 ms) so motion stays smooth; `?interp=0` turns that off.
- **Controller frame:** the page converts the right controller's grip pose into the robot frame through the
  current scene placement and sends it with `frame: "robot"`. Your hand therefore moves the gripper the way you
  see it, wherever you put the scene.
- **Panel fallback:** `--view panel` streams the teammate-style front | wrist | rear camera panel to
  `panel.html` instead.
- **Offline:** three.js r186 is vendored in `fsim/static/vendor/three` (MIT), so the headset needs no internet.
- **Debugging:** `chrome://inspect` on the Linux box shows the headset page's console, including
  `window.fsim.stats()`. The page also works in a desktop browser as a preview.

### Data

One HDF5 file per episode, in the teammate's layout plus extras. The full key list is in
`fsim/data/hdf5_writer.py`.

- **Teammate layout:**
  - `actions` (OSC_POSE, normalized)
  - `rewards`, `dones`
  - `states`
  - `obs/{agentview_rgb, eye_in_hand_rgb, joint_states, gripper_states, ee_pos, ee_quat}`
- **Extras:**
  - `actions_fb`: FurnitureBench action convention
  - `target_pose`
  - `states_full`: `mj_getState`, including `eq_active` for the screw welds
  - `skills`, `phase`
  - the raw `controller` stream
  - `obs/rear_rgb` and `obs/parts_poses`
  - `obs/corner_rgb`: a camera above the table's far corner on the robot's left, looking down at the
    workspace (this copy only).
  - `obs/side_rgb`: a camera beside the table's long edge on the robot's left. Only this copy
    (`Mujoco_data_collect_sidecam`, branch `side-camera`) has it; it defaults to port 8004 and
    `~/fsim_dataset_sidecam` so it runs next to the original (port 8003, `~/fsim_dataset`).
  - with `--record-operator`: `obs/operator_rgb`, the scene rendered from the operator's headset pose (fovy
    90), and `obs/operator_head_pose`. It is a MuJoCo render, so it has no ghost gripper or HUD.
  - attrs with the layout, screw events, start/end snapshots, versions and the real `env_args`
  - attr `source`: `teleop` for VR demos, `scripted` for expert demos

```bash
python -m fsim.data.inspect <file.hdf5>                       # fields and shapes
python -m fsim.data.replay <file.hdf5> --mode states          # reproduces the recorded run exactly
python -m fsim.data.replay <file.hdf5> --mode actions         # open loop from the first state
bash stack.sh merge -o one_leg.hdf5 --success-only           # robomimic-style file with masks
```

### Scripted expert

`fsim/env/expert.py` completes one_leg without a headset. It reads part poses from the simulation state, so
it is a privileged script, not a vision policy.

```bash
bash stack.sh expert --randomness med --episodes 20                  # success rate (no cameras, about 15 s each)
bash stack.sh expert --randomness low --episodes 5 --record ~/fsim_expert   # demos in the VR file format
```

- **Phases:** it follows the FurnitureBench phases with a parallel gripper:
  1. Clamp the tabletop's rim wall that faces the robot.
  2. Square the tabletop and slide it into the fence corner.
  3. Grasp the lying leg with the gripper tilted 30 degrees and lift it.
  4. Carry it over the nearest hole, turn it upright, and lower the stud until it rests on the thread catch.
     Release it standing in the hole.
  5. Regrasp from above and turn the wrist at 90 degrees/s until the leg seats. The Kinova's last joint is
     continuous, so no regrasp is needed while screwing.
- **Results (WSL2):** 40/40 at low (one seed) and 100/100 at med (three seeds). Episodes take about 950 to 1200 steps
  (the limit is 3000).
- **Limits:** the "high" presets are not covered. One of the three works. In another the target leg lies in
  the corner where the tabletop must go, and in the third the leg is out of reach of the tilted grasp.
- **Why the tilted grasp:** with a horizontal gripper the wrist links hang 5 to 9 cm below the grip point and
  land on the lying legs or the 3 cm fence. Tilted, the gripper points 30 degrees down when the leg is
  upright, and the wrist stays above the grip point.

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

## Adding a task (generic parts and mechanisms)

- **Parts:** `PrimitivePartSpec(name, mass, geoms, studs, sockets)` in `fsim/spec/types.py`. Geoms are
  box / cylinder / capsule / sphere with local pos + quat; the origin must be the centre of the geoms'
  bounding box. Each stud adds a collision cylinder. Sockets are visual only, so the collision geoms must
  leave the bore clear (`fsim.spec.common.ring_geoms`). The visual mesh is CSG (geoms - sockets + studs,
  helical ridges), with primitive visuals as the fallback. Inertia comes from the geoms unless `com` /
  `inertia` override it.
- **Feature frames:** the origin is on the part surface and +z points out of the part. Mated, the child
  feature sits on the parent site with z anti-parallel.
- **Screws:** `ScrewSpec(..., sites=(site_at(socket_or_stud), ...), child_feature=...)` takes any parent
  and any axis. The child can own the stud (catch cylinder inside the bore), or the parent can own it
  (nut: the catch is a collar of `catch_radius` around the stud). Chains work, so a child of one screw can
  be the parent of another. Without `sites`, the tray `hole_centers` are used as before.
- **Inserts:** `InsertSpec` is the slide-only version, managed by `env.inserts` next to `env.screw`.
- **Phases:** `Phase(kind, part, target, label)`, with the kinds listed in `Phase`.
- **Assembled poses:** `seated_rel_pose()` gives the seated poses for `AssemblyTarget.rel_poses`.
- **Tests:** `tests/test_tasks_generic.py` checks every registered task. `tests/synthetic_task.py` shows
  every mechanism.
- **Mesh paths:** `fsim/models/meshes.py` defines the mesh directory hash of the tray/leg tasks. Leave it
  unchanged, or recorded demos lose their mesh paths. New generator code goes in
  `fsim/models/part_meshes.py`.

## Notes for the Linux collection box

- robosuite 1.4.0 cannot run on the Windows dev laptop: Windows Application Control blocks numba's DLL, and
  robosuite requires numba. Develop the robosuite layer on Linux, or on WSL2 as a test bench. WSL2 renders
  EGL in software: three 224 px cameras run at about 12 steps/s, below the 20 Hz control rate, so collect on
  the box's GPU.
- An EGL context renders only in the thread where it is current. When the env is built in one thread and
  stepped in another, every camera image (recorded and streamed) comes back as noise. The server therefore
  builds the env inside its collect thread (`fsim.server.start_collect`), and `tests/test_server_robosuite.py`
  checks that streamed images are real renders. Shutdown stops that thread before exit, because tearing EGL
  down under a running render segfaults.
- robosuite 1.4.0 calls `mujoco.mj_fullM(model, M, data.qM)`. MuJoCo 3.10 changed that signature and 3.11
  removed `qM`, so keep `mujoco<3.10` (see requirements.txt).
- robosuite does not merge `<default>` classes, so every generated element carries explicit attributes.
  Equalities must be appended to the task model after `ManipulationTask` is built.
- Call `HelicalScrewManager.update()` every physics substep (from `_pre_action`) and `reset()` in
  `_reset_internal` (robosuite uses `mj_resetData` when `hard_reset=False`).
- Several processes may build the generated assets at once (the server and a replay, parallel jobs). The
  writers use per-process temp files and an atomic rename (`fsim/fileutil.py`).
- MuJoCo appends solver warnings to `MUJOCO_LOG.TXT` in the working directory (gitignored). An occasional
  "Linesearch objective is not convex" during hard contact is harmless.
- Generated mesh paths are absolute. robosuite's `edit_model_xml` (used by its playback, and by robomimic's
  `reset_to`) only relocates robosuite assets. `KinovaFurnitureEnv.edit_model_xml` also relocates the
  generated part meshes, rebuilding them from the stored spec hash if needed, and the Menagerie gripper.
- Robot base: (-0.3, 0, 0.43), on a 15 mm clamp. The rear camera is derived from it.
