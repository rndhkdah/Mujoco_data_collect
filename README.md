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

## stool and lamp design

Both are measured from the design's part geometry and rebuilt from primitives. Threads use the square_table
family (9.375 mm pitch, 11.5 mm stud, 12.5 mm bore, 1.5 turns from catch to seat). In the design the
stud tips touch the hole floors when seated; here studs are 0.5 mm shorter so the screw stop sets the
pose. Neither task has a scripted expert, auto-screw or reach check yet.

- **stool** (`fsim/spec/stool.py`): 3 targets, phases grasp_seat, place_seat_corner, then pick / insert /
  screw for leg3, leg2, leg1.
  - Seat: Ø100 x 30 mm open tray, holes up, 57.3 g. 2.5 mm plate, 2 mm rim (the grasp: Ø100 is too wide
    for the 2F-85), threaded bosses at r 32.5 mm, 120 degrees apart. The place_corner check ignores yaw.
  - Legs: 35 x 35 x 60 mm body sheared 15 degrees (12 stepped slabs), 27.1 g, vertical stud under the
    bottom-face centre. Any leg fits any socket; each site's yaw seats the leg leaning outward, 40.5 mm
    from the seat axis and 45 mm above it (matches the design's assembled poses to 0.4 mm).
  - The leg's grip point is 3 to 8 mm off its screw axis, so twisting about the gripper axis fights the
    thread; grasp the leg low.
- **lamp** (`fsim/spec/lamp.py`): targets bulb (screwed) and hood (position-only, requires the bulb);
  phases grasp_base, place_base_corner, pick / insert / screw_bulb, pick_hood, place_hood.
  - Base: 84.7 mm square foot, stepped loft, neck and a 55.2 mm square top plate (the grasp) around the
    bore, 63 mm tall, 59.7 g.
  - Bulb: stud, collar, 35 mm body and globe, 38.5 g; lying, it rests on globe and collar at about 8
    degrees (design 8.3). Estimated: globe 29.5 mm (design 30) and collar 21.6 mm (design 22.2), which
    keep that tilt while giving the hood more room around the globe.
  - Hood: open conical shell (Ø88 to Ø60 over 100 mm, 4 mm wall) of 24 tilted staves, 60 g. It rests
    with its inside on the top plate's corners, 72.2 to 74.6 mm above the base origin depending on its
    yaw (design 75.4); the target (73.4 mm, 5 mm tolerance) holds at every yaw. Grasp it outside near the
    narrow top (about 60 mm): the design's wall pinch puts a finger inside, where the globe is.
- Estimated, not measured precisely: the base loft steps, the seat hub, plate thicknesses, and the
  stepped leg shear. Layouts, footprints and the three measured "high" layouts follow the design.

## desk design

`fsim/spec/desk.py`. The square_table construction at another size, with FurnitureBench's desk numbers.

- **Parts:**
  - `desk_top`: a 150 x 210 x 31.25 mm open tray (`TrayPartSpec`), 169.4 g. It has a 6.25 mm plate, a
    6.25 mm rim, and a threaded boss (bore radius 12.5 mm, outer radius 16.25 mm) around each of 4 blind
    holes at (+-50, +-80) mm.
  - `desk_leg1..4` (`LegPartSpec`): a 30 x 30 x 100 mm body and a 25 mm stud.
- **Frames:** our part frames are the design's (x, -z, y) turned 90 degrees about the assembly axis, so
  the 210 mm side runs along W y at zero yaw.
- **Mechanism:** one tray screw with 4 holes. The seat height is 65.625 mm, the pitch 9.375 mm, and the
  threads right-handed. Each leg counts as assembled in any hole at identity rotation (5 mm per axis,
  cosine >= 0.94).
- **Layout:**
  - The tabletop starts at (0.2415, 0) holes up, with its long side along W y. The legs lie along +x at
    y = -0.15, -0.08, 0.08 and 0.15, with the stud toward the robot.
  - Footprints are the design's reset lengths: 150 x 210 mm and 125 x 30 mm.
  - The tabletop's position noise is [-5, +2] cm on both axes at low and med (the design's override). The
    3 high presets are converted from the design.
- **Phases:** `grasp_top`, `place_top_corner` (corner (0.2965, 0.06), yaw checked modulo 180 degrees),
  then `pick_legN`, `insert_legN`, `screw_legN` for N = 1..4.
- **Estimated:**
  - The leg mass is 56.2 g as the design states. Its simulator's density times mesh volume gives 29.4 g.
  - The stud collision radius is 11.5 mm, as on square_table.
  - The design's (10, 5, 10) mm position tolerance is replaced by 5 mm on every axis.
  - The corner fixture is assumed: the design has no desk script.
  - max_steps is 4500.

## round_table design

`fsim/spec/round_table.py`. Built from primitives (`PrimitivePartSpec`), with dimensions measured on
FurnitureBench's meshes.

- **Parts** (origin at the centre of each part's collision-geom bounding box):
  - `round_table_top`, 121.69 g: a 200 mm disc with a 6.25 mm plate and a 6.25 mm rim, 18.75 mm high. A
    42.5 mm boss in the middle, 32.5 mm high, holds a threaded 25 mm socket, 26.25 mm deep. The table is
    built upside down, socket up.
  - `round_table_leg`, 32.28 g: a 65 mm body with a 4-flat collar (33.4 mm across the flats, made of two
    crossed boxes), then a 42.5 mm cylinder. A 26.25 mm stud is below and a threaded 25 mm socket, 27.5 mm
    deep, is on top.
  - `round_table_base`, 33.96 g: a "+"-shaped foot, 130 mm tip to tip and 12.5 mm thick. Its arms are
    30 mm wide at the ends and narrow to 22 mm at a 40 mm hub. A 27.5 mm stud is below.
- **Mechanisms:** a chain of two screws, both with a socket site on the parent and the stud on the child:
  - The leg screws into the top: seated 49.05 mm above the top origin at -175 degrees yaw.
  - The base screws onto the leg: seated 39.05 mm above the leg origin at +90 degrees yaw.
  - Both poses match the design's assembled poses, converted into our frames (within the 0.3 mm seat gap).
- **Layout:**
  - The top starts socket up at (0.2415, 0). The leg lies along +x with the stud toward the robot. The
    base stands on its stud tip, as in the design.
  - Footprints are the design's: 200 x 200 mm for the top, 91 x 42.5 mm for the leg (centred on our
    origin), and the design's 20 x 20 mm for the base. The base's real 130 mm cross never fits next to the
    top at the nominal layout.
  - The top's position noise is [-5, +1] cm at low and med. The 3 high presets are converted.
- **Phases:** `grasp_top`, `place_top_corner` (corner (0.2715, 0.065), no yaw check), `pick_leg`,
  `insert_leg`, `screw_leg`, `pick_base`, `insert_base`, `screw_base`.
- **Spinning top:**
  - Nothing holds the round top against twisting except table friction, about 0.013-0.015 N*m.
  - With `thread_friction` 0.01 N*m, a leg or base driven along the thread leaves a free-lying top in place
    (under 1 degree). At 0.02 N*m the top starts to slip.
  - Twisting past the seat turns the whole table (the test driver's 24 degrees of overtightening turns it
    9-13 degrees). See `tests/test_task_round_table.py`.
- **Estimated:**
  - Stud collision radius 11.5 mm.
  - The number of polygon segments in each ring, and where the base arms change width.
  - The base screw counts as seated within 8 degrees (default 1). When the hand lets go, the hinge stop
    springs the wide base back about 5 degrees before thread friction holds it. Assembly allows about 20
    degrees.
  - 5 mm position tolerance; max_steps 4000.
  - A lying leg balances on the edge of its cylinder and tips 8-10 degrees onto the collar. The design's
    measured presets show the same.

## chair design

Spec: `fsim/spec/chair.py`. Tests: `tests/test_task_chair.py` plus the generic suite. The FurnitureBench meshes
were measured; the parts here are primitives.

- **Parts** (origin = bounding-box centre of the collision geoms):

| Part | Size (mm) | Mass | Notes |
|---|---|---|---|
| `chair_seat` | 100 x 100 x 30 | 61.87 g | open tray, 5 mm floor and 3.75 mm rim; 35 x 30 mm tongue between the studs. Two 25 mm deep holes for the legs (bore r 12.5) in bosses. Two studs, 47 mm long, on the rear edge: r 7.8 shank, then r 6.6 thread (6 mm pitch) |
| `chair_leg1/2` | 30 x 30 x 60 body + 25 mm stud | 22.44 g | |
| `chair_back` | 125 x 210 x 30 | 123.16 g | two 30 x 30 posts with 19 mm square through-holes, a 12.5 mm top rail and centre panel; the slot between the posts takes the tongue |
| `chair_nut1/2` | 30 mm cube | 10.25 g | blind 14.4 mm square bore, 18 mm deep |

- **Mechanisms:**
  - **Legs:** screwed into the seat holes, 9.375 mm right-handed pitch, as in square_table.
  - **Back:** an `InsertSpec` along the stud axis.
    - It engages once both studs are 7 mm into its holes (travel 40 mm) and seats on the stud face.
    - It is accepted the right way up or turned 180 degrees about the stud axis.
    - The face-flipped back would need a second child mate. That is not supported, so it does not engage.
  - **Nuts:** inverted screws (the seat owns the studs), 6 mm pitch.
    - The site is the nut mouth on the back's top face.
    - A r 8.2 mm catch collar fits inside the back's holes, so the back slides over it.
    - Thread friction is 0.004 N*m. The standing chair is not braced by the fence; twisting a nut on turns it
      less than 0.3 degrees.
- **Targets:** both legs into either hole, the back, both nuts on either stud.
  - Tolerance 5 mm on every axis and orientation cosine 0.94.
  - The nuts count only once the back is in (`requires`, as in the design).
- **Phases** (17, the design's chair skill count):
  1. grasp_seat
  2. place_seat_corner: tray up, studs toward the robot, rear face on the front fence at (0.3215, 0.115), within
     2.5 cm and 15 degrees of yaw -90
  3. pick / insert / screw leg1 and leg2
  4. flip_seat: the studs within 20 degrees of up, released
  5. pick_back, insert_back (seated)
  6. pick / insert / screw nut1 and nut2
- **Why the flip:** the back cannot go on while the seat lies flat, because it would go through the table. So
  the seat is stood on its rear face, studs up, which is the design's own init-assembled seat pose.
  - Once the back is on, its centre of mass is past the 30 mm rear face. The chair leans back onto the top rail
    (about 33 degrees) and rests there, so the nuts go on along a tilted axis.
  - Standing the seat with the legs away from the robot tilts that axis toward the robot.
- **Layout and randomization:**
  - Layout: the design's reset poses. Footprints are its reset rectangles. High: its three measured layouts.
  - At med, the design's narrower seat range applies (-5..+3.5 cm). It applies it at every level; here it is med only.
  - med accepts about 1 in 100k layouts, so `max_tries` is 3M.
- **Estimates and deviations:**
  - leg pitch 9.375 mm (measured 9.29)
  - nut pitch unified with the studs at 6 mm (the nut mesh says 5.62)
  - nut bore 18 mm deep; the design's 15.2 mm bottoms on the stud before the nut reaches the back, which is why
    the design has both 0.0795 and 0.0815
  - collision clearances: tongue narrowed to 35 mm, back holes 19 mm square, stud thread r 6.6
  - the leg-hole female thread radii
  - the snug corner
  - max_steps 6000
- **TODO (`tools/reach_check.py`):** add the chair's snug-corner leg-hole poses, top-down, about z 0.03-0.09:
  - hole W (0.355, 0.081) and W (0.355, 0.149)
  - the farthest is about 1 cm beyond square_table's farthest hole
  - Fall back to the design's corner (seat 2.5 cm off the front fence) if they are not reachable.

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

## drawer / cabinet design

Both tasks follow the FurnitureBench drawer and cabinet designs. The dimensions were measured from the
design's meshes and rebuilt as primitives (`fsim/spec/drawer.py`, `fsim/spec/cabinet.py`). Layouts are the
design's skill-0 reset poses and its 3 measured high layouts, converted with `tag_pose_to_placement`.
Footprints are the real part extents. Assembly tolerances are 5 mm per axis and cosine 0.94, as for
square_table. Numbers marked EST are our choices, not measurements.

### drawer

- **Parts:**
  - **drawer_box:** a chest, 112.5 x 121.5 x 97.5 mm, 151.6 g, standing on its +y end (+y points down). Its
    opening (+z face) holds two drawer slots, with an open tray on top. Each slot has a pair of side rails,
    r 3 mm (EST; the design uses 3.75).
  - **drawer_container_top / _bottom:** one shared geometry, 110.25 x 42.75 x 99.75 mm, 59.4 g. It is an open
    box with a 7.5 mm front panel that is wider than the opening, side grooves for the rails, and a knob.
- **Mechanism:** one `InsertSpec` with one slot per opening; either container fits either slot.
  - The slide axis is the box's z, horizontal when the box stands.
  - travel 85 mm: the container is captured 1 cm before its back reaches the opening, so it never slides in by
    contact.
  - Engage tolerances (EST): 4 mm radial, 8 degrees tilt, 8 degrees twist.
  - The seat is the front panel stopping on the side-wall front edges, with a 0.3 mm gap.
  - Sliding clearances are 0.5 to 2.25 mm (EST). The walls are widened to 1.5 mm per side.
  - The design's single slot z fits only one of its two meshes. Our physical seat lies within 3.5 mm of both.
- **Phases:**
  1. grasp_box
  2. place_box_corner: the box snug in the +y fence corner with its opening facing -W y, so the left fence
     backs it while drawers are pushed in.
  3. box_upright
  4. pick / insert / push of container_bottom
  5. pick / insert / push of container_top

  max_steps is 4500.
- **Gripper:** the box is wider than 85 mm on every side, so clamp a wall and slide it. With a top pinch the
  lower drawer can be held all the way in, because the upper slot is still empty. The upper drawer must be
  released about 2 cm before the seat and pushed home.

### cabinet

- **Parts:**
  - **cabinet_body:** 60.5 x 120 x 135 mm plus a 25 mm stud, 131 g. It has two hinge poles (r 3.1 mm) at the
    front corners, a shelf, and a top plate cut back so the door clips can pass. Its z points to the top end.
  - **cabinet_door_left / _right:** 30 x 53.75 x 123.75 mm, 30.2 g. Each is a panel with a pull handle and a
    C-clip on the outer edge: two 1.5 mm jaws (EST) and a 9.5 mm slot around the 6.2 mm pole.
  - **cabinet_top:** 57.5 x 120 x 30 mm, 60.3 g. A block with a blind 12 mm bore.
- **Mechanisms:**
  - **Doors:** one `InsertSpec` per pole.
    - travel 130 mm: the door is captured 6 mm before the clip reaches the pole tip.
    - Engage tolerances (EST): 3 mm radial, 4 degrees tilt (it is a 124 mm part), 10 degrees twist.
    - The door is locked once seated, with no hinge. The design scores a fixed pose, and a free door would
      swing open when the body is tipped up.
    - The design's left-door pose is 4.75 mm off its pole, so ours mirrors the right door.
  - **Top:** an inverted screw on the body's stud.
    - Thread: 9.375 mm pitch, right-handed (measured), 1.5 turns.
    - Catch collar r 14 mm, 1.25 mm clear of the doors.
    - `thread_friction` 0.03 N*m (EST): the long top otherwise springs back off the seat after a fast twist.
    - A second site rotated 180 degrees accepts the top's other placement. Holes at the same spot share one
      catch.
    - The top counts only after **both** doors. The design required only the left one.
- **Phases:**
  1. grasp_body
  2. place_body_corner: the body on its back, top end toward the robot, snug in the +y corner, so the front
     fence backs it while the doors are pushed on along +W x.
  3. pick / insert / slide of door_right
  4. pick / insert / slide of door_left
  5. grasp_body_flip
  6. tip_body_upright (body z within 15 degrees of up)
  7. pick_top
  8. insert_top
  9. screw_top

  max_steps is 6000.
- **Gripper:** pinch the door handles (7.5 mm) top-down. A door lies in the same orientation it has on the
  lying body. Grasp the top across its 57.5 mm. Pinch a side wall to tip the body up; this is the hardest step.

**Insert spike** (`drive` along the axis from the capture point, 20-40 trials per site and child):
- Drawer, both slots, both containers: 20/20 seated, with up to 2.5 mm lateral offset and 6 degrees twist, or
  up to 4 degrees tilt.
- Doors: 20/20 seated with up to 2.5 mm lateral offset and 6 degrees twist, and 40/40 with up to 3.5 degrees
  tilt.
- No child-parent contact while sliding, so the 0.5 mm rail gap and the 1.5 mm jaws hold.
- Seat error is at most 0.08 mm.
- With the earlier 8 degrees door tilt tolerance, about 1 in 20 tilted captures flung the door. That is why the
  tolerance is now 4 degrees.

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
