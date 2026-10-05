# Architecture

One Python process does everything: `sim/collect_circle_server.py`. It runs the robosuite
env in a background thread, streams rendered frames to the Quest over a WebSocket, takes
the controller state back over the same socket, and appends successful episodes to an
HDF5 file. There is no ROS, and there are no other processes.

Every shell script first sources [`_env.sh`](_env.sh). It is the one place that sets the
paths (`REPO`, `SIM_DIR`, `VENV_PY`), the port (`SIM_PORT=8002`), the output directory
(`SIM_DATA_DIR`), `MUJOCO_GL=egl`, and the `server_pids` helper that `status` and
`kill.sh` use.

## Dataflow

```
bash stack.sh start [flags]
  ├─ gripper preflight   (kinova_env.menagerie_gripper_xml(); exit 1 if missing)
  ├─ adb reverse tcp:8002 tcp:8002
  └─ exec libero_venv python sim/collect_circle_server.py --port 8002 [flags]
        │  env_config.apply_config(--config)  ← sim/env_configs/<name>.json  (geometry defaults)
        │
        ├─ collect thread:  make_circle_env()
        │     kinova_circle_env  ←  kinova_grid_env  ←  kinova_env
        │     (KinovaTwoObjCircle)  (objects, basket,   (MenagerieRobotiq2f85 gripper,
        │                            strict success)     registered in robosuite)
        │     robosuite 1.4.0 Kinova3 + OSC_POSE, LIBERO objects, EGL offscreen cameras
        │     circle_schedule   → the episode list (funnel lattice or voxel grid)
        │     hdf5_writer.DemoWriter → $SIM_DATA_DIR/data/circle.hdf5 (or --out)
        │
        └─ aiohttp on 127.0.0.1:8002
              GET /          → sim/static/index.html   (the WebXR client)
              GET /static/*  → sim/static/
              GET /ws        → WebSocket: JPEG frames out, controller JSON in

RUNTIME LOOP:
  Quest controllers ══/ws══▶ shared.controller ──▶ collect thread: clutched 1:1 → OSC_POSE action ──▶ env.step
  env obs [agent | wrist | top view] ──JPEG──▶ shared.frame ══/ws══▶ headset panel
  strict success ──▶ settle capture ──▶ DemoWriter.add_demo ──▶ HDF5 (appended, one file)
```

## Episode driver (`collect_loop`)

1. **Schedule.** `funnel` mode (default) uses `circle_schedule.build_circle_schedule`:
   2 (which object is on the left) x 2 (target cell) x 2 (approach: start over the target
   or over the other object) x `n_layers*9` start points, in layer-major order with the
   lowest layer first. With 3 layers that is 216 episodes. `voxel` mode first runs a
   **reach pre-pass**: it drives the arm to every voxel center, keeps the reachable ones,
   and builds one episode per reachable voxel per target, in both arrangements
   (`build_voxel_schedule`).
2. **Resume.** If the output file exists, the `(obj_left_idx, target_cell, approach,
   init_idx)` of every `demo_*` already in it is skipped. The number of `demo_` groups is
   the pointer into the remaining list.
3. **SETUP.** The server places the objects for this spec (`set_episode` + `reset`). In
   funnel mode it rebuilds the start lattices around the objects' actual positions, then
   drives the EE to this episode's start point with a P-controller.
4. **WAIT** (`READY-grip to start`). The arm holds the start pose until the right grip
   passes `--engage-threshold`.
5. **RECORD.** Each step turns the controller state into a 7-D OSC_POSE action. Grip held
   = clutched motion relative to the anchor taken at grip-down (`A` switches to rotation).
   Trigger = gripper. Rising `B` = discard. Every step records obs, action, the full
   MuJoCo state, reward and done, at about 20 Hz.
6. **Success.** `_check_success()` is strict: the target is centered in the basket, below
   the rim, and settled. It also requires the gripper to be commanded open. Then
   `--settle-capture` (12) extra hold-still steps are recorded, the last step gets
   reward 1 and done 1, and the demo is written with its per-episode attrs (layer, start
   xyz, object positions and heights, basket, voxel ijk, ...). If the episode reaches
   `--max-steps` or `B` is pressed, it is discarded and the same spec is retried.

## WebXR client (`sim/static/index.html`)

It shows the frames as a head-locked quad in AR passthrough (or VR if AR is unavailable)
and sends ONE controller snapshot per XR frame: right-hand pose, trigger, grip, A, B, plus
the left X/Y. When the runtime enumerates several input sources per hand, the client picks
the best one, and it holds the last good read across a dropped frame so a missing hand
never looks like a button release. The left and right sticks resize and move the panel on
the client only. The left X/Y buttons are sent but unused in the sim.

## Design notes

- **No `set -e`/`set -u`** in the shell scripts by design: `pgrep`, `curl` and `adb`
  legitimately return non-zero.
- **Scoped stop.** `server_pids` matches only python processes whose script argument
  resolves to *this* repo's `sim/collect_circle_server.py` (the script python runs, not a
  file named in `-c`/`-m` arguments). `kill.sh` therefore cannot hit another clone of the
  server or any other process, including scripts that import this repo's `sim/` modules.
- **Stop between episodes.** `kill.sh` sends SIGTERM, then SIGKILL ~2 s later. SIGTERM
  lets aiohttp shut down, but nothing waits for the collect thread: it is a daemon, it
  is never joined, and there is no `on_shutdown` hook. With the Quest page connected,
  aiohttp also waits up to 60 s for the open WebSocket, so the kill usually ends in
  SIGKILL. All demos live in one HDF5 file, and a kill in the middle of `add_demo` can
  corrupt the whole file. Stop the server after a `[save] demo_N` line, while the panel
  shows `READY-grip to start`.
- **Never open the output file while collecting.** `DemoWriter` reopens the file in
  `'a'` mode for every demo. If another process holds an h5py handle, that open fails on
  the HDF5 lock and the collect thread dies, while aiohttp keeps serving a frozen frame.
  `stack.sh status` therefore only lists files.
- **127.0.0.1 only.** The headset gets in over `adb reverse`, so there is nothing to
  firewall. Another WebXR server on a different port can run at the same time, and the
  Quest switches between the pages.
- **Image convention.** Recorded images are robosuite's raw OpenGL-convention frames. Only
  the VR preview is flipped (`[::-1]`). A robosuite `macros_private.py` that sets
  `IMAGE_CONVENTION = "opencv"` would silently flip every recorded image and break the
  output contract in README.md.
- **Pinned env chain.** The 8 files listed in `sim/env_chain.sha256` are pinned by hash.
  Change them only together with that file, and keep any other copy of the chain (with
  `sim/assets/README.md` and `sim/assets/build_gripper.py`) byte-identical.
  `collect_circle_server.py`, `hdf5_writer.py`, `static/` and `sim/README.md` are not part
  of the chain.
