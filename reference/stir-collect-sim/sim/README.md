# sim/: robosuite simulation and the Quest WebXR collection server

The code that runs. The launcher (`stack.sh`, `_env.sh`, `kill.sh`) is at the repo root.

**This repo only:**
- `collect_circle_server.py`: the WebXR collection server (`:8002`, `bash stack.sh start`).
- `hdf5_writer.py`: `DemoWriter`, the robomimic/LIBERO HDF5 layout (the output contract in
  the top-level README).
- `static/index.html`: the Quest WebXR client.

External scripts can import from this directory at runtime:
`collect_circle_server.draw_birdeye_circle`, `hdf5_writer.DemoWriter` and the env chain
below. Renaming `sim/` or those names breaks them.

**Env chain, pinned by hash** (`sha256sum -c env_chain.sha256`). Keep any other copy
byte-identical:
- `kinova_circle_env.py` → `kinova_grid_env.py` → `kinova_env.py`: the MuJoCo env chain.
- `circle_schedule.py`, `env_config.py`, `env_configs/`: episode schedule and scene configs.
- `assets/README.md`, `assets/build_gripper.py`: the generator of the Menagerie Robotiq
  2F-85 gripper (`assets/grippers/`, gitignored) and its notes. They are not in
  `env_chain.sha256`; keep both identical in every copy (`cmp`).
- `env_chain.sha256`: hashes of the 8 `.py`/`.json` chain files above. It does not cover
  this repo's own files, so do not regenerate it over the whole directory.

See the top-level [README.md](../README.md) and [ARCHITECTURE.md](../ARCHITECTURE.md).
