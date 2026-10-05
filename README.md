# Workspace extras

Snapshot of the other folders that sit next to `Mujoco_data_collect` on the collection laptop
(taken 2026-10-05). They are third-party code, kept here with their own licenses.

| Folder | Upstream | State |
|---|---|---|
| `LIBERO/` | https://github.com/Lifelong-Robot-Learning/LIBERO at `8f1084e` (MIT) | plus an empty `libero/__init__.py` |
| `stir-collect-sim/` | https://github.com/stir-replanning/stir-collect-sim at `a9501a5` (Apache-2.0) | local edits to `sim/collect_circle_server.py` and `sim/hdf5_writer.py` |
| `platform-tools/` | Android SDK Platform-Tools 37.0.1 (Linux), see `NOTICE.txt` | unmodified |

The virtualenvs (`fsim_venv`, `libero_venv`) are not included; rebuild them from the requirements files.
The recorded episodes are attached to the repository's releases.
