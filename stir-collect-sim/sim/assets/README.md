# sim/assets: the Menagerie Robotiq 2F-85 gripper

Every env in this chain (`kinova_env` -> `kinova_grid_env` -> `kinova_circle_env`) is built
with `gripper_types="MenagerieRobotiq2f85"`, which loads `grippers/robotiq_2f85_menagerie.xml`
and the meshes it names. robosuite does not ship these files and the repository does not
track them: `build_gripper.py` in this directory generates them into `grippers/`
(gitignored). Until it has run, the modules still import, but building any env
(`make_kinova_env`, `make_grid_env`, `make_circle_env`) stops with a `FileNotFoundError`
from `kinova_env.menagerie_gripper_xml()` that lists the paths it checked.

## Generate it

```bash
cd <repo> && ~/workspace/libero_venv/bin/python sim/assets/build_gripper.py
```

- Any Python with `mujoco` and `numpy` works (the venv from SETUP.md has both). It needs
  HTTPS access to raw.githubusercontent.com (about 3 MB).
- Re-running is safe: it downloads the pinned files again and rewrites the same output.
- `--out DIR` writes `DIR/grippers/` instead (then export `KINOVA_SIM_ASSETS=DIR`);
  `--ref <sha>` builds from another Menagerie commit.

## Source and license

`robotiq_2f85/` of MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie),
pinned to commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`. The Menagerie repository is
Apache-2.0, but each model directory has its own license: `robotiq_2f85/LICENSE` is
**BSD-2-Clause** ("Copyright (c) 2013, ROS-Industrial"). The script copies it to
`grippers/LICENSE`; keep it next to any copy of the meshes.

## Output

```
sim/assets/
  README.md                          this file
  build_gripper.py                   the generator
  grippers/                          generated, gitignored
    robotiq_2f85_menagerie.xml
    meshes/robotiq_2f85/*.stl        base_mount, base, driver, coupler, follower, pad,
                                     silicone_pad, spring_link
    LICENSE                          BSD-2-Clause, from mujoco_menagerie/robotiq_2f85/
```

`robotiq_2f85_menagerie.xml` is upstream `2f85.xml` adapted for robosuite 1.4.0:

- **Defaults flattened** into every element. robosuite's `MujocoXML.merge` does not copy
  `<default>`, so joint ranges, armature, damping, pad friction and the actuator's
  `biastype` would otherwise be lost.
- **Meshes named** after their files, with `file=` relative to the XML. robosuite 1.4.0
  ignores `<compiler meshdir>` and resolves every `file=` next to the XML it loads.
- **Geom groups** 1 = visual and 0 = collision, robosuite's convention (Menagerie uses 2
  and 3).
- **Explicit `<inertial>`** on the bodies that had none. robosuite's world compiles with
  `inertiagrouprange="0 0"`, which would drop the mass Menagerie derives from its geoms.
- **robosuite frames**, laid out like robosuite's `robotiq_gripper_85.xml`: an `eef` body
  at Menagerie's `pinch` point holding `grip_site`, `grip_site_cylinder` and `ee_x/y/z`,
  and an `ft_frame` site with the `force_ee`/`torque_ee` sensors. The root body gets a
  +90 deg yaw, so on Kinova3 the fingers point along the approach axis and open across the
  wrist camera's view, like robosuite's `Robotiq85Gripper`.
- **Kept as upstream**: every name `kinova_env.py` relies on (8 hinge joints, pad geoms
  `left_pad1/2` and `right_pad1/2`, `fingers_actuator` with ctrlrange `0 255`), the
  tendon, the equality constraints and the contact excludes. Menagerie's `<option>` is
  dropped, since robosuite uses its world's.

## How it is verified

`build_gripper.py` writes nothing unless both of its checks pass:

1. Upstream `2f85.xml` and the flattened model compile (`mujoco.MjModel`) to the same 63
   compiled arrays (bodies, joints, dofs, geoms, meshes, actuator, equality, tendon,
   contact excludes).
2. The wrapped XML, compiled with `inertiagrouprange="0 0"` like robosuite's world, still
   matches except for the root yaw and the geom groups, and `grip_site` sits on `pinch`.

Then check that the env chain finds and builds it:

```bash
cd <repo>/sim && ~/workspace/libero_venv/bin/python -c \
  "import kinova_env; print(kinova_env.menagerie_gripper_xml())"
MUJOCO_GL=egl ~/workspace/libero_venv/bin/python kinova_circle_env.py
```

The second command builds and resets the env. `bash stack.sh check` reports the gripper
too. Recorded datasets keep the full MJCF they were recorded with in each demo's
`model_file` attribute; compare against it when exact replay matters.

## Where the code looks (first hit wins)

1. `$KINOVA_SIM_ASSETS/grippers/robotiq_2f85_menagerie.xml`
2. `<this sim/ dir>/assets/grippers/robotiq_2f85_menagerie.xml`, the default output of
   `build_gripper.py`
3. `<robosuite>/models/assets/grippers/robotiq_2f85_menagerie.xml` (robosuite's
   `xml_path_completion`)

To share one copy between checkouts, run `build_gripper.py --out <dir>` once and export
`KINOVA_SIM_ASSETS=<dir>`. Avoid option 3 when the venv is shared with other projects;
nothing here installs into it.

The `.py`/`.json` files of the chain are pinned by `../env_chain.sha256` (check with
`cd sim && sha256sum -c env_chain.sha256`). Keep this README, `build_gripper.py` and those
files byte-identical in every copy of the chain.
