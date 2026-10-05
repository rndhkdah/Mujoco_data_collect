# Setup

No ROS. Three directories sit side by side:

```
~/workspace/
├── stir-collect-sim/   this repo (collection server + MuJoCo env chain)
├── libero_venv/        Python 3.12 venv the server runs in
└── LIBERO/             LIBERO source checkout, installed editable into libero_venv
```

`_env.sh` finds the venv as a sibling of the repo (`VENV_PY`). If yours lives elsewhere,
export `VENV_PY` before running `stack.sh`.

## 1. This repo

```bash
git clone <repo-url> ~/workspace/stir-collect-sim
```

## 2. Python venv

```bash
python3.12 -m venv ~/workspace/libero_venv
~/workspace/libero_venv/bin/pip install -r ~/workspace/stir-collect-sim/requirements.txt
```

The venv can be shared with other projects; anything installed into it is seen by every
project that uses it. Keep `robosuite==1.4.0`: robosuite 1.5 removed
`load_controller_config`, which the env chain imports.

## 3. LIBERO (object assets)

Upstream LIBERO has no `libero/__init__.py`, so
`pip install git+https://github.com/Lifelong-Robot-Learning/LIBERO.git` installs nothing
importable. Install a local checkout (revision `8f1084e`) in editable mode, with an empty
`libero/__init__.py` added:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/workspace/LIBERO
git -C ~/workspace/LIBERO checkout 8f1084e
touch ~/workspace/LIBERO/libero/__init__.py
~/workspace/libero_venv/bin/pip install -e ~/workspace/LIBERO
```

LIBERO's `setup.py` declares no dependencies. Do **not** install LIBERO's own
`requirements.txt`: it pins old versions (numpy 1.22, a torch/robomimic training stack)
that this venv does not use. What LIBERO's env package imports is already in this repo's
`requirements.txt`.

Create LIBERO's config **before** the first import. Without it, `import libero.libero`
stops on an interactive `input()` prompt, which crashes with EOFError when there is no tty.
The file is `~/.libero/config.yaml` (or `$LIBERO_CONFIG_PATH/config.yaml`):

```bash
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
assets: $HOME/workspace/LIBERO/libero/libero/./assets
bddl_files: $HOME/workspace/LIBERO/libero/libero/./bddl_files
benchmark_root: $HOME/workspace/LIBERO/libero/libero
datasets: $HOME/workspace/LIBERO/libero/libero/../datasets
init_states: $HOME/workspace/LIBERO/libero/libero/./init_files
EOF
```

Because the install is editable, moving or deleting `~/workspace/LIBERO` breaks the venv.

## 4. The Menagerie Robotiq 2F-85 gripper (generate it)

```bash
~/workspace/libero_venv/bin/python ~/workspace/stir-collect-sim/sim/assets/build_gripper.py
```

Every env is built with the `MenagerieRobotiq2f85` gripper, which loads
`grippers/robotiq_2f85_menagerie.xml` and its meshes, and robosuite does not ship them.
The script downloads MuJoCo Menagerie's `robotiq_2f85` at a pinned commit (network
needed), checks the conversion, and writes the XML, the meshes and the BSD-2-Clause
`LICENSE` to `sim/assets/grippers/` (gitignored). [sim/assets/README.md](sim/assets/README.md)
has the details: the pinned commit, what the conversion changes, how it is verified, and
the lookup order.

To keep one copy for several checkouts, run it with `--out <dir>` and export
`KINOVA_SIM_ASSETS=<dir>`. Avoid copying it into a shared venv's robosuite tree.

`bash stack.sh status` shows where it resolves (the `gripper asset` line). The gripper
files are not covered by `sim/env_chain.sha256`, so `cd sim && sha256sum -c
env_chain.sha256` still passes after they are generated.

## 5. Headless rendering (EGL)

The server renders the agent and wrist cameras offscreen:

- `MUJOCO_GL=egl` (`_env.sh` exports it, and the server sets it too).
- `PYOPENGL_PLATFORM` must be unset or `egl`. robosuite's EGL context rejects any other value.
- Choose the GPU with `MUJOCO_EGL_DEVICE_ID` or `CUDA_VISIBLE_DEVICES`.
- On Jetson Thor this uses the NVIDIA Tegra libEGL. GPU memory is unified and shared with
  any JAX policy server running on the same machine. If EGL context creation fails, stop
  the policy server or lower its `XLA_PYTHON_CLIENT_MEM_FRACTION`.

## 6. Meta Quest

Install adb (`sudo apt install adb`), enable Developer Mode on the Quest, connect it over
USB and accept the debugging prompt. `adb devices` should then list the headset.

The server binds `127.0.0.1` only. `bash stack.sh start` runs
`adb reverse tcp:8002 tcp:8002` so the headset browser can open `http://localhost:8002`.
With more than one adb device attached, set `ANDROID_SERIAL`. `adb reverse` rules are per
port, so rules for other ports coexist with this one. Nothing in this repo removes a rule.

## 7. Run it

```bash
cd ~/workspace/stir-collect-sim
bash stack.sh check
bash stack.sh start
```

`check` builds the env and resets it 3 times without a Quest. After `start`, open
`http://localhost:8002` on the Quest and press **Enter VR**.

See [README.md](README.md) for the controls, the output contract and the env vars, and
[ARCHITECTURE.md](ARCHITECTURE.md) for how the server works.
