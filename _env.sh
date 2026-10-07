#!/usr/bin/env bash
# Sourced by stack.sh and kill.sh: paths, port, data dir, rendering backend.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
_ROOT="$(dirname "$(dirname "$REPO")")"   # data-collection/ (repos live in code/, venvs in envs/)
VENV_DIR=${VENV_DIR:-"$_ROOT/envs/fsim_venv"}
VENV_PY=${VENV_PY:-"$VENV_DIR/bin/python"}
export PATH="$_ROOT/tools/platform-tools:$PATH"   # adb
FSIM_PORT=${FSIM_PORT:-8006}   # 8004 collection, 8005 policy-eval
export FSIM_DATA_DIR=${FSIM_DATA_DIR:-$HOME/fsim_dataset_tasks}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

server_pids() {
  # processes running the fsim.server module on our port
  local pid
  for pid in $(pgrep -f "fsim\.server" 2>/dev/null); do
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--port $FSIM_PORT" && echo "$pid"
  done
}
