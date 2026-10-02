#!/usr/bin/env bash
# Sourced by stack.sh and kill.sh: paths, port, data dir, rendering backend.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
VENV_DIR=${VENV_DIR:-"$(dirname "$REPO")/fsim_venv"}
VENV_PY=${VENV_PY:-"$VENV_DIR/bin/python"}
FSIM_PORT=${FSIM_PORT:-8003}
export FSIM_DATA_DIR=${FSIM_DATA_DIR:-$HOME/fsim_dataset}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

server_pids() {
  # processes running the fsim.server module on our port
  local pid
  for pid in $(pgrep -f "fsim\.server" 2>/dev/null); do
    tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null | grep -q -- "--port $FSIM_PORT" && echo "$pid"
  done
}
