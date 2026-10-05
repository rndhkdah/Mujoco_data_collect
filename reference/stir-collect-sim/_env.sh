#!/usr/bin/env bash
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SIM_DIR="$REPO/sim"
VENV_PY=${VENV_PY:-"$(dirname "$REPO")/libero_venv/bin/python"}
SIM_PORT=${SIM_PORT:-8002}
export SIM_DATA_DIR=${SIM_DATA_DIR:-$HOME/kinova_dataset}
export MUJOCO_GL=${MUJOCO_GL:-egl}
SERVER="$SIM_DIR/collect_circle_server.py"
server_pids() {
  local pid path argv i
  for pid in $(pgrep -f 'collect_circle_server\.py' 2>/dev/null); do
    mapfile -d '' -t argv 2>/dev/null <"/proc/$pid/cmdline" || continue
    case "${argv[0]##*/}" in python*) ;; *) continue ;; esac
    path=""
    i=1
    while [ "$i" -lt "${#argv[@]}" ]; do
      case "${argv[i]}" in
        --)
          path="${argv[i + 1]}"
          break
          ;;
        -W | -X | --check-hash-based-pycs) i=$((i + 2)) ;;
        -W* | -X* | --*) i=$((i + 1)) ;;
        - | -*[cm]*) break ;;
        -*) i=$((i + 1)) ;;
        *)
          path="${argv[i]}"
          break
          ;;
      esac
    done
    case "$path" in *collect_circle_server.py) ;; *) continue ;; esac
    case "$path" in /*) ;; *) path="$(readlink "/proc/$pid/cwd" 2>/dev/null)/$path" ;; esac
    [ "$(readlink -f "$path" 2>/dev/null)" = "$SERVER" ] && echo "$pid"
  done
}
