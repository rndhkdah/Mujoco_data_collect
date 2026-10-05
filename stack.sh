#!/usr/bin/env bash
usage() {
  cat <<'USAGE'
Furniture assembly data collection: Kinova Gen3 in robosuite/MuJoCo, Quest WebXR teleop.

usage: bash stack.sh <command> [flags]

  setup            create the venv (../fsim_venv), install requirements, build the gripper and meshes
  check            run the tests (incl. a recorded scripted-expert episode) and the reach check; no Quest needed
  start [flags]    adb reverse + collection server on :8004 (Ctrl-C stops it); flags go to fsim.server,
                   e.g. --task one_leg --randomness low --num-demos 50 --operator alice
                   (--view panel: camera panel instead of the 3D view)
  status           server, port, adb reverse, assets, episode counts
  kill             stop the server
  expert [flags]   scripted one_leg expert, no Quest needed: success rate, or demos with --record DIR,
                   e.g. --randomness med --episodes 20
  merge [flags]    merge episodes into one robomimic file: bash stack.sh merge -o one_leg.hdf5 [--success-only]
  replay <file>    replay a recorded episode from its states and actions

Episodes go to $FSIM_DATA_DIR/<task>/<randomness>/*.hdf5 (default ~/fsim_dataset).
The Quest opens http://localhost:8004 through adb reverse.
USAGE
  exit "${1:-1}"
}
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"

need_venv() { [ -x "$VENV_PY" ] || { echo "venv python not found: $VENV_PY (run: bash stack.sh setup)"; exit 1; }; }
need_assets() {
  "$VENV_PY" -c "import fsim.env.gripper as g; print(g.menagerie_gripper_xml())" >/dev/null 2>&1 ||
    { echo "gripper asset missing: $VENV_PY fsim/assets/build_gripper.py --out fsim/assets"; exit 1; }
}

setup() {
  [ -x "$VENV_PY" ] || python3 -m venv "$VENV_DIR"
  "$VENV_PY" -m pip install -q --upgrade pip
  "$VENV_PY" -m pip install -q -r "$REPO/requirements.txt" termcolor pytest
  "$VENV_PY" "$REPO/fsim/assets/build_gripper.py" --out "$REPO/fsim/assets"
  "$VENV_PY" -m fsim.models.build
  echo "[setup] done: $VENV_PY"
}

check() {
  need_venv; need_assets
  (cd "$REPO" && "$VENV_PY" -m pytest -q -p no:cacheprovider) || exit 1
  (cd "$REPO" && "$VENV_PY" tools/reach_check.py | tail -3)
}

start() {
  need_venv; need_assets
  if [ -n "$(server_pids)" ]; then echo "[start] already running: bash stack.sh kill"; exit 1; fi
  adb reverse tcp:"$FSIM_PORT" tcp:"$FSIM_PORT" || echo "  (adb reverse failed: Quest connected and authorized?)"
  echo "[start] http://localhost:$FSIM_PORT  data -> $FSIM_DATA_DIR"
  cd "$REPO" && exec "$VENV_PY" -m fsim.server --port "$FSIM_PORT" "$@"
}

status() {
  local pids code
  pids=$(server_pids)
  code=$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://127.0.0.1:$FSIM_PORT/" 2>/dev/null)
  printf "  %-14s %s\n" "server" "${pids:+UP (pid $(echo $pids))}${pids:-down}"
  printf "  %-14s http://127.0.0.1:%s/ -> %s\n" "port" "$FSIM_PORT" "${code:-000}"
  if adb reverse --list 2>/dev/null | grep -q "tcp:$FSIM_PORT "; then
    printf "  %-14s tcp:%s set\n" "adb reverse" "$FSIM_PORT"
  else
    printf "  %-14s tcp:%s not set\n" "adb reverse" "$FSIM_PORT"
  fi
  need_venv
  printf "  %-14s %s\n" "gripper" "$("$VENV_PY" -c 'import fsim.env.gripper as g; print(g.menagerie_gripper_xml())' 2>/dev/null || echo MISSING)"
  if [ -d "$FSIM_DATA_DIR" ]; then
    find "$FSIM_DATA_DIR" -name '*.hdf5' 2>/dev/null | awk -F/ '{print $(NF-2)"/"$(NF-1)}' | sort | uniq -c |
      awk '{printf "  %-14s %s episodes\n", $2, $1}'
  fi
}

cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  setup) setup ;;
  check) check ;;
  start) start "$@" ;;
  status) status ;;
  kill) exec bash "$HERE/kill.sh" ;;
  expert) need_venv; need_assets; cd "$REPO" && exec "$VENV_PY" -m fsim.env.expert "$@" ;;
  merge) need_venv; cd "$REPO" && exec "$VENV_PY" -m fsim.data.merge "$FSIM_DATA_DIR" "$@" ;;
  replay) need_venv; cd "$REPO" && "$VENV_PY" -m fsim.data.replay "$@" --mode states && "$VENV_PY" -m fsim.data.replay "$@" --mode actions ;;
  -h | --help | help | "") usage 0 ;;
  *) echo "unknown command: $cmd"; usage 1 ;;
esac
