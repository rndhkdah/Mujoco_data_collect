#!/usr/bin/env bash
usage_text() {
  cat <<'USAGE'
STIR sim data collection: Kinova Gen3 in robosuite/MuJoCo, Quest WebXR teleop, no ROS.

usage: bash stack.sh <command> [flags]

  start [flags]   collection server on :8002 (foreground, Ctrl-C stops it) + adb reverse;
                  flags pass through to sim/collect_circle_server.py
  sim [flags]     same as start
  check           headless smoke test: build the env and reset it 3x (no Quest)
  configs         list the --config presets (sim/env_configs/*.json)
  status          server, HTTP port, adb reverse, gripper asset, output files
  kill            stop this repo's server (same as kill.sh)

Example: bash stack.sh start --config voxel --out ~/kinova_dataset/data/voxel.hdf5

Output goes to $SIM_DATA_DIR/data/circle.hdf5 unless --out is given. An existing file
is resumed, so give each --config its own --out. Type the --out path in full:
$SIM_DATA_DIR is set inside _env.sh, after the shell has expanded the arguments.

The server binds 127.0.0.1; the Quest opens http://localhost:8002 through adb reverse.
USAGE
}
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
usage() {
  usage_text
  exit "${1:-1}"
}
gripper_check() {
  local out rc
  if [ ! -x "$VENV_PY" ]; then
    echo "venv python not found: $VENV_PY (SETUP.md step 2)"
    return 2
  fi
  out=$(cd "$SIM_DIR" && "$VENV_PY" -B -c \
    'import kinova_env; print(kinova_env.menagerie_gripper_xml())' 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "$out" | tail -1
    return 0
  fi
  if echo "$out" | grep -q '^FileNotFoundError: Menagerie'; then
    echo "$out" | sed -n '/^FileNotFoundError: Menagerie/,$p'
    return 1
  fi
  echo "$out" | tail -n 8
  return 2
}
gripper_fail() {
  echo "$2"
  if [ "$3" -eq 1 ]; then
    echo "[$1] FAIL: no gripper asset -> no env. Generate it: $VENV_PY $SIM_DIR/assets/build_gripper.py (see $SIM_DIR/assets/README.md)"
  else
    echo "[$1] FAIL: could not run the gripper lookup under VENV_PY=$VENV_PY (venv or import problem above; see SETUP.md)"
  fi
}
start() {
  local pids xml rc
  pids=$(server_pids)
  if [ -n "$pids" ]; then
    echo "[sim] already running (pid $(echo $pids)). Stop it first: bash stack.sh kill"
    exit 1
  fi
  xml=$(gripper_check)
  rc=$?
  [ $rc -ne 0 ] && {
    gripper_fail sim "$xml" $rc
    exit 1
  }
  echo "[sim] gripper: $xml"
  echo "[sim] adb reverse tcp:$SIM_PORT"
  adb reverse tcp:"$SIM_PORT" tcp:"$SIM_PORT" || echo "  (Quest connected?)"
  echo "[sim] robosuite kinova_circle_env via collect_circle_server.py -> http://localhost:$SIM_PORT"
  echo "[sim] output dir: $SIM_DATA_DIR/data (default file circle.hdf5; --out overrides)"
  exec "$VENV_PY" "$SERVER" --port "$SIM_PORT" "$@"
}
check() {
  local xml rc
  echo "[check] VENV_PY=$VENV_PY  MUJOCO_GL=$MUJOCO_GL"
  xml=$(gripper_check)
  rc=$?
  [ $rc -ne 0 ] && {
    gripper_fail check "$xml" $rc
    exit 1
  }
  echo "[check] gripper: $xml"
  echo "[check] building kinova_circle_env and resetting 3 episode specs..."
  PYTHONUNBUFFERED=1 exec "$VENV_PY" "$SIM_DIR/kinova_circle_env.py"
}
configs() {
  local f py="$VENV_PY"
  [ -x "$py" ] || py=python3
  echo "[configs] $SIM_DIR/env_configs/  (--config <name>, or --config <path.json>)"
  for f in "$SIM_DIR"/env_configs/*.json; do
    printf "  %-12s %s\n" "$(basename "$f" .json)" \
      "$("$py" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("_comment", ""))' "$f" 2>/dev/null)"
  done
}
status() {
  local pids code xml rc
  echo "[status] repo $REPO"
  pids=$(server_pids)
  code=$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://127.0.0.1:$SIM_PORT/" 2>/dev/null)
  if [ -n "$pids" ]; then
    printf "  %-15s UP (pid %s)\n" "server" "$(echo $pids)"
  else printf "  %-15s down\n" "server"; fi
  printf "  %-15s http://127.0.0.1:%s/ -> %s\n" "port" "$SIM_PORT" "${code:-000}"
  if [ -z "$pids" ] && [ -n "$code" ] && [ "$code" != 000 ]; then
    echo "  WARNING: :$SIM_PORT answers but not from this repo's server (another copy running?)"
  fi
  if pgrep -x adb >/dev/null 2>&1; then
    if adb reverse --list 2>/dev/null | grep -q "tcp:$SIM_PORT "; then
      printf "  %-15s tcp:%s set\n" "adb reverse" "$SIM_PORT"
    else
      printf "  %-15s tcp:%s NOT set (bash stack.sh start sets it; Quest plugged in?)\n" "adb reverse" "$SIM_PORT"
    fi
  else
    printf "  %-15s adb server not running (not started by status)\n" "adb reverse"
  fi
  xml=$(gripper_check)
  rc=$?
  case $rc in
    0) printf "  %-15s %s\n" "gripper asset" "$xml" ;;
    1) printf "  %-15s MISSING -> %s\n" "gripper asset" "$SIM_DIR/assets/README.md" ;;
    *) printf "  %-15s UNKNOWN, lookup failed: %s\n" "gripper asset" "$(echo "$xml" | tail -1)" ;;
  esac
  echo "[status] output files in $SIM_DATA_DIR/data (size, mtime):"
  if ls "$SIM_DATA_DIR"/data/*.hdf5 >/dev/null 2>&1; then
    ls -la "$SIM_DATA_DIR"/data/*.hdf5 | awk '{print "         " $5, $6, $7, $8, $9}'
  else
    echo "         (none yet; the server creates the dir)"
  fi
  echo "[status] Quest page: http://localhost:$SIM_PORT"
}
cmd="${1:-}"
[ $# -gt 0 ] && shift
case "$cmd" in
  start | sim) start "$@" ;;
  check) check ;;
  configs) configs ;;
  status) status ;;
  kill) exec bash "$HERE/kill.sh" ;;
  -h | --help | help | "") usage 0 ;;
  *)
    echo "unknown command: $cmd"
    usage 1
    ;;
esac
