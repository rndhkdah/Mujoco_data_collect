#!/usr/bin/env bash
# Stop this repo's collection server (SIGTERM, then SIGKILL after 2 s). Episodes are written
# atomically, so a kill can only lose the episode in progress.
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
pids=$(server_pids)
if [ -z "$pids" ]; then
  echo "[kill] no server on port $FSIM_PORT"
  exit 0
fi
echo "[kill] SIGTERM $(echo $pids)"
kill $pids 2>/dev/null
sleep 2
left=$(server_pids)
[ -n "$left" ] && { echo "[kill] SIGKILL $(echo $left)"; kill -9 $left 2>/dev/null; }
exit 0
