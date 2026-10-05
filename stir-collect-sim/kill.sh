#!/usr/bin/env bash
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/_env.sh"
pids=$(server_pids)
if [ -z "$pids" ]; then
  echo "[kill] collect-sim: nothing running."
  exit 0
fi
echo "[kill] SIGTERM $(echo $pids)"
kill -TERM $pids 2>/dev/null
for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 0.2
  [ -z "$(server_pids)" ] && break
done
left=$(server_pids)
[ -n "$left" ] && {
  echo "[kill] SIGKILL $(echo $left)"
  kill -9 $left 2>/dev/null
  sleep 1
}
echo "[kill] collect-sim stopped."
