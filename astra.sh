#!/bin/bash
# astra.sh — start, stop, or check the ASTRA Terminal web server.
#
#   ./astra.sh start     launch the server on port 8000 (stays running)
#   ./astra.sh stop      shut it down
#   ./astra.sh status    is it running, and is it answering?
#   ./astra.sh restart   stop then start
#   ./astra.sh warm      pre-load the caches so the first page view is instant
#   ./astra.sh logs      tail the server log
#
# The server keeps running after you close the terminal.

cd /root/crypto-quant-lab/scalper || exit 1
PORT=8000
PY=.venv/bin/python
LOG=web/server.log

start() {
  if pgrep -f "[w]eb/app.py" > /dev/null; then
    echo "already running on port $PORT"
    return
  fi
  setsid nohup $PY web/app.py $PORT > "$LOG" 2>&1 < /dev/null &
  disown
  sleep 4
  if pgrep -f "[w]eb/app.py" > /dev/null; then
    echo "ASTRA Terminal is up"
    echo "  open  http://localhost:$PORT"
  else
    echo "failed to start — check $LOG"
    tail -20 "$LOG"
  fi
}

stop() {
  if pgrep -f "[w]eb/app.py" > /dev/null; then
    pkill -f "web/app.py"
    # Wait for it to actually go. Returning early made `restart` report "already
    # running" and skip the start. A hung worker can also survive SIGTERM while
    # no longer listening, which looks identical from the outside — so escalate.
    for _ in $(seq 1 20); do
      pgrep -f "[w]eb/app.py" > /dev/null || break
      sleep 0.3
    done
    if pgrep -f "[w]eb/app.py" > /dev/null; then
      pkill -9 -f "web/app.py"
      sleep 1
    fi
    echo "stopped"
  else
    echo "was not running"
  fi
}

status() {
  if pgrep -f "[w]eb/app.py" > /dev/null; then
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/" || echo 000)
    echo "running (pid $(pgrep -f 'web/app.py' | head -1)), HTTP $code"
  else
    echo "not running"
  fi
}

warm() {
  # The first request for a symbol bridges the gap between the daily archive and
  # live data, which is rate-limited and can take a minute. Doing it here means
  # the browser never waits on it.
  for s in BTCUSDT ETHUSDT; do
    for ep in "chart?symbol=$s&tf=15&count=320" \
              "assess?symbol=$s&horizon=15&cost=none" \
              "signal?symbol=$s&horizon=15" \
              "model-stats?symbol=$s&horizon=15"; do
      printf "  %-52s " "${ep%%\?*} $s"
      curl -s -o /dev/null -w "%{http_code}  %{time_total}s\n" \
        --max-time 600 "http://127.0.0.1:$PORT/api/$ep"
    done
  done
}

case "${1:-start}" in
  start)   start ;;
  stop)    stop ;;
  restart) stop; sleep 1; start ;;
  status)  status ;;
  warm)    warm ;;
  logs)    tail -f "$LOG" ;;
  *)       echo "usage: $0 {start|stop|restart|status|warm|logs}" ;;
esac
