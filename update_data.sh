#!/usr/bin/env bash
#
# Keep the market-data archive fresh. Run daily from cron — it re-checks the last
# few days and downloads any newly-published daily files from Binance, so the
# backtests / scanner / forward test never go stale.
#
#   crontab -e   ->   30 2 * * * /home/USER/scalper/update_data.sh >> /home/USER/scalper/ingest.log 2>&1
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY=./.venv/bin/python
SYMBOLS="${SYMBOLS:-BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT}"
DAYS="${DAYS:-5}"          # re-scan the last N days (fills whatever is missing)

echo "==> $(date -u +%FT%TZ) refreshing archive ($DAYS days)"
for s in $SYMBOLS; do
  "$PY" ingest.py "$s" "$DAYS" 4 || echo "!! $s failed"
done
echo "==> done"
