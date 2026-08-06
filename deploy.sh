#!/usr/bin/env bash
#
# ASTRA Terminal — one-shot VPS setup.
#
# On the VPS:
#     git clone <your-repo> /opt/astra
#     cd /opt/astra
#     sudo ./deploy.sh                 # (or: sudo PORT=8080 ./deploy.sh)
#
# It creates the virtualenv, installs deps, installs a systemd service and starts
# it. Run it again any time to update (git pull && sudo ./deploy.sh).
#
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8000}"
RUN_USER="${SUDO_USER:-$(id -un)}"
PYBIN="${PYBIN:-python3.12}"

echo "==> ASTRA deploy"
echo "    dir=$APP_DIR  port=$PORT  user=$RUN_USER"

# ---- 1) Python + virtualenv --------------------------------------------------
if ! command -v "$PYBIN" >/dev/null 2>&1; then
  echo "==> installing $PYBIN"
  apt-get update -y
  apt-get install -y "$PYBIN" "${PYBIN}-venv"
fi
if [ ! -x "$APP_DIR/.venv/bin/python" ]; then
  echo "==> creating virtualenv"
  "$PYBIN" -m venv "$APP_DIR/.venv"
fi
echo "==> installing dependencies"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip >/dev/null
"$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

# ---- 2) market data check ----------------------------------------------------
if [ ! -d "$APP_DIR/data/minute" ] || [ -z "$(ls -A "$APP_DIR/data/minute" 2>/dev/null || true)" ]; then
  echo ""
  echo "!!  No market data found in data/minute/ — the app will have nothing to score."
  echo "!!  Copy it from the machine that has it, e.g. (run THERE):"
  echo "!!      rsync -avz ./data/ ${RUN_USER}@<this-vps-ip>:${APP_DIR}/data/"
  echo ""
fi

# ---- 3) systemd service ------------------------------------------------------
echo "==> installing systemd service"
sed -e "s#__APP_DIR__#${APP_DIR}#g" \
    -e "s#__RUN_USER__#${RUN_USER}#g" \
    -e "s#__PORT__#${PORT}#g" \
    "$APP_DIR/deploy/scalper.service" > /etc/systemd/system/scalper.service

systemctl daemon-reload
systemctl enable scalper >/dev/null 2>&1 || true
systemctl restart scalper
sleep 2

echo ""
if systemctl is-active --quiet scalper; then
  echo "==> scalper is running on http://<this-vps-ip>:${PORT}"
else
  echo "==> service did not start — check: journalctl -u scalper -n 40 --no-pager"
fi
echo "    live logs:  journalctl -u scalper -f"
echo "    subdomain + SSL:  see deploy/scalper.nginx.conf"
echo "    SECURITY: the app has no login — firewall the port to your IP, or use"
echo "              the nginx password gate in deploy/scalper.nginx.conf."
