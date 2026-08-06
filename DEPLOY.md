# Deploying scalper to a VPS

The app is a FastAPI server (`web/app.py`) that listens on a port (default **8000**).
It runs alongside any existing website with **zero conflict** — it just uses a
different port.

## Requirements

- Linux VPS (Ubuntu 22/24), **Python 3.12**
- ~1.5 GB disk (code + models + market data), 1–2 GB RAM
- Outbound HTTPS to Binance for live data *(archive-based pages work without it)*

## What is NOT in git

`data/` (the market archive, ~350 MB+) is git-ignored — too big. Ship it separately
after cloning:

```bash
# run on the machine that already has data/
rsync -avz ./data/ USER@VPS_IP:/opt/scalper/data/
```

Without it, the pages have nothing to score.

## Deploy

```bash
# on the VPS
git clone <your-repo-url> /opt/scalper
cd /opt/scalper
# copy the data/ folder over (see above), then:
sudo ./deploy.sh                 # creates venv, installs deps, installs+starts systemd service
```

Updating later:

```bash
cd /opt/scalper && git pull && sudo ./deploy.sh
```

App is now at `http://VPS_IP:8000`. Manage it with:

```bash
systemctl status scalper
journalctl -u scalper -f
```

## Address options (existing site stays untouched)

- **Simplest** — open the port and use the IP: `sudo ufw allow 8000/tcp` → `http://VPS_IP:8000`
- **Subdomain + SSL** — use `deploy/scalper.nginx.conf` (a separate nginx server block):
  ```bash
  sudo cp deploy/scalper.nginx.conf /etc/nginx/sites-available/scalper
  sudo sed -i 's/__DOMAIN__/scalper.yourdomain.com/' /etc/nginx/sites-available/scalper
  sudo ln -s /etc/nginx/sites-available/scalper /etc/nginx/sites-enabled/
  sudo nginx -t && sudo systemctl reload nginx
  sudo certbot --nginx -d scalper.yourdomain.com
  ```
  With nginx, don't expose 8000 publicly — firewall it and let nginx proxy `127.0.0.1:8000`.

## ⚠️ Security

The app has **no login** — anyone with the address sees your signals. Pick one:

- Firewall to your own IP: `sudo ufw allow from YOUR_IP to any port 8000`
- nginx basic-auth (see the commented block in `deploy/scalper.nginx.conf`)
- Put it behind a VPN
