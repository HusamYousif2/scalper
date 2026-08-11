"""
forward.py — a real forward test (paper record), not a backtest.

A backtest re-scores history; this freezes trades as they actually settle going
forward. On the first tick we stamp `start_ts = now`; from then on, every strategy
trade whose entry is after that stamp and whose exit has settled (a later candle
exists) is appended once to an append-only log and never rewritten. The log only
grows with genuinely out-of-sample, forward-observed trades.

Store: data/forward/{symbol}_{tf}m.jsonl  (+ .state.json for the start stamp)
"""

from __future__ import annotations

import json
import os
import time

import strategy_pro as SP

TRACKED = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
           "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]
LOOKBACK_DAYS = 60      # window each tick re-scans (faster than 120)
SEED_DAYS = 55          # on first tick, seed the record with this much recent history


def _dir():
    import ingest
    d = os.path.join(ingest.ROOT, "data", "forward")
    os.makedirs(d, exist_ok=True)
    return d


def _paths(symbol, tf):
    d = _dir()
    return (os.path.join(d, f"{symbol}_{tf}m.jsonl"),
            os.path.join(d, f"{symbol}_{tf}m.state.json"))


def _read_log(lp):
    out = []
    try:
        with open(lp) as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out


def tick(symbol, tf=240):
    """Advance the forward log: append any newly-settled forward trades. Safe to
    call as often as you like — it dedups by entry time."""
    lp, sp = _paths(symbol, tf)
    try:
        state = json.load(open(sp))
    except Exception:
        state = {}
    now = int(time.time())
    try:
        rep = SP.backtest(symbol, tf, LOOKBACK_DAYS)
    except Exception as e:
        return {"symbol": symbol, "tf": tf, "error": f"{type(e).__name__}: {e}"}

    latest = rep["to"]                       # newest exit time we have data for
    logged = {t["entry_time"] for t in _read_log(lp)}
    # seed from recent history so it's useful immediately (also re-seeds a still-
    # empty log, e.g. a fresh deploy); then it grows forward from here
    if "start_ts" not in state or not logged:
        state["start_ts"] = (latest or now) - SEED_DAYS * 86400
    start_ts = state["start_ts"]
    added = 0
    with open(lp, "a") as f:
        for t in rep["trades_list"]:
            if t["entry_time"] < start_ts:          # only trades observed after we started
                continue
            if t["entry_time"] in logged:           # already frozen
                continue
            if t["exit_time"] >= latest:            # not settled yet (still the newest bar)
                continue
            rec = dict(t)
            rec["recorded_at"] = now
            f.write(json.dumps(rec) + "\n")
            logged.add(t["entry_time"])
            added += 1

    state["last_tick"] = now
    json.dump(state, open(sp, "w"))
    return {"symbol": symbol, "tf": tf, "start_ts": start_ts, "added": added}


def _metrics(trades):
    n = len(trades)
    if not n:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_r": 0.0, "profit_factor": None, "max_drawdown_r": 0.0,
                "avg_r": 0.0, "equity": []}
    wins = [t for t in trades if t["outcome"] == "win"]
    gw = sum(t["r"] for t in trades if t["r"] > 0)
    gl = -sum(t["r"] for t in trades if t["r"] < 0)
    cum = peak = mdd = 0.0
    eq = []
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        cum += t["r"]
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
        eq.append({"time": t["exit_time"], "r": round(cum, 3)})
    total = sum(t["r"] for t in trades)
    return {
        "trades": n, "wins": len(wins), "losses": n - len(wins),
        "win_rate": round(len(wins) / n * 100, 1),
        "total_r": round(total, 2), "avg_r": round(total / n, 3),
        "profit_factor": round(gw / gl, 2) if gl > 0 else None,
        "max_drawdown_r": round(mdd, 2), "equity": eq,
    }


def read(tf=240, symbols=None):
    """The merged forward record for the tracked symbols, plus per-symbol rows and
    each symbol's current open lean."""
    symbols = symbols or TRACKED
    all_trades, per, start = [], [], None
    for s in symbols:
        lp, sp = _paths(s, tf)
        try:
            st = json.load(open(sp)).get("start_ts")
        except Exception:
            st = None
        if st and (start is None or st < start):
            start = st
        ts = _read_log(lp)
        for t in ts:
            t["symbol"] = s
        all_trades.extend(ts)
        per.append({"symbol": s, **_metrics(ts)})

    all_trades.sort(key=lambda t: t["exit_time"])
    return {
        "tf": tf, "start_ts": start, "symbols": symbols,
        "metrics": _metrics(all_trades),
        "per_symbol": per,
        "trades_list": sorted(all_trades, key=lambda t: t["entry_time"], reverse=True),
    }


def tick_all(tf=240, symbols=None):
    return [tick(s, tf) for s in (symbols or TRACKED)]
