"""
decisions.py — a LIVE, honest scorecard of the AI read's calls.

Unlike a backtest (which re-scores history), this records each decision the
strategy makes ANCHORED to the bar it was made on, then watches the real price
that came after and freezes the outcome:

  - price reached the TARGET before the stop  -> win
  - price hit the STOP first                   -> loss
  - the timeframe horizon passed, neither hit  -> expired (judged by direction:
      price ended on the profitable side = a directional win, else a loss)

Because the decision at bar i only uses data up to bar i (via
strategy_pro.decision_at), replaying past bars and scoring them on LATER price
is genuinely out-of-sample — there's no look-ahead. New decisions accrue forward
from a background monitor, so the scorecard is honest and always fresh.

Store: data/decisions.db (SQLite, stdlib — no new dependency).
"""

from __future__ import annotations

import os
import sqlite3
import threading
import time

import numpy as np

import strategy_pro as SP

# what we track
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
           "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]
TFS = [5, 15, 60, 240, 1440]

# how many bars a decision is given to reach its target before we call it
# "expired" and judge it by direction
HORIZON_BARS = {5: 12, 15: 12, 60: 12, 240: 8, 1440: 5}
# how far back to seed on first run (so the page isn't empty on day one)
SEED_DAYS = {5: 3, 15: 7, 60: 15, 240: 45, 1440: 90}
# how much minute history to load to GENERATE a decision on each tf (the
# 100/200-period studies need it); also covers the seed window + horizon
GEN_DAYS = {5: 6, 15: 12, 60: 30, 240: 75, 1440: 330}

_LOCK = threading.Lock()
_DB_PATH = None


def _db_path():
    global _DB_PATH
    if _DB_PATH is None:
        import ingest
        d = os.path.join(ingest.ROOT, "data")
        os.makedirs(d, exist_ok=True)
        _DB_PATH = os.path.join(d, "decisions.db")
    return _DB_PATH


def _conn():
    c = sqlite3.connect(_db_path(), timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            tf INTEGER NOT NULL,
            bias TEXT NOT NULL,
            entry REAL NOT NULL,
            stop REAL NOT NULL,
            target REAL NOT NULL,
            risk REAL NOT NULL,
            rr REAL NOT NULL,
            passed INTEGER,
            total_checks INTEGER,
            opened_at INTEGER NOT NULL,
            horizon_ts INTEGER NOT NULL,
            recorded_at INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            outcome_r REAL,
            exit_price REAL,
            resolved_at INTEGER,
            UNIQUE(symbol, tf, opened_at)
        )
    """)
    return c


# ---------------------------------------------------------------- recording ---

def _row_from_decision(symbol, d):
    """Normalise a strategy_pro.decision_at() dict into a stored row."""
    entry = float(d["entry"]); stop = float(d["stop"]); target = float(d["target"])
    risk = abs(entry - stop)
    if risk <= 0:
        return None
    rr = abs(target - entry) / risk
    tf = int(d["tf"])
    opened_at = int(d["as_of"])
    horizon_ts = opened_at + HORIZON_BARS.get(tf, 8) * tf * 60
    return {
        "symbol": symbol, "tf": tf, "bias": d["bias"],
        "entry": entry, "stop": stop, "target": target, "risk": risk, "rr": rr,
        "passed": int(d.get("passed", 0)), "total_checks": int(d.get("total_checks", 6)),
        "opened_at": opened_at, "horizon_ts": horizon_ts,
    }


def _insert(conn, row):
    try:
        conn.execute("""
            INSERT OR IGNORE INTO decisions
              (symbol, tf, bias, entry, stop, target, risk, rr, passed,
               total_checks, opened_at, horizon_ts, recorded_at)
            VALUES (:symbol,:tf,:bias,:entry,:stop,:target,:risk,:rr,:passed,
                    :total_checks,:opened_at,:horizon_ts,:recorded_at)
        """, {**row, "recorded_at": int(time.time())})
        return conn.total_changes
    except sqlite3.Error:
        return 0


def record(symbol, decision):
    """Record a single live decision (from strategy_pro.current). Deduped by the
    bar it was anchored to, so calling it often is safe."""
    row = _row_from_decision(symbol, decision)
    if not row:
        return 0
    with _LOCK, _conn() as conn:
        before = conn.total_changes
        _insert(conn, row)
        conn.commit()
        return conn.total_changes - before


# --------------------------------------------------------------- evaluation ---

def _resolve_one(bias, entry, stop, target, risk, opened_at, horizon_ts,
                 ts, hi, lo, cl):
    """Walk the minute path strictly AFTER opened_at (up to horizon) and decide
    the outcome. Returns (status, outcome_r, exit_price, resolved_at) or None if
    it can't be resolved yet (not enough data past the bar)."""
    lo_i = int(np.searchsorted(ts, opened_at, side="right"))
    hi_i = int(np.searchsorted(ts, horizon_ts, side="right"))
    if lo_i >= len(ts):
        return None                       # no data after the decision yet
    end = min(hi_i, len(ts))
    seg_hi = hi[lo_i:end]; seg_lo = lo[lo_i:end]
    seg_ts = ts[lo_i:end]; seg_cl = cl[lo_i:end]
    if len(seg_hi) == 0:
        return None
    long = bias == "long"
    if long:
        stop_hit = seg_lo <= stop
        tp_hit = seg_hi >= target
    else:
        stop_hit = seg_hi >= stop
        tp_hit = seg_lo <= target
    i_stop = int(np.argmax(stop_hit)) if stop_hit.any() else None
    i_tp = int(np.argmax(tp_hit)) if tp_hit.any() else None

    # if both touched, the STOP is assumed first within a bar (conservative —
    # never overstates the win rate)
    if i_stop is not None and (i_tp is None or i_stop <= i_tp):
        return ("loss", -1.0, float(stop), int(seg_ts[i_stop]))
    if i_tp is not None:
        return ("win", round(abs(target - entry) / risk, 3), float(target), int(seg_ts[i_tp]))

    # neither hit — only "expired" once the horizon has actually passed
    now = int(time.time())
    if now < horizon_ts and end >= len(ts):
        return None                       # still open, horizon not reached yet
    exit_px = float(seg_cl[-1])
    signed = (exit_px - entry) if long else (entry - exit_px)
    r = round(signed / risk, 3)
    status = "expired_win" if r > 0 else "expired_loss"
    return (status, r, exit_px, int(seg_ts[-1]))


def _evaluate_symbol(conn, symbol, minute_df):
    """Resolve every still-open decision for this symbol using the minute path."""
    rows = conn.execute(
        "SELECT * FROM decisions WHERE symbol=? AND status='open'", (symbol,)
    ).fetchall()
    if not rows:
        return 0
    # to epoch SECONDS, independent of the index resolution (ns vs us)
    ts = np.asarray(minute_df.index.astype("datetime64[s]").astype("int64"), dtype="int64")
    hi = minute_df["high"].to_numpy(dtype=float)
    lo = minute_df["low"].to_numpy(dtype=float)
    cl = minute_df["close"].to_numpy(dtype=float)
    resolved = 0
    for r in rows:
        res = _resolve_one(r["bias"], r["entry"], r["stop"], r["target"], r["risk"],
                           r["opened_at"], r["horizon_ts"], ts, hi, lo, cl)
        if res is None:
            continue
        status, outcome_r, exit_px, resolved_at = res
        conn.execute(
            "UPDATE decisions SET status=?, outcome_r=?, exit_price=?, resolved_at=? WHERE id=?",
            (status, outcome_r, exit_px, resolved_at, r["id"]))
        resolved += 1
    return resolved


# ------------------------------------------------------------ tick / seed  ---

def _bar_indices_for_seed(n, tf):
    """Non-overlapping historical bar indices to seed, newest-friendly. Steps by
    the horizon so seeded decisions don't overlap each other."""
    step = HORIZON_BARS.get(tf, 8)
    approx = int(SEED_DAYS.get(tf, 30) * 1440 / tf)
    start = max(210, n - approx)          # keep 200+ warmup bars for the studies
    return list(range(start, n - 1, step))


def tick_symbol(symbol, frame_fn, seed=False):
    """Generate + record decisions for one symbol across all timeframes, then
    resolve its open decisions against real price. `frame_fn(symbol, days)`
    returns a minute-indexed OHLC DataFrame."""
    added = resolved = 0
    with _LOCK, _conn() as conn:
        widest = None
        for tf in TFS:
            try:
                m = frame_fn(symbol, GEN_DAYS.get(tf, 60))
                if m is None or len(m) < 210:
                    continue
                c, ind, L, S = SP.prepare(m, tf)
                if len(c) < 210:
                    continue
                idxs = _bar_indices_for_seed(len(c), tf) if seed else [len(c) - 2]
                for i in idxs:
                    d = SP.decision_at(c, ind, L, S, i, tf)
                    row = _row_from_decision(symbol, d)
                    if row:
                        b = conn.total_changes
                        _insert(conn, row)
                        added += conn.total_changes - b
                # keep the widest minute frame we loaded for evaluation
                if widest is None or len(m) > len(widest):
                    widest = m
            except Exception:
                continue
        if widest is not None:
            try:
                resolved = _evaluate_symbol(conn, symbol, widest)
            except Exception:
                resolved = 0
        conn.commit()
    return {"symbol": symbol, "added": added, "resolved": resolved}


def tick_all(frame_fn, seed=False, symbols=None):
    return [tick_symbol(s, frame_fn, seed=seed) for s in (symbols or SYMBOLS)]


def _default_frame_fn(symbol, days):
    import live_data as LD
    return LD.load_recent_archive(symbol, days)


def ensure_seeded(frame_fn=None):
    """Seed once if the DB is empty, so the scorecard is populated immediately."""
    with _LOCK, _conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
    if n == 0:
        tick_all(frame_fn or _default_frame_fn, seed=True)
        return True
    return False


# --------------------------------------------------------------- read/agg  ---

def _agg(rows):
    resolved = [r for r in rows if r["status"] != "open"]
    n = len(resolved)
    if n == 0:
        return {"decisions": len(rows), "open": len(rows), "resolved": 0,
                "win_rate": 0.0, "target_hit_rate": 0.0, "net_r": 0.0,
                "avg_r": 0.0, "wins": 0, "losses": 0, "expired": 0}
    wins = [r for r in resolved if (r["outcome_r"] or 0) > 0]
    target_hits = [r for r in resolved if r["status"] == "win"]
    losses = [r for r in resolved if r["status"] in ("loss", "expired_loss")]
    expired = [r for r in resolved if r["status"] in ("expired_win", "expired_loss")]
    net = sum((r["outcome_r"] or 0) for r in resolved)
    return {
        "decisions": len(rows),
        "open": len(rows) - n,
        "resolved": n,
        "wins": len(wins), "losses": len(losses), "expired": len(expired),
        "win_rate": round(len(wins) / n * 100, 1),                 # profitable %
        "target_hit_rate": round(len(target_hits) / n * 100, 1),   # reached TP %
        "net_r": round(net, 2),
        "avg_r": round(net / n, 3),
    }


def read(symbols=None, tfs=None):
    """The whole scorecard: headline, per-timeframe, per-symbol, recent list."""
    symbols = symbols or SYMBOLS
    tfs = tfs or TFS
    with _LOCK, _conn() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM decisions").fetchall()]
    rows = [r for r in rows if r["symbol"] in symbols and r["tf"] in tfs]

    by_tf = []
    for tf in tfs:
        a = _agg([r for r in rows if r["tf"] == tf])
        by_tf.append({"tf": tf, **a})
    by_symbol = []
    for s in symbols:
        a = _agg([r for r in rows if r["symbol"] == s])
        by_symbol.append({"symbol": s, **a})

    # the marketable story: does the tool do better when it's more confident?
    by_conviction = []
    for lo, hi, label in [(5, 6, "high (5-6/6)"), (4, 4, "solid (4/6)"),
                          (0, 3, "lower (≤3/6)")]:
        a = _agg([r for r in rows if lo <= (r["passed"] or 0) <= hi])
        by_conviction.append({"band": label, "min": lo, "max": hi, **a})

    resolved = [r for r in rows if r["status"] != "open"]
    resolved.sort(key=lambda r: r["resolved_at"] or 0, reverse=True)
    recent = [{
        "symbol": r["symbol"], "tf": r["tf"], "bias": r["bias"],
        "entry": r["entry"], "stop": r["stop"], "target": r["target"],
        "status": r["status"], "outcome_r": r["outcome_r"],
        "opened_at": r["opened_at"], "resolved_at": r["resolved_at"],
        "passed": r["passed"], "total_checks": r["total_checks"],
    } for r in resolved[:60]]

    return {
        "overall": _agg(rows),
        "by_tf": by_tf,
        "by_symbol": by_symbol,
        "by_conviction": by_conviction,
        "recent": recent,
        "generated_at": int(time.time()),
    }


if __name__ == "__main__":
    print("seeding…", ensure_seeded())
    import json
    print(json.dumps(read()["overall"], indent=2))
