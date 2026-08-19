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

import json as _json
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
# this is a LIVE test, not a backtest: we keep a rolling 5-day window and start
# measuring from now. We seed the last 5 days once (genuine out-of-sample) so the
# page isn't empty, then it grows forward.
WINDOW_DAYS = 5
SEED_DAYS = {5: 5, 15: 5, 60: 5, 240: 5, 1440: 5}
# how much minute history to load to GENERATE a decision on each tf (the
# 100/200-period studies need it); also covers the seed window + horizon
GEN_DAYS = {5: 6, 15: 12, 60: 30, 240: 75, 1440: 330}

_LOCK = threading.Lock()
_DB_PATH = None
_WEIGHTS_PATH = None
_WEIGHTS_CACHE = {"ts": 0, "data": None}

# the 5 directional indicators the bias is a weighted vote over (volatility is a
# non-directional confirmation, kept out of the vote)
DIRECTIONAL = ["T3 trend", "Range Filter", "EMA200 macro",
               "ADX / DMI", "Momentum (Squeeze)"]
MIN_SAMPLES = 40        # need this many resolved votes before we trust a weight
W_FLOOR, W_CAP = 0.05, 1.3


def _weights_path():
    global _WEIGHTS_PATH
    if _WEIGHTS_PATH is None:
        import ingest
        _WEIGHTS_PATH = os.path.join(ingest.ROOT, "data", "decision_weights.json")
    return _WEIGHTS_PATH


def _acc_to_weight(acc_pct):
    # edge over a coin-flip, clamped; 50% -> ~0, 65% -> 0.3, floored so nothing dies
    return round(min(W_CAP, max(W_FLOOR, (acc_pct / 100.0 - 0.5) * 2.0)), 4)


def compute_weights():
    """Recompute per-timeframe indicator weights from the rolling window and
    persist them. Weights measure each indicator's INTRINSIC hit rate vs real
    price, so they're stable no matter how the votes were combined."""
    glob = indicator_accuracy(None)
    out = {"global": {n: _acc_to_weight(v["acc"]) for n, v in glob.items()},
           "acc": {"global": glob}, "updated": int(time.time())}
    for tf in TFS:
        a = indicator_accuracy(tf)
        w = {}
        for name in DIRECTIONAL:
            if name in a and a[name]["n"] >= MIN_SAMPLES:
                w[name] = _acc_to_weight(a[name]["acc"])
            elif name in glob and glob[name]["n"] >= MIN_SAMPLES:
                w[name] = _acc_to_weight(glob[name]["acc"])
            else:
                w[name] = 0.2                      # neutral prior until we learn
        out[str(tf)] = w
        out["acc"][str(tf)] = a
    with open(_weights_path(), "w") as f:
        _json.dump(out, f)
    _WEIGHTS_CACHE["ts"] = 0                        # force reload
    return out


def get_weights(tf=None):
    """Current weights for a timeframe (dict {indicator: weight}); {} if not yet
    learned (callers then fall back to an equal-weight vote)."""
    now = time.time()
    if _WEIGHTS_CACHE["data"] is None or now - _WEIGHTS_CACHE["ts"] > 30:
        try:
            with open(_weights_path()) as f:
                _WEIGHTS_CACHE["data"] = _json.load(f)
        except Exception:
            _WEIGHTS_CACHE["data"] = {}
        _WEIGHTS_CACHE["ts"] = now
    data = _WEIGHTS_CACHE["data"] or {}
    if tf is not None and str(tf) in data:
        return data[str(tf)]
    return data.get("global", {})


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
            votes TEXT,
            dir_correct INTEGER,
            UNIQUE(symbol, tf, opened_at)
        )
    """)
    # tolerate older DBs created before these columns existed
    for col, decl in (("votes", "TEXT"), ("dir_correct", "INTEGER")):
        try:
            c.execute(f"ALTER TABLE decisions ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass
    c.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    return c


def _get_meta(conn, key, default=None):
    r = conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
    return r["v"] if r else default


def _set_meta(conn, key, val):
    conn.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (key, str(val)))


def tracking_since():
    with _LOCK, _conn() as conn:
        v = _get_meta(conn, "tracking_since")
    return int(v) if v else None


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
    votes = {c["label"]: c.get("vote") for c in d.get("checks", [])}
    return {
        "symbol": symbol, "tf": tf, "bias": d["bias"],
        "entry": entry, "stop": stop, "target": target, "risk": risk, "rr": rr,
        "passed": int(d.get("passed", 0)), "total_checks": int(d.get("total_checks", 6)),
        "opened_at": opened_at, "horizon_ts": horizon_ts,
        "votes": _json.dumps(votes),
    }


def _insert(conn, row):
    try:
        conn.execute("""
            INSERT OR IGNORE INTO decisions
              (symbol, tf, bias, entry, stop, target, risk, rr, passed,
               total_checks, opened_at, horizon_ts, recorded_at, votes)
            VALUES (:symbol,:tf,:bias,:entry,:stop,:target,:risk,:rr,:passed,
                    :total_checks,:opened_at,:horizon_ts,:recorded_at,:votes)
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
        dir_correct = 1 if outcome_r > 0 else 0
        conn.execute(
            "UPDATE decisions SET status=?, outcome_r=?, exit_price=?, resolved_at=?, "
            "dir_correct=? WHERE id=?",
            (status, outcome_r, exit_px, resolved_at, dir_correct, r["id"]))
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


def tick_symbol(symbol, frame_fn, since=None):
    """Generate + record decisions for one symbol across all timeframes (only for
    bars that closed at/after `since`), then resolve open decisions against real
    price. Heavy work is done OUTSIDE the DB lock so scorecard readers never
    block; only the fast INSERT/UPDATE section is locked."""
    # ---- heavy compute, no lock ----
    rows_to_insert = []
    widest = None
    for tf in TFS:
        try:
            m = frame_fn(symbol, GEN_DAYS.get(tf, 60))
            if m is None or len(m) < 210:
                continue
            c, ind, L, S = SP.prepare(m, tf)
            if len(c) < 210:
                continue
            w = get_weights(tf)
            # capture every closed bar since the last ~2 monitor ticks (dedup
            # makes overlaps free), but NEVER a bar that closed before we started
            # tracking — this is a live test, we do not backfill history.
            k = min(24, max(1, (2 * 1800) // (tf * 60) + 2))
            idxs = list(range(max(210, len(c) - 1 - k), len(c) - 1))
            for i in idxs:
                bar_ts = int(c.index[i].timestamp())
                if since and bar_ts < since:
                    continue
                d = SP.decision_at(c, ind, L, S, i, tf, weights=w)
                row = _row_from_decision(symbol, d)
                if row:
                    rows_to_insert.append(row)
            if widest is None or len(m) > len(widest):
                widest = m
        except Exception:
            continue

    # ---- brief locked DB section ----
    added = resolved = 0
    with _LOCK, _conn() as conn:
        for row in rows_to_insert:
            b = conn.total_changes
            _insert(conn, row)
            added += conn.total_changes - b
        if widest is not None:
            try:
                resolved = _evaluate_symbol(conn, symbol, widest)
            except Exception:
                resolved = 0
        conn.commit()
    return {"symbol": symbol, "added": added, "resolved": resolved}


def prune_old():
    """Drop decisions older than the rolling window so the DB (and the numbers)
    only ever reflect the last WINDOW_DAYS."""
    cutoff = int(time.time()) - (WINDOW_DAYS + 2) * 86400
    with _LOCK, _conn() as conn:
        conn.execute("DELETE FROM decisions WHERE opened_at < ?", (cutoff,))
        conn.commit()


def tick_all(frame_fn, symbols=None):
    since = tracking_since()
    res = [tick_symbol(s, frame_fn, since=since) for s in (symbols or SYMBOLS)]
    try:
        compute_weights()      # learn which indicators are predicting, feed back
        prune_old()
    except Exception:
        pass
    return res


def _default_frame_fn(symbol, days):
    import live_data as LD
    return LD.load_recent_archive(symbol, days)


def ensure_started(frame_fn=None):
    """Stamp the moment we start measuring. NO backfill — this is a genuine live
    test that begins now and grows forward. Returns True if this is the first
    start."""
    with _LOCK, _conn() as conn:
        v = _get_meta(conn, "tracking_since")
        if v:
            return False
        _set_meta(conn, "tracking_since", int(time.time()))
        conn.commit()
    return True


# --------------------------------------------------------------- read/agg  ---

def indicator_accuracy(tf=None):
    """Per-indicator hit rate: when indicator X voted a direction, how often did
    price actually go that way? This is what tells us whether re-weighting the
    vote can lift the edge above coin-flip."""
    with _LOCK, _conn() as conn:
        q = "SELECT tf, bias, dir_correct, votes FROM decisions WHERE status!='open' AND votes IS NOT NULL"
        rows = conn.execute(q).fetchall()
    tally = {}
    for r in rows:
        if tf is not None and r["tf"] != tf:
            continue
        if r["dir_correct"] is None:
            continue
        # the direction price ACTUALLY went
        actual = r["bias"] if r["dir_correct"] else ("short" if r["bias"] == "long" else "long")
        try:
            votes = _json.loads(r["votes"] or "{}")
        except Exception:
            continue
        for name, v in votes.items():
            if v not in ("long", "short"):
                continue
            t = tally.setdefault(name, [0, 0])
            t[1] += 1
            if v == actual:
                t[0] += 1
    return {name: {"acc": round(hit / n * 100, 1), "n": n}
            for name, (hit, n) in tally.items() if n}


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
    cutoff = int(time.time()) - WINDOW_DAYS * 86400
    with _LOCK, _conn() as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM decisions WHERE opened_at >= ?", (cutoff,)).fetchall()]
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

    # calls being watched RIGHT NOW — real live activity, shown from minute one
    opens = [r for r in rows if r["status"] == "open"]
    opens.sort(key=lambda r: r["opened_at"], reverse=True)
    open_calls = [{
        "symbol": r["symbol"], "tf": r["tf"], "bias": r["bias"],
        "entry": r["entry"], "stop": r["stop"], "target": r["target"],
        "opened_at": r["opened_at"], "horizon_ts": r["horizon_ts"],
        "passed": r["passed"], "total_checks": r["total_checks"],
    } for r in opens[:40]]

    # the learned weights, as a ranked list for display (which indicators the
    # tool has learned to trust)
    acc = indicator_accuracy(None)
    weights = []
    gw = get_weights(None)
    for name in DIRECTIONAL:
        weights.append({"indicator": name, "weight": gw.get(name, 0.0),
                        "accuracy": acc.get(name, {}).get("acc"),
                        "n": acc.get(name, {}).get("n", 0)})
    weights.sort(key=lambda x: -(x["accuracy"] or 0))

    return {
        "overall": _agg(rows),
        "by_tf": by_tf,
        "by_symbol": by_symbol,
        "by_conviction": by_conviction,
        "weights": weights,
        "window_days": WINDOW_DAYS,
        "tracking_since": tracking_since(),
        "open_calls": open_calls,
        "recent": recent,
        "generated_at": int(time.time()),
    }


if __name__ == "__main__":
    print("started fresh:", ensure_started())
    tick_all(_default_frame_fn)
    print(_json.dumps(read()["overall"], indent=2))
