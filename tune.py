"""
tune.py — grid-search the engine's parameters per symbol/timeframe, honestly.

For each parameter set we replay the whole window once (net of cost), then split
the trades by TIME into a train slice (first 2/3) and a validation slice (last
1/3). We pick the best set on TRAIN and report how it did on VALIDATION — a set
that only shines on train and dies on validation is overfit and worthless. We
also print the current default for comparison, so "better" is measured, not
assumed.

Data is loaded once per (symbol, tf); the engine's module-level constants are
monkey-patched per combo (deterministic, no state leaks — they're restored).
"""

from __future__ import annotations

import itertools

import backtest as BT
import rule_engine as RE

COST_BPS = 2 * (4.5 + 1.0)      # 11 bps round-trip, same as the report

GRID = {
    "SCORE_ENTRY": [0.18, 0.30, 0.45],
    "ADX_TREND":   [20, 25, 30],
    "RR_TARGET":   [1.8, 2.5, 3.5],
    "SL_BAND":     [(1.0, 2.5), (1.5, 3.5)],
}

_DEFAULTS = {k: getattr(RE, k) for k in ("SCORE_ENTRY", "ADX_TREND", "RR_TARGET",
                                         "SL_ATR_MIN", "SL_ATR_MAX")}


def _apply(p):
    RE.SCORE_ENTRY = p["SCORE_ENTRY"]
    RE.ADX_TREND = p["ADX_TREND"]
    RE.RR_TARGET = p["RR_TARGET"]
    RE.SL_ATR_MIN, RE.SL_ATR_MAX = p["SL_BAND"]


def _restore():
    for k, v in _DEFAULTS.items():
        setattr(RE, k, v)


def _score(cW, indW, sim_start, mh, split_ts):
    trades = BT._simulate(cW, indW, sim_start, mh, COST_BPS)
    tr = [t for t in trades if t["entry_time"] < split_ts]
    va = [t for t in trades if t["entry_time"] >= split_ts]
    def agg(ts):
        n = len(ts)
        w = sum(1 for t in ts if t["outcome"] == "win")
        net = sum(t["r"] for t in ts)
        return {"n": n, "win": round(w / n * 100, 1) if n else 0, "net": round(net, 1)}
    return agg(tr), agg(va), len(trades)


def tune(symbol, tf, days=90):
    cW, indW, sim_start, mh = BT.prepare(symbol, tf, days)
    split_ts = int(cW.index[sim_start].timestamp()
                   + (cW.index[-1].timestamp() - cW.index[sim_start].timestamp()) * 2 / 3)

    # baseline (current defaults)
    _restore()
    b_tr, b_va, _ = _score(cW, indW, sim_start, mh, split_ts)

    results = []
    keys = ["SCORE_ENTRY", "ADX_TREND", "RR_TARGET", "SL_BAND"]
    for combo in itertools.product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        _apply(p)
        tr, va, _ = _score(cW, indW, sim_start, mh, split_ts)
        results.append((p, tr, va))
    _restore()

    # rank by TRAIN net, require a minimum number of trades on both slices so a
    # 2-trade fluke can't win
    ranked = sorted(
        [r for r in results if r[1]["n"] >= 15 and r[2]["n"] >= 6],
        key=lambda r: r[1]["net"], reverse=True)
    best = ranked[0] if ranked else None

    print(f"\n===== {symbol} {tf}m  (90d, cost {COST_BPS:.0f}bps) =====")
    print(f"  DEFAULT   train net {b_tr['net']:+6.1f}R ({b_tr['n']:>3}t {b_tr['win']:.0f}%) "
          f"| valid net {b_va['net']:+6.1f}R ({b_va['n']:>3}t {b_va['win']:.0f}%)")
    if best:
        p, tr, va = best
        print(f"  BEST(tr)  train net {tr['net']:+6.1f}R ({tr['n']:>3}t {tr['win']:.0f}%) "
              f"| valid net {va['net']:+6.1f}R ({va['n']:>3}t {va['win']:.0f}%)")
        print(f"    params: SCORE_ENTRY={p['SCORE_ENTRY']} ADX={p['ADX_TREND']} "
              f"RR={p['RR_TARGET']} SL={p['SL_BAND']}")
        # also the config that best survives on validation
        bv = max(ranked, key=lambda r: r[2]["net"])
        pv, tv, vv = bv
        print(f"  BEST(val) valid net {vv['net']:+6.1f}R ({vv['n']:>3}t {vv['win']:.0f}%) "
              f"| train net {tv['net']:+6.1f}R  -> SCORE_ENTRY={pv['SCORE_ENTRY']} "
              f"ADX={pv['ADX_TREND']} RR={pv['RR_TARGET']} SL={pv['SL_BAND']}")
    else:
        print("  no combo cleared the minimum-trades bar")


if __name__ == "__main__":
    import sys
    tfs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [15, 60, 240]
    syms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["BTCUSDT", "ETHUSDT"]
    for s in syms:
        for tf in tfs:
            try:
                tune(s, tf)
            except Exception as e:
                import traceback
                print(f"\n{s} {tf}m FAIL:", type(e).__name__, e)
                traceback.print_exc()
