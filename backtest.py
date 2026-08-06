"""
backtest.py — replay the decision engine over history and score it.

The engine is deterministic, so its call on any past candle is exactly what it
would have said live. We walk the candles, and whenever the engine gives an
ACTIONABLE signal we open one position (one at a time), then step forward candle
by candle until the take-profit or the stop is touched — using only that
candle's high/low, never the future — and record win or loss in R multiples
(R = risk = entry-to-stop distance). From those trades we compute the usual
scorecard: win rate, profit factor, expectancy, max drawdown, equity curve.

No look-ahead: every decision at index i is `decide(candles[:i+1], ind[:i+1])`,
and every exit is decided from candles strictly after entry. If a single candle
touches both stop and target, the stop is assumed first (conservative).
"""

from __future__ import annotations

import pandas as pd

import indicators as IND
import live_data as LD
import rule_engine as RE

WARMUP = 300          # candles kept before the window so 200-period studies are valid


def _agg():
    return {"open": "first", "high": "max", "low": "min", "close": "last",
            "volume": "sum"}


def _simulate(candles, ind, start_i, max_hold, cost_bps):
    """One position at a time. `cost_bps` = round-trip fee+slippage in bps.
    Returns a list of closed trades with net R (after cost)."""
    highs = candles["high"].to_numpy()
    lows = candles["low"].to_numpy()
    closes = candles["close"].to_numpy()
    times = candles.index
    n = len(candles)

    trades = []
    i = start_i
    while i < n - 1:
        plan = RE.decide(candles.iloc[:i + 1], ind.iloc[:i + 1])
        if not plan.get("actionable") or plan.get("entry") is None:
            i += 1
            continue

        side = plan["side"]
        entry, sl, tp = plan["entry"], plan["stop"], plan["take"]
        risk = abs(entry - sl)
        if risk <= 0:
            i += 1
            continue

        outcome, exit_i, exit_px = None, None, None
        jmax = min(n - 1, i + max_hold)
        for j in range(i + 1, jmax + 1):
            hi, lo = highs[j], lows[j]
            if side == "long":
                hit_sl, hit_tp = lo <= sl, hi >= tp
            else:
                hit_sl, hit_tp = hi >= sl, lo <= tp
            if hit_sl:                       # stop assumed first on an inside-bar tie
                outcome, exit_i, exit_px = "loss", j, sl
                break
            if hit_tp:
                outcome, exit_i, exit_px = "win", j, tp
                break

        if outcome is None:                  # timed out — mark to market at the last close
            exit_i, exit_px = jmax, closes[jmax]

        r_gross = ((exit_px - entry) if side == "long" else (entry - exit_px)) / risk
        # cost in R = round-trip cost as a fraction of the stop distance. Tight
        # stops (small stop %) pay a bigger fraction of R, so scalping the low
        # timeframes is where fees bite hardest.
        stop_bps = risk / entry * 1e4
        cost_r = (cost_bps / stop_bps) if stop_bps > 0 else 0.0
        r = r_gross - cost_r
        if outcome is None:                       # outcome is which level was hit (gross)
            outcome = "win" if r_gross > 0 else "loss"

        trades.append({
            "entry_time": int(times[i].timestamp()),
            "exit_time": int(times[exit_i].timestamp()),
            "side": side,
            "entry": round(entry, 6), "sl": round(sl, 6), "tp": round(tp, 6),
            "exit": round(float(exit_px), 6),
            "r": round(float(r), 3),               # net of cost — the money number
            "r_gross": round(float(r_gross), 3),
            "cost_r": round(float(cost_r), 3),
            "outcome": outcome,
        })
        i = exit_i + 1                       # flat again only after the trade closes
    return trades


def _metrics(symbol, tf, days, trades):
    n = len(trades)
    wins = [t for t in trades if t["outcome"] == "win"]
    losses = [t for t in trades if t["outcome"] == "loss"]
    total_r = sum(t["r"] for t in trades)
    gross_win = sum(t["r"] for t in wins)
    gross_loss = -sum(t["r"] for t in losses)          # positive number

    equity, cum, peak, mdd = [], 0.0, 0.0, 0.0
    for t in trades:
        cum += t["r"]
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
        equity.append({"time": t["exit_time"], "r": round(cum, 3)})

    pf = (gross_win / gross_loss) if gross_loss > 0 else None   # None = no losers
    r_before_cost = sum(t.get("r_gross", t["r"]) for t in trades)
    cost_total = sum(t.get("cost_r", 0.0) for t in trades)
    return {
        "symbol": symbol, "tf": tf, "days": days,
        "trades": n,
        "total_r_before_cost": round(r_before_cost, 2),
        "cost_r_total": round(cost_total, 2),
        "wins": len(wins), "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "total_r": round(total_r, 2),
        "avg_r": round(total_r / n, 3) if n else 0.0,
        "expectancy_r": round(total_r / n, 3) if n else 0.0,
        "profit_factor": round(pf, 2) if pf is not None else None,
        "gross_win_r": round(gross_win, 2),
        "gross_loss_r": round(gross_loss, 2),
        "max_drawdown_r": round(mdd, 2),
        "equity": equity,
        "trades_list": trades[-200:],
    }


def prepare(symbol: str, tf: int, days: int):
    """Load archive candles + indicators for the window, trimmed to window+warmup.
    Returns (candles, ind, sim_start_index, default_max_hold). Loading is done
    once here so a tuner can reuse it across many parameter sets."""
    tail_days = min(400, max(150, days + 60))
    m = LD.load_recent_archive(symbol, tail_days)

    agg = _agg()
    for extra in ("buy_qty", "sell_qty", "n_trades", "whale_buy_qty",
                  "whale_sell_qty", "sum_open_interest",
                  "count_long_short_ratio", "sum_toptrader_long_short_ratio"):
        if extra in m.columns:
            agg[extra] = "last" if "ratio" in extra or extra == "sum_open_interest" else "sum"

    c = m.resample(f"{tf}min").agg(agg).dropna(subset=["open", "close"])
    ind = IND.compute_all(c)

    cutoff = c.index.max() - pd.Timedelta(days=days)
    start_i = int(c.index.searchsorted(cutoff))
    w0 = max(0, start_i - WARMUP)
    cW = c.iloc[w0:]
    indW = ind.iloc[w0:].reset_index(drop=True)
    sim_start = start_i - w0
    max_hold = max(8, int(24 * 60 / tf))
    return cW, indW, sim_start, max_hold


def run(symbol: str, tf: int, days: int, max_hold: int | None = None,
        fee_bps: float = 4.5, slip_bps: float = 1.0) -> dict:
    """Backtest the engine on `symbol` at `tf`-minute candles over the last `days`.
    `fee_bps`/`slip_bps` are per-side; round-trip cost = 2×(fee+slip)."""
    cW, indW, sim_start, mh = prepare(symbol, tf, days)
    if max_hold is None:
        max_hold = mh

    cost_bps = 2 * (fee_bps + slip_bps)
    trades = _simulate(cW, indW, sim_start, max_hold, cost_bps)
    out = _metrics(symbol, tf, days, trades)
    out["fee_bps"] = fee_bps
    out["slip_bps"] = slip_bps
    out["cost_bps_roundtrip"] = round(cost_bps, 2)
    out["from"] = int(cW.index[min(sim_start, len(cW) - 1)].timestamp())
    out["to"] = int(cW.index[-1].timestamp())
    return out


if __name__ == "__main__":
    import json
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    tf = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    r = run(sym, tf, days)
    r.pop("equity", None)
    r.pop("trades_list", None)
    print(json.dumps(r, indent=2))
