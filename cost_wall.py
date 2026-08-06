"""
cost_wall.py — the ceiling test.

Before asking whether a model can predict direction, ask a cheaper question:
if a PERFECT predictor existed — one that always knows the next move's sign —
how much would it earn after fees at each holding period?

That number is the ceiling. No model can beat it. If the ceiling is negative at
some holding period and fee level, then no amount of machine learning, retraining
frequency, or data resolution makes that combination tradeable, and we should
stop before building anything.

Three ceilings are computed, from strict to generous:

  close-to-close : perfect sign call, enter and exit at the bar close
  perfect-timed  : perfect sign AND perfect entry/exit inside the window, i.e.
                   capturing the whole high-to-low range. Physically impossible
                   to achieve; it is the absolute upper bound.
  selective      : perfect sign, and it also skips every window whose move would
                   not cover the fee. The realistic ceiling for a scalper.
"""

import sys

import numpy as np
import pandas as pd

import features as FE

HORIZONS = [1, 5, 15, 30, 60, 240]

# round-trip cost in return units, both sides included
FEES = {
    "spot taker 0.10%": 0.0020,
    "spot w/ BNB 0.075%": 0.0015,
    "futures taker 0.045%": 0.0009,
    "futures maker 0.02%": 0.0004,
}


def analyse(symbol: str) -> None:
    m = FE.build_minute_frame(symbol)
    close = m["close"]
    high = m["high"]
    low = m["low"]

    print(f"\n{'=' * 86}")
    print(f"{symbol}   {m.index.min()} -> {m.index.max()}   {len(m):,} minutes")

    for h in HORIZONS:
        fwd = np.log(close.shift(-h) / close).dropna()
        # best possible entry and exit inside the window
        win_hi = high[::-1].rolling(h, min_periods=1).max()[::-1].shift(-1)
        win_lo = low[::-1].rolling(h, min_periods=1).min()[::-1].shift(-1)
        rng = np.log(win_hi / win_lo).dropna()

        abs_move = fwd.abs()
        print(f"\n--- holding period: {h} minute(s) ---")
        print(f"    median absolute move : {abs_move.median() * 1e4:7.2f} bps")
        print(f"    mean absolute move   : {abs_move.mean() * 1e4:7.2f} bps")
        print(f"    mean high-low range  : {rng.mean() * 1e4:7.2f} bps")

        rows = []
        for name, fee in FEES.items():
            rows.append({
                "fee model": name,
                "cost_bps": fee * 1e4,
                "ceiling_c2c": (abs_move.mean() - fee) * 1e4,
                "ceiling_perfect_timing": (rng.mean() - fee) * 1e4,
                "ceiling_selective": np.maximum(abs_move - fee, 0).mean() * 1e4,
                "tradeable_%": (abs_move > fee).mean() * 100,
            })
        print(pd.DataFrame(rows).round(2).to_string(index=False))


def required_accuracy(symbol: str) -> None:
    """
    The break-even directional accuracy.

    A trader who is right with probability p on a move of average size M earns
    (2p - 1) * M gross. Setting that equal to the round-trip fee F gives

        p_breakeven = (1 + F / M) / 2

    This is the cleanest statement of the problem: it says exactly how good the
    model has to be, before any model exists.

    The second half repeats the calculation on the subset of windows that follow
    an unusually volatile hour. That filter is causal — it uses only past
    volatility, which is genuinely forecastable — and it raises M, which lowers
    the accuracy needed.
    """
    m = FE.build_minute_frame(symbol)
    close = m["close"]
    # realised volatility of the PAST hour, known at decision time
    past_vol = np.sqrt(m["realized_var"].rolling(60, min_periods=30).sum())

    print(f"\n{'=' * 86}")
    print(f"BREAK-EVEN DIRECTIONAL ACCURACY — {symbol}")
    print("(share of calls that must be correct just to cover fees)")

    for label, mask in (("all windows", None),
                        ("top 25% by prior-hour volatility", "hi")):
        rows = []
        for h in HORIZONS:
            fwd = np.log(close.shift(-h) / close)
            sel = fwd.notna() & past_vol.notna()
            if mask == "hi":
                sel &= past_vol >= past_vol.quantile(0.75)
            M = fwd[sel].abs().mean()
            row = {"horizon_min": h, "mean_move_bps": M * 1e4}
            for name, fee in FEES.items():
                p = (1 + fee / M) / 2
                row[name] = p * 100 if p <= 1 else np.nan
            rows.append(row)
        print(f"\n-- {label} --")
        t = pd.DataFrame(rows).round(2)
        print(t.to_string(index=False))
        print("   nan = impossible: the fee exceeds the entire average move,")
        print("         so even a flawless predictor loses money.")


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]):
        analyse(s)
        required_accuracy(s)
