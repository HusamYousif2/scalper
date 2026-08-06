"""
calibrate_rv.py — find the constant that turns a candle's high-low range into
the same realised variance the tick tape produces.

Needed because the live path bridges long gaps with one-minute candles, which
carry no tick data, while every model in this project was trained on realised
variance summed from individual trades. Feeding the model a differently-scaled
variance would bias every forecast by a constant factor that nobody would notice.

The Parkinson estimator says variance is (ln(high/low))^2 / (4 ln 2). Whether
that theoretical constant holds on this instrument at this sampling frequency is
an empirical question, so it is measured here against two years of archive data
where both quantities exist.

Runs entirely on local parquet — no network.
"""

import sys

import numpy as np
import pandas as pd

import ingest

THEORETICAL = 1.0 / (4.0 * np.log(2.0))


def calibrate(symbol: str, agg_minutes: list[int] = (1, 15, 60)) -> dict:
    m = ingest.load_minutes(symbol)
    rng2 = np.log(m["high"] / m["low"].replace(0.0, np.nan)) ** 2
    tick = m["realized_var"]

    print(f"{symbol}: {len(m):,} minutes")
    print(f"  theoretical Parkinson constant 1/(4 ln2) = {THEORETICAL:.6f}\n")

    out = {}
    for w in agg_minutes:
        if w == 1:
            a, b = rng2, tick
        else:
            a = rng2.rolling(w, min_periods=w).sum()
            b = tick.rolling(w, min_periods=w).sum()
        ok = a.notna() & b.notna() & (a > 0) & (b > 0)
        a, b = a[ok], b[ok]

        # The relationship is multiplicative and heavily right-skewed, so the
        # ratio of sums fits the MEAN and is badly wrong for a typical window.
        # Everything downstream works in log volatility, so the constant is
        # fitted in log space, where the median is the unbiased choice.
        k_sums = float(b.sum() / a.sum())
        k_log = float(np.exp(np.median(np.log(b) - np.log(a))))
        corr = float(np.corrcoef(np.log(a), np.log(b))[0, 1])
        err_sums = ((k_sums * a - b) / b).abs().median() * 100
        err_log = ((k_log * a - b) / b).abs().median() * 100
        out[w] = {"k_log": k_log, "k_sums": k_sums, "log_corr": corr,
                  "median_abs_err_%": float(err_log)}
        print(f"  aggregated over {w:>3} min:"
              f"  k_log = {k_log:.6f} (x{k_log / THEORETICAL:.2f} theoretical)"
              f"   log-corr {corr:.3f}"
              f"   median err {err_log:.1f}%  (ratio-of-sums fit: {err_sums:.0f}%)")
    return out


if __name__ == "__main__":
    syms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTCUSDT", "ETHUSDT"]
    res = {s: calibrate(s) for s in syms}
    print("\nPARKINSON_K per symbol (15-minute log fit) — these differ enough")
    print("between assets that a single shared constant would bias one of them:")
    for s in syms:
        print(f"    {s}: {res[s][15]['k_log']:.6f}")
