"""
check_leak.py — does any feature see the future?

The test: build the feature matrix twice, once from the full history and once
from a history truncated at some date T. For every timestamp before T the two
must agree to the last bit. If a feature peeks forward, removing the future
changes its past values and the comparison fails.

This is the only reliable way to tell a real signal from a leak, and it is
cheap: no model is trained.

A second check confirms the label is aligned as intended: fwd_ret at t must
equal the log return from close[t] to close[t + horizon], recomputed directly
from the raw price series.
"""

import os
import sys

import numpy as np
import pandas as pd

import features as FE
import ingest

TRUNCATE_AT = pd.Timestamp("2026-01-01")


def _restricted_loader(symbol: str, cutoff: pd.Timestamp):
    """load_minutes, but only from day files strictly before the cutoff."""
    original = ingest.load_minutes

    def loader(sym: str) -> pd.DataFrame:
        d = os.path.join(ingest.MINUTE_DIR, sym)
        files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
        files = [f for f in files if f[:-8] < cutoff.strftime("%Y-%m-%d")]
        parts = [pd.read_parquet(os.path.join(d, f)) for f in files]
        df = pd.concat(parts).sort_index()
        return df[~df.index.duplicated(keep="first")]

    return original, loader


def run(symbol: str = "BTCUSDT", horizon: int = 5) -> None:
    print(f"no-lookahead test | {symbol} | horizon {horizon}m "
          f"| truncating at {TRUNCATE_AT.date()}")

    full = FE.hourly_dataset(symbol, horizon=horizon)

    original, loader = _restricted_loader(symbol, TRUNCATE_AT)
    ingest.load_minutes = loader
    try:
        trunc = FE.hourly_dataset(symbol, horizon=horizon)
    finally:
        ingest.load_minutes = original

    # compare only the region both versions cover, minus a margin at the edge
    # where the truncated version legitimately has no forward data
    edge = TRUNCATE_AT - pd.Timedelta(days=2)
    common = full.index.intersection(trunc.index)
    common = common[common < edge]
    print(f"  overlapping rows compared: {len(common):,}")

    cols = FE.feature_names(full)
    bad = []
    for c in cols:
        a = full.loc[common, c].to_numpy(dtype="float64")
        b = trunc.loc[common, c].to_numpy(dtype="float64")
        if not np.allclose(a, b, rtol=1e-6, atol=1e-9, equal_nan=True):
            diff = np.nanmax(np.abs(a - b))
            bad.append((c, float(diff)))

    if bad:
        print(f"\n  LEAK DETECTED in {len(bad)} of {len(cols)} features:")
        for c, d in sorted(bad, key=lambda x: -x[1])[:20]:
            print(f"    {c:<28} max abs diff {d:.6g}")
    else:
        print(f"\n  PASS: all {len(cols)} features identical before the cutoff.")

    # label alignment, recomputed from raw prices
    m = FE.build_minute_frame(symbol)
    close = m["close"]
    sample = full.index[:: max(1, len(full) // 5000)]
    expected = np.log(
        close.reindex(sample + pd.Timedelta(minutes=horizon)).to_numpy()
        / close.reindex(sample).to_numpy()
    )
    got = full.loc[sample, "fwd_ret"].to_numpy()
    ok = np.allclose(expected, got, rtol=1e-8, atol=1e-12, equal_nan=True)
    print(f"  label alignment on {len(sample):,} sampled rows: "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        i = int(np.nanargmax(np.abs(expected - got)))
        print(f"    worst at {sample[i]}: expected {expected[i]:.8f}, "
              f"got {got[i]:.8f}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    h = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    run(sym, h)
