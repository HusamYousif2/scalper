"""
diag.py — diagnose WHERE the edge is, now that we know a small one exists.

The development run showed a genuine but small signal (AUC ~0.57, about 5 basis
points gross per trade) that a 11 basis point round trip destroys. Three levers
could close that gap, and this script measures all three at once:

  1. selectivity  - is the edge bigger in the most confident predictions?
  2. horizon      - a longer hold captures a bigger move at the same fixed cost
  3. fee model    - taker (market order) vs maker (resting limit order)

Nothing here is a strategy yet. It is a measurement of where, if anywhere, the
gross edge exceeds the cost floor.
"""

import sys

import numpy as np
import pandas as pd

import features as FE
import model as M

HORIZONS = [60, 120, 240]
TOP_K = [1.00, 0.50, 0.25, 0.10, 0.05, 0.02]

# round-trip cost, in return units
FEE_TAKER = 0.0011   # 0.045% x2 taker + slippage
FEE_MAKER = 0.0005   # 0.02% x2 maker + partial-fill allowance


def decile_table(pred: pd.DataFrame) -> pd.DataFrame:
    """Split predictions by confidence and show the gross edge inside each bucket."""
    conf = (pred["p_up"] - 0.5).abs()
    side = np.sign(pred["p_up"] - 0.5)
    gross = side * pred["fwd_ret"]
    q = pd.qcut(conf, 10, labels=False, duplicates="drop")
    rows = []
    for b in sorted(pd.Series(q).dropna().unique()):
        sel = q == b
        rows.append(
            {
                "decile": int(b) + 1,
                "n": int(sel.sum()),
                "min_conf": float(conf[sel].min()),
                "hit_rate_%": float((gross[sel] > 0).mean() * 100),
                "gross_bps": float(gross[sel].mean() * 1e4),
                "abs_move_bps": float(pred["fwd_ret"][sel].abs().mean() * 1e4),
            }
        )
    return pd.DataFrame(rows)


def selectivity_table(pred: pd.DataFrame, periods_per_year: float) -> pd.DataFrame:
    """Take only the top k fraction by confidence and score it after fees."""
    conf = (pred["p_up"] - 0.5).abs()
    side = np.sign(pred["p_up"] - 0.5)
    gross = (side * pred["fwd_ret"]).to_numpy()
    rows = []
    for k in TOP_K:
        thr = conf.quantile(1 - k)
        sel = (conf >= thr).to_numpy()
        if sel.sum() < 30:
            continue
        g = gross[sel]
        for name, fee in (("taker", FEE_TAKER), ("maker", FEE_MAKER)):
            net = g - fee
            sr = net.mean() / net.std(ddof=1) if net.std(ddof=1) > 0 else 0.0
            rows.append(
                {
                    "top_k_%": k * 100,
                    "fees": name,
                    "n_trades": int(sel.sum()),
                    "hit_%": float((g > 0).mean() * 100),
                    "gross_bps": float(g.mean() * 1e4),
                    "net_bps": float(net.mean() * 1e4),
                    "sharpe_ann": float(
                        sr * np.sqrt(periods_per_year * sel.mean())
                    ),
                    "total_net_%": float(net.sum() * 100),
                }
            )
    return pd.DataFrame(rows)


def run(symbol: str = "BTCUSDT") -> None:
    for h in HORIZONS:
        df = FE.hourly_dataset(symbol, horizon=h)
        cols = FE.feature_names(df)
        if len(df) < 800:
            print(f"\n### horizon {h}m: only {len(df)} samples, skipped")
            continue
        ppy = (365 * 24 * 60) / h
        pred = M.walk_forward(df, cols)
        auc = M._auc(pred["p_up"].to_numpy(), pred["fwd_ret"].to_numpy())

        print(f"\n{'=' * 78}")
        print(f"### horizon {h} minutes | {len(df)} non-overlapping samples "
              f"| walk-forward AUC {auc:.4f}")
        print(f"    median absolute move over the horizon: "
              f"{df['fwd_ret'].abs().median() * 1e4:.1f} bps"
              f"   (taker cost = {FEE_TAKER * 1e4:.0f} bps,"
              f" maker cost = {FEE_MAKER * 1e4:.0f} bps)")
        print("\n-- gross edge by confidence decile --")
        print(decile_table(pred).round(2).to_string(index=False))
        print("\n-- take only the most confident predictions --")
        print(selectivity_table(pred, ppy).round(2).to_string(index=False))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
