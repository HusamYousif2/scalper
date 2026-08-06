"""
compare_features.py — does adding indicators / levels / volume profile / candles
actually help, or does it only help in sample?

Method: the exact validation protocol from validate_fast.py (daily retrain,
90-day rolling window, causal trailing-window thresholds, development/holdout
split) run once per feature set. The only thing that changes between runs is the
list of columns handed to the model.

Feature sets tested:
    base            the original 77 microstructure features
    base+IND        plus classic indicators
    base+SR         plus support / resistance and round levels
    base+VP         plus volume profile
    base+CDL        plus candle geometry
    base+BOOK       plus order-book wall geometry
    all             everything at once

Reading the output: compare the HOLDOUT rows, not the development rows. A family
that lifts development and drops holdout is fitting noise, which is the normal
outcome of adding features and the reason this script exists.
"""

import os
import sys

import numpy as np
import pandas as pd

import features as FE
import features2 as F2
import model as M
import validate_fast as V

HORIZON = 15
SETS = {
    "base": [],
    "base+IND": ["IND"],
    "base+SR": ["SR"],
    "base+VP": ["VP"],
    "base+CDL": ["CDL"],
    "base+BOOK": ["BOOK"],
    "all": ["IND", "SR", "VP", "CDL", "BOOK"],
}


def cols_for(df: pd.DataFrame, families: list[str]) -> list[str]:
    """Base columns always, plus the requested families only."""
    out = []
    for c in F2.feature_names(df):
        prefix = c.split("_")[0]
        if prefix in F2.FAMILIES:
            if prefix in families:
                out.append(c)
        else:
            out.append(c)
    return out


def run(symbol: str = "BTCUSDT") -> pd.DataFrame:
    # build the full matrix once; each set is a column subset of it, so no
    # feature is ever recomputed differently between runs
    df = F2.dataset(symbol, horizon=HORIZON)
    vol = df["sigma_60"]
    print(f"{symbol} | horizon {HORIZON}m | {len(df):,} samples "
          f"| {len(F2.feature_names(df))} features available", flush=True)

    M.MIN_TRAIN_DAYS = V.ROLLING_DAYS
    M.PARAMS["n_jobs"] = 4

    rows = []
    for name, fams in SETS.items():
        cols = cols_for(df, fams)
        pred = M.walk_forward(df, cols, rolling_days=V.ROLLING_DAYS,
                              test_days=V.RETRAIN_DAYS)
        auc = M._auc(pred["p_up"].to_numpy(), pred["fwd_ret"].to_numpy())
        trades = V.build_trades(pred, vol)
        if len(trades) < 100:
            print(f"  {name}: too few trades ({len(trades)})", flush=True)
            continue
        split = trades.index[int(len(trades) * V.DEV_FRACTION)]
        for phase, sub in (("development", trades[trades.index < split]),
                           ("HOLDOUT", trades[trades.index >= split])):
            s = V.score(sub, phase)
            if s:
                rows.append({"set": name, "n_features": len(cols),
                             "auc": auc, **s})
        done = pd.DataFrame(rows)
        print(f"\n>>> {name} ({len(cols)} features), AUC {auc:.4f}", flush=True)
        print(done[done["set"] == name][
            ["phase", "n_trades", "accuracy_%", "net_bps", "sharpe_ann", "psr"]
        ].round(3).to_string(index=False), flush=True)

    t = pd.DataFrame(rows)
    print(f"\n{'=' * 92}\nSUMMARY — holdout only (this is the row that matters)")
    h = t[t["phase"] == "HOLDOUT"][
        ["set", "n_features", "auc", "n_trades", "accuracy_%",
         "net_bps", "sharpe_ann", "psr"]
    ]
    print(h.round(3).to_string(index=False))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"compare_{symbol}.csv")
    t.to_csv(out, index=False)
    print(f"\nfull table written to {out}")
    return t


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
