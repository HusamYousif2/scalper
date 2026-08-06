"""
model.py — walk-forward training and honest evaluation of the 1-hour direction model.

Protocol (fixed before looking at any result, to limit researcher degrees of freedom):

  1. The timeline is split once:  first 70% = DEVELOPMENT, last 30% = HOLDOUT.
     The holdout is opened only after the development phase is finished.
  2. Inside development we walk forward: train on everything up to a point,
     predict the next TEST_DAYS days, roll forward. This mirrors live use and
     never lets the model see its own future.
  3. A position is taken only when the model's probability is far enough from
     0.5. Every taken hour pays the full round-trip cost.
  4. Results are reported against the lab's trust bar:
        OOS Sharpe > 0  AND  PSR > 0.95  AND  the shuffled control gives ~0.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.expanduser("~/crypto-quant-lab/research"))
from validation import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_per_period,
)

import features as FE  # noqa: E402

DEV_FRACTION = 0.70
MIN_TRAIN_DAYS = 180
TEST_DAYS = 14
HOURS_PER_YEAR = 24 * 365

# round-trip cost on BTC perpetual futures: 0.045% taker per side + slippage
COST_RT = 0.0011
COST_GRID = [0.0008, 0.0011, 0.0015, 0.0020]
MARGINS = [0.00, 0.02, 0.04, 0.06]

PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.6,
    reg_lambda=5.0,
    n_jobs=-1,
    verbose=-1,
)


def walk_forward(df: pd.DataFrame, cols: list[str], seed: int = 0,
                 rolling_days: int | None = None,
                 test_days: int | None = None) -> pd.DataFrame:
    """
    Walk forward, returning out-of-sample probabilities.

    rolling_days=None trains on everything up to the cut (expanding window).
    rolling_days=N trains only on the last N days, so the model forgets old
    regimes — this is the "a fresh model for the current market" idea.
    test_days controls how often the model is rebuilt; 1 means daily.
    """
    y = (df["fwd_ret"] > 0).astype(int).to_numpy()
    X = df[cols].to_numpy(dtype="float32")
    idx = df.index
    step_days = test_days if test_days is not None else TEST_DAYS

    start = idx.min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    preds, stamps, importances = [], [], []
    cursor = start
    while cursor < idx.max():
        stop = cursor + pd.Timedelta(days=step_days)
        tr = idx < cursor
        if rolling_days is not None:
            tr &= idx >= cursor - pd.Timedelta(days=rolling_days)
        te = (idx >= cursor) & (idx < stop)
        if te.sum() == 0 or tr.sum() < 1000:
            cursor = stop
            continue
        clf = LGBMClassifier(random_state=seed, **PARAMS)
        clf.fit(X[tr], y[tr])
        preds.append(clf.predict_proba(X[te])[:, 1])
        stamps.append(idx[te])
        importances.append(clf.feature_importances_)
        cursor = stop

    out = pd.DataFrame(
        {"p_up": np.concatenate(preds)}, index=stamps[0].append(stamps[1:])
    )
    out["fwd_ret"] = df.loc[out.index, "fwd_ret"]
    out.attrs["importance"] = pd.Series(
        np.mean(importances, axis=0), index=cols
    ).sort_values(ascending=False)
    return out


def evaluate(pred: pd.DataFrame, margin: float, cost: float) -> dict:
    """Turn probabilities into positions and score the resulting return stream."""
    p = pred["p_up"].to_numpy()
    r = pred["fwd_ret"].to_numpy()
    pos = np.where(p > 0.5 + margin, 1.0, np.where(p < 0.5 - margin, -1.0, 0.0))
    net = pos * r - cost * np.abs(pos)

    traded = pos != 0
    n = int(traded.sum())
    sr = sharpe_per_period(net)
    ann = sr * np.sqrt(HOURS_PER_YEAR)
    s = pd.Series(net)
    psr = (
        probabilistic_sharpe_ratio(sr, 0.0, len(net), s.skew(), s.kurt() + 3.0)
        if len(net) > 10
        else np.nan
    )
    hit = float((np.sign(r[traded]) == pos[traded]).mean()) if n else np.nan
    return {
        "margin": margin,
        "cost": cost,
        "n_hours": len(net),
        "n_trades": n,
        "trade_rate": n / len(net),
        "hit_rate": hit,
        "gross_bps": float((pos * r)[traded].mean() * 1e4) if n else np.nan,
        "net_bps": float(net[traded].mean() * 1e4) if n else np.nan,
        "total_net": float(net.sum()),
        "sharpe_ann": float(ann),
        "psr": float(psr),
        "auc": _auc(p, r),
    }


def _auc(p: np.ndarray, r: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = (r > 0).astype(int)
    if y.min() == y.max():
        return np.nan
    return float(roc_auc_score(y, p))


def report(title: str, rows: list[dict]) -> pd.DataFrame:
    t = pd.DataFrame(rows)
    print(f"\n===== {title} =====")
    show = t[
        ["margin", "cost", "n_trades", "trade_rate", "hit_rate",
         "gross_bps", "net_bps", "sharpe_ann", "psr", "auc"]
    ].copy()
    show["trade_rate"] = (show["trade_rate"] * 100).round(1)
    show["hit_rate"] = (show["hit_rate"] * 100).round(2)
    for c in ["gross_bps", "net_bps"]:
        show[c] = show[c].round(2)
    show["sharpe_ann"] = show["sharpe_ann"].round(3)
    show["psr"] = show["psr"].round(3)
    show["auc"] = show["auc"].round(4)
    print(show.to_string(index=False))
    return t


def main(symbol: str = "BTCUSDT") -> None:
    df = FE.hourly_dataset(symbol)
    cols = FE.feature_names(df)
    split = df.index[int(len(df) * DEV_FRACTION)]
    dev, hold = df[df.index < split], df[df.index >= split]

    print(f"symbol       : {symbol}")
    print(f"samples      : {len(df)} hourly rows, {len(cols)} features")
    print(f"period       : {df.index.min()} -> {df.index.max()}")
    print(f"development  : {dev.index.min()} -> {dev.index.max()}  ({len(dev)})")
    print(f"holdout      : {hold.index.min()} -> {hold.index.max()}  ({len(hold)})")
    print(f"base rate up : {(df['fwd_ret'] > 0).mean():.4f}")

    # ---------- development ------------------------------------------------
    pred_dev = walk_forward(dev, cols)
    rows = [evaluate(pred_dev, m, c) for m in MARGINS for c in COST_GRID]
    tdev = report("DEVELOPMENT walk-forward", rows)

    print("\ntop 20 features by average gain:")
    print(pred_dev.attrs["importance"].head(20).to_string())

    # ---------- shuffled control ------------------------------------------
    rng = np.random.default_rng(7)
    sh = dev.copy()
    sh["fwd_ret"] = rng.permutation(sh["fwd_ret"].to_numpy())
    pred_sh = walk_forward(sh, cols)
    report(
        "SHUFFLED CONTROL (must be ~0 / AUC ~0.5)",
        [evaluate(pred_sh, m, COST_RT) for m in MARGINS],
    )

    # ---------- holdout, opened once --------------------------------------
    # same walk forward over the full timeline; we only read the holdout part,
    # and every holdout prediction still comes from a model trained on its past
    pred_full = walk_forward(df, cols)
    pred_hold = pred_full[pred_full.index >= split]
    rows_h = [evaluate(pred_hold, m, c) for m in MARGINS for c in COST_GRID]
    thold = report("HOLDOUT (never used for any decision)", rows_h)

    # ---------- deflated Sharpe over the whole search ----------------------
    trials = tdev["sharpe_ann"].to_numpy() / np.sqrt(HOURS_PER_YEAR)
    best_row = thold.loc[thold["sharpe_ann"].idxmax()]
    dsr, sr_star = deflated_sharpe_ratio(
        best_row["sharpe_ann"] / np.sqrt(HOURS_PER_YEAR),
        trials,
        best_row["n_hours"],
        0.0,
        3.0,
    )
    print(
        f"\nDeflated Sharpe of best holdout config: {dsr:.3f} "
        f"(benchmark SR* = {sr_star * np.sqrt(HOURS_PER_YEAR):.3f} annualised)"
    )

    # ---------- buy and hold reference ------------------------------------
    bh = hold["fwd_ret"].to_numpy()
    print(
        f"buy-and-hold over holdout : total {bh.sum():.4f}, "
        f"annualised Sharpe {sharpe_per_period(bh) * np.sqrt(HOURS_PER_YEAR):.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
