"""
fast_scalp.py — test the "fresh model every day" idea at short horizons.

Three changes from the previous study, all of them the user's hypothesis:

  1. the model is rebuilt EVERY DAY, not every two weeks
  2. it trains on a rolling recent window and forgets older regimes, instead of
     an expanding window that averages over two years
  3. horizons are 5 / 15 / 30 minutes instead of 1 / 2 / 4 hours

Two filters are applied on top, because the cost-wall analysis showed they are
the only conditions under which the arithmetic can work at all:

  - trade only after an unusually volatile hour (the move has to be big enough
    to pay the fee, and volatility, unlike direction, is forecastable)
  - trade only the most confident predictions

The break-even accuracy from cost_wall.py is printed next to the achieved
accuracy, so the gap is explicit rather than buried in a Sharpe ratio.
"""

import sys
import time

import numpy as np
import pandas as pd

import features as FE
import model as M

# ordered by how plausible the cost-wall analysis says they are, so the most
# informative results arrive first
HORIZONS = [15, 30, 5]
ROLLING_DAYS = [30, 90]
RETRAIN_DAYS = 1
VOL_QUANTILES = [0.0, 0.75]      # 0.0 = no filter, 0.75 = top quarter only
TOP_K = [1.00, 0.25, 0.10]

FEE_MAKER = 0.0004   # 0.02% per side, both sides passive
FEE_TAKER = 0.0009   # 0.045% per side


def breakeven(mean_move: float, fee: float) -> float:
    return (1 + fee / mean_move) / 2


def evaluate(pred: pd.DataFrame, vol: pd.Series, vq: float, k: float) -> dict:
    p = pred["p_up"]
    r = pred["fwd_ret"]
    sel = pd.Series(True, index=pred.index)
    if vq > 0:
        thr_v = vol.reindex(pred.index).quantile(vq)
        sel &= vol.reindex(pred.index) >= thr_v
    conf = (p - 0.5).abs()
    if k < 1.0:
        sel &= conf >= conf[sel].quantile(1 - k)
    if sel.sum() < 100:
        return {}

    side = np.sign(p[sel] - 0.5)
    gross = side * r[sel]
    mean_move = r[sel].abs().mean()
    acc = float((gross > 0).mean())
    return {
        "vol_filter": vq,
        "top_k": k,
        "n": int(sel.sum()),
        "mean_move_bps": mean_move * 1e4,
        "accuracy_%": acc * 100,
        "need_maker_%": breakeven(mean_move, FEE_MAKER) * 100,
        "need_taker_%": breakeven(mean_move, FEE_TAKER) * 100,
        "gross_bps": float(gross.mean() * 1e4),
        "net_maker_bps": float(gross.mean() * 1e4 - FEE_MAKER * 1e4),
        "net_taker_bps": float(gross.mean() * 1e4 - FEE_TAKER * 1e4),
    }


def run(symbol: str = "BTCUSDT") -> None:
    # start testing once the longest rolling window is full, so every
    # configuration is scored over the exact same period
    M.MIN_TRAIN_DAYS = max(ROLLING_DAYS)
    # thousands of tiny fits: spawning 12 threads per fit costs more than it
    # saves, so cap the thread pool
    M.PARAMS["n_jobs"] = 4
    for h in HORIZONS:
        df = FE.hourly_dataset(symbol, horizon=h)
        cols = FE.feature_names(df)
        vol = df["sigma_60"]
        print(f"\n{'=' * 96}", flush=True)
        print(f"{symbol} | horizon {h}m | {len(df):,} samples "
              f"| {len(cols)} features", flush=True)

        for rd in ROLLING_DAYS:
            t0 = time.time()
            pred = M.walk_forward(df, cols, rolling_days=rd, test_days=RETRAIN_DAYS)
            print(f"    (fitted in {time.time() - t0:.0f}s)", flush=True)
            auc = M._auc(pred["p_up"].to_numpy(), pred["fwd_ret"].to_numpy())
            rows = [evaluate(pred, vol, vq, k)
                    for vq in VOL_QUANTILES for k in TOP_K]
            rows = [r for r in rows if r]
            print(f"\n-- rolling window {rd}d, retrained every {RETRAIN_DAYS}d "
                  f"| walk-forward AUC {auc:.4f} --", flush=True)
            if rows:
                print(pd.DataFrame(rows).round(2).to_string(index=False), flush=True)
            else:
                print("   too few samples after filtering", flush=True)


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
