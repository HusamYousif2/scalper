"""
run_full.py — the complete honest evaluation.

Order of operations matters here, so it is fixed in code:

  1. build the hourly dataset with non-overlapping labels
  2. walk forward over the whole timeline; every prediction is made by a model
     that only ever saw its own past
  3. split the RESULTS into development and holdout. The holdout is scored but
     no choice anywhere in this file depends on it
  4. score with the realistic limit-order fill model, not with an assumed fill
  5. repeat the identical, frozen recipe on symbols the model has never seen

Trust bar (inherited from the lab's earlier work):
  holdout Sharpe > 0  AND  PSR > 0.95  AND  the same recipe holds on unseen
  symbols  AND  the shuffled control stays at zero.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/crypto-quant-lab/research"))
from validation import probabilistic_sharpe_ratio, sharpe_per_period  # noqa: E402

import execution as EX  # noqa: E402
import features as FE  # noqa: E402
import model as M  # noqa: E402

DEV_FRACTION = 0.70
TOP_K = [1.00, 0.50, 0.25, 0.10]
# a passive order must sit genuinely away from the last price; an order placed
# at the touch is really a market order wearing a maker fee
OFFSETS = [0.0002, 0.0005, 0.0010]
HORIZONS = [60, 120, 240]

_minute_cache: dict[str, pd.DataFrame] = {}


def minute_frame(symbol: str) -> pd.DataFrame:
    """Price path only — that is all the fill simulator needs, and it keeps the
    cached frame small enough to hold alongside the feature matrix."""
    if symbol not in _minute_cache:
        _minute_cache[symbol] = FE.build_minute_frame(symbol)[["high", "low", "close"]]
    return _minute_cache[symbol]


def signals_from(pred: pd.DataFrame, k: float) -> pd.DataFrame:
    """Keep the top k fraction by confidence and turn them into +1 / -1 sides."""
    conf = (pred["p_up"] - 0.5).abs()
    thr = conf.quantile(1 - k)
    sel = pred[conf >= thr]
    return pd.DataFrame(
        {"side": np.where(sel["p_up"] > 0.5, 1.0, -1.0)}, index=sel.index
    )


def phase_report(name: str, symbol: str, pred: pd.DataFrame, horizon: int) -> pd.DataFrame:
    mf = minute_frame(symbol)
    rows = []
    for k in TOP_K:
        sig = signals_from(pred, k)
        for off in OFFSETS:
            tr = EX.simulate(mf, sig, horizon=horizon, offset=off)
            s = EX.score(tr, horizon)
            if not s or "net_bps" not in s:
                continue
            net = tr[tr["filled"]]["net"]
            s["psr"] = probabilistic_sharpe_ratio(
                sharpe_per_period(net), 0.0, len(net), net.skew(), net.kurt() + 3.0
            )
            rows.append({"top_k_%": k * 100, "offset_bps": off * 1e4, **s})
    t = pd.DataFrame(rows)
    print(f"\n----- {name} | {symbol} | horizon {horizon}m -----")
    if len(t):
        print(t.round(2).to_string(index=False))
    else:
        print("  not enough filled trades to score")
    return t


def monthly_breakdown(symbol: str, pred: pd.DataFrame, horizon: int,
                      k: float = 0.25, offset: float = 0.0005) -> pd.DataFrame:
    """
    Month by month behaviour of one fixed configuration.

    A strategy with no edge scatters randomly around zero. A strategy whose edge
    depends on market regime is positive in blocks and negative in blocks. The
    two look identical in a single aggregate number, which is why this table
    exists.
    """
    sig = signals_from(pred, k)
    tr = EX.simulate(minute_frame(symbol), sig, horizon=horizon, offset=offset)
    if len(tr) == 0:
        return pd.DataFrame()
    tr = tr[tr["filled"]].copy()
    tr["month"] = tr.index.to_period("M")
    g = tr.groupby("month")["net"]
    out = pd.DataFrame({
        "n_trades": g.size(),
        "net_bps": g.mean() * 1e4,
        "total_%": g.sum() * 100,
        "hit_%": g.apply(lambda s: (s > 0).mean() * 100),
    })
    # what the market itself did that month, for context
    px = minute_frame(symbol)["close"]
    mret = px.resample("MS").last().pct_change() * 100
    mret.index = mret.index.to_period("M")
    out["market_%"] = mret.reindex(out.index)
    return out


def run_symbol(symbol: str, horizon: int, split_ts=None) -> dict:
    df = FE.hourly_dataset(symbol, horizon=horizon)
    cols = FE.feature_names(df)
    pred = M.walk_forward(df, cols)
    auc = M._auc(pred["p_up"].to_numpy(), pred["fwd_ret"].to_numpy())
    print(f"\n{'=' * 78}")
    print(f"{symbol} | horizon {horizon}m | {len(df)} samples "
          f"| {len(cols)} features | walk-forward AUC {auc:.4f}")
    if split_ts is None:
        split_ts = pred.index[int(len(pred) * DEV_FRACTION)]
    dev = pred[pred.index < split_ts]
    hold = pred[pred.index >= split_ts]
    print(f"development {dev.index.min()} -> {dev.index.max()} ({len(dev)})")
    print(f"holdout     {hold.index.min()} -> {hold.index.max()} ({len(hold)})")
    tdev = phase_report("DEVELOPMENT", symbol, dev, horizon)
    thold = phase_report("HOLDOUT", symbol, hold, horizon)

    mb = monthly_breakdown(symbol, pred, horizon)
    if len(mb):
        print(f"\n----- MONTH BY MONTH | {symbol} | horizon {horizon}m "
              f"| top 25% | 5 bps passive entry -----")
        print(mb.round(2).to_string())
        pos_months = (mb["net_bps"] > 0).mean() * 100
        print(f"positive months: {pos_months:.0f}%   "
              f"correlation of monthly edge with market direction: "
              f"{mb['net_bps'].corr(mb['market_%']):.2f}")

    return {"auc": auc, "dev": tdev, "hold": thold, "monthly": mb,
            "pred": pred, "split": split_ts}


def main() -> None:
    primary = "BTCUSDT"
    unseen = [s for s in ["ETHUSDT", "SOLUSDT"]
              if os.path.isdir(os.path.expanduser(
                  f"~/crypto-quant-lab/scalper/data/minute/{s}"))]

    results = {}
    for h in HORIZONS:
        results[h] = run_symbol(primary, h)

    # shuffled control on the primary symbol at the shortest horizon
    df = FE.hourly_dataset(primary, horizon=HORIZONS[0])
    rng = np.random.default_rng(11)
    sh = df.copy()
    sh["fwd_ret"] = rng.permutation(sh["fwd_ret"].to_numpy())
    pred_sh = M.walk_forward(sh, FE.feature_names(sh))
    print(f"\n{'=' * 78}\nSHUFFLED CONTROL AUC: "
          f"{M._auc(pred_sh['p_up'].to_numpy(), pred_sh['fwd_ret'].to_numpy()):.4f} "
          f"(must be ~0.50)")

    if not unseen:
        print("\nNo unseen symbols downloaded yet — frozen test skipped.")
        return
    print(f"\n{'=' * 78}\nFROZEN RECIPE ON UNSEEN SYMBOLS: {unseen}")
    for s in unseen:
        for h in HORIZONS:
            try:
                run_symbol(s, h)
            except Exception as e:
                print(f"  {s} h={h}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        HORIZONS = [int(x) for x in sys.argv[1].split(",")]
    main()
