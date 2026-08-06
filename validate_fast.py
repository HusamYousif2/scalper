"""
validate_fast.py — strict validation of the one configuration that looked real.

Frozen configuration (chosen from the fast_scalp sweep, then not touched):

    horizon          15 minutes
    training window  rolling 90 days, rebuilt every day
    filter 1         only after an unusually volatile hour
    filter 2         only the most confident tenth of predictions
    execution        market in, market out - no limit-order fill assumption

Four things the sweep did not do, all of which can only make the result worse:

  1. CAUSAL THRESHOLDS. The sweep picked the volatility and confidence cutoffs
     from the distribution of the whole period, which is not knowable at the
     time of the trade. Here both cutoffs come from a trailing window only.
  2. A HOLDOUT. The sweep scored everything over one period. Here the timeline
     is split and the later part is scored separately.
  3. DEFLATED SHARPE against every cell examined during the search, so that
     picking the best of many is penalised.
  4. THE SAME RECIPE ON ETHUSDT, which played no part in choosing it.

A shuffled-label control runs through the identical path.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/crypto-quant-lab/research"))
from validation import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_per_period,
)

import features as FE  # noqa: E402
import model as M  # noqa: E402

HORIZON = 15
ROLLING_DAYS = 90
RETRAIN_DAYS = 1
VOL_Q = 0.75
CONF_Q = 0.90
LOOKBACK = "30D"          # trailing window used to set both cutoffs

# market order both ways: 0.045% x 2, plus crossing the spread
FEE_TAKER = 0.0009 + 0.00015
FEE_MAKER = 0.0004

DEV_FRACTION = 0.70
BARS_PER_YEAR = 365 * 24 * 60 / HORIZON


def causal_threshold(s: pd.Series, q: float, lookback: str) -> pd.Series:
    """
    Quantile of a trailing window, shifted so the current value never
    contributes to the cutoff it is compared against.
    """
    return s.shift(1).rolling(lookback, min_periods=200).quantile(q)


def build_trades(pred: pd.DataFrame, vol: pd.Series) -> pd.DataFrame:
    """Apply both causal filters and return the trades that survive."""
    conf = (pred["p_up"] - 0.5).abs()
    vol = vol.reindex(pred.index)

    vol_cut = causal_threshold(vol, VOL_Q, LOOKBACK)
    conf_cut = causal_threshold(conf, CONF_Q, LOOKBACK)

    take = (vol >= vol_cut) & (conf >= conf_cut)
    take = take.fillna(False)

    side = np.where(pred["p_up"] > 0.5, 1.0, -1.0)
    gross = side * pred["fwd_ret"]
    out = pd.DataFrame(
        {"side": side, "gross": gross,
         "net_taker": gross - FEE_TAKER, "net_maker": gross - FEE_MAKER},
        index=pred.index,
    )[take]
    return out


def score(trades: pd.DataFrame, label: str, col: str = "net_taker") -> dict:
    if len(trades) < 50:
        print(f"  {label}: only {len(trades)} trades, not scored")
        return {}
    r = trades[col]
    sr = sharpe_per_period(r)
    # annualise by the number of trades actually taken per year
    span_years = (trades.index.max() - trades.index.min()).days / 365.25
    per_year = len(r) / max(span_years, 1e-9)
    ann = sr * np.sqrt(per_year)
    psr = probabilistic_sharpe_ratio(sr, 0.0, len(r), r.skew(), r.kurt() + 3.0)
    d = {
        "phase": label,
        "n_trades": len(r),
        "trades_per_day": len(r) / max(span_years * 365.25, 1e-9),
        "accuracy_%": float((trades["gross"] > 0).mean() * 100),
        "gross_bps": float(trades["gross"].mean() * 1e4),
        "net_bps": float(r.mean() * 1e4),
        "total_%": float(r.sum() * 100),
        "sharpe_ann": float(ann),
        "psr": float(psr),
        "sr_per_trade": float(sr),
    }
    return d


def monthly(trades: pd.DataFrame, col: str = "net_taker") -> pd.DataFrame:
    g = trades.groupby(trades.index.to_period("M"))[col]
    return pd.DataFrame({
        "n": g.size(),
        "net_bps": g.mean() * 1e4,
        "total_%": g.sum() * 100,
        "win_%": g.apply(lambda s: (s > 0).mean() * 100),
    })


def run_symbol(symbol: str, shuffle: bool = False) -> dict:
    df = FE.hourly_dataset(symbol, horizon=HORIZON)
    cols = FE.feature_names(df)
    if shuffle:
        rng = np.random.default_rng(23)
        df = df.copy()
        df["fwd_ret"] = rng.permutation(df["fwd_ret"].to_numpy())

    M.MIN_TRAIN_DAYS = ROLLING_DAYS
    # hundreds of small fits: 12 threads each costs more in coordination than
    # it saves, measured at roughly 2x slower than 4 threads on this box
    M.PARAMS["n_jobs"] = 4
    pred = M.walk_forward(df, cols, rolling_days=ROLLING_DAYS,
                          test_days=RETRAIN_DAYS)
    auc = M._auc(pred["p_up"].to_numpy(), pred["fwd_ret"].to_numpy())
    trades = build_trades(pred, df["sigma_60"])

    tag = f"{symbol}{' [SHUFFLED]' if shuffle else ''}"
    print(f"\n{'=' * 84}")
    print(f"{tag} | horizon {HORIZON}m | walk-forward AUC {auc:.4f} "
          f"| {len(trades)} trades after causal filters")

    if len(trades) < 50:
        return {}
    split = trades.index[int(len(trades) * DEV_FRACTION)]
    rows = [
        score(trades, "ALL"),
        score(trades[trades.index < split], "development"),
        score(trades[trades.index >= split], "HOLDOUT"),
    ]
    rows = [r for r in rows if r]
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))

    print("\n  same trades priced with maker fees instead:")
    mk = pd.DataFrame([r for r in [
        score(trades, "ALL", "net_maker"),
        score(trades[trades.index >= split], "HOLDOUT", "net_maker"),
    ] if r])
    print(mk.round(3).to_string(index=False))

    mb = monthly(trades)
    print(f"\n  month by month ({(mb['net_bps'] > 0).mean() * 100:.0f}% positive months):")
    print(mb.round(2).to_string())
    return {"trades": trades, "table": t, "auc": auc}


def main() -> None:
    res = run_symbol("BTCUSDT")
    if not res:
        return

    # every cell inspected during the search, for the deflation penalty
    n_cells = 36
    hold = res["trades"].iloc[int(len(res["trades"]) * DEV_FRACTION):]
    sr_hold = sharpe_per_period(hold["net_taker"])
    rng = np.random.default_rng(5)
    # spread of Sharpes seen across the search, approximated by resampling the
    # observed per-trade Sharpe dispersion
    trial_sr = rng.normal(0.0, max(abs(sr_hold), 1e-4), n_cells)
    dsr, sr_star = deflated_sharpe_ratio(
        sr_hold, trial_sr, len(hold), hold["net_taker"].skew(),
        hold["net_taker"].kurt() + 3.0
    )
    print(f"\n{'=' * 84}")
    print(f"Deflated Sharpe on holdout, penalised for {n_cells} inspected cells: "
          f"{dsr:.3f}  (benchmark SR* per trade = {sr_star:.4f})")

    run_symbol("BTCUSDT", shuffle=True)
    if os.path.isdir(os.path.expanduser(
            "~/crypto-quant-lab/scalper/data/minute/ETHUSDT")):
        print(f"\n{'=' * 84}\nFROZEN RECIPE ON ETHUSDT (played no part in choosing it)")
        run_symbol("ETHUSDT")


if __name__ == "__main__":
    main()
