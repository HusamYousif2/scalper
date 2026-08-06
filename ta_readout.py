"""
ta_readout.py — the technical picture, in terms of what each reading precedes.

The indicators themselves are already inside the model: RSI, MACD, Bollinger,
ATR, Stochastic, channel position, fresh breaks, round-number distance, volume
point of control, candle geometry — all 59 of them are part of the 131 features
the volatility model uses. This layer does not add predictive power. Its job is
to make the technical picture legible, which the model cannot do on its own.

The design choice that matters: a raw indicator value is not information. "RSI is
72" tells a reader nothing they can act on. What is informative is where 72 sits
in this market's recent distribution, and what historically followed readings in
that region:

    RSI 72   88th percentile of the last 30 days
             moves that followed readings this high averaged 24 bps
             against 14 bps unconditionally  ->  amplification 1.7x

Two rules are enforced throughout:

  1. Every historical statistic comes from a TRAILING window. Computing them over
     the whole sample is what made three separate candidate signals look real
     earlier in this project before collapsing on a second asset.
  2. Nothing here reports a direction. Amplification of the coming move is
     measurable; its sign is not, and this project has the measurements to say so.

`validate()` answers the question that decides whether any of this may be shown:
does an amplification measured on the trailing window still hold in the period
that follows? An indicator whose amplification does not persist is decoration.
"""

import os
import sys

import numpy as np
import pandas as pd

import features as FE
import features2 as F2

LOOKBACK_DAYS = 30
BUCKETS = 5
HORIZON = 15
STEP = 15                 # sample every 15 minutes when building statistics

# families that are readable as "technical analysis" to a trader
TA_PREFIXES = ("IND", "SR", "VP", "CDL")

# Readings that passed BOTH tests on BTCUSDT: persistence above 0.5 and
# next-period spread above 0.5, over twelve folds. 21 of 53 qualified.
#
# Every one of the strongest ten is a range or volatility measure. Momentum and
# position readings failed: RSI-60 was stable (0.89) but its top and bottom
# buckets precede almost the same move (spread 0.29), candle body and channel
# position scored under 0.16 persistence, and Stochastic-14 came out NEGATIVE at
# -0.158 — its relationship inverts from one period to the next. Those are
# excluded and named in the documentation rather than quietly dropped.
HEADLINE = [
    "IND_atr_rel_14", "IND_atr_rel_60", "CDL_range_rel_15",
    "SR_channel_width_60", "CDL_range_rel_60", "IND_bb_width_14",
    "IND_atr_rel_240", "SR_channel_width_240", "VP_dist_poc_1440",
    "SR_channel_width_1440", "VP_dist_poc_240", "IND_macd_rel",
    "SR_to_low_1440", "SR_round_dist_1000", "SR_round_dist_5000",
]

# Failed the tests on BTCUSDT and are deliberately not shown. Kept here so the
# exclusion is documented rather than invisible.
REJECTED = {
    "IND_stoch_14": "persistence -0.158 — the relationship inverts between periods",
    "CDL_body_5": "persistence 0.083",
    "IND_dist_sma_atr_14": "persistence 0.133",
    "CDL_lower_wick_60": "persistence 0.150",
    "CDL_body_15": "persistence 0.183",
    "SR_channel_pos_60": "persistence 0.200",
    "CDL_streak": "persistence 0.233",
    "SR_to_high_60": "persistence 0.233",
    "IND_rsi_60": "stable at 0.892 but spread only 0.292 — no discrimination",
}


def ta_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(TA_PREFIXES)]


def build(symbol: str, minute: pd.DataFrame | None = None) -> pd.DataFrame:
    """Indicator values on a sampled grid, with the forward absolute move."""
    m = FE.build_minute_frame(symbol, minute=minute)
    extra = F2.build_extra(m, ["IND", "SR", "VP", "CDL"])
    close = m["close"]
    fwd = (np.log(close.shift(-HORIZON) / close).abs() * 1e4).rename("fwd_bps")
    df = extra.join(fwd)
    grid = ((df.index.hour * 60 + df.index.minute) % STEP) == 0
    return df[grid].replace([np.inf, -np.inf], np.nan)


def amplification(df: pd.DataFrame, col: str, value: float,
                  buckets: int = BUCKETS) -> dict | None:
    """
    Where `value` sits in this window, and how big the moves were that followed
    readings in the same bucket, relative to the window as a whole.
    """
    s = df[col].dropna()
    if len(s) < 200:
        return None
    joint = df[[col, "fwd_bps"]].dropna()
    if len(joint) < 200:
        return None

    try:
        edges = np.unique(np.quantile(joint[col], np.linspace(0, 1, buckets + 1)))
        if len(edges) < 3:
            return None
        b = pd.cut(joint[col], edges, labels=False, include_lowest=True)
    except Exception:
        return None

    base = float(joint["fwd_bps"].mean())
    if base <= 0:
        return None
    means = joint.groupby(b)["fwd_bps"].mean()
    counts = joint.groupby(b).size()

    cur_b = int(np.clip(np.searchsorted(edges, value, side="right") - 1,
                        0, len(edges) - 2))
    if cur_b not in means.index:
        return None
    return {
        "indicator": col,
        "value": float(value),
        "percentile": float((s < value).mean() * 100),
        "bucket": cur_b + 1,
        "n_bucket": int(counts[cur_b]),
        "followed_bps": float(means[cur_b]),
        "baseline_bps": base,
        "amplification": float(means[cur_b] / base),
    }


def readout(symbol: str, minute: pd.DataFrame | None = None,
            columns: list[str] | None = None) -> pd.DataFrame:
    """The current technical picture, ranked by how much amplification it implies."""
    df = build(symbol, minute)
    cutoff = df.index.max() - pd.Timedelta(days=LOOKBACK_DAYS)
    # the trailing window excludes the newest rows, whose forward move is unknown
    window = df[(df.index >= cutoff) & df["fwd_bps"].notna()]
    latest = df.iloc[-1]

    cols = columns or [c for c in HEADLINE if c in df.columns]
    rows = []
    for c in cols:
        v = latest.get(c)
        if v is None or not np.isfinite(v):
            continue
        r = amplification(window, c, float(v))
        if r:
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    t = pd.DataFrame(rows).sort_values("amplification", ascending=False)
    return t.reset_index(drop=True)


def validate(symbol: str, folds: int = 12) -> pd.DataFrame:
    """
    Does an amplification measured on the trailing window survive into the next?

    For each fold, every indicator is bucketed on the trailing window, the
    amplification per bucket is recorded, and the SAME buckets are then scored on
    the following period. If the relationship is real the two agree; if it is
    noise they do not. Reported as the rank correlation between the two, per
    indicator, averaged over folds.
    """
    df = build(symbol).dropna(subset=["fwd_bps"])
    cols = [c for c in ta_columns(df) if df[c].notna().mean() > 0.9]
    start, end = df.index.min(), df.index.max()
    span = (end - start) / (folds + 1)

    rows = []
    for c in cols:
        corrs, spreads = [], []
        for k in range(1, folds + 1):
            tr_lo, tr_hi = start + span * (k - 1), start + span * k
            te_lo, te_hi = tr_hi, start + span * (k + 1)
            tr = df[(df.index >= tr_lo) & (df.index < tr_hi)][[c, "fwd_bps"]].dropna()
            te = df[(df.index >= te_lo) & (df.index < te_hi)][[c, "fwd_bps"]].dropna()
            if len(tr) < 300 or len(te) < 300:
                continue
            try:
                edges = np.unique(np.quantile(tr[c], np.linspace(0, 1, BUCKETS + 1)))
                if len(edges) < 3:
                    continue
                btr = pd.cut(tr[c], edges, labels=False, include_lowest=True)
                bte = pd.cut(te[c], edges, labels=False, include_lowest=True)
            except Exception:
                continue
            a = tr.groupby(btr)["fwd_bps"].mean() / tr["fwd_bps"].mean()
            b = te.groupby(bte)["fwd_bps"].mean() / te["fwd_bps"].mean()
            common = a.index.intersection(b.index)
            if len(common) < 3:
                continue
            corrs.append(float(pd.Series(a[common].to_numpy()).corr(
                pd.Series(b[common].to_numpy()), method="spearman")))
            spreads.append(float(b[common].max() - b[common].min()))

        if corrs:
            p = float(np.nanmean(corrs))
            sp = float(np.nanmean(spreads))
            rows.append({
                "indicator": c,
                "folds": len(corrs),
                "persistence": p,
                "next_period_spread": sp,
                # Persistence alone saturates: rank correlation over five buckets
                # is exactly 1.0 for ANY monotone relationship, and every
                # volatility proxy is monotone in forward volatility by
                # construction. RSI scored 0.89 persistence with a spread of only
                # 0.29 — perfectly stable and practically useless, because its top
                # and bottom buckets precede almost the same move. The product of
                # the two is what separates a usable reading from a truism.
                "usefulness": p * sp,
            })

    t = pd.DataFrame(rows).sort_values("usefulness", ascending=False)
    return t.reset_index(drop=True)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    mode = sys.argv[2] if len(sys.argv) > 2 else "validate"

    if mode == "validate":
        t = validate(sym)
        print(f"{sym}: does trailing-window amplification persist into the next "
              f"period?\n")
        print("  persistence is the rank correlation between the amplification")
        print("  measured on the trailing window and the amplification actually")
        print("  realised in the period that follows. 1.0 = holds perfectly,")
        print("  0.0 = the trailing measurement said nothing about the future.\n")
        print("  ranked by usefulness = persistence x next-period spread,")
        print("  because a perfectly stable indicator whose buckets precede the")
        print("  same move tells a reader nothing.\n")
        print(t.head(20).round(3).to_string(index=False))
        print(f"\n  indicators tested: {len(t)}")
        print(f"  median persistence: {t['persistence'].median():.3f}")
        print(f"  median spread:      {t['next_period_spread'].median():.3f}")
        print(f"  usable (persistence > 0.5 AND spread > 0.5): "
              f"{int(((t['persistence'] > 0.5) & (t['next_period_spread'] > 0.5)).sum())}"
              f" of {len(t)}")
        print("\n  weakest by usefulness (these will not be displayed):")
        print(t.tail(8).round(3).to_string(index=False))
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "reports", f"ta_persistence_{sym}.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        t.to_csv(out, index=False)
        print(f"  written to {out}")
    else:
        print(readout(sym).round(3).to_string(index=False))
