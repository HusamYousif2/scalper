"""
vol_extra.py — the volatility features the first model was missing.

The current model feeds realised variance to the learner as a single lump. The
volatility literature is clear that taking it apart helps, and four
decompositions are worth testing here. Each is computed from one-minute returns
over the archive we already have, so nothing needs re-downloading.

  SEMI  realised semivariance (Barndorff-Nielsen, Kinnebrock, Shephard 2010).
        Variance is split into the part built from negative returns and the part
        built from positive ones. Downside variance is the stronger predictor of
        what comes next: markets remember fear longer than greed.

  JUMP  bipower variation (Barndorff-Nielsen and Shephard 2004) measures the
        continuous part of variance while ignoring isolated jumps. The gap
        between total variance and bipower variation is the jump component. The
        two persist very differently — continuous volatility carries forward,
        a jump is over once it has happened — so a model that cannot tell them
        apart is averaging two unrelated processes.

  SEAS  intraday seasonality. Crypto volatility has a repeating shape across the
        hours of the week. The model currently sees the clock only as a sine and
        cosine; this instead supplies the volatility NORMALLY seen at this hour,
        measured over a trailing window so it stays causal.

  XASS  cross-asset. BTC and ETH volatility move together, and neither model
        currently looks at the other asset at all.
"""

import os
import sys

import numpy as np
import pandas as pd

import features as FE

WINDOWS = [15, 60, 240, 1440]
SEASONAL_WEEKS = 8          # trailing weeks used to learn the hour-of-week shape
OTHER = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}


def semivariance(r: pd.Series, w: int) -> tuple[pd.Series, pd.Series]:
    """Variance built from down moves and from up moves, separately."""
    mp = max(2, w // 2)
    neg = (r.clip(upper=0.0) ** 2).rolling(w, min_periods=mp).sum()
    pos = (r.clip(lower=0.0) ** 2).rolling(w, min_periods=mp).sum()
    return neg, pos


def bipower(r: pd.Series, w: int) -> pd.Series:
    """
    Bipower variation: the continuous part of variance.

    Multiplying neighbouring absolute returns makes a single large jump
    contribute only through its two products rather than its square, so isolated
    jumps wash out while ordinary volatility survives.
    """
    mp = max(2, w // 2)
    mu1 = np.sqrt(2.0 / np.pi)
    prod = r.abs() * r.abs().shift(1)
    return prod.rolling(w, min_periods=mp).sum() / (mu1 ** 2)


def seasonal_profile(rv: pd.Series, weeks: int = SEASONAL_WEEKS) -> pd.Series:
    """
    Typical variance for this minute of the week, from trailing weeks only.

    Implemented as a rolling mean over the same slot one week apart, which keeps
    it causal: the value at time t uses only weeks that already finished.
    """
    week = 7 * 24 * 60
    parts = [rv.shift(week * k) for k in range(1, weeks + 1)]
    return pd.concat(parts, axis=1).mean(axis=1)


def build(symbol: str, minute: pd.DataFrame | None = None,
          with_cross: bool = True) -> pd.DataFrame:
    m = FE.build_minute_frame(symbol, minute=minute)
    close = m["close"]
    r = np.log(close / close.shift(1))
    rv_min = m["realized_var"]

    f = pd.DataFrame(index=m.index)

    for w in WINDOWS:
        mp = max(2, w // 2)
        rv = rv_min.rolling(w, min_periods=mp).sum()
        neg, pos = semivariance(r, w)
        bv = bipower(r, w)
        total = (r ** 2).rolling(w, min_periods=mp).sum()

        # SEMI: which side built the variance, and how lopsided it was
        f[f"SEMI_down_{w}"] = np.log(neg.replace(0.0, np.nan))
        f[f"SEMI_skew_{w}"] = (neg - pos) / (neg + pos).replace(0.0, np.nan)

        # JUMP: continuous share, and the size of what could not be explained
        # by continuous movement
        jump = (total - bv).clip(lower=0.0)
        f[f"JUMP_share_{w}"] = jump / total.replace(0.0, np.nan)
        f[f"JUMP_cont_ratio_{w}"] = bv / total.replace(0.0, np.nan)
        # tick-level variance against minute-level: how much of the measured
        # variance is bid-ask bounce rather than real movement
        f[f"JUMP_micro_ratio_{w}"] = np.log(
            (rv / total.replace(0.0, np.nan)).replace(0.0, np.nan)
        )

    # SEAS: how this moment compares with what this hour of the week usually does
    prof = seasonal_profile(rv_min.rolling(60, min_periods=30).sum())
    cur = rv_min.rolling(60, min_periods=30).sum()
    f["SEAS_profile"] = np.log(prof.replace(0.0, np.nan))
    f["SEAS_vs_profile"] = np.log(
        (cur / prof.replace(0.0, np.nan)).replace(0.0, np.nan)
    )

    if with_cross and symbol in OTHER:
        other = OTHER[symbol]
        d = os.path.join(FE.ingest.MINUTE_DIR, other)
        if os.path.isdir(d):
            om = FE.build_minute_frame(other)
            orv = om["realized_var"]
            for w in (60, 240):
                mp = max(2, w // 2)
                a = rv_min.rolling(w, min_periods=mp).sum()
                b = orv.rolling(w, min_periods=mp).sum().reindex(m.index)
                f[f"XASS_other_vol_{w}"] = np.log(np.sqrt(b.replace(0.0, np.nan)))
                f[f"XASS_ratio_{w}"] = np.log(
                    (a / b.replace(0.0, np.nan)).replace(0.0, np.nan)
                )
                oc = om["close"].reindex(m.index)
                f[f"XASS_other_ret_{w}"] = np.log(oc / oc.shift(w))
            del om

    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


FAMILIES = ("SEMI", "JUMP", "SEAS", "XASS")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    f = build(sym)
    counts = {fam: sum(c.startswith(fam) for c in f.columns) for fam in FAMILIES}
    print(f"{sym}: {len(f):,} minutes, {len(f.columns)} new features")
    for k, v in counts.items():
        print(f"   {k:<6} {v}")
    print(f"\n  non-null share of the last 100k rows:")
    print((f.tail(100000).notna().mean().groupby(
        [c.split('_')[0] for c in f.columns]).mean() * 100).round(1).to_string())
