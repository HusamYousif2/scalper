"""
edge_features.py — indicators that cannot be built from a price chart.

Everything on a standard trading screen is a transform of open, high, low, close
and volume. Anyone can compute them, so nothing computed from them can
distinguish this tool. What we have that a chart does not is the order flow tape,
the futures open interest, and the positioning of retail against professionals.

Six indicators are defined here. None is standard; each is built because there is
a mechanism behind it, and each is measured before being believed.

  ABSORPTION      price movement per unit of one-sided aggression. When heavy
                  buying moves price very little, someone large is selling into
                  it from resting orders. Absorption ends abruptly, and when it
                  does the move is violent — so low absorption readings should
                  precede volatility.

  FLOW_PERSIST    autocorrelation of signed order flow. Informed size arrives in
                  a sequence because it is being worked over minutes; noise
                  alternates. Persistent flow marks a participant with a reason.

  CROWD_STRESS    how one-sided retail positioning is, multiplied by how fast
                  open interest is growing. Crowded and still building is the
                  configuration that unwinds violently.

  SMART_DUMB      retail long/short against top-trader long/short. The two
                  usually move together; when they separate, one group is on the
                  wrong side and the resolution tends to be sharp.

  CASCADE         a proxy for forced liquidation: open interest falling hard
                  while price moves hard on heavy volume. Positions are being
                  closed involuntarily, which begets more of the same.

  WHALE_CONC      share of volume done in large prints, and how lopsided it is.
                  A tape dominated by a few large orders behaves differently from
                  the same volume spread across many small ones.
"""

import sys

import numpy as np
import pandas as pd

import features as FE

WINDOWS = [15, 60, 240]
EPS = 1e-12


def _z(s: pd.Series, w: int = 1440) -> pd.Series:
    """Standardise against a trailing window so the scale is stable over time."""
    mp = max(30, w // 4)
    return (s - s.rolling(w, min_periods=mp).mean()) / (
        s.rolling(w, min_periods=mp).std().replace(0.0, np.nan)
    )


def build(symbol: str, minute: pd.DataFrame | None = None) -> pd.DataFrame:
    m = FE.build_minute_frame(symbol, minute=minute)
    close = m["close"]
    r = np.log(close / close.shift(1))
    f = pd.DataFrame(index=m.index)

    signed = m["buy_qty"] - m["sell_qty"]
    total = m["buy_qty"] + m["sell_qty"]

    for w in WINDOWS:
        mp = max(2, w // 2)
        sig_w = signed.rolling(w, min_periods=mp).sum()
        tot_w = total.rolling(w, min_periods=mp).sum()
        move_w = np.log(close / close.shift(w))

        # ABSORPTION: how far price travelled per unit of net aggression.
        # Small values mean the book ate the flow without moving.
        imb = (sig_w / tot_w.replace(0.0, np.nan)).abs()
        f[f"ABSORPTION_{w}"] = _z(
            move_w.abs() / (imb + EPS)
        )
        # the same idea in reverse: aggression that produced nothing
        f[f"ABSORB_FAIL_{w}"] = (imb > imb.rolling(1440, min_periods=360)
                                 .quantile(0.8)) & (
            move_w.abs() < move_w.abs().rolling(1440, min_periods=360)
            .quantile(0.3)
        )
        f[f"ABSORB_FAIL_{w}"] = f[f"ABSORB_FAIL_{w}"].astype("float32")

        # FLOW_PERSIST: is the same side arriving repeatedly?
        s = signed / total.replace(0.0, np.nan)
        lag = s.shift(1)
        m1 = s.rolling(w, min_periods=mp).mean()
        m2 = lag.rolling(w, min_periods=mp).mean()
        cov = (s * lag).rolling(w, min_periods=mp).mean() - m1 * m2
        sd1 = s.rolling(w, min_periods=mp).std(ddof=0)
        sd2 = lag.rolling(w, min_periods=mp).std(ddof=0)
        f[f"FLOW_PERSIST_{w}"] = cov / (sd1 * sd2).replace(0.0, np.nan)

        # WHALE_CONC: how much of the tape is large prints, and their direction
        wq = m["whale_buy_qty"] + m["whale_sell_qty"]
        f[f"WHALE_CONC_{w}"] = (
            wq.rolling(w, min_periods=mp).sum() / tot_w.replace(0.0, np.nan)
        )
        f[f"WHALE_SKEW_{w}"] = (
            (m["whale_buy_qty"] - m["whale_sell_qty"]).rolling(w, min_periods=mp).sum()
            / wq.rolling(w, min_periods=mp).sum().replace(0.0, np.nan)
        )

    if "sum_open_interest" in m.columns:
        oi = m["sum_open_interest"]
        for w in (15, 60, 240):
            mp = max(2, w // 2)
            oi_chg = np.log((oi / oi.shift(w)).replace(0.0, np.nan))
            move = np.log(close / close.shift(w))
            vol_rel = (
                total.rolling(w, min_periods=mp).sum()
                / total.rolling(1440, min_periods=360).sum().replace(0.0, np.nan)
                * (1440 / w)
            )
            # CASCADE: open interest collapsing while price runs on heavy volume
            f[f"CASCADE_{w}"] = _z(
                (-oi_chg).clip(lower=0) * move.abs() * vol_rel.clip(upper=10)
            )
            # the benign opposite: new positions opening into a move
            f[f"BUILDUP_{w}"] = _z(
                oi_chg.clip(lower=0) * move.abs() * vol_rel.clip(upper=10)
            )

        if "count_long_short_ratio" in m.columns:
            retail = np.log(pd.to_numeric(m["count_long_short_ratio"],
                                          errors="coerce").clip(lower=EPS))
            oi_growth = np.log((oi / oi.shift(240)).replace(0.0, np.nan))
            # CROWD_STRESS: one-sidedness times how fast the crowd is growing
            f["CROWD_STRESS"] = _z(retail.abs() * oi_growth.clip(lower=0))
            f["CROWD_LEAN"] = _z(retail)

            if "sum_toptrader_long_short_ratio" in m.columns:
                top = np.log(pd.to_numeric(m["sum_toptrader_long_short_ratio"],
                                           errors="coerce").clip(lower=EPS))
                # SMART_DUMB: the gap between the two groups, standardised
                f["SMART_DUMB_GAP"] = _z(retail - top)
                f["SMART_DUMB_CHG"] = f["SMART_DUMB_GAP"] - f["SMART_DUMB_GAP"].shift(60)

    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


FAMILIES = ("ABSORPTION", "ABSORB", "FLOW", "WHALE", "CASCADE", "BUILDUP",
            "CROWD", "SMART")

GROUPS = {
    "ABSORB": ("ABSORPTION", "ABSORB"),
    "FLOW": ("FLOW",),
    "WHALE": ("WHALE",),
    "OIFLOW": ("CASCADE", "BUILDUP"),
    "CROWD": ("CROWD", "SMART"),
}


def group_of(col: str) -> str | None:
    for g, prefixes in GROUPS.items():
        if col.startswith(prefixes):
            return g
    return None


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    f = build(sym)
    print(f"{sym}: {len(f):,} minutes, {len(f.columns)} indicators")
    counts: dict[str, int] = {}
    for c in f.columns:
        g = group_of(c) or "other"
        counts[g] = counts.get(g, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"   {k:<8} {v}")
    tail = f.tail(200000)
    print("\n  non-null share over the last 200k minutes:")
    for k in sorted(counts):
        cols = [c for c in f.columns if (group_of(c) or "other") == k]
        print(f"   {k:<8} {tail[cols].notna().mean().mean() * 100:5.1f} %")
