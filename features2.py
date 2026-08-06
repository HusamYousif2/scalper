"""
features2.py — the extra feature families: classic indicators, support and
resistance, volume profile, candle shape, and order-book wall geometry.

These sit ON TOP of the 77 features in features.py. Nothing is removed, so the
comparison "old set vs old + new" is clean.

A warning that belongs in the file, not just in conversation: adding features
reliably improves in-sample fit and usually damages out-of-sample performance.
This lab has already demonstrated that (`research/feature_explosion.py` reached
in-sample Sharpe 9 and negative out-of-sample). Every family here is therefore
tagged, so the evaluation can report which family, if any, actually paid.

Family tags:
  IND  classic technical indicators
  SR   support / resistance and round-number levels
  VP   volume profile
  CDL  candle geometry
  BOOK order-book wall shape
"""

import numpy as np
import pandas as pd

import features as FE

# windows in minutes used across the families
IND_WINDOWS = [14, 60, 240]
SR_WINDOWS = [60, 240, 1440]
VP_WINDOWS = [240, 1440]
CDL_WINDOWS = [5, 15, 60]

# psychological round levels in USD; distance to these is a real, if weak,
# effect in crypto because stop and limit orders cluster on them
ROUND_LEVELS = [100.0, 500.0, 1000.0, 5000.0]


def _safe(a, b):
    return a / b.replace(0.0, np.nan) if isinstance(b, pd.Series) else a / b


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = _safe(up, dn)
    return 100 - 100 / (1 + rs)


def _atr(m: pd.DataFrame, n: int) -> pd.Series:
    prev = m["close"].shift(1)
    tr = pd.concat(
        [m["high"] - m["low"], (m["high"] - prev).abs(), (m["low"] - prev).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def indicators(m: pd.DataFrame) -> dict[str, pd.Series]:
    """IND — the classic screen indicators, in normalised (unitless) form."""
    close = m["close"]
    out = {}
    for n in IND_WINDOWS:
        # centred on zero so the model does not have to learn the 50 line
        out[f"IND_rsi_{n}"] = _rsi(close, n) - 50.0

        sma = close.rolling(n, min_periods=n // 2).mean()
        sd = close.rolling(n, min_periods=n // 2).std()
        # position inside the Bollinger band, and how wide the band is
        out[f"IND_bb_pos_{n}"] = _safe(close - sma, 2 * sd)
        out[f"IND_bb_width_{n}"] = _safe(4 * sd, sma)

        atr = _atr(m, n)
        out[f"IND_atr_rel_{n}"] = _safe(atr, close)
        # how far price sits from its average, measured in average true ranges
        out[f"IND_dist_sma_atr_{n}"] = _safe(close - sma, atr)

        hi = m["high"].rolling(n, min_periods=n // 2).max()
        lo = m["low"].rolling(n, min_periods=n // 2).min()
        out[f"IND_stoch_{n}"] = _safe(close - lo, hi - lo) - 0.5

    ema_fast = close.ewm(span=12 * 5, adjust=False).mean()
    ema_slow = close.ewm(span=26 * 5, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9 * 5, adjust=False).mean()
    out["IND_macd_hist"] = _safe(macd - signal, close)
    out["IND_macd_rel"] = _safe(macd, close)
    return out


def support_resistance(m: pd.DataFrame) -> dict[str, pd.Series]:
    """SR — where price sits relative to recent extremes and round numbers."""
    close = m["close"]
    atr = _atr(m, 60)
    out = {}
    for w in SR_WINDOWS:
        hi = m["high"].rolling(w, min_periods=w // 2).max()
        lo = m["low"].rolling(w, min_periods=w // 2).min()
        # distance to the ceiling and the floor, in units of volatility, so the
        # value means the same thing at any price level
        out[f"SR_to_high_{w}"] = _safe(hi - close, atr)
        out[f"SR_to_low_{w}"] = _safe(close - lo, atr)
        # position inside the channel: 0 at the floor, 1 at the ceiling
        out[f"SR_channel_pos_{w}"] = _safe(close - lo, hi - lo)
        # a channel that is tightening often precedes expansion
        out[f"SR_channel_width_{w}"] = _safe(hi - lo, close)
        # fresh break of the previous extreme
        out[f"SR_break_up_{w}"] = (close > hi.shift(1)).astype("float32")
        out[f"SR_break_dn_{w}"] = (close < lo.shift(1)).astype("float32")

    for lv in ROUND_LEVELS:
        nearest = (close / lv).round() * lv
        out[f"SR_round_dist_{int(lv)}"] = _safe(close - nearest, atr)
    return out


def volume_profile(m: pd.DataFrame) -> dict[str, pd.Series]:
    """
    VP — where the traded volume actually sits.

    The point of control is the price level that absorbed the most volume in the
    window. Price tends to be attracted to it and to react at its edges, which is
    information the plain return and volume features cannot express.

    Levels are bucketed in units of the recent average true range so the bucket
    size adapts to volatility instead of being a fixed dollar amount.
    """
    close = m["close"]
    atr = _atr(m, 60)
    bucket = (atr * 0.5).replace(0.0, np.nan)
    lvl = (close / bucket).round()

    out = {}
    for w in VP_WINDOWS:
        # rolling mode of the level, weighted by traded volume
        vol = m["volume"]
        df = pd.DataFrame({"lvl": lvl, "vol": vol})

        def poc(block: pd.DataFrame) -> float:
            g = block.groupby("lvl")["vol"].sum()
            return float(g.idxmax()) if len(g) else np.nan

        # an exact rolling groupby is far too slow over a million rows, so the
        # window is stepped hourly and forward filled between steps
        step = 60
        idx = np.arange(0, len(df), step)
        vals = np.full(len(df), np.nan)
        for i in idx:
            lo = max(0, i - w)
            if i - lo < w // 2:
                continue
            vals[i] = poc(df.iloc[lo:i])
        poc_lvl = pd.Series(vals, index=df.index).ffill()

        poc_price = poc_lvl * bucket
        out[f"VP_dist_poc_{w}"] = _safe(close - poc_price, atr)
        # dispersion of volume across levels: concentrated vs spread out
        out[f"VP_conc_{w}"] = (
            m["volume"].rolling(w, min_periods=w // 2).max()
            / m["volume"].rolling(w, min_periods=w // 2).sum()
        )
    return out


def candles(m: pd.DataFrame) -> dict[str, pd.Series]:
    """CDL — the shape of the aggregated candle: body, wicks, and streaks."""
    out = {}
    for w in CDL_WINDOWS:
        o = m["open"].shift(w - 1)
        c = m["close"]
        hi = m["high"].rolling(w, min_periods=1).max()
        lo = m["low"].rolling(w, min_periods=1).min()
        rng = (hi - lo).replace(0.0, np.nan)
        body = c - o
        out[f"CDL_body_{w}"] = body / rng
        # a long upper wick means buyers were rejected up there
        out[f"CDL_upper_wick_{w}"] = (hi - np.maximum(o, c)) / rng
        out[f"CDL_lower_wick_{w}"] = (np.minimum(o, c) - lo) / rng
        out[f"CDL_range_rel_{w}"] = _safe(hi - lo, c)

    # how many consecutive minutes have closed in the same direction
    sign = np.sign(m["close"].diff()).fillna(0.0)
    grp = (sign != sign.shift(1)).cumsum()
    out["CDL_streak"] = sign * sign.groupby(grp).cumcount().add(1)
    return out


def book_shape(m: pd.DataFrame) -> dict[str, pd.Series]:
    """BOOK — geometry of the resting liquidity, beyond simple imbalance."""
    out = {}
    # the ±0.2 % level exists only from late 2025, so picking it as the "near"
    # level would make every feature here mostly empty and get them all dropped.
    # Choose the closest level that is actually populated across the history.
    tags = [t for t in ["0p2", "1p0", "2p0", "5p0"]
            if f"bid_notional_{t}" in m.columns
            and m[f"bid_notional_{t}"].isna().mean() < 0.05]
    if len(tags) < 2:
        return out

    near, far = tags[0], tags[-1]
    bid_n, ask_n = m[f"bid_notional_{near}"], m[f"ask_notional_{near}"]
    bid_f, ask_f = m[f"bid_notional_{far}"], m[f"ask_notional_{far}"]

    # how quickly liquidity thickens away from the price: a steep book means a
    # thin surface that gaps easily, a flat book absorbs size
    out["BOOK_bid_slope"] = _safe(bid_f, bid_n)
    out["BOOK_ask_slope"] = _safe(ask_f, ask_n)
    out["BOOK_slope_asym"] = out["BOOK_bid_slope"] - out["BOOK_ask_slope"]

    total = bid_n + ask_n
    out["BOOK_near_depletion"] = _safe(
        total, total.rolling(240, min_periods=120).mean()
    )
    # is the imbalance building or fading
    imb = _safe(bid_n - ask_n, bid_n + ask_n)
    for w in [15, 60]:
        out[f"BOOK_imb_trend_{w}"] = imb - imb.rolling(w, min_periods=w // 2).mean()
    return out


FAMILIES = {
    "IND": indicators,
    "SR": support_resistance,
    "VP": volume_profile,
    "CDL": candles,
    "BOOK": book_shape,
}


def build_extra(m: pd.DataFrame, families: list[str] | None = None) -> pd.DataFrame:
    """Compute the requested families on a 1-minute frame."""
    names = families or list(FAMILIES)
    cols: dict[str, pd.Series] = {}
    for n in names:
        cols.update(FAMILIES[n](m))
    out = pd.DataFrame(cols, index=m.index)
    return out.astype("float32")


def dataset(symbol: str, horizon: int = 15,
            families: list[str] | None = None) -> pd.DataFrame:
    """
    The original dataset plus the extra families, sampled on the same grid and
    subject to the same usability mask, so the only difference is the columns.
    """
    base = FE.build_features(symbol, horizon=horizon)
    m = FE.build_minute_frame(symbol)
    extra = build_extra(m, families)
    joined = base.join(extra, how="left")

    step_ok = ((joined.index.hour * 60 + joined.index.minute) % horizon) == 0 \
        if horizon < 60 else \
        (joined.index.minute == 0) & (joined.index.hour % max(1, horizon // 60) == 0)
    joined = joined[step_ok & joined["usable"]].drop(columns=["usable"])
    joined = joined.replace([np.inf, -np.inf], np.nan)
    keep = joined.dropna(subset=["fwd_ret", "close", "sigma_60"])
    good = [c for c in keep.columns if keep[c].isna().mean() < 0.05]
    return keep[good].dropna()


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FE.FEATURE_EXCLUDE]


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    d = dataset(sym, horizon=15)
    cols = feature_names(d)
    by_fam: dict[str, int] = {}
    for c in cols:
        fam = c.split("_")[0] if c.split("_")[0] in FAMILIES else "base"
        by_fam[fam] = by_fam.get(fam, 0) + 1
    print(f"{sym}: {len(d):,} samples, {len(cols)} features")
    for k, v in sorted(by_fam.items()):
        print(f"   {k:<6} {v}")
