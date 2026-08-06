"""
features.py — turn the 1-minute microstructure table into an hourly supervised
learning matrix.

Every feature at timestamp t is computed from data at or before t only.
The label looks forward from t to t + HORIZON minutes.

Sampling is strictly hourly so that consecutive labels do NOT overlap. Overlapping
labels correlate the samples and make any out-of-sample result look more
significant than it is; that is one of the ways a backtest lies.
"""

import numpy as np
import pandas as pd

import ingest

HORIZON = 60          # predict this many minutes ahead
WINDOWS = [5, 15, 30, 60, 240]
BASELINE = 1440       # one day, used to normalise activity levels


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def _imbalance(pos: pd.Series, neg: pd.Series) -> pd.Series:
    """Signed imbalance in [-1, 1]: +1 all on the positive side, -1 all negative."""
    return _safe_ratio(pos - neg, pos + neg)


def build_minute_frame(symbol: str, minute: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Place a minute table on a gap-free grid, filling the gaps sensibly.

    `minute` lets a caller supply a frame it built itself — the live path stitches
    archive and REST data together and passes the result in here, so that live and
    historical features are computed by exactly the same code.
    """
    df = ingest.load_minutes(symbol) if minute is None else minute.copy()
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    df = df.reindex(full)
    df.index.name = "ts"

    # mark which minutes came from a real archive file. Days can be missing from
    # the archive; a rolling window that spans a hole would silently mix periods
    # that are hours or months apart, so those rows are dropped later.
    df["present"] = df["n_trades"].notna().astype(float)

    # positioning metrics are published every 5 minutes; carrying the last known
    # value forward uses only past information, so it introduces no lookahead
    oi_cols = [c for c in df.columns if c.startswith(("sum_", "count_"))]
    df[oi_cols] = df[oi_cols].ffill(limit=15)

    # a minute with no trades is a real event, not missing data
    for c in ["volume", "quote_volume", "n_trades", "buy_qty", "sell_qty",
              "buy_cnt", "sell_cnt", "whale_buy_qty", "whale_sell_qty",
              "realized_var"]:
        df[c] = df[c].fillna(0.0)
    df["close"] = df["close"].ffill()

    # only the price path needs full precision; halving everything else halves
    # the memory of every rolling window computed from it
    for c in df.columns:
        if c not in ("open", "high", "low", "close", "vwap"):
            df[c] = df[c].astype("float32")
    return df


def build_features(symbol: str, horizon: int = HORIZON,
                   minute: pd.DataFrame | None = None) -> pd.DataFrame:
    m = build_minute_frame(symbol, minute=minute)
    close = m["close"]
    f = pd.DataFrame(index=m.index)

    # ---- rolling sums we reuse across several features -------------------
    vol_base = m["volume"].rolling(BASELINE, min_periods=BASELINE // 2).sum()
    trd_base = m["n_trades"].rolling(BASELINE, min_periods=BASELINE // 2).sum()
    rv_base = m["realized_var"].rolling(BASELINE, min_periods=BASELINE // 2).sum()

    for w in WINDOWS:
        mp = max(2, w // 2)
        buy = m["buy_qty"].rolling(w, min_periods=mp).sum()
        sell = m["sell_qty"].rolling(w, min_periods=mp).sum()
        bcnt = m["buy_cnt"].rolling(w, min_periods=mp).sum()
        scnt = m["sell_cnt"].rolling(w, min_periods=mp).sum()
        wbuy = m["whale_buy_qty"].rolling(w, min_periods=mp).sum()
        wsell = m["whale_sell_qty"].rolling(w, min_periods=mp).sum()
        vol = m["volume"].rolling(w, min_periods=mp).sum()
        qvol = m["quote_volume"].rolling(w, min_periods=mp).sum()
        trd = m["n_trades"].rolling(w, min_periods=mp).sum()
        rv = m["realized_var"].rolling(w, min_periods=mp).sum()

        # price move over the window, expressed in units of its own volatility
        ret = np.log(close / close.shift(w))
        f[f"ret_{w}"] = ret
        f[f"ret_z_{w}"] = _safe_ratio(ret, np.sqrt(rv))

        # who is hitting the book: taker buy vs taker sell aggression
        f[f"ofi_vol_{w}"] = _imbalance(buy, sell)
        f[f"ofi_cnt_{w}"] = _imbalance(bcnt, scnt)
        f[f"ofi_whale_{w}"] = _imbalance(wbuy, wsell)

        # activity relative to a normal day: is this window unusually busy?
        f[f"vol_rel_{w}"] = _safe_ratio(vol, vol_base) * (BASELINE / w)
        f[f"trd_rel_{w}"] = _safe_ratio(trd, trd_base) * (BASELINE / w)
        f[f"rv_rel_{w}"] = _safe_ratio(rv, rv_base) * (BASELINE / w)

        # average trade size: many small trades vs few large ones
        f[f"avg_trade_{w}"] = _safe_ratio(qvol, trd)

        # where price sits versus the window's volume weighted average price
        vwap_w = _safe_ratio(qvol, vol)
        f[f"vwap_dev_{w}"] = _safe_ratio(close - vwap_w, close)

    # ---- order book shape -------------------------------------------------
    for tag in ["0p2", "1p0", "2p0", "5p0"]:
        b, a = f"bid_notional_{tag}", f"ask_notional_{tag}"
        if b not in m.columns:
            continue
        imb = _imbalance(m[b], m[a])
        f[f"depth_imb_{tag}"] = imb
        f[f"depth_imb_chg60_{tag}"] = imb - imb.shift(60)
        total = m[b] + m[a]
        f[f"depth_rel_{tag}"] = _safe_ratio(
            total, total.rolling(BASELINE, min_periods=BASELINE // 2).mean()
        )

    # book pressure very close to the touch versus deep in the book
    if "depth_imb_0p2" in f and "depth_imb_5p0" in f:
        f["depth_imb_slope"] = f["depth_imb_0p2"] - f["depth_imb_5p0"]

    # ---- futures positioning ---------------------------------------------
    if "sum_open_interest" in m.columns:
        oi = m["sum_open_interest"]
        for w in [15, 60, 240]:
            oi_chg = np.log(_safe_ratio(oi, oi.shift(w)))
            f[f"oi_chg_{w}"] = oi_chg
            # rising price + rising OI = new longs; rising price + falling OI =
            # shorts covering. The interaction separates the two regimes.
            f[f"oi_x_ret_{w}"] = oi_chg * f[f"ret_{w}"]
        f["oi_rel"] = _safe_ratio(oi, oi.rolling(BASELINE, min_periods=BASELINE // 2).mean())

    for c in ["count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
              "count_long_short_ratio", "sum_taker_long_short_vol_ratio"]:
        if c in m.columns:
            s = pd.to_numeric(m[c], errors="coerce")
            short = c.replace("_long_short", "").replace("_ratio", "")
            f[f"{short}"] = np.log(s.clip(lower=1e-6))
            f[f"{short}_chg60"] = f[f"{short}"] - f[f"{short}"].shift(60)

    # ---- session clock ----------------------------------------------------
    hour = f.index.hour + f.index.minute / 60.0
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["dow"] = f.index.dayofweek.astype(float)

    # ---- forward label ----------------------------------------------------
    f["fwd_ret"] = np.log(close.shift(-horizon) / close)
    # highest and lowest price reached inside the label window, so a later step
    # can ask "was the move big enough to clear costs" rather than only its sign.
    # reversing the series turns a trailing rolling window into a forward one.
    fwd_high = close[::-1].rolling(horizon, min_periods=1).max()[::-1].shift(-1)
    fwd_low = close[::-1].rolling(horizon, min_periods=1).min()[::-1].shift(-1)
    f["fwd_up"] = np.log(fwd_high / close)
    f["fwd_dn"] = np.log(fwd_low / close)
    f["close"] = close
    # realised volatility of the last hour, used to scale the trading threshold
    f["sigma_60"] = np.sqrt(m["realized_var"].rolling(60, min_periods=30).sum())

    # a row is usable only if the whole trailing feature window AND the whole
    # forward label window come from days that actually exist in the archive
    # window ending at t covers the trailing day; shifting a window back by
    # HORIZON makes it cover the label period (t, t + HORIZON]
    past_ok = m["present"].rolling(BASELINE, min_periods=BASELINE).min()
    fwd_ok = m["present"].rolling(horizon, min_periods=horizon).min().shift(-horizon)
    usable = (past_ok.fillna(0) > 0) & (fwd_ok.fillna(0) > 0)

    # two years at minute resolution is over a million rows; float64 across ~90
    # columns is around 700 MB, which does not fit comfortably here. Every value
    # stored is a ratio or a small return, well inside float32 precision.
    keep_f64 = ["close", "fwd_ret", "fwd_up", "fwd_dn"]
    for c in f.columns:
        if c not in keep_f64:
            f[c] = f[c].astype("float32")
    f["usable"] = usable
    return f


def hourly_dataset(symbol: str, horizon: int = HORIZON,
                   overlap: bool = False) -> pd.DataFrame:
    """
    Sample the feature frame on the hour and drop incomplete rows.

    With overlap=False the sampling step equals the horizon, so no two labels
    share a minute. Overlapping labels are correlated and make a result look
    more statistically solid than it is.
    """
    f = build_features(symbol, horizon=horizon)
    if overlap:
        on_grid = pd.Series(True, index=f.index)
    elif horizon >= 60:
        step = max(1, horizon // 60)
        on_grid = (f.index.minute == 0) & (f.index.hour % step == 0)
    else:
        # sub-hourly horizons: one sample every `horizon` minutes
        on_grid = ((f.index.hour * 60 + f.index.minute) % horizon) == 0
    f = f[on_grid & f["usable"]].drop(columns=["usable"])
    f = f.replace([np.inf, -np.inf], np.nan)
    keep = f.dropna(subset=["fwd_ret", "fwd_up", "fwd_dn", "close", "sigma_60"])
    # drop feature columns that are mostly empty, then drop remaining bad rows
    good_cols = [c for c in keep.columns if keep[c].isna().mean() < 0.05]
    keep = keep[good_cols].dropna()
    return keep


FEATURE_EXCLUDE = {"fwd_ret", "fwd_up", "fwd_dn", "close", "sigma_60"}


def feature_names(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_EXCLUDE]


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    d = hourly_dataset(sym)
    print(f"{sym}: {d.shape[0]} hourly samples, {len(feature_names(d))} features")
    print(f"period: {d.index.min()} -> {d.index.max()}")
    print(f"fwd_ret std: {d['fwd_ret'].std():.5f}  mean: {d['fwd_ret'].mean():.6f}")
    print("\nfirst 20 feature names:")
    for c in feature_names(d)[:20]:
        print("  ", c)
