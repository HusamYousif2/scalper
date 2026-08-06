"""
indicators.py — the full technical suite, computed server-side.

Why server-side rather than in the charting library: the free tier of a hosted
charting product limits how many studies you may apply. Computing them here
removes the limit entirely — the browser receives finished series and only has to
draw them. It also means the custom studies below, which no charting package
ships, sit alongside the classics as equals.

Two families are produced.

CLASSIC — what every trader already knows how to read:
    moving averages (SMA/EMA ribbons), VWAP, Bollinger Bands, Keltner Channels,
    Donchian Channels, SuperTrend, Ichimoku, Parabolic SAR, RSI, Stochastic,
    Stochastic RSI, MACD, ADX/DMI, CCI, Williams %R, MFI, ATR, OBV, ROC,
    volume with its moving average.

DESK — built from data a candlestick chart does not contain, and which the
exchange only exposes through the raw trade tape and the futures endpoints:
    cumulative volume delta, aggressor imbalance, large-print (whale) flow,
    absorption, trade intensity, open-interest change, crowd positioning, and
    the model's own volatility forecast plotted as a band around price.

Everything is causal. No series uses a value it could not have known at the time.
"""

import numpy as np
import pandas as pd

EPS = 1e-12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing, used by RSI, ATR and ADX."""
    return s.ewm(alpha=1 / n, adjust=False).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"],
                      (df["high"] - prev).abs(),
                      (df["low"] - prev).abs()], axis=1).max(axis=1)


# --------------------------------------------------------------------------- #
# classic studies
# --------------------------------------------------------------------------- #
def moving_averages(df: pd.DataFrame) -> dict:
    c = df["close"]
    return {
        "ema9": _ema(c, 9), "ema21": _ema(c, 21), "ema50": _ema(c, 50),
        "ema200": _ema(c, 200),
        "sma20": c.rolling(20, min_periods=5).mean(),
        "sma50": c.rolling(50, min_periods=10).mean(),
    }


def vwap(df: pd.DataFrame) -> dict:
    """Session VWAP, reset each UTC day, with one standard-deviation bands."""
    day = df.index.floor("D")
    pv = (df["close"] * df["volume"]).groupby(day).cumsum()
    vv = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    vw = pv / vv
    dev = (df["close"] - vw) ** 2
    var = (dev * df["volume"]).groupby(day).cumsum() / vv
    sd = np.sqrt(var)
    return {"vwap": vw, "vwap_up": vw + sd, "vwap_dn": vw - sd}


def bollinger(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> dict:
    c = df["close"]
    mid = c.rolling(n, min_periods=n // 2).mean()
    sd = c.rolling(n, min_periods=n // 2).std()
    return {"bb_mid": mid, "bb_up": mid + k * sd, "bb_dn": mid - k * sd,
            "bb_width": (2 * k * sd) / mid.replace(0, np.nan) * 100}


def keltner(df: pd.DataFrame, n: int = 20, k: float = 1.5) -> dict:
    mid = _ema(df["close"], n)
    atr = _rma(_true_range(df), n)
    return {"kc_mid": mid, "kc_up": mid + k * atr, "kc_dn": mid - k * atr}


def donchian(df: pd.DataFrame, n: int = 20) -> dict:
    up = df["high"].rolling(n, min_periods=n // 2).max()
    dn = df["low"].rolling(n, min_periods=n // 2).min()
    return {"dc_up": up, "dc_dn": dn, "dc_mid": (up + dn) / 2}


def supertrend(df: pd.DataFrame, n: int = 10, k: float = 3.0) -> dict:
    atr = _rma(_true_range(df), n)
    hl2 = (df["high"] + df["low"]) / 2
    upper, lower = hl2 + k * atr, hl2 - k * atr
    # .to_numpy() can return a read-only view; copy so the trailing logic below
    # can write back into these arrays
    c = df["close"].to_numpy()
    up, lo = upper.to_numpy().copy(), lower.to_numpy().copy()
    trend = np.ones(len(df))
    line = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        if np.isnan(up[i]) or np.isnan(lo[i]):
            continue
        up[i] = min(up[i], up[i - 1]) if c[i - 1] <= up[i - 1] else up[i]
        lo[i] = max(lo[i], lo[i - 1]) if c[i - 1] >= lo[i - 1] else lo[i]
        trend[i] = 1 if c[i] > up[i - 1] else -1 if c[i] < lo[i - 1] else trend[i - 1]
        line[i] = lo[i] if trend[i] > 0 else up[i]
    return {"supertrend": pd.Series(line, index=df.index),
            "supertrend_dir": pd.Series(trend, index=df.index)}


def ichimoku(df: pd.DataFrame) -> dict:
    def mid(n):
        return (df["high"].rolling(n, min_periods=n // 2).max()
                + df["low"].rolling(n, min_periods=n // 2).min()) / 2
    tenkan, kijun = mid(9), mid(26)
    return {
        "tenkan": tenkan, "kijun": kijun,
        "senkou_a": ((tenkan + kijun) / 2).shift(26),
        "senkou_b": mid(52).shift(26),
    }


def psar(df: pd.DataFrame, step: float = 0.02, cap: float = 0.2) -> dict:
    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 3:
        return {"psar": pd.Series(out, index=df.index)}
    bull, af, ep, sar = True, step, h[0], l[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if bull:
            sar = min(sar, l[i - 1], l[max(0, i - 2)])
            if l[i] < sar:
                bull, sar, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, cap)
        else:
            sar = max(sar, h[i - 1], h[max(0, i - 2)])
            if h[i] > sar:
                bull, sar, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, cap)
        out[i] = sar
    return {"psar": pd.Series(out, index=df.index)}


def rsi(df: pd.DataFrame, n: int = 14) -> dict:
    d = df["close"].diff()
    up = _rma(d.clip(lower=0), n)
    dn = _rma(-d.clip(upper=0), n)
    return {"rsi": 100 - 100 / (1 + up / dn.replace(0, np.nan))}


def stochastic(df: pd.DataFrame, n: int = 14, d: int = 3) -> dict:
    hi = df["high"].rolling(n, min_periods=n // 2).max()
    lo = df["low"].rolling(n, min_periods=n // 2).min()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return {"stoch_k": k, "stoch_d": k.rolling(d, min_periods=1).mean()}


def stoch_rsi(df: pd.DataFrame, n: int = 14) -> dict:
    r = rsi(df, n)["rsi"]
    lo = r.rolling(n, min_periods=n // 2).min()
    hi = r.rolling(n, min_periods=n // 2).max()
    k = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    return {"stochrsi_k": k, "stochrsi_d": k.rolling(3, min_periods=1).mean()}


def macd(df: pd.DataFrame) -> dict:
    line = _ema(df["close"], 12) - _ema(df["close"], 26)
    sig = _ema(line, 9)
    return {"macd": line, "macd_signal": sig, "macd_hist": line - sig}


def adx(df: pd.DataFrame, n: int = 14) -> dict:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _rma(_true_range(df), n)
    pdi = 100 * _rma(pd.Series(plus, index=df.index), n) / tr.replace(0, np.nan)
    mdi = 100 * _rma(pd.Series(minus, index=df.index), n) / tr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return {"adx": _rma(dx, n), "di_plus": pdi, "di_minus": mdi}


def cci(df: pd.DataFrame, n: int = 20) -> dict:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(n, min_periods=n // 2).mean()
    md = (tp - ma).abs().rolling(n, min_periods=n // 2).mean()
    return {"cci": (tp - ma) / (0.015 * md.replace(0, np.nan))}


def williams_r(df: pd.DataFrame, n: int = 14) -> dict:
    hi = df["high"].rolling(n, min_periods=n // 2).max()
    lo = df["low"].rolling(n, min_periods=n // 2).min()
    return {"williams_r": -100 * (hi - df["close"]) / (hi - lo).replace(0, np.nan)}


def mfi(df: pd.DataFrame, n: int = 14) -> dict:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    pos = flow.where(tp > tp.shift(1), 0.0).rolling(n, min_periods=n // 2).sum()
    neg = flow.where(tp < tp.shift(1), 0.0).rolling(n, min_periods=n // 2).sum()
    return {"mfi": 100 - 100 / (1 + pos / neg.replace(0, np.nan))}


def atr(df: pd.DataFrame, n: int = 14) -> dict:
    a = _rma(_true_range(df), n)
    return {"atr": a, "atr_pct": a / df["close"].replace(0, np.nan) * 100}


def obv(df: pd.DataFrame) -> dict:
    sign = np.sign(df["close"].diff()).fillna(0.0)
    return {"obv": (sign * df["volume"]).cumsum()}


def roc(df: pd.DataFrame, n: int = 12) -> dict:
    return {"roc": df["close"].pct_change(n) * 100}


# --------------------------------------------------------------------------- #
# desk studies — these need the trade tape and the futures endpoints
# --------------------------------------------------------------------------- #
def cvd(df: pd.DataFrame) -> dict:
    """Cumulative volume delta: running total of aggressive buys minus sells."""
    if "buy_qty" not in df.columns:
        return {}
    delta = df["buy_qty"].fillna(0) - df["sell_qty"].fillna(0)
    return {"cvd": delta.cumsum(), "delta": delta}


def aggressor(df: pd.DataFrame, n: int = 14) -> dict:
    if "buy_qty" not in df.columns:
        return {}
    b = df["buy_qty"].rolling(n, min_periods=1).sum()
    s = df["sell_qty"].rolling(n, min_periods=1).sum()
    return {"aggressor": (b - s) / (b + s).replace(0, np.nan)}


def whale_flow(df: pd.DataFrame, n: int = 14) -> dict:
    if "whale_buy_qty" not in df.columns:
        return {}
    b = df["whale_buy_qty"].fillna(0).rolling(n, min_periods=1).sum()
    s = df["whale_sell_qty"].fillna(0).rolling(n, min_periods=1).sum()
    tot = (b + s).replace(0, np.nan)
    return {"whale_flow": (b - s) / tot,
            "whale_share": tot / df["volume"].rolling(n, min_periods=1)
                              .sum().replace(0, np.nan)}


def absorption(df: pd.DataFrame, n: int = 14) -> dict:
    """
    Price movement per unit of one-sided aggression.

    A low reading means heavy one-way pressure that price refused to follow —
    someone large is filling the other side from resting orders.
    """
    if "buy_qty" not in df.columns:
        return {}
    b = df["buy_qty"].rolling(n, min_periods=1).sum()
    s = df["sell_qty"].rolling(n, min_periods=1).sum()
    imb = ((b - s) / (b + s).replace(0, np.nan)).abs()
    move = (np.log(df["close"] / df["close"].shift(n))).abs() * 1e4
    raw = move / (imb + 0.05)
    z = (raw - raw.rolling(240, min_periods=60).mean()) / \
        raw.rolling(240, min_periods=60).std().replace(0, np.nan)
    return {"absorption": -z}          # positive = absorption happening


def intensity(df: pd.DataFrame, n: int = 14) -> dict:
    if "n_trades" not in df.columns:
        return {}
    fast = df["n_trades"].rolling(n, min_periods=1).mean()
    slow = df["n_trades"].rolling(240, min_periods=60).mean()
    return {"intensity": fast / slow.replace(0, np.nan)}


def open_interest(df: pd.DataFrame, n: int = 15) -> dict:
    if "sum_open_interest" not in df.columns:
        return {}
    oi = df["sum_open_interest"].ffill()
    return {"oi": oi, "oi_chg": oi.pct_change(n) * 100}


def crowd(df: pd.DataFrame) -> dict:
    out = {}
    if "count_long_short_ratio" in df.columns:
        out["retail_ls"] = pd.to_numeric(df["count_long_short_ratio"],
                                         errors="coerce").ffill()
    if "sum_toptrader_long_short_ratio" in df.columns:
        out["top_ls"] = pd.to_numeric(df["sum_toptrader_long_short_ratio"],
                                      errors="coerce").ffill()
    if "retail_ls" in out and "top_ls" in out:
        out["crowd_gap"] = np.log(out["retail_ls"].clip(lower=EPS)) - \
                           np.log(out["top_ls"].clip(lower=EPS))
    return out


CLASSIC = {
    "moving_averages": moving_averages, "vwap": vwap, "bollinger": bollinger,
    "keltner": keltner, "donchian": donchian, "supertrend": supertrend,
    "ichimoku": ichimoku, "psar": psar, "rsi": rsi, "stochastic": stochastic,
    "stoch_rsi": stoch_rsi, "macd": macd, "adx": adx, "cci": cci,
    "williams_r": williams_r, "mfi": mfi, "atr": atr, "obv": obv, "roc": roc,
}
DESK = {
    "cvd": cvd, "aggressor": aggressor, "whale_flow": whale_flow,
    "absorption": absorption, "intensity": intensity,
    "open_interest": open_interest, "crowd": crowd,
}


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """Every study, aligned to the candle index that was passed in."""
    out = {}
    for name, fn in {**CLASSIC, **DESK}.items():
        try:
            out.update(fn(df))
        except Exception:
            continue
    res = pd.DataFrame(out, index=df.index)
    return res.replace([np.inf, -np.inf], np.nan)
