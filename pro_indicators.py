"""
pro_indicators.py — the specific study set of a discretionary trend/volatility
setup, kept separate from the live-chart indicators so nothing there is disturbed.

  T3 (Tillson)            — trend permission
  Range Filter (Donovan)  — trend/entry filter
  Squeeze Momentum (TTM)  — momentum + squeeze
  Change of Volatility    — volatility-expansion edge (approximated)
  Chaikin Volatility      — volatility-expansion edge
  ADX / DMI               — trend strength + direction
  ATR                     — stop / trail distance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import _ema, _rma, _true_range


def t3(close: pd.Series, n: int = 8, v: float = 0.7) -> pd.Series:
    e1 = _ema(close, n); e2 = _ema(e1, n); e3 = _ema(e2, n)
    e4 = _ema(e3, n); e5 = _ema(e4, n); e6 = _ema(e5, n)
    a = v
    c1 = -a ** 3
    c2 = 3 * a ** 2 + 3 * a ** 3
    c3 = -6 * a ** 2 - 3 * a - 3 * a ** 3
    c4 = 1 + 3 * a + a ** 3 + 3 * a ** 2
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def range_filter(src: pd.Series, period: int = 50, mult: float = 3.0):
    """Returns (filter line, direction ∈ {-1,0,+1})."""
    x = src.to_numpy(dtype=float)
    d = np.abs(np.diff(x, prepend=x[0]))
    avrng = _ema(pd.Series(d, index=src.index), period).to_numpy()
    smrng = _ema(pd.Series(avrng, index=src.index), period * 2 - 1).to_numpy() * mult

    filt = np.empty_like(x)
    filt[0] = x[0]
    for i in range(1, len(x)):
        prev, s, r = filt[i - 1], x[i], smrng[i]
        if s > prev:
            filt[i] = prev if (s - r) < prev else (s - r)
        else:
            filt[i] = prev if (s + r) > prev else (s + r)
    fdir = np.sign(np.diff(filt, prepend=filt[0]))
    return pd.Series(filt, index=src.index), pd.Series(fdir, index=src.index)


def _linreg(y: pd.Series, n: int) -> pd.Series:
    x = np.arange(n)
    xm = x.mean()
    denom = ((x - xm) ** 2).sum()

    def f(w):
        ym = w.mean()
        slope = ((x - xm) * (w - ym)).sum() / denom
        return ym + slope * ((n - 1) - xm)

    return y.rolling(n).apply(f, raw=True)


def squeeze_momentum(df: pd.DataFrame, n: int = 16, bb: float = 2.0, kc: float = 1.5):
    """LazyBear/TTM. Returns (momentum value, squeeze_on bool series)."""
    c = df["close"]
    basis = c.rolling(n).mean()
    dev = bb * c.rolling(n).std(ddof=0)
    ubb, lbb = basis + dev, basis - dev

    rangema = _true_range(df).rolling(n).mean()
    ma = c.rolling(n).mean()
    ukc, lkc = ma + rangema * kc, ma - rangema * kc

    sqz_on = (lbb > lkc) & (ubb < ukc)

    hh = df["high"].rolling(n).max()
    ll = df["low"].rolling(n).min()
    val_src = c - ((hh + ll) / 2 + ma) / 2
    val = _linreg(val_src, n)
    return val, sqz_on


def change_of_vol(df: pd.DataFrame, fast: int = 6, slow: int = 100, ema: int = 55) -> pd.Series:
    """Approximation: short-window return volatility relative to its own longer
    baseline, smoothed. > 0 means volatility is expanding — 'the edge'."""
    ret = df["close"].pct_change()
    volf = ret.rolling(fast).std(ddof=0)
    base = volf.rolling(slow).mean()
    cov = volf / base.replace(0, np.nan) - 1.0
    return _ema(cov.fillna(0.0), ema)


def chaikin_vol(df: pd.DataFrame, ema: int = 10, roc: int = 12) -> pd.Series:
    hl = df["high"] - df["low"]
    e = _ema(hl, ema)
    return (e - e.shift(roc)) / e.shift(roc).replace(0, np.nan) * 100


def dmi(df: pd.DataFrame, n: int = 10):
    """Returns (adx, +DI, -DI). Same math as the live ADX, length configurable."""
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = _true_range(df)
    atr = _rma(tr, n)
    pdi = 100 * _rma(pd.Series(plus, index=df.index), n) / atr.replace(0, np.nan)
    mdi = 100 * _rma(pd.Series(minus, index=df.index), n) / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return _rma(dx, n), pdi, mdi


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _rma(_true_range(df), n)


def compute(df: pd.DataFrame,
            t3_n=8, t3_v=0.7, rf_period=50, rf_mult=3.0,
            sq_n=16, sq_bb=2.0, sq_kc=1.5, dmi_n=10, atr_n=14) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["t3"] = t3(df["close"], t3_n, t3_v)
    rf, fdir = range_filter(df["close"], rf_period, rf_mult)
    out["rf"], out["rf_dir"] = rf, fdir
    val, sqz = squeeze_momentum(df, sq_n, sq_bb, sq_kc)
    out["sq_val"], out["sq_on"] = val, sqz.astype(float)
    out["cov"] = change_of_vol(df)
    out["cvol"] = chaikin_vol(df)
    adx, pdi, mdi = dmi(df, dmi_n)
    out["adx"], out["pdi"], out["mdi"] = adx, pdi, mdi
    out["atr"] = atr(df, atr_n)

    # candidate confirmation indicators (tested as optional gates; kept only if
    # they improve out-of-sample results)
    c = df["close"]
    out["ema200"] = _ema(c, 200)
    delta = c.diff()
    gain = _rma(delta.clip(lower=0), 14)
    loss = _rma(-delta.clip(upper=0), 14)
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["vol_sma"] = df["volume"].rolling(20).mean()
    macd = _ema(c, 12) - _ema(c, 26)
    out["macd_hist"] = macd - _ema(macd, 9)
    return out.replace([np.inf, -np.inf], np.nan)
