"""
indicator_ai.py — scans the ENTIRE indicator suite (≈30 classic + order-flow
studies) and surfaces the strongest opportunities right now.

For each indicator it derives a current directional signal (long / short) and a
0–1 strength, then aggregates them into a consensus bias and the top-N indicators
firing this bar, with ATR-based entry / stop / target. A decision-support view
over the whole toolbox — refresh it on a timer to watch opportunities move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import indicators as IND
import live_data as LD

STOP_ATR = 2.0
TARGET_R = 2.0

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
        "buy_qty": "sum", "sell_qty": "sum", "whale_buy_qty": "sum", "whale_sell_qty": "sum",
        "n_trades": "sum", "sum_open_interest": "last",
        "bid_notional_1p0": "last", "ask_notional_1p0": "last"}


def _frame(symbol, tf, minute_df=None):
    if minute_df is None:
        minute_df = LD.load_recent_archive(symbol, 25)
    agg = {k: v for k, v in _AGG.items() if k in minute_df.columns}
    c = minute_df.resample(f"{tf}min").agg(agg).dropna(subset=["open", "close"])
    return c


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _signals(cur, prev, close, atr):
    """Return a list of {name, dir, strength, reason} for indicators firing now."""
    out = []
    g = lambda k: (float(cur[k]) if k in cur and not pd.isna(cur[k]) else None)
    gp = lambda k: (float(prev[k]) if k in prev and not pd.isna(prev[k]) else None)
    a = atr or (close * 0.01)

    def add(name, d, s, reason):
        if d and s > 0.02:
            out.append({"name": name, "dir": "long" if d > 0 else "short",
                        "strength": round(_clip(s), 3), "reason": reason})

    rsi = g("rsi")
    if rsi is not None:
        if rsi < 40: add("RSI", +1, (40 - rsi) / 40, f"RSI {rsi:.0f} — oversold, turning up")
        elif rsi > 60: add("RSI", -1, (rsi - 60) / 40, f"RSI {rsi:.0f} — overbought, turning down")

    k = g("stoch_k")
    if k is not None:
        if k < 25: add("Stochastic", +1, (25 - k) / 25, f"Stoch {k:.0f} — oversold")
        elif k > 75: add("Stochastic", -1, (k - 75) / 25, f"Stoch {k:.0f} — overbought")

    sk = g("stochrsi_k")
    if sk is not None:
        if sk < 20: add("Stoch RSI", +1, (20 - sk) / 20, "Stoch-RSI oversold")
        elif sk > 80: add("Stoch RSI", -1, (sk - 80) / 20, "Stoch-RSI overbought")

    mh, mhp = g("macd_hist"), gp("macd_hist")
    if mh is not None:
        add("MACD momentum", np.sign(mh), _clip(abs(mh) / (0.4 * a)),
            f"MACD histogram {'positive' if mh > 0 else 'negative'}")
    md, ms = g("macd"), g("macd_signal")
    mdp, msp = gp("macd"), gp("macd_signal")
    if None not in (md, ms, mdp, msp):
        if md > ms and mdp <= msp: add("MACD cross", +1, 0.85, "MACD crossed above signal")
        elif md < ms and mdp >= msp: add("MACD cross", -1, 0.85, "MACD crossed below signal")

    st, stp = g("supertrend_dir"), gp("supertrend_dir")
    if st is not None:
        fresh = (stp is not None and st != stp)
        add("SuperTrend", np.sign(st), 0.9 if fresh else 0.55,
            "SuperTrend flipped " + ("up" if st > 0 else "down") if fresh
            else "SuperTrend " + ("bullish" if st > 0 else "bearish"))

    adx, dp, dm = g("adx"), g("di_plus"), g("di_minus")
    if None not in (adx, dp, dm) and adx > 18:
        s = _clip((adx - 18) / 22)
        if dp > dm: add("ADX / DMI", +1, s, f"ADX {adx:.0f}, +DI leading — strong uptrend")
        else: add("ADX / DMI", -1, s, f"ADX {adx:.0f}, −DI leading — strong downtrend")

    e50, e200 = g("ema50"), g("ema200")
    if None not in (e50, e200):
        if close > e50 > e200: add("EMA trend", +1, 0.5, "Price above EMA50 above EMA200")
        elif close < e50 < e200: add("EMA trend", -1, 0.5, "Price below EMA50 below EMA200")

    vwap = g("vwap")
    if vwap is not None:
        add("VWAP", np.sign(close - vwap), _clip(abs(close - vwap) / (1.2 * a)),
            "Price " + ("above" if close > vwap else "below") + " VWAP")

    bbu, bbd, bbm = g("bb_up"), g("bb_dn"), g("bb_mid")
    if None not in (bbu, bbd, bbm) and bbu > bbd:
        pctb = (close - bbd) / (bbu - bbd)
        if close > bbu: add("Bollinger", +1, 0.7, "Breakout above upper band")
        elif close < bbd: add("Bollinger", -1, 0.7, "Breakdown below lower band")
        elif pctb < 0.15: add("Bollinger", +1, 0.4, "Tagging lower band — mean-revert up")
        elif pctb > 0.85: add("Bollinger", -1, 0.4, "Tagging upper band — mean-revert down")

    cci = g("cci")
    if cci is not None:
        if cci < -100: add("CCI", +1, _clip((-cci - 100) / 150), f"CCI {cci:.0f} oversold")
        elif cci > 100: add("CCI", -1, _clip((cci - 100) / 150), f"CCI {cci:.0f} overbought")

    wr = g("williams_r")
    if wr is not None:
        if wr < -80: add("Williams %R", +1, (-80 - wr) / 20, "Williams %R oversold")
        elif wr > -20: add("Williams %R", -1, (wr + 20) / 20, "Williams %R overbought")

    mfi = g("mfi")
    if mfi is not None:
        if mfi < 25: add("Money Flow", +1, (25 - mfi) / 25, f"MFI {mfi:.0f} — money leaving is exhausted")
        elif mfi > 75: add("Money Flow", -1, (mfi - 75) / 25, f"MFI {mfi:.0f} — overbought money flow")

    roc = g("roc")
    if roc is not None:
        add("Rate of Change", np.sign(roc), _clip(abs(roc) / 3), f"ROC {roc:+.1f}%")

    psar = g("psar")
    if psar is not None:
        add("Parabolic SAR", np.sign(close - psar), 0.5,
            "SAR " + ("below price — bullish" if psar < close else "above price — bearish"))

    dcu, dcd = g("dc_up"), g("dc_dn")
    if None not in (dcu, dcd):
        if close >= dcu * 0.999: add("Donchian", +1, 0.75, "New breakout high")
        elif close <= dcd * 1.001: add("Donchian", -1, 0.75, "New breakdown low")

    sa, sb = g("senkou_a"), g("senkou_b")
    if None not in (sa, sb):
        top, bot = max(sa, sb), min(sa, sb)
        if close > top: add("Ichimoku cloud", +1, 0.55, "Price above the cloud")
        elif close < bot: add("Ichimoku cloud", -1, 0.55, "Price below the cloud")

    tk, kj = g("tenkan"), g("kijun")
    if None not in (tk, kj):
        add("Ichimoku TK cross", np.sign(tk - kj), 0.45,
            "Tenkan " + ("above" if tk > kj else "below") + " Kijun")

    cvd, cvdp = g("cvd"), gp("cvd")
    if None not in (cvd, cvdp):
        add("Cumulative Delta", np.sign(cvd - cvdp), 0.5,
            "CVD " + ("rising — buyers pressing" if cvd > cvdp else "falling — sellers pressing"))

    agg = g("aggressor")
    if agg is not None:
        add("Aggressor flow", np.sign(agg), _clip(abs(agg)), "Aggressive " + ("buyers" if agg > 0 else "sellers") + " dominate")

    wf = g("whale_flow")
    if wf is not None:
        add("Whale flow", np.sign(wf), _clip(abs(wf)), "Whales net " + ("buying" if wf > 0 else "selling"))

    oic = g("oi_chg")
    if oic is not None and prev is not None:
        pc = close - float(prev.get("close", close)) if hasattr(prev, "get") else 0
        if oic > 0 and pc != 0:
            add("Open interest", np.sign(pc), 0.5,
                "Rising OI with price " + ("up — new longs" if pc > 0 else "down — new shorts"))

    obv, obvp = g("obv"), gp("obv")
    if None not in (obv, obvp):
        add("On-Balance Volume", np.sign(obv - obvp), 0.4,
            "OBV " + ("rising" if obv > obvp else "falling"))

    return out


def scan(symbol, tf, minute_df=None):
    c = _frame(symbol, tf, minute_df)
    ind = IND.compute_all(c)
    i = len(ind) - 1
    cur, prev = ind.iloc[i], (ind.iloc[i - 1] if i > 0 else ind.iloc[i])
    # carry close onto the prev row for the OI rule
    prev = dict(prev)
    prev["close"] = float(c["close"].iloc[i - 1]) if i > 0 else float(c["close"].iloc[i])
    close = float(c["close"].iloc[i])
    atr = float(cur["atr"]) if "atr" in cur and not pd.isna(cur["atr"]) else close * 0.01

    sigs = _signals(cur, prev, close, atr)
    long_s = sum(s["strength"] for s in sigs if s["dir"] == "long")
    short_s = sum(s["strength"] for s in sigs if s["dir"] == "short")
    n_long = sum(1 for s in sigs if s["dir"] == "long")
    n_short = sum(1 for s in sigs if s["dir"] == "short")
    net = long_s - short_s
    total = long_s + short_s
    bias = "long" if net >= 0 else "short"
    up = bias == "long"
    consensus = round(100 * (max(long_s, short_s) / total), 0) if total > 0 else 0
    # score: how strong AND how one-sided the whole suite is right now
    score = round(_clip(abs(net) / 6) * 60 + (consensus / 100) * 40)

    risk = STOP_ATR * atr
    stop = close - risk if up else close + risk
    target = close + TARGET_R * risk if up else close - TARGET_R * risk

    # top signals: those agreeing with the consensus, strongest first
    aligned = sorted([s for s in sigs if s["dir"] == bias],
                     key=lambda s: s["strength"], reverse=True)
    top = [{"name": s["name"], "dir": s["dir"],
            "strength": round(s["strength"] * 100), "reason": s["reason"]}
           for s in aligned[:6]]

    rating = "Strong" if score >= 70 else "Building" if score >= 45 else "Watch"
    return {
        "symbol": symbol, "tf": tf, "as_of": int(c.index[i].timestamp()),
        "bias": bias, "score": min(100, score), "rating": rating,
        "consensus_pct": consensus, "n_long": n_long, "n_short": n_short,
        "n_total": len(sigs),
        "entry": round(close, 6), "stop": round(stop, 6), "target": round(target, 6),
        "atr_pct": round(atr / close * 100, 3),
        "top": top,
    }
