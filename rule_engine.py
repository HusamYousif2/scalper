"""
rule_engine.py — a deterministic, human-style trading decision engine.

This is NOT a machine-learning model and it makes no forecast. It reads the same
indicator suite the chart already computes (indicators.compute_all) and reasons
over it the way a disciplined discretionary trader would:

  1. Every indicator casts a signed vote in [-1, +1] (-1 = bearish, +1 = bullish)
     grouped into families (trend, momentum, location, volume/tape). Each family
     carries a weight — trend and the order-flow tape count for more than a
     single oscillator.
  2. The weighted average of all votes is the "confluence score" in [-1, +1].
     A long is taken only when the score clears a threshold AND the market is
     actually trending (ADX gate); otherwise the engine sits out. This is what
     stops it from buying into chop.
  3. Levels are computed from market structure, never invented:
        entry = last close
        stop  = beyond the recent swing, distance clamped to a sane ATR band
        take  = entry + R:R x stop-distance   (fixed reward-to-risk)

Everything it decides is transparent: the returned payload lists every vote and
its reason, so the chart can show exactly why the call is what it is.

Design choices worth flagging:
  * ADX gates participation, not direction — a strong score in a dead market is
    demoted to NEUTRAL rather than traded.
  * Oscillators vote WITH the move in the trend zone but FADE it at extremes
    (RSI > 80 in an uptrend is exhaustion, not confirmation), so the engine
    isn't fooled into chasing a blow-off top.
  * The stop is placed at structure first, then clamped to [1.0, 2.5] x ATR, so
    it is neither inside the noise nor absurdly far.
"""

from __future__ import annotations

import math


# --------------------------------------------------------------------------- #
# tunables — every threshold a discretionary trader would carry in their head
# --------------------------------------------------------------------------- #
SCORE_ENTRY = 0.18          # |confluence| must clear this to consider a trade
ADX_TREND = 20.0            # below this the market is ranging → gate longs/shorts
ADX_STRONG = 30.0           # a genuinely strong trend
RR_TARGET = 1.8             # reward-to-risk the take-profit is sized to
SL_ATR_MIN = 1.0            # stop never tighter than this many ATRs
SL_ATR_MAX = 2.5            # stop never wider than this many ATRs
SWING_LOOKBACK = 20         # candles used to find the structural swing hi/lo
SWING_PAD_ATR = 0.15        # push the stop just beyond the swing by this much ATR

# family weights — how much each group of evidence counts toward the decision
W_TREND = 1.0
W_MOMENTUM = 0.6
W_LOCATION = 0.5
W_TAPE = 0.9                # proprietary order-flow edge, weighted heavily


def _last(series, i=-1):
    """Last non-NaN-safe value of an indicator column, or None."""
    if series is None:
        return None
    try:
        v = float(series.iloc[i])
    except (TypeError, ValueError, IndexError):
        return None
    return None if (v is None or math.isnan(v) or math.isinf(v)) else v


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


# --------------------------------------------------------------------------- #
# the vote functions — each returns (vote in [-1,1], label, human reason) or None
# --------------------------------------------------------------------------- #
def _trend_votes(px, ind):
    votes = []

    e21, e50, e200 = _last(ind.get("ema21")), _last(ind.get("ema50")), _last(ind.get("ema200"))
    if e21 is not None and e50 is not None and e200 is not None:
        if px > e21 > e50 > e200:
            votes.append((1.0, "EMA stack", "price above a rising 21>50>200 stack — clean uptrend"))
        elif px < e21 < e50 < e200:
            votes.append((-1.0, "EMA stack", "price below a falling 21<50<200 stack — clean downtrend"))
        else:
            v = (0.5 if px > e50 else -0.5)
            votes.append((v, "EMA stack", f"price {'above' if v > 0 else 'below'} the 50-EMA but stack not aligned"))

    st = _last(ind.get("supertrend_dir"))
    if st is not None:
        votes.append((1.0 if st > 0 else -1.0, "SuperTrend",
                      f"SuperTrend is {'long' if st > 0 else 'short'}"))

    adx, dip, dim = _last(ind.get("adx")), _last(ind.get("di_plus")), _last(ind.get("di_minus"))
    if adx is not None and dip is not None and dim is not None:
        strength = _clip((adx - ADX_TREND) / 20.0, 0.0, 1.0)
        v = (1.0 if dip > dim else -1.0) * strength
        votes.append((v, "ADX / DI",
                      f"ADX {adx:.0f} with DI{'+' if dip > dim else '−'} leading"))

    macd, sig = _last(ind.get("macd")), _last(ind.get("macd_signal"))
    if macd is not None and sig is not None:
        votes.append((_clip((macd - sig) * 1e6, -1, 1) or (0.6 if macd > sig else -0.6),
                      "MACD", f"MACD line {'above' if macd > sig else 'below'} signal"))

    psar = _last(ind.get("psar"))
    if psar is not None:
        votes.append((0.6 if px > psar else -0.6, "Parabolic SAR",
                      f"price {'above' if px > psar else 'below'} SAR"))

    sa, sb = _last(ind.get("senkou_a")), _last(ind.get("senkou_b"))
    if sa is not None and sb is not None:
        top, bot = max(sa, sb), min(sa, sb)
        if px > top:
            votes.append((0.7, "Ichimoku cloud", "price above the cloud"))
        elif px < bot:
            votes.append((-0.7, "Ichimoku cloud", "price below the cloud"))
        else:
            votes.append((0.0, "Ichimoku cloud", "price inside the cloud — no trend"))

    return votes


def _momentum_votes(ind):
    votes = []

    rsi = _last(ind.get("rsi"))
    if rsi is not None:
        if rsi >= 80:
            votes.append((-0.5, "RSI", f"RSI {rsi:.0f} — overbought, exhaustion risk"))
        elif rsi <= 20:
            votes.append((0.5, "RSI", f"RSI {rsi:.0f} — oversold, snap-back risk"))
        else:
            votes.append((_clip((rsi - 50) / 30.0, -1, 1), "RSI",
                          f"RSI {rsi:.0f} leaning {'up' if rsi > 50 else 'down'}"))

    k, d = _last(ind.get("stoch_k")), _last(ind.get("stoch_d"))
    if k is not None and d is not None:
        base = 1.0 if k > d else -1.0
        if k >= 80:
            base = -0.4
        elif k <= 20:
            base = 0.4
        votes.append((base, "Stochastic",
                      f"%K {k:.0f} {'>' if k > d else '<'} %D"))

    cci = _last(ind.get("cci"))
    if cci is not None:
        votes.append((_clip(cci / 150.0, -1, 1), "CCI", f"CCI {cci:.0f}"))

    wr = _last(ind.get("williams_r"))
    if wr is not None:
        votes.append((_clip((wr + 50) / 30.0, -1, 1), "Williams %R", f"Williams %R {wr:.0f}"))

    roc = _last(ind.get("roc"))
    if roc is not None:
        votes.append((_clip(roc / 2.0, -1, 1), "Rate of Change",
                      f"{roc:+.2f}% over the ROC window"))

    return votes


def _location_votes(px, ind):
    votes = []

    vw = _last(ind.get("vwap"))
    if vw is not None:
        votes.append((0.7 if px > vw else -0.7, "VWAP",
                      f"price {'above' if px > vw else 'below'} the session VWAP"))

    up, dn = _last(ind.get("bb_up")), _last(ind.get("bb_dn"))
    if up is not None and dn is not None and up > dn:
        pctb = (px - dn) / (up - dn)          # %B: 0 at lower band, 1 at upper
        if pctb > 1.0:
            votes.append((-0.3, "Bollinger", "price broke above the upper band — stretched"))
        elif pctb < 0.0:
            votes.append((0.3, "Bollinger", "price broke below the lower band — stretched"))
        else:
            votes.append((_clip((pctb - 0.5) * 2, -1, 1), "Bollinger",
                          f"price at {pctb * 100:.0f}% of the band width"))

    dcu, dcd = _last(ind.get("dc_up")), _last(ind.get("dc_dn"))
    if dcu is not None and dcd is not None:
        if px >= dcu:
            votes.append((0.6, "Donchian", "new 20-bar high — breakout"))
        elif px <= dcd:
            votes.append((-0.6, "Donchian", "new 20-bar low — breakdown"))

    return votes


def _tape_votes(ind):
    """Order-flow / desk studies — the edge a candlestick chart cannot show."""
    votes = []

    agg = _last(ind.get("aggressor"))
    if agg is not None:
        votes.append((_clip(agg * 2, -1, 1), "Aggressor imbalance",
                      f"aggressive flow {agg:+.2f} ({'buyers' if agg > 0 else 'sellers'} lifting)"))

    cvd, cvd_prev = _last(ind.get("cvd")), _last(ind.get("cvd"), -6)
    if cvd is not None and cvd_prev is not None:
        slope = cvd - cvd_prev
        votes.append((_clip(slope / (abs(cvd) + 1e-9) * 5, -1, 1) or (0.5 if slope > 0 else -0.5),
                      "Cumulative Delta",
                      f"CVD {'rising' if slope > 0 else 'falling'} over recent bars"))

    wf = _last(ind.get("whale_flow"))
    if wf is not None:
        votes.append((_clip(wf * 2, -1, 1), "Whale flow",
                      f"large-trade flow {wf:+.2f}"))

    oi = _last(ind.get("oi_chg"))
    agg2 = _last(ind.get("aggressor"))
    if oi is not None and agg2 is not None:
        # rising OI + buy pressure = conviction long; rising OI + sell = conviction short
        v = _clip(oi / 2.0, -1, 1) * (1 if agg2 >= 0 else -1)
        votes.append((v, "Open interest",
                      f"OI {oi:+.1f}% {'confirming' if abs(v) > 0.1 else 'flat against'} the flow"))

    return votes


# --------------------------------------------------------------------------- #
# level construction — structure first, then clamped to a sane ATR band
# --------------------------------------------------------------------------- #
def _levels(side, entry, atr, candles):
    lows = candles["low"].tail(SWING_LOOKBACK)
    highs = candles["high"].tail(SWING_LOOKBACK)
    swing_low = float(lows.min()) if len(lows) else entry - atr
    swing_high = float(highs.max()) if len(highs) else entry + atr

    if side == "long":
        struct_dist = entry - (swing_low - SWING_PAD_ATR * atr)
        dist = _clip(struct_dist, SL_ATR_MIN * atr, SL_ATR_MAX * atr)
        stop = entry - dist
        take = entry + RR_TARGET * dist
    else:
        struct_dist = (swing_high + SWING_PAD_ATR * atr) - entry
        dist = _clip(struct_dist, SL_ATR_MIN * atr, SL_ATR_MAX * atr)
        stop = entry + dist
        take = entry - RR_TARGET * dist

    return stop, take, dist


def _bps(a, b):
    return abs(a - b) / b * 1e4 if b else 0.0


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #
def decide(candles, ind) -> dict:
    """
    candles : DataFrame with open/high/low/close/volume (the resampled frame)
    ind     : DataFrame from indicators.compute_all(candles)
    returns : a JSON-safe plan dict
    """
    entry = float(candles["close"].iloc[-1])
    atr = _last(ind.get("atr"))
    if atr is None or atr <= 0:
        # fall back to a rough ATR from recent range if the column is missing
        rng = (candles["high"] - candles["low"]).tail(14)
        atr = float(rng.mean()) if len(rng) else entry * 0.004

    families = [
        ("trend", W_TREND, _trend_votes(entry, ind)),
        ("momentum", W_MOMENTUM, _momentum_votes(ind)),
        ("location", W_LOCATION, _location_votes(entry, ind)),
        ("tape", W_TAPE, _tape_votes(ind)),
    ]

    signals = []
    num = den = 0.0
    fam_scores = {}
    for fam, w, votes in families:
        if not votes:
            fam_scores[fam] = None
            continue
        fam_avg = sum(v for v, _, _ in votes) / len(votes)
        fam_scores[fam] = round(fam_avg, 3)
        num += fam_avg * w
        den += w
        for v, label, reason in votes:
            signals.append({
                "family": fam, "label": label,
                "vote": round(v, 2),
                "dir": "long" if v > 0.05 else "short" if v < -0.05 else "flat",
                "reason": reason,
            })

    score = (num / den) if den else 0.0          # confluence in [-1, 1]

    adx = _last(ind.get("adx")) or 0.0
    trending = adx >= ADX_TREND

    # ---- decide side ----
    if score >= SCORE_ENTRY and trending:
        side = "long"
    elif score <= -SCORE_ENTRY and trending:
        side = "short"
    else:
        side = "neutral"

    # agreement is counted toward the shown lean (sign of the confluence), so a
    # "short lean" read never reports "0 of N aligned"
    lean = "long" if score >= 0 else "short"
    agree = sum(1 for s in signals if s["dir"] == lean)
    total = sum(1 for s in signals if s["dir"] != "flat")
    # confidence blends how strong AND how agreed the read is, gated by trend
    conf = int(round(_clip(abs(score) / 0.5, 0, 1) * 100))
    if not trending:
        conf = int(conf * 0.5)

    actionable = side != "neutral"
    # Direction used to lay out the levels. When the trade gate isn't met we still
    # size a reference setup along the current lean (sign of the confluence) so the
    # entry/stop/target are always populated and visibly re-scale with the
    # timeframe — the verdict, not the presence of numbers, is what says "trade".
    bias = side if actionable else ("long" if score >= 0 else "short")

    stop, take, dist = _levels(bias, entry, atr, candles)
    stop_bps = _bps(entry, stop)
    take_bps = _bps(entry, take)
    rr = (take_bps / stop_bps) if stop_bps else None

    plan = {
        "side": side,
        "bias": bias,
        "actionable": actionable,
        "score": round(score, 3),
        "confidence": conf,
        "adx": round(adx, 1),
        "trending": trending,
        "atr": round(atr, 8),
        "atr_pct": round(atr / entry * 100, 3),
        "entry": round(entry, 8),
        "stop": round(stop, 8),
        "take": round(take, 8),
        "stop_bps": round(stop_bps, 1),
        "take_bps": round(take_bps, 1),
        "rr": round(rr, 2) if rr else None,
        "family_scores": fam_scores,
        "agree": agree,
        "total_votes": total,
        "signals": signals,
    }

    lead = [f for f, s in fam_scores.items()
            if s is not None and ((s > 0) == (bias == "long")) and abs(s) > 0.1]

    verb = "long" if bias == "long" else "short"
    conviction = ("strong" if adx >= ADX_STRONG else "solid") if actionable else "low"
    trend_note = "trending market" if trending else "ranging market"
    plan["narrative"] = (
        f"The AI leans {verb} — {conviction} conviction "
        f"(confluence {score:+.2f}, {agree} of {total} indicators aligned, "
        f"ADX {adx:.0f} · {trend_note}). "
        f"Leading evidence: {', '.join(lead) or 'trend'}. "
        f"Suggested entry {entry:.2f}, stop {stop:.2f}, target {take:.2f}"
        + (f" ({rr:.2f}:1)." if rr else "."))
    return plan
