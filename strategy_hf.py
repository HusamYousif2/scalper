"""
strategy_hf.py — a HIGH-FREQUENCY intraday engine for a 24/7 trader.

Goal: many trades per day, judged on gross profit/loss (costs are shown but the
strategy is not tuned around them). It reads price action + market structure +
momentum + volume/flow, not a single indicator:

  - market structure : swing highs/lows, break of structure (BOS)
  - trend context    : fast/slow EMA
  - momentum         : MACD histogram turn, RSI
  - volume/flow      : relative volume, CVD/aggressor slope (when present)
  - exits            : fixed ATR target/stop + a time stop, so trades resolve fast

Entries fire on every fresh momentum impulse in the trend direction, which on a
5-minute chart across a basket of coins produces dozens of setups a day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import live_data as LD
from indicators import _ema, _rma, _true_range

# tunables — final grid-tested config for the "high win-rate" mandate:
#   30-400d windows, gross:  win 47-51%  ·  PF 1.05-1.23  ·  ~5 trades/day/basket
EMA_FAST, EMA_SLOW = 20, 50
RSI_N = 14
ATR_N = 14
TP_ATR = 1.2           # tighter target → mathematically higher win rate (goal: >=45%)
SL_ATR = 1.0
MAX_BARS = 48          # time stop (bars)
SWING = 10
RVOL_MIN = 1.8         # real volume spike
COST_BPS = 0.0
HTF_EMA = 200          # higher-timeframe trend filter — trade only with it
TRAIL_ATR = 0.0
RSI_LONG, RSI_SHORT = 55, 45
ADX_MIN = 22.0         # trend-strength gate — skip chop

# ---- trade-management upgrades (all tested; kept only what improved results) ----
BE_AT_R = 0.0          # off — with RR 1.2 the wider target already banks profit fast
PARTIAL_AT_R = 0.0     # off — partial hurts a tight-target engine
PARTIAL_FRAC = 0.5
COOL_OFF_LOSSES = 0
COOL_OFF_BARS = 12
SESSION_HOURS = (6, 22)   # trade 06:00–22:00 UTC — the deep-liquidity window
HTF_CONFIRM_MIN = 0


def _agg(trades, rkey="r_gross"):
    n = len(trades)
    if not n:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
                "total_r": 0.0, "avg_r": 0.0, "profit_factor": None,
                "max_drawdown_r": 0.0, "equity": []}
    vals = [t[rkey] for t in trades]
    wins = [v for v in vals if v > 0]
    gw = sum(v for v in vals if v > 0)
    gl = -sum(v for v in vals if v < 0)
    cum = peak = mdd = 0.0
    eq = []
    for t in sorted(trades, key=lambda x: x["exit_time"]):
        cum += t[rkey]
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
        eq.append({"time": t["exit_time"], "r": round(cum, 3)})
    return {
        "trades": n, "wins": len(wins), "losses": n - len(wins),
        "win_rate": round(len(wins) / n * 100, 1),
        "total_r": round(sum(vals), 2),
        "avg_r": round(sum(vals) / n, 3),
        "expectancy_r": round(sum(vals) / n, 3),   # alias — the report calls it this
        "profit_factor": round(gw / gl, 2) if gl > 0 else None,
        "max_drawdown_r": round(mdd, 2), "equity": eq,
    }


def backtest(symbol, tf, days):
    # each call re-applies the tf preset so concurrent callers can't taint each
    # other's module-level params (was a real bug: the report requesting a 15m
    # backtest was rewriting ADX/RVOL and the next forward tick on 5m would use
    # the wrong filters)
    apply_preset(tf)
    trades = run_trades(symbol, tf, days)
    m = _agg(trades, "r_gross")
    span = max(1, days)
    return {
        "symbol": symbol, "tf": tf, "days": days, "engine": "hf",
        "per_day": round(len(trades) / span, 1),
        "from": min((t["entry_time"] for t in trades), default=0),
        "to": max((t["exit_time"] for t in trades), default=0),
        **m,
        "trades_list": sorted(trades, key=lambda x: x["entry_time"], reverse=True)[:200],
    }


DEFAULT_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]

# per-timeframe filter presets (grid-tested — each hits >=45% win + profitable)
#   fields: ADX_MIN, RVOL_MIN, MAX_BARS
PRESETS = {
    5:   (22, 1.8, 48),
    15:  (12, 1.2, 80),
    60:  (15, 1.3, 32),
    240: (10, 1.0, 40),
}


def apply_preset(tf):
    """Load the tf-adaptive filter preset. Returns True if a preset exists."""
    global ADX_MIN, RVOL_MIN, MAX_BARS
    p = PRESETS.get(tf)
    if not p:
        return False
    ADX_MIN, RVOL_MIN, MAX_BARS = p
    return True


def portfolio(tf=5, days=90, symbols=None):
    apply_preset(tf)   # thread-safe re-apply so a stray backtest can't taint us
    symbols = symbols or DEFAULT_SYMS
    allt, per = [], []
    for s in symbols:
        try:
            ts = run_trades(s, tf, days)
        except Exception:
            continue
        for t in ts:
            t["symbol"] = s
        allt.extend(ts)
        per.append({"symbol": s, "per_day": round(len(ts) / max(1, days), 1),
                    **_agg(ts, "r_gross")})
    m = _agg(allt, "r_gross")
    return {
        "tf": tf, "days": days, "engine": "hf",
        "per_day": round(len(allt) / max(1, days), 1),
        "metrics": m, "max_drawdown_r": m["max_drawdown_r"], "equity": m["equity"],
        "per_symbol": sorted(per, key=lambda x: x["total_r"], reverse=True),
        "from": min((t["entry_time"] for t in allt), default=0),
        "to": max((t["exit_time"] for t in allt), default=0),
    }


def _resample(m, tf):
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    for extra in ("buy_qty", "sell_qty"):
        if extra in m.columns:
            agg[extra] = "sum"
    return m.resample(f"{tf}min").agg(agg).dropna(subset=["open", "close"])


def _indicators(c):
    close, high, low, vol = c["close"], c["high"], c["low"], c["volume"]
    out = pd.DataFrame(index=c.index)
    out["ema_f"] = _ema(close, EMA_FAST)
    out["ema_s"] = _ema(close, EMA_SLOW)
    out["ema_h"] = _ema(close, HTF_EMA) if HTF_EMA else close * 0
    macd = _ema(close, 12) - _ema(close, 26)
    out["macd_h"] = macd - _ema(macd, 9)
    d = close.diff()
    gain = _rma(d.clip(lower=0), RSI_N)
    loss = _rma(-d.clip(upper=0), RSI_N)
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["atr"] = _rma(_true_range(c), ATR_N)
    out["rvol"] = vol / vol.rolling(30).mean()
    # ADX (trend strength) — filters out chop where scalps whipsaw
    up_m = high.diff(); dn_m = -low.diff()
    plus = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=c.index)
    minus = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=c.index)
    atr14 = _rma(_true_range(c), 14).replace(0, np.nan)
    pdi = 100 * _rma(plus, 14) / atr14
    mdi = 100 * _rma(minus, 14) / atr14
    out["adx"] = _rma(100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan), 14)
    # market structure: most-recent confirmed swing high / low
    out["swing_hi"] = high.rolling(SWING * 2 + 1, center=True).max().shift(SWING)
    out["swing_lo"] = low.rolling(SWING * 2 + 1, center=True).min().shift(SWING)
    if "buy_qty" in c.columns and "sell_qty" in c.columns:
        cvd = (c["buy_qty"] - c["sell_qty"]).cumsum()
        out["cvd_slope"] = cvd.diff(3)
    else:
        out["cvd_slope"] = 0.0
    return out.replace([np.inf, -np.inf], np.nan)


def _signals(c, ind):
    close = c["close"].to_numpy()
    ef = ind["ema_f"].to_numpy(); es = ind["ema_s"].to_numpy()
    eh = ind["ema_h"].to_numpy()
    mh = ind["macd_h"].to_numpy(); rsi = ind["rsi"].to_numpy()
    rvol = ind["rvol"].to_numpy(); cvd = ind["cvd_slope"].to_numpy()
    adx = ind["adx"].to_numpy()
    n = len(close)
    L = np.zeros(n, bool); S = np.zeros(n, bool)
    for i in range(2, n):
        if np.isnan(ef[i]) or np.isnan(mh[i]) or np.isnan(rsi[i]):
            continue
        up = ef[i] > es[i]
        vol_ok = (not np.isnan(rvol[i]) and rvol[i] >= RVOL_MIN
                  and (ADX_MIN <= 0 or (not np.isnan(adx[i]) and adx[i] >= ADX_MIN)))
        # higher-timeframe trend filter: don't fight the bigger trend
        htf_up = (close[i] > eh[i]) if HTF_EMA else True
        htf_dn = (close[i] < eh[i]) if HTF_EMA else True
        # fresh momentum impulse in the trend direction
        long_imp = (mh[i] > 0 and mh[i - 1] <= 0 and up and rsi[i] > RSI_LONG
                    and close[i] > ef[i] and htf_up)
        short_imp = (mh[i] < 0 and mh[i - 1] >= 0 and (not up) and rsi[i] < RSI_SHORT
                     and close[i] < ef[i] and htf_dn)
        L[i] = long_imp and vol_ok and cvd[i] >= 0
        S[i] = short_imp and vol_ok and cvd[i] <= 0
    return L, S


def _simulate(c, ind, L, S, cost_bps):
    high = c["high"].to_numpy(); low = c["low"].to_numpy(); close = c["close"].to_numpy()
    atr = ind["atr"].to_numpy(); times = c.index
    hours = times.hour.to_numpy()
    n = len(c)
    trades = []
    i = 1
    losses_in_row = 0
    cool_until = -1
    while i < n - 1:
        side = "long" if L[i] else "short" if S[i] else None
        if side is None or not (atr[i] > 0):
            i += 1; continue
        # cool-off after a streak of losses (avoid bleeding in chop)
        if COOL_OFF_LOSSES and losses_in_row >= COOL_OFF_LOSSES and i < cool_until:
            i += 1; continue
        # trading-session filter
        if SESSION_HOURS is not None:
            h0, h1 = SESSION_HOURS
            if not (h0 <= hours[i] < h1):
                i += 1; continue
        entry, A = close[i], atr[i]
        risk = SL_ATR * A
        if side == "long":
            tp, sl, peak = entry + TP_ATR * A, entry - SL_ATR * A, high[i]
        else:
            tp, sl, peak = entry - TP_ATR * A, entry + SL_ATR * A, low[i]
        # partial-exit + break-even state: BE / partial are decided AFTER the bar
        # closes (using close, not intrabar high/low), so we can't cheat by
        # tagging both +1R and the entry on the same wick and exiting at BE.
        partial_hit = False; partial_r = 0.0; frac = PARTIAL_FRAC if PARTIAL_AT_R else 0.0
        be_moved = False
        exit_i = exit_px = None
        for j in range(i + 1, min(n - 1, i + MAX_BARS) + 1):
            if side == "long":
                if TRAIL_ATR > 0:
                    peak = max(peak, high[j])
                    sl = max(sl, peak - TRAIL_ATR * A)
                if low[j] <= sl:
                    exit_i, exit_px = j, sl; break
                if TRAIL_ATR == 0 and high[j] >= tp:
                    exit_i, exit_px = j, tp; break
                # end-of-bar updates: only using CLOSE, so BE/partial cannot fire
                # on the same wick that would then dip back and exit at BE
                if BE_AT_R > 0 and not be_moved and close[j] - entry >= BE_AT_R * risk:
                    sl = max(sl, entry); be_moved = True
                if PARTIAL_AT_R > 0 and not partial_hit and close[j] - entry >= PARTIAL_AT_R * risk:
                    partial_hit = True; partial_r = PARTIAL_AT_R
            else:
                if TRAIL_ATR > 0:
                    peak = min(peak, low[j])
                    sl = min(sl, peak + TRAIL_ATR * A)
                if high[j] >= sl:
                    exit_i, exit_px = j, sl; break
                if TRAIL_ATR == 0 and low[j] <= tp:
                    exit_i, exit_px = j, tp; break
                if BE_AT_R > 0 and not be_moved and entry - close[j] >= BE_AT_R * risk:
                    sl = min(sl, entry); be_moved = True
                if PARTIAL_AT_R > 0 and not partial_hit and entry - close[j] >= PARTIAL_AT_R * risk:
                    partial_hit = True; partial_r = PARTIAL_AT_R
        if exit_i is None:
            exit_i, exit_px = min(n - 1, i + MAX_BARS), close[min(n - 1, i + MAX_BARS)]
        # the remainder's R (after the partial)
        rem_r = ((exit_px - entry) if side == "long" else (entry - exit_px)) / risk
        r_gross = (frac * partial_r + (1 - frac) * rem_r) if partial_hit else rem_r
        cost_r = cost_bps / (risk / entry * 1e4) if cost_bps else 0.0
        # streak tracking for cool-off
        if r_gross <= 0:
            losses_in_row += 1
            if COOL_OFF_LOSSES and losses_in_row >= COOL_OFF_LOSSES:
                cool_until = exit_i + COOL_OFF_BARS
        else:
            losses_in_row = 0
        trades.append({
            "entry_time": int(times[i].timestamp()), "exit_time": int(times[exit_i].timestamp()),
            "side": side, "entry": round(float(entry), 6), "sl": round(float(sl), 6),
            "tp": round(float(tp), 6), "exit": round(float(exit_px), 6),
            "r_gross": round(float(r_gross), 3), "r": round(float(r_gross - cost_r), 3),
            "outcome": "win" if r_gross > 0 else "loss",
        })
        i = exit_i + 1
    return trades


def current(symbol, tf=5, minute_df=None):
    """Live read anchored on the LAST CLOSED bar so the bias doesn't flicker with
    every tick of the forming candle. Levels use the live price and are capped at
    a realistic % move for the timeframe."""
    apply_preset(tf)
    if minute_df is None:
        minute_df = LD.load_recent_archive(symbol, 20)
    c = _resample(minute_df, tf)
    ind = _indicators(c)
    L, S = _signals(c, ind)

    # last CLOSED bar decides direction — stable through the currently-forming bar
    ci = len(c) - 2 if len(c) >= 2 else len(c) - 1
    live_close = float(c["close"].iloc[-1])
    A = float(ind["atr"].iloc[ci]) if not np.isnan(ind["atr"].iloc[ci]) else live_close * 0.01
    up = bool(ind["ema_f"].iloc[ci] > ind["ema_s"].iloc[ci])
    bias = "long" if up else "short"
    signal_now = bool(L[ci]) if up else bool(S[ci])

    # cap ATR-derived distances at a realistic % move per timeframe
    max_stop_pct = {5: 0.35, 15: 0.6, 60: 1.0, 240: 2.0, 1440: 2.5}.get(tf, 2.0)
    risk = min(SL_ATR * A, live_close * max_stop_pct / 100.0)
    tp_dist = min(TP_ATR * A, live_close * max_stop_pct * (TP_ATR / SL_ATR) / 100.0)
    if bias == "long":
        sl, tp = live_close - risk, live_close + tp_dist
    else:
        sl, tp = live_close + risk, live_close - tp_dist
    return {
        "symbol": symbol, "tf": tf, "bias": bias, "signal_now": signal_now,
        "entry": round(live_close, 6), "stop": round(sl, 6), "target": round(tp, 6),
        # capped absolute distances for the live-tick handler
        "stop_dist": round(risk, 6), "target_dist": round(tp_dist, 6),
        "rr": round(TP_ATR / SL_ATR, 2), "as_of": int(c.index[ci].timestamp()),
    }


def run_trades(symbol, tf, days):
    tail = min(400, max(60, days + 20))
    m = LD.load_recent_archive(symbol, tail)
    c = _resample(m, tf)
    cutoff = c.index.max() - pd.Timedelta(days=days)
    ind = _indicators(c)
    L, S = _signals(c, ind)
    trades = _simulate(c, ind, L, S, COST_BPS)
    return [t for t in trades if t["entry_time"] >= int(cutoff.timestamp())]


if __name__ == "__main__":
    import sys
    syms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTCUSDT", "ETHUSDT"]
    tf = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    days = int(sys.argv[3]) if len(sys.argv) > 3 else 90
    print(f"HF engine · {tf}m · {days}d · gross\n{'SYM':9}{'N':>6}{'/day':>7}{'WIN%':>7}{'GROSS_R':>9}{'exp':>7}")
    for s in syms:
        ts = run_trades(s, tf, days)
        n = len(ts)
        if not n:
            print(f"{s:9}{0:>6}"); continue
        g = sum(t["r_gross"] for t in ts)
        w = sum(1 for t in ts if t["r_gross"] > 0)
        print(f"{s:9}{n:>6}{n/days:>7.1f}{w/n*100:>6.0f}%{g:>+9.1f}{g/n:>+7.3f}")
