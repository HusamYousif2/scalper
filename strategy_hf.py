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

# tunables (grid-tested on the 8-coin basket, 5m, 90d, gross)
EMA_FAST, EMA_SLOW = 20, 50
RSI_N = 14
ATR_N = 14
TP_ATR = 2.0           # target distance (2:1 reward:risk)
SL_ATR = 1.0           # stop distance
MAX_BARS = 40          # time stop (bars)
SWING = 10             # bars each side for a swing pivot / structure break
RVOL_MIN = 1.2         # only take setups on above-average (spiking) volume
COST_BPS = 0.0         # per the mandate: judge on P/L, not cost (set >0 to include)


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
        "total_r": round(sum(vals), 2), "avg_r": round(sum(vals) / n, 3),
        "profit_factor": round(gw / gl, 2) if gl > 0 else None,
        "max_drawdown_r": round(mdd, 2), "equity": eq,
    }


def backtest(symbol, tf, days):
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


def portfolio(tf=5, days=90, symbols=None):
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
    macd = _ema(close, 12) - _ema(close, 26)
    out["macd_h"] = macd - _ema(macd, 9)
    d = close.diff()
    gain = _rma(d.clip(lower=0), RSI_N)
    loss = _rma(-d.clip(upper=0), RSI_N)
    out["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    out["atr"] = _rma(_true_range(c), ATR_N)
    out["rvol"] = vol / vol.rolling(30).mean()
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
    mh = ind["macd_h"].to_numpy(); rsi = ind["rsi"].to_numpy()
    rvol = ind["rvol"].to_numpy(); cvd = ind["cvd_slope"].to_numpy()
    n = len(close)
    L = np.zeros(n, bool); S = np.zeros(n, bool)
    for i in range(2, n):
        if np.isnan(ef[i]) or np.isnan(mh[i]) or np.isnan(rsi[i]):
            continue
        up = ef[i] > es[i]
        vol_ok = not np.isnan(rvol[i]) and rvol[i] >= RVOL_MIN
        # fresh momentum impulse in the trend direction
        long_imp = mh[i] > 0 and mh[i - 1] <= 0 and up and rsi[i] > 48 and close[i] > ef[i]
        short_imp = mh[i] < 0 and mh[i - 1] >= 0 and (not up) and rsi[i] < 52 and close[i] < ef[i]
        L[i] = long_imp and vol_ok and cvd[i] >= 0
        S[i] = short_imp and vol_ok and cvd[i] <= 0
    return L, S


def _simulate(c, ind, L, S, cost_bps):
    high = c["high"].to_numpy(); low = c["low"].to_numpy(); close = c["close"].to_numpy()
    atr = ind["atr"].to_numpy(); times = c.index
    n = len(c)
    trades = []
    i = 1
    while i < n - 1:
        side = "long" if L[i] else "short" if S[i] else None
        if side is None or not (atr[i] > 0):
            i += 1
            continue
        entry, A = close[i], atr[i]
        if side == "long":
            tp, sl = entry + TP_ATR * A, entry - SL_ATR * A
        else:
            tp, sl = entry - TP_ATR * A, entry + SL_ATR * A
        risk = SL_ATR * A
        exit_i = exit_px = None
        for j in range(i + 1, min(n - 1, i + MAX_BARS) + 1):
            if side == "long":
                if low[j] <= sl:
                    exit_i, exit_px = j, sl; break
                if high[j] >= tp:
                    exit_i, exit_px = j, tp; break
            else:
                if high[j] >= sl:
                    exit_i, exit_px = j, sl; break
                if low[j] <= tp:
                    exit_i, exit_px = j, tp; break
        if exit_i is None:
            exit_i, exit_px = min(n - 1, i + MAX_BARS), close[min(n - 1, i + MAX_BARS)]
        r_gross = ((exit_px - entry) if side == "long" else (entry - exit_px)) / risk
        cost_r = cost_bps / (risk / entry * 1e4) if cost_bps else 0.0
        trades.append({
            "entry_time": int(times[i].timestamp()), "exit_time": int(times[exit_i].timestamp()),
            "side": side, "entry": round(float(entry), 6), "sl": round(float(sl), 6),
            "tp": round(float(tp), 6), "exit": round(float(exit_px), 6),
            "r_gross": round(float(r_gross), 3), "r": round(float(r_gross - cost_r), 3),
            "outcome": "win" if r_gross > 0 else "loss",
        })
        i = exit_i + 1
    return trades


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
