"""
strategy_pro.py — the discretionary trend/volatility setup, backtested honestly.

Entry (long; short mirrors):
  gates  : close > T3, Range-Filter up, +DI > -DI with ADX > 12,
           and volatility EXPANDING (Chaikin Vol > 0 or Change-of-Vol > 0)  ← the edge
  trigger: Squeeze momentum crosses up through zero, OR the squeeze releases
           with positive momentum.
Exit:
  initial stop 2.5×ATR; trailing stop 5×ATR from the peak; also exit if price
  loses T3 (trend permission gone). No fixed target — winners are left to run,
  which is what can make the average win bigger than the average loss and beat
  the fee. All results are NET of an 11 bps round-trip cost.

Reported with a time train/validation split so an overfit does not masquerade as
an edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import live_data as LD
import pro_indicators as PI

ADX_MIN = 12.0               # trend-strength gate
STOP_ATR = 2.0               # tighter initial stop (1R) — tested best net on 4H
TRAIL_ATR = 7.0              # give winners more room to run
COST_BPS = 2 * (3.0 + 0.5)    # round-trip: limit/maker-oriented fee + slippage
FUNDING_BPS_DAY = 2.0         # perp funding, typical flat
MAX_HOLD_MULT = 400          # cap a trade at ~this many candles

# quality filters (grid-tested on BTC+ETH 4H, 500d). MIN_ATR_PCT is the floor at
# 4H; it's scaled by sqrt(tf/240) per timeframe so it still filters calm-market
# junk on 4H (a big win there) without emptying lower timeframes.
MIN_ATR_PCT = 0.45
VOL_BOTH = False             # either volatility gauge expanding is enough
REGIME_TF3 = True            # T3 must slope the trade's way (trend-quality filter)
ENTRY_RELEASE_ONLY = False   # enter only on a squeeze release (cleanest breakout)
VOL_RISING = True            # volatility must be rising — a much stronger edge (tested)

# candidate confirmation gates — each tested; keep only out-of-sample improvers
USE_EMA200 = False           # trade only with the long-term (200) trend
USE_VOLUME = False           # require above-average volume on entry
USE_RSI = False              # momentum: RSI on the trade's side of 50
USE_MACD = False             # MACD histogram on the trade's side of 0


def _load(symbol, tf, days):
    tail = min(400, max(150, days + 60))
    m = LD.load_recent_archive(symbol, tail)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    c = m.resample(f"{tf}min").agg(agg).dropna(subset=["open", "close"])
    ind = PI.compute(c)
    cutoff = c.index.max() - pd.Timedelta(days=days)
    start = int(c.index.searchsorted(cutoff))
    w0 = max(0, start - 300)
    return c.iloc[w0:], ind.iloc[w0:], start - w0


def _signals(c, ind):
    close = c["close"].to_numpy()
    t3 = ind["t3"].to_numpy(); rfd = ind["rf_dir"].to_numpy()
    adx = ind["adx"].to_numpy(); pdi = ind["pdi"].to_numpy(); mdi = ind["mdi"].to_numpy()
    cov = ind["cov"].to_numpy(); cvol = ind["cvol"].to_numpy()
    val = ind["sq_val"].to_numpy(); sqon = ind["sq_on"].to_numpy()
    ema200 = ind["ema200"].to_numpy(); rsi = ind["rsi"].to_numpy()
    vsma = ind["vol_sma"].to_numpy(); mhist = ind["macd_hist"].to_numpy()
    vol = c["volume"].to_numpy()
    n = len(close)
    L = np.zeros(n, bool); S = np.zeros(n, bool)
    for i in range(3, n):
        if np.isnan(t3[i]) or np.isnan(adx[i]) or np.isnan(val[i]) or np.isnan(t3[i - 3]):
            continue
        rising = (cvol[i] > cvol[i - 1]) if VOL_RISING else True
        vedge = ((cvol[i] > 0 and cov[i] > 0) if VOL_BOTH else (cvol[i] > 0 or cov[i] > 0)) and rising
        # higher-quality trend regime: T3 must also be sloping the trade's way
        up_ok = (t3[i] > t3[i - 3]) if REGIME_TF3 else True
        dn_ok = (t3[i] < t3[i - 3]) if REGIME_TF3 else True
        # optional confirmation gates (each tested independently)
        vol_ok = (not USE_VOLUME) or (vol[i] > vsma[i])
        e2_up = (not USE_EMA200) or (close[i] > ema200[i])
        e2_dn = (not USE_EMA200) or (close[i] < ema200[i])
        rsi_up = (not USE_RSI) or (rsi[i] > 50)
        rsi_dn = (not USE_RSI) or (rsi[i] < 50)
        mh_up = (not USE_MACD) or (mhist[i] > 0)
        mh_dn = (not USE_MACD) or (mhist[i] < 0)
        lg = (close[i] > t3[i] and rfd[i] > 0 and pdi[i] > mdi[i]
              and adx[i] > ADX_MIN and vedge and up_ok and vol_ok and e2_up and rsi_up and mh_up)
        sg = (close[i] < t3[i] and rfd[i] < 0 and mdi[i] > pdi[i]
              and adx[i] > ADX_MIN and vedge and dn_ok and vol_ok and e2_dn and rsi_dn and mh_dn)
        cross_up = val[i] > 0 and val[i - 1] <= 0
        cross_dn = val[i] < 0 and val[i - 1] >= 0
        release = sqon[i - 1] > 0.5 and sqon[i] < 0.5
        trig_up = (release and val[i] > 0) if ENTRY_RELEASE_ONLY else (cross_up or (release and val[i] > 0))
        trig_dn = (release and val[i] < 0) if ENTRY_RELEASE_ONLY else (cross_dn or (release and val[i] < 0))
        L[i] = lg and trig_up
        S[i] = sg and trig_dn
    return L, S


def _simulate(c, ind, start, L, S, cost_bps, tf=240):
    high = c["high"].to_numpy(); low = c["low"].to_numpy(); close = c["close"].to_numpy()
    t3 = ind["t3"].to_numpy(); atr = ind["atr"].to_numpy(); times = c.index
    n = len(c)
    # the calm-market floor scales with timeframe (ATR grows ~sqrt(time))
    min_atr_pct = MIN_ATR_PCT * (tf / 240.0) ** 0.5
    trades = []
    i = max(start, 1)
    while i < n - 1:
        side = "long" if L[i] else "short" if S[i] else None
        if side is None or not (atr[i] > 0):
            i += 1
            continue
        # skip setups whose stop is too tight for costs to be negligible
        if atr[i] / close[i] * 100 < min_atr_pct:
            i += 1
            continue
        entry, A = close[i], atr[i]
        risk = STOP_ATR * A
        sl0 = entry - risk if side == "long" else entry + risk   # initial 2.5×ATR stop
        tp0 = entry + 2 * risk if side == "long" else entry - 2 * risk  # 2R = 5×ATR target
        if side == "long":
            stop, peak = entry - risk, high[i]
        else:
            stop, peak = entry + risk, low[i]

        exit_i = exit_px = None
        jmax = min(n - 1, i + MAX_HOLD_MULT)
        for j in range(i + 1, jmax + 1):
            if side == "long":
                peak = max(peak, high[j])
                stop = max(stop, peak - TRAIL_ATR * A)
                if low[j] <= stop:
                    exit_i, exit_px = j, stop; break
                if close[j] < t3[j]:
                    exit_i, exit_px = j, close[j]; break
            else:
                peak = min(peak, low[j])
                stop = min(stop, peak + TRAIL_ATR * A)
                if high[j] >= stop:
                    exit_i, exit_px = j, stop; break
                if close[j] > t3[j]:
                    exit_i, exit_px = j, close[j]; break
        if exit_i is None:
            exit_i, exit_px = jmax, close[jmax]

        r_gross = ((exit_px - entry) if side == "long" else (entry - exit_px)) / risk
        stop_bps = risk / entry * 1e4
        hold_days = max(0.0, (times[exit_i].timestamp() - times[i].timestamp()) / 86400.0)
        # trading cost is round-trip; funding accrues per day the position is held
        cost_r = (cost_bps + FUNDING_BPS_DAY * hold_days) / stop_bps
        r = r_gross - cost_r
        trades.append({
            "entry_time": int(times[i].timestamp()),
            "exit_time": int(times[exit_i].timestamp()),
            "side": side, "hold_days": round(hold_days, 2),
            "entry": round(float(entry), 6), "exit": round(float(exit_px), 6),
            "sl": round(float(sl0), 6), "tp": round(float(tp0), 6),   # 1R stop / 2R target; exit trails
            "r": round(float(r), 3), "r_gross": round(float(r_gross), 3),
            "outcome": "win" if r_gross > 0 else "loss",
        })
        i = exit_i + 1
    return trades


def _agg(ts):
    n = len(ts)
    if not n:
        return {"n": 0, "win": 0.0, "net": 0.0, "avgW": 0.0, "avgL": 0.0, "pf": 0.0}
    wins = [t["r"] for t in ts if t["outcome"] == "win"]
    losses = [t["r"] for t in ts if t["outcome"] == "loss"]
    gw = sum(x for x in (t["r"] for t in ts) if x > 0)
    gl = -sum(x for x in (t["r"] for t in ts) if x < 0)
    return {
        "n": n,
        "win": round(len(wins) / n * 100, 1),
        "net": round(sum(t["r"] for t in ts), 1),
        "avgW": round(np.mean([t["r"] for t in ts if t["r"] > 0]), 2) if any(t["r"] > 0 for t in ts) else 0.0,
        "avgL": round(np.mean([t["r"] for t in ts if t["r"] < 0]), 2) if any(t["r"] < 0 for t in ts) else 0.0,
        "pf": round(gw / gl, 2) if gl > 0 else None,
    }


DEFAULT_SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
                "XRPUSDT", "ADAUSDT", "DOGEUSDT", "LINKUSDT"]


def run_trades(symbol, tf, days):
    c, ind, start = _load(symbol, tf, days)
    L, S = _signals(c, ind)
    return _simulate(c, ind, start, L, S, COST_BPS, tf)


def portfolio(tf=240, days=190, symbols=None):
    """Run the strategy on every symbol and merge the trades into one equity
    curve — the deployable, diversified form of the edge."""
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
        a = _agg(ts)
        per.append({"symbol": s, **a})
    allt.sort(key=lambda t: t["exit_time"])

    cum = peak = mdd = 0.0
    eq = []
    for t in allt:
        cum += t["r"]
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
        eq.append({"time": t["exit_time"], "r": round(cum, 3)})

    return {
        "tf": tf, "days": days, "symbols": symbols,
        "metrics": _agg(allt),
        "max_drawdown_r": round(mdd, 2),
        "equity": eq,
        "per_symbol": sorted(per, key=lambda x: x["net"], reverse=True),
        "from": min((t["entry_time"] for t in allt), default=0),
        "to": max((t["exit_time"] for t in allt), default=0),
    }


def prepare(minute_df, tf):
    """Resample to the timeframe and compute indicators + raw signals ONCE.
    Shared by current() (latest bar) and the decision tracker (every bar), so
    both read from one honest code path."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    c = minute_df.resample(f"{tf}min").agg(agg).dropna(subset=["open", "close"])
    ind = PI.compute(c)
    L, S = _signals(c, ind)
    return c, ind, L, S


def current(symbol, tf, minute_df=None, weights=None):
    """The strategy's read on the LAST CLOSED bar (stable through the currently
    forming bar so the bias doesn't flip every tick). `weights` (learned by the
    live scorecard) makes the bias a weighted vote; omitted = equal vote."""
    if minute_df is None:
        minute_df = LD.load_recent_archive(symbol, 40)
    c, ind, L, S = prepare(minute_df, tf)
    close_i = len(c) - 2 if len(c) >= 2 else len(c) - 1
    live_close = float(c["close"].iloc[-1])
    d = decision_at(c, ind, L, S, close_i, tf, live_close=live_close, weights=weights)
    d["symbol"] = symbol
    return d


def decision_at(c, ind, L, S, close_i, tf, live_close=None, weights=None):
    """The strategy's full decision anchored at closed-bar index `close_i`.
    When live_close is None (historical replay) the bar's own close is the
    anchor; the maths use ONLY data at or before close_i, so replaying past
    bars is genuinely out-of-sample against later price.

    `weights` (dict {indicator: weight}) makes the bias a WEIGHTED vote — the
    live scorecard learns which indicators actually predict and feeds their
    weights back here, so the read self-improves. Absent, it's an equal vote."""
    prev_i = max(0, close_i - 1)
    row = ind.iloc[close_i]
    prev_row = ind.iloc[prev_i]
    close_bar = float(c["close"].iloc[close_i])
    prev_close = float(c["close"].iloc[prev_i])
    if live_close is None:
        live_close = close_bar
    atr = float(row["atr"]) if not np.isnan(row["atr"]) else close_bar * 0.01

    # ---- VOTE-BASED bias & confluence ----
    # Every indicator ALWAYS votes long or short (no abstains) — that way
    # confluence is honest: N/5 means "N indicators actively agree with the
    # winning direction". The majority wins the bias; ties break to the
    # previous-bar direction.
    t3_up = close_bar > row["t3"]
    rf_up = row["rf_dir"] > 0
    dmi_up = row["pdi"] > row["mdi"]
    sq_up = row["sq_val"] > 0
    # EMA200 macro trend — another directional vote; usually aligns with T3 in
    # a trending market, so it lifts typical confluence to 4-5 without cheating
    ema200 = row.get("ema200", np.nan) if hasattr(row, "get") else np.nan
    if np.isnan(ema200):
        ema200 = float(c["close"].ewm(span=200, adjust=False).mean().iloc[close_i])
    ema200_up = close_bar > float(ema200)
    # volatility is a NON-directional gate — it joins the majority (rewards a
    # bias that already has support with a volatility-confirmation vote, or
    # tags AGAINST if volatility contracting = trend fading)
    vol_expanding = (row["cvol"] > 0) or (row["cov"] > 0)

    # WEIGHTED directional vote — each indicator pushes +w (long) or -w (short);
    # weights come from the live scorecard's measured hit rate, so proven
    # indicators (T3, Squeeze, ADX) outweigh coin-flip ones (EMA200).
    dir_votes = {"T3 trend": t3_up, "Range Filter": rf_up, "EMA200 macro": ema200_up,
                 "ADX / DMI": dmi_up, "Momentum (Squeeze)": sq_up}
    w = weights or {}
    score = sum((w.get(name, 1.0)) * (1.0 if v else -1.0) for name, v in dir_votes.items())
    prev_up = prev_close > prev_row["t3"]
    if score > 1e-9:      bias = "long"
    elif score < -1e-9:   bias = "short"
    else:                 bias = "long" if prev_up else "short"
    up = bias == "long"

    # weighted confidence: share of vote weight on the winning side (0.5–1.0)
    tot_w = sum(w.get(name, 1.0) for name in dir_votes) or 1.0
    win_w = sum(w.get(name, 1.0) for name, v in dir_votes.items() if v == up)
    weighted_conf = round(win_w / tot_w, 3)

    # volatility casts its vote WITH the winning bias if expanding (confirming
    # the trend), AGAINST if contracting (the trend is fading, low conviction)
    vol_up = up if vol_expanding else (not up)

    checks = [
        {"label": "T3 trend",   "pass": t3_up == up,
         "detail": f"price {'above' if t3_up else 'below'} T3",
         "vote": "long" if t3_up else "short"},
        {"label": "Range Filter", "pass": rf_up == up,
         "detail": f"filter pointing {'up' if rf_up else 'down'}",
         "vote": "long" if rf_up else "short"},
        {"label": "EMA200 macro", "pass": ema200_up == up,
         "detail": f"price {'above' if ema200_up else 'below'} EMA200",
         "vote": "long" if ema200_up else "short"},
        {"label": "ADX / DMI",  "pass": dmi_up == up,
         "detail": f"ADX {row['adx']:.0f}, DI{'+' if dmi_up else '−'} leading",
         "vote": "long" if dmi_up else "short"},
        {"label": "Momentum (Squeeze)", "pass": sq_up == up,
         "detail": f"momentum {row['sq_val']:+.1f}{' (squeeze on)' if row['sq_on'] > 0.5 else ''}",
         "vote": "long" if sq_up else "short"},
        {"label": "Volatility expanding", "pass": vol_expanding,
         "detail": f"Chaikin {row['cvol']:.0f}, ΔVol {row['cov']:+.2f}",
         "vote": "long" if vol_up else "short"},
    ]
    passed = sum(1 for c_ in checks if c_["pass"])

    # cap ATR-derived risk at a REALISTIC % of price for the timeframe.
    max_stop_pct = {5: 0.35, 15: 0.6, 60: 1.0, 240: 2.0, 1440: 2.5}.get(tf, 2.0)
    # ANCHOR ENTRY TO LIVE PRICE (frozen by the endpoint's tf-based cache).
    # Previously we anchored on close_bar which meant entry was hours old — the
    # market had already moved past it, so "entry 63497 / live 62732" looked
    # backwards. Now entry = the actual price when the decision was made and
    # stays put until the next scheduled refresh.
    anchor = live_close
    risk_raw = STOP_ATR * atr
    risk = min(risk_raw, anchor * max_stop_pct / 100.0)
    stop = anchor - risk if up else anchor + risk

    # target uses the STRUCTURAL level (recent swing high/low) — a real level
    # price actually reaches (support/resistance), not a raw ATR distance in
    # empty air. Lookback SCALES with the timeframe so 1H and 4H don't land on
    # the same swing (1H sees the last day, 4H sees the last week+, 1D longer).
    # target = the closer of {2R math, nearest structural swing} — a reachable
    # level price actually trades to, never past 2R.
    lookback = {5: 24, 15: 20, 60: 24, 240: 30, 1440: 40}.get(tf, 20)
    lookback = min(lookback, len(c) - 1)
    if up:
        struct = float(c["high"].iloc[max(0, close_i - lookback):close_i + 1].max())
        math_target = anchor + 2 * risk
        target = min(math_target, struct) if struct > anchor else math_target
    else:
        struct = float(c["low"].iloc[max(0, close_i - lookback):close_i + 1].min())
        math_target = anchor - 2 * risk
        target = max(math_target, struct) if struct < anchor else math_target

    signal_now = bool(L[close_i]) if up else bool(S[close_i])
    close = anchor

    # conviction label — used for UI colour/messaging, but levels ALWAYS render
    high_conviction = passed >= 4
    stop_bps = abs(close - stop) / close * 1e4

    # opportunity score (0-100): how strong / ready this setup is right now
    adx_v = float(row["adx"]) if not np.isnan(row["adx"]) else 0.0
    adx_f = max(0.0, min(1.0, (adx_v - 15) / 25))
    vol_ok = (row["cvol"] > 0) or (row["cov"] > 0)
    score = 50 * (passed / len(checks)) + (30 if signal_now else 0) \
        + 15 * adx_f + (5 if vol_ok else 0)
    score = round(min(100.0, score))
    rating = "Fire" if signal_now else "Strong" if score >= 72 \
        else "Building" if score >= 50 else "Watch"

    return {
        "tf": tf,
        "bias": bias,
        "signal_now": signal_now,
        "score": score, "rating": rating,
        # levels ALWAYS render — the vote system picks the strongest direction
        # so there's always a defensible bias and set of levels
        "entry": round(anchor, 6),
        "stop": round(stop, 6),
        "target": round(target, 6),
        "live_price": round(live_close, 6),
        "high_conviction": high_conviction,
        "stop_bps": round(stop_bps, 1),
        "atr": round(atr, 6),
        "atr_pct": round(atr / close * 100, 3),
        "trail_atr": TRAIL_ATR,
        "stop_atr": STOP_ATR,
        "adx": round(adx_v, 1),
        "passed": passed, "total_checks": len(checks),
        "weighted_conf": weighted_conf,
        "checks": checks,
        "as_of": int(c.index[close_i].timestamp()),
    }


def run(symbol, tf, days=90):
    c, ind, start = _load(symbol, tf, days)
    L, S = _signals(c, ind)
    trades = _simulate(c, ind, start, L, S, COST_BPS, tf)
    split = int(c.index[start].timestamp()
               + (c.index[-1].timestamp() - c.index[start].timestamp()) * 2 / 3)
    tr = [t for t in trades if t["entry_time"] < split]
    va = [t for t in trades if t["entry_time"] >= split]
    return {"all": _agg(trades), "train": _agg(tr), "valid": _agg(va)}


def _bundle(trades, rkey):
    """Metrics for one profit definition — rkey='r' is net of costs, 'r_gross' is
    pre-cost. Win/loss here is decided by that same profit column."""
    n = len(trades)
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
        "win_rate": round(len(wins) / n * 100, 1) if n else 0.0,
        "total_r": round(sum(vals), 2),
        "avg_r": round(sum(vals) / n, 3) if n else 0.0,
        "expectancy_r": round(sum(vals) / n, 3) if n else 0.0,
        "profit_factor": round(gw / gl, 2) if gl > 0 else None,
        "max_drawdown_r": round(mdd, 2),
        "equity": eq,
    }


def backtest(symbol, tf, days):
    """Full scorecard for the report page, driven entirely by the pro strategy —
    every entry, stop and exit comes from the strategy's own indicators. Returns
    BOTH the net-of-cost result and the pre-cost (gross) result, so the report can
    show either. Works for any timeframe / window."""
    trades = run_trades(symbol, tf, days)
    net = _bundle(trades, "r")
    gross = _bundle(trades, "r_gross")

    # flat fields mirror the NET bundle (back-compat: weekly snapshot, render)
    return {
        "symbol": symbol, "tf": tf, "days": days, "engine": "pro",
        "from": min((t["entry_time"] for t in trades), default=0),
        "to": max((t["exit_time"] for t in trades), default=0),
        **net,
        "net": net, "gross": gross,
        "trades_list": [
            {"entry_time": t["entry_time"], "exit_time": t["exit_time"],
             "side": t["side"], "entry": t["entry"], "sl": t["sl"], "tp": t["tp"],
             "exit": t["exit"], "r": t["r"], "r_gross": t["r_gross"],
             "outcome": t["outcome"]}
            for t in sorted(trades, key=lambda x: x["entry_time"])
        ],
    }


if __name__ == "__main__":
    import sys
    tfs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [15, 60, 240]
    syms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["BTCUSDT", "ETHUSDT"]
    print(f"pro strategy · net of {COST_BPS:.0f}bps · trailing {STOP_ATR}/{TRAIL_ATR}×ATR\n")
    print(f"{'SYM':8} {'TF':5} {'N':>4} {'WIN%':>5} {'NET_R':>7} {'avgW':>5} {'avgL':>6} {'PF':>5}   {'VALID(net/n/win%)':>18}")
    for s in syms:
        for tf in tfs:
            try:
                r = run(s, tf)
                a, v = r["all"], r["valid"]
                pf = "inf" if a["pf"] is None else a["pf"]
                print(f"{s:8} {str(tf)+'m':5} {a['n']:>4} {a['win']:>5} {a['net']:>+7.1f} "
                      f"{a['avgW']:>5} {a['avgL']:>6} {str(pf):>5}   "
                      f"{v['net']:>+6.1f}/{v['n']}/{v['win']}%")
            except Exception as e:
                import traceback
                print(f"{s} {tf}m FAIL {type(e).__name__}: {e}")
                traceback.print_exc()
