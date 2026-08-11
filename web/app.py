"""
web/app.py — HTTP layer over the assessment engine.

Design constraints that shaped this file:

  * assess() takes 15-20 seconds cold and a few seconds warm, which is far too
    slow to block a page load. Results are therefore cached per symbol, horizon
    and cost for CACHE_SECONDS, and the front end polls rather than waiting.
  * Nothing is computed here. This module only serves what the engine produced,
    so there is exactly one implementation of every number in the product.
  * Every figure is returned with the reference it must be read against — the
    benchmark it beat, the coverage it was measured at, the sample it came from.
    That pairing is enforced in the payload shape rather than left to the
    template, because the one time it was left to presentation it was forgotten.
"""

import os
import sys
import threading
import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import assess as A  # noqa: E402
import ingest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")

CACHE_SECONDS = 120
_cache: dict[tuple, tuple[float, dict]] = {}
_locks: dict[tuple, threading.Lock] = {}
_guard = threading.Lock()

app = FastAPI(title="Microstructure Desk", docs_url=None, redoc_url=None)


def _lock_for(key: tuple) -> threading.Lock:
    with _guard:
        if key not in _locks:
            _locks[key] = threading.Lock()
        return _locks[key]


def _clean(obj):
    """JSON-safe: numpy scalars, NaN and infinities all become plain values."""
    import math

    import numpy as np

    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@app.get("/api/symbols")
def symbols(data: int = Query(0, description="1 = any symbol with archived data (for the pro strategy)")):
    """By default, symbols that have both stored data and a trained ML model.
    With data=1, every symbol that has archived candles — the pro strategy needs
    no model, so the report can score any of them."""
    out = []
    if os.path.isdir(ingest.MINUTE_DIR):
        for s in sorted(os.listdir(ingest.MINUTE_DIR)):
            if data:
                if os.path.isdir(os.path.join(ingest.MINUTE_DIR, s)):
                    out.append({"symbol": s, "horizons": []})
                continue
            horizons = [h for h in (15, 60)
                        if os.path.exists(os.path.join(
                            A.TV.MODEL_DIR, f"vol_{s}_{h}m.pkl"))]
            if horizons:
                out.append({"symbol": s, "horizons": horizons})
    return {"symbols": out}


@app.get("/api/assess")
def api_assess(
    symbol: str = Query("BTCUSDT"),
    horizon: int = Query(15),
    cost: str = Query("none", description="preset name, basis points, or none"),
):
    fees = None if cost.lower() in ("none", "-", "off", "") else cost
    if fees is not None and fees not in A.FEE_MODELS:
        try:
            fees = float(fees)
        except ValueError:
            raise HTTPException(400, f"bad cost {cost!r}")

    key = (symbol, horizon, str(fees))
    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_SECONDS:
        return JSONResponse(_clean({**hit[1], "cached": True}))

    with _lock_for(key):
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < CACHE_SECONDS:
            return JSONResponse(_clean({**hit[1], "cached": True}))
        try:
            # bounded, small frame: features only need a few weeks, not months —
            # 150 days made this ~15s. 25 is plenty and ~6x faster.
            data = A.assess(symbol, horizon, fees, minute=_market_frame(symbol, 25))
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")
        data["fee_models"] = A.FEE_MODELS
        _cache[key] = (time.time(), data)
        return JSONResponse(_clean({**data, "cached": False}))


@app.get("/api/candles")
def api_candles(
    symbol: str = Query("BTCUSDT"),
    minutes: int = Query(1, description="minutes per candle"),
    count: int = Query(90, description="how many candles"),
):
    """
    Recent candles for the chart.

    Read from the same combined archive-plus-live frame the assessment uses, so
    the chart and the numbers can never disagree about what price is doing.
    """
    import live_data as LD

    key = ("candles", symbol, minutes, count)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return JSONResponse(_clean(hit[1]))

    with _lock_for(key):
        try:
            # 150 archive days so higher timeframes have enough candles (4H needs
            # ~53 days for 320 bars plus warm-up); the archive load is ~1.5s and
            # is shared across every timeframe, so switching tf never re-bridges.
            m = _market_frame(symbol, 150)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")

        need = minutes * count
        tail = m.tail(need)
        if minutes > 1:
            agg = tail.resample(f"{minutes}min").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum",
                 "buy_qty": "sum", "sell_qty": "sum"}
            )
        else:
            agg = tail[["open", "high", "low", "close", "volume",
                        "buy_qty", "sell_qty"]]
        agg = agg.dropna(subset=["open", "high", "low", "close"])

        out = {
            "symbol": symbol,
            "minutes": minutes,
            "candles": [
                {
                    "t": str(ix),
                    "o": float(r["open"]), "h": float(r["high"]),
                    "l": float(r["low"]), "c": float(r["close"]),
                    "v": float(r["volume"]),
                    "buy": float(r.get("buy_qty", 0) or 0),
                    "sell": float(r.get("sell_qty", 0) or 0),
                }
                for ix, r in agg.iterrows()
            ],
        }
        _cache[key] = (time.time(), out)
        return JSONResponse(_clean(out))


import concurrent.futures as _futures

_BRIDGE_POOL = _futures.ThreadPoolExecutor(max_workers=2)
_BRIDGE_BUDGET = 3.0     # seconds we'll wait for the live bridge before serving archive


def _market_frame(symbol: str, tail_days: int = 150):
    """Live frame when the bridge is quick, else the archive — never hang.

    The live bridge pages rate-limited REST calls; when the archive trails by a
    lot (e.g. an upstream daily dump hasn't published) a cold bridge can stall for
    minutes. We give it a few seconds on a worker thread; if it isn't done we
    serve the archive (slightly stale) and let the bridge keep warming the cache
    in the background, so a later request gets the live data. The endpoints cache
    their output, so this wait is paid at most once per cache window.
    """
    import live_data as LD

    fut = _BRIDGE_POOL.submit(LD.combined_frame_cached, symbol, 3, tail_days)
    try:
        return fut.result(timeout=_BRIDGE_BUDGET)
    except Exception:
        return LD.load_recent_archive(symbol, tail_days)


@app.get("/api/chart")
def api_chart(
    symbol: str = Query("BTCUSDT"),
    tf: int = Query(15, description="candle size in minutes"),
    count: int = Query(320, description="how many candles"),
):
    """
    Candles plus the full indicator suite, computed server-side.

    Doing the maths here rather than in the browser is what removes the study
    limit that a hosted charting subscription imposes: the client receives
    finished series and only has to draw them.
    """
    import indicators as IND
    import live_data as LD

    key = ("chart", symbol, tf, count)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return JSONResponse(_clean({**hit[1], "cached": True}))

    with _lock_for(key):
        try:
            # 150 archive days so higher timeframes have enough candles (4H needs
            # ~53 days for 320 bars plus warm-up); the archive load is ~1.5s and
            # is shared across every timeframe, so switching tf never re-bridges.
            m = _market_frame(symbol, 150)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")

        agg = {"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum"}
        for extra in ("buy_qty", "sell_qty", "n_trades", "whale_buy_qty",
                      "whale_sell_qty", "sum_open_interest",
                      "count_long_short_ratio", "sum_toptrader_long_short_ratio"):
            if extra in m.columns:
                agg[extra] = "last" if "ratio" in extra or extra == "sum_open_interest" \
                    else "sum"

        # 400 extra candles of warm-up so the 200-period studies are valid on
        # the first candle actually returned
        need = tf * (count + 400)
        tail = m.tail(need)
        c = tail.resample(f"{tf}min").agg(agg).dropna(subset=["open", "close"])

        ind = IND.compute_all(c)

        # the pro trend/volatility study set (the indicators from the strategy),
        # drawn on the chart alongside the classic suite
        try:
            import pro_indicators as PI

            pro = PI.compute(c)
            ind["pro_t3"] = pro["t3"]
            ind["pro_rf"] = pro["rf"]
            ind["pro_sq"] = pro["sq_val"]
            ind["pro_cov"] = pro["cov"]
            ind["pro_cvol"] = pro["cvol"]
            ind["pro_adx"] = pro["adx"]
            ind["pro_dip"] = pro["pdi"]
            ind["pro_dim"] = pro["mdi"]
            ind["pro_atr_pct"] = pro["atr"] / c["close"] * 100
        except Exception:
            pass

        c = c.tail(count)
        ind = ind.tail(count)

        def series(name):
            if name not in ind.columns:
                return None
            s = ind[name]
            return [None if pd.isna(v) else round(float(v), 8) for v in s]

        import pandas as pd  # noqa: F811  (local import keeps startup light)

        out = {
            "symbol": symbol,
            "tf": tf,
            "candles": [
                {"time": int(ix.timestamp()),
                 "open": float(r["open"]), "high": float(r["high"]),
                 "low": float(r["low"]), "close": float(r["close"]),
                 "volume": float(r.get("volume", 0) or 0)}
                for ix, r in c.iterrows()
            ],
            "indicators": {n: series(n) for n in ind.columns},
            "classic": sorted(IND.CLASSIC.keys()),
            "desk": sorted(IND.DESK.keys()),
        }
        _cache[key] = (time.time(), out)
        return JSONResponse(_clean({**out, "cached": False}))


@app.get("/api/plan")
def api_plan(
    symbol: str = Query("BTCUSDT"),
    tf: int = Query(15, description="candle size in minutes"),
):
    """
    Rule-based trade plan — a deterministic confluence engine, no ML, no LLM.

    Reads the same indicator suite the chart draws and reasons over it the way a
    disciplined trader would: every study votes, the weighted confluence decides
    direction (gated by ADX so it never trades chop), and entry/stop/target come
    from market structure. Fast, transparent, and fully back-testable.
    """
    import indicators as IND
    import live_data as LD
    import rule_engine as RE

    key = ("plan", symbol, tf)
    hit = _cache.get(key)
    # 60s cache to match the once-a-minute live-watch poll — recomputing the plan
    # from the (separately cached) frame is cheap, only the live bridge is slow
    if hit and time.time() - hit[0] < 60:
        return JSONResponse(_clean({**hit[1], "cached": True}))

    with _lock_for(key):
        try:
            # 150 archive days so higher timeframes have enough candles (4H needs
            # ~53 days for 320 bars plus warm-up); the archive load is ~1.5s and
            # is shared across every timeframe, so switching tf never re-bridges.
            m = _market_frame(symbol, 150)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")

        agg = {"open": "first", "high": "max", "low": "min", "close": "last",
               "volume": "sum"}
        for extra in ("buy_qty", "sell_qty", "n_trades", "whale_buy_qty",
                      "whale_sell_qty", "sum_open_interest",
                      "count_long_short_ratio", "sum_toptrader_long_short_ratio"):
            if extra in m.columns:
                agg[extra] = "last" if "ratio" in extra or extra == "sum_open_interest" \
                    else "sum"

        # warm-up so the 200-period studies are valid on the last candle
        tail = m.tail(tf * 600)
        c = tail.resample(f"{tf}min").agg(agg).dropna(subset=["open", "close"])
        ind = IND.compute_all(c)

        plan = RE.decide(c, ind)
        out = {"symbol": symbol, "tf": tf,
               "as_of": int(c.index[-1].timestamp()), "plan": plan}
        _cache[key] = (time.time(), out)
        return JSONResponse(_clean({**out, "cached": False}))


@app.get("/api/backtest")
def api_backtest(symbol: str = Query("BTCUSDT"), tf: int = Query(15),
                 days: int = Query(7)):
    """
    Score the decision engine over the last `days` of archived candles.

    Deterministic replay: every past signal is exactly what the engine would have
    said live, and every exit is decided from later candles only. Also writes a
    per-ISO-week summary snapshot so weekly performance accumulates over time.
    """
    import datetime as _dt
    import json as _json

    import strategy_pro as SP

    days = max(1, min(days, 180))
    key = ("backtest", symbol, tf, days)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 300:
        return JSONResponse(_clean({**hit[1], "cached": True}))

    with _lock_for(key):
        try:
            # every entry / stop / exit comes from the pro strategy's own indicators
            rep = SP.backtest(symbol, tf, days)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")

        # weekly snapshot (summary only), keyed by the ISO week of the window end.
        # Only the 7-day window is a "weekly evaluation" — longer windows must not
        # overwrite the week's snapshot.
        try:
            if days != 7:
                raise StopIteration
            wk = _dt.datetime.utcfromtimestamp(rep["to"]).isocalendar()
            wid = f"{wk[0]}-W{wk[1]:02d}"
            pdir = os.path.join(ingest.ROOT, "data", "perf")
            os.makedirs(pdir, exist_ok=True)
            summ = {k: rep[k] for k in (
                "symbol", "tf", "days", "trades", "wins", "losses", "win_rate",
                "total_r", "avg_r", "profit_factor", "max_drawdown_r", "from", "to")}
            summ["week"] = wid
            with open(os.path.join(pdir, f"{symbol}_{tf}m_{wid}.json"), "w") as f:
                _json.dump(summ, f)
        except Exception:
            pass

        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/perf/weeks")
def api_perf_weeks(symbol: str = Query("BTCUSDT"), tf: int = Query(15)):
    """The stored weekly summaries for a symbol+timeframe, oldest first."""
    import json as _json

    pdir = os.path.join(ingest.ROOT, "data", "perf")
    out = []
    if os.path.isdir(pdir):
        pref = f"{symbol}_{tf}m_"
        for fn in sorted(f for f in os.listdir(pdir)
                         if f.startswith(pref) and f.endswith(".json")):
            try:
                with open(os.path.join(pdir, fn)) as f:
                    out.append(_json.load(f))
            except Exception:
                continue
    return {"symbol": symbol, "tf": tf, "weeks": out[-26:]}


@app.get("/api/portfolio")
def api_portfolio(tf: int = Query(240), days: int = Query(190),
                  symbols: str = Query("")):
    """The pro trend/volatility strategy run across the chosen symbols and merged
    into one equity curve. Defaults to the full symbol universe. Archive-only."""
    import strategy_pro as SP

    days = max(30, min(days, 700))
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] or SP.DEFAULT_SYMS
    key = ("portfolio", tf, days, tuple(syms))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return JSONResponse(_clean({**hit[1], "cached": True}))
    with _lock_for(key):
        try:
            rep = SP.portfolio(tf=tf, days=days, symbols=syms)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")
        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/hf_portfolio")
def api_hf_portfolio(tf: int = Query(5), days: int = Query(90), symbols: str = Query("")):
    """The high-frequency engine across the basket — many trades a day, judged on
    gross P/L (costs excluded by mandate). Archive-only."""
    import strategy_hf as HF

    days = max(14, min(days, 400))
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()] or HF.DEFAULT_SYMS
    key = ("hfport", tf, days, tuple(syms))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 600:
        return JSONResponse(_clean({**hit[1], "cached": True}))
    with _lock_for(key):
        try:
            rep = HF.portfolio(tf=tf, days=days, symbols=syms)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")
        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/signals")
def api_signals(tf: int = Query(240)):
    """The pro strategy's current read on every symbol — where to act right now:
    direction, entry, stop, target, checks, and whether a signal just fired."""
    import strategy_pro as SP

    key = ("signals", tf)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 60:
        return JSONResponse(_clean({**hit[1], "cached": True}))
    with _lock_for(key):
        rows = []
        for s in SP.DEFAULT_SYMS:
            try:
                rows.append(SP.current(s, tf))
            except Exception:
                continue
        rows.sort(key=lambda r: (r.get("score", 0), bool(r.get("signal_now"))), reverse=True)
        rep = {"tf": tf, "rows": rows}
        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/indicator_scan")
def api_indicator_scan(symbol: str = Query("BTCUSDT"), tf: int = Query(15)):
    """Scan the whole indicator suite for one symbol: consensus bias, score, and
    the top indicators firing right now, with entry/stop/target. Refresh on a timer."""
    import indicator_ai as IA

    key = ("indscan", symbol, tf)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 90:
        return JSONResponse(_clean({**hit[1], "cached": True}))
    with _lock_for(key):
        try:
            rep = IA.scan(symbol, tf, _market_frame(symbol, 25))
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")
        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/hf_signal")
def api_hf_signal(symbol: str = Query("BTCUSDT"), tf: int = Query(5)):
    """The high-frequency engine's live read for one coin: direction, whether a
    setup is firing, and entry / stop / target."""
    import strategy_hf as HF

    key = ("hfsig", symbol, tf)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 60:
        return JSONResponse(_clean({**hit[1], "cached": True}))
    with _lock_for(key):
        try:
            rep = HF.current(symbol, tf, _market_frame(symbol, 20))
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")
        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/forward")
def api_forward(tf: int = Query(5)):
    """The live forward test: trades frozen as they settle in real time (BTC+ETH),
    net of all costs. Advances the log on each call, then returns the record."""
    import forward as FW

    key = ("forward", tf)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 120:
        return JSONResponse(_clean({**hit[1], "cached": True}))
    with _lock_for(key):
        try:
            FW.tick_all(tf)
            rep = FW.read(tf)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")
        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/pro_plan")
def api_pro_plan(symbol: str = Query("BTCUSDT"), tf: int = Query(240)):
    """The pro strategy's read on the latest candle for one symbol: lean, the
    seven checks, entry and 2.5×ATR stop, and whether a signal just fired."""
    import strategy_pro as SP

    key = ("pro_plan", symbol, tf)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < 60:
        return JSONResponse(_clean({**hit[1], "cached": True}))
    with _lock_for(key):
        try:
            rep = SP.current(symbol, tf, _market_frame(symbol, 40))
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")
        _cache[key] = (time.time(), rep)
        return JSONResponse(_clean({**rep, "cached": False}))


@app.get("/api/signal")
def api_signal(symbol: str = Query("BTCUSDT"), horizon: int = Query(15)):
    """
    Directional read plus expected magnitude.

    The probability comes from the calibrated classifier; the magnitude comes
    from the conformalised quantile models. The reliability table saved at
    training time travels with it so the interface can show what the stated
    probabilities actually delivered.
    """
    import train_direction as TD

    key = ("signal", symbol, horizon)
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return JSONResponse(_clean({**hit[1], "cached": True}))

    with _lock_for(key):
        try:
            a = A.assess(symbol, horizon, None)
        except Exception as e:
            raise HTTPException(503, f"{type(e).__name__}: {e}")

        out = {"symbol": symbol, "horizon": horizon, "price": a["price"],
               "as_of": a["as_of_utc"], "quantiles": a.get("quantiles_bps")}

        try:
            bundle, meta = TD.load(symbol, horizon)
            df, m = A.build_live_features(symbol, horizon,
                                          LD_frame(symbol))
            row = df.dropna(subset=[c for c in meta["features"]
                                    if c in df.columns]).tail(1)
            sig = TD.predict(bundle, meta, row)
            q = a.get("quantiles_bps") or {}
            px = a["price"]
            side = 1 if sig["p_up"] > 0.5 else -1
            sig["target_price"] = px * (1 + side * (q.get("q75", 0)) / 1e4)
            sig["stop_price"] = px * (1 - side * (q.get("q90", 0)) / 1e4)
            sig["target_bps"] = q.get("q75")
            sig["stop_bps"] = q.get("q90")
            sig["reliability"] = meta.get("reliability")
            sig["accuracy_all"] = meta.get("accuracy_all_%")
            sig["accuracy_high"] = meta.get("accuracy_high_conviction_%")
            sig["trained_at"] = meta.get("trained_at")
            out["signal"] = sig
        except Exception as e:
            out["signal_error"] = f"{type(e).__name__}: {e}"

        _cache[key] = (time.time(), out)
        return JSONResponse(_clean({**out, "cached": False}))


def LD_frame(symbol: str):
    import live_data as LD

    return LD.combined_frame_cached(symbol, live_hours=3)


@app.get("/api/model-stats")
def api_model_stats(symbol: str = Query("BTCUSDT"), horizon: int = Query(15)):
    """
    Validation record for the models actually being served.

    Every figure here is read from a model artifact or a measurement report on
    disk — none of it is written into this file by hand. If a model is retrained
    and does worse, this endpoint says so on the next request.
    """
    import json as _json

    out = {"symbol": symbol, "horizon": horizon}

    stem = os.path.join(A.TV.MODEL_DIR, f"vol_{symbol}_{horizon}m.json")
    if os.path.exists(stem):
        with open(stem) as f:
            meta = _json.load(f)
        out["point_model"] = {
            "r2": meta.get("valid_r2"),
            "benchmark_r2": meta.get("har_r2"),
            "n_features": meta.get("n_features"),
            "n_train": meta.get("n_train"),
            "trained_at": meta.get("trained_at"),
        }

    qstem = os.path.join(A.TV.MODEL_DIR, f"quant_{symbol}_{horizon}m.json")
    if os.path.exists(qstem):
        with open(qstem) as f:
            qmeta = _json.load(f)
        out["quantile_model"] = {
            "coverage": qmeta.get("verified_coverage_%"),
            "ci": qmeta.get("coverage_95ci_pts"),
            "walkforward_worst_error_pts": qmeta.get("walkforward_worst_error_pts"),
            "walkforward_n": qmeta.get("walkforward_n"),
            "recalibrate_every_days": qmeta.get("recalibrate_every_days"),
        }

    # weekly skill curve from the prequential monitor, if it has been run
    rep = os.path.join(os.path.dirname(HERE), "reports",
                       f"decay_{symbol}_{horizon}m.csv")
    if os.path.exists(rep):
        import pandas as pd

        d = pd.read_csv(rep)
        col = "skill_%" if "skill_%" in d.columns else None
        if col and "ts" in d.columns:
            d = d.dropna(subset=[col]).tail(40)
            out["skill_curve"] = {
                "weeks": [str(x)[:10] for x in d["ts"]],
                "skill": [round(float(v), 2) for v in d[col]],
                "positive_share": round(float((d[col] > 0).mean() * 100), 1),
                "recent_mean": round(float(d[col].tail(8).mean()), 2),
                "overall_mean": round(float(d[col].mean()), 2),
            }

    # how much history backs the models
    try:
        ndays = len(os.listdir(os.path.join(ingest.MINUTE_DIR, symbol)))
    except Exception:
        ndays = None
    out["data"] = {"days": ndays, "granularity": "1 minute",
                   "sources": ["aggregated trades", "order book depth",
                               "open interest", "long/short positioning"]}
    return JSONResponse(_clean(out))


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"),
                        headers={"Cache-Control": "no-store"})


@app.get("/report")
def report():
    return FileResponse(os.path.join(STATIC, "report.html"),
                        headers={"Cache-Control": "no-store"})


class NoCacheStatic(StaticFiles):
    """
    Serve the front end with caching disabled.

    Browsers hold on to JS and CSS aggressively, which meant edits to the chart
    silently did not appear and it looked as though the change had not worked.
    For a locally served app there is nothing to gain from caching these files.
    """

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp


app.mount("/static", NoCacheStatic(directory=STATIC), name="static")


def _forward_ticker():
    """Advance the live forward log every 30 min so it accrues even with no page
    views. Cheap and idempotent (dedups by entry time)."""
    import forward as FW

    while True:
        try:
            FW.tick_all(240)
        except Exception:
            pass
        time.sleep(1800)


@app.on_event("startup")
def _start_bg():
    import threading
    threading.Thread(target=_forward_ticker, daemon=True).start()


if __name__ == "__main__":
    import uvicorn

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
