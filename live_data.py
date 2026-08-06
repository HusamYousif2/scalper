"""
live_data.py — bring the minute frame up to the present moment.

The daily archive is complete but lags by a day or two, which is useless for
scalping. The public REST endpoints carry the last hours but paging two years of
trades through them would take hours and hit rate limits.

So the two are stitched: the archive supplies everything up to its last stored
day, and REST fills the gap from there to now. The result is one continuous
minute frame with exactly the columns `features.py` expects.

No API key is needed for any endpoint used here.

Order-book depth is the one exception. Binance publishes depth snapshots only in
the daily archive; live there is a current snapshot and nothing else. So a fresh
install has no trailing depth window. `poll_depth` starts building one, and until
it has a day of history the tool runs on the depth-free feature set.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

import ingest
import rate_limit as RL

FAPI = "https://fapi.binance.com"
DEPTH_DIR = os.path.join(ingest.ROOT, "data", "depth_live")

# aggTrades allows a one hour window per query and 1000 rows per page
AGG_LIMIT = 1000
DEPTH_LEVELS_PCT = [0.2, 1.0, 2.0, 5.0]

# Range-to-realised-variance constants, fitted per symbol in calibrate_rv.py.
#
# They are recorded but NOT used to feed the model. The fit showed the constant
# drifts with the aggregation window (0.037 / 0.054 / 0.104 at 1 / 15 / 60
# minutes on BTC) and leaves a 65-76 % median error whichever value is chosen.
# Tick-summed realised variance is inflated by bid-ask bounce in a way a
# high-low range simply does not capture, so substituting one for the other
# would quietly corrupt every forecast. Candles are therefore used only where
# ticks are unavailable and precision does not matter.
PARKINSON_K = {"BTCUSDT": 0.054069, "ETHUSDT": 0.090092}
DEFAULT_PARKINSON_K = 0.07


def _get(path: str, **params) -> list | dict:
    """
    One budgeted request.

    418 and 429 are treated as hard stops, not as something to retry through:
    Binance escalates the ban for every request received while it is in force, so
    the correct response is to record the ban and refuse to send until it lapses.
    """
    for attempt in range(4):
        RL.BUDGET.spend(path)
        r = requests.get(f"{FAPI}{path}", params=params, timeout=30)
        if r.status_code in (418, 429):
            retry_after = float(r.headers.get("Retry-After", 120))
            RL.BUDGET.ban_for(retry_after)
            raise RuntimeError(
                f"rate limited ({r.status_code}) on {path}; "
                f"pausing all requests for {retry_after:.0f}s"
            )
        try:
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return []


def _epoch_ms(dt: datetime) -> int:
    """
    Epoch milliseconds from a datetime.

    datetime.timestamp() interprets a NAIVE datetime as local time. On a machine
    at UTC+8 that silently shifted every request eight hours into the past and
    returned stale trades that looked perfectly valid. Timezone-aware input is
    required here, and asserted rather than assumed.
    """
    assert dt.tzinfo is not None, "timestamps must be timezone-aware UTC"
    return int(dt.timestamp() * 1000)


def _fetch_hour(symbol: str, cursor: datetime, stop: datetime) -> list:
    """All aggregated trades inside one hour window."""
    rows, from_id = [], None
    while True:
        params = dict(symbol=symbol, limit=AGG_LIMIT)
        if from_id is None:
            params.update(startTime=_epoch_ms(cursor), endTime=_epoch_ms(stop))
        else:
            params.update(fromId=from_id)
        batch = _get("/fapi/v1/aggTrades", **params)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < AGG_LIMIT or batch[-1]["T"] >= _epoch_ms(stop):
            break
        from_id = batch[-1]["a"] + 1
    return rows


def fetch_agg_trades(symbol: str, start: datetime, end: datetime,
                     workers: int = 3) -> pd.DataFrame:
    """
    Aggregated trades between two instants (timezone-aware UTC).

    Windows run in parallel, but every call still passes through the shared
    weight budget in rate_limit.py, so the pool cannot outrun the exchange's
    limit — an earlier six-worker version without that budget earned a 418 IP
    ban within minutes.

    A cold start bridging a day-long archive gap therefore takes several minutes;
    that cost is paid once, because recent_minutes caches what it fetches.
    """
    windows = []
    cursor = start
    while cursor < end:
        stop = min(cursor + timedelta(hours=1), end)
        windows.append((cursor, stop))
        cursor = stop

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch_hour, symbol, a, b): a for a, b in windows}
        for fut in as_completed(futs):
            rows.extend(fut.result())

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return pd.DataFrame({
        "price": df["p"].astype("float64"),
        "quantity": df["q"].astype("float64"),
        "transact_time": df["T"].astype("int64"),
        "is_buyer_maker": df["m"].astype(bool),
    }).drop_duplicates(subset=["transact_time", "price", "quantity"])


def fetch_klines(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    One-minute candles — the cheap bulk route for bridging a long gap.

    Weight 5 for up to 1000 candles, against weight 20 for at most 1000 trades.
    Bridging a day costs two calls here and roughly five hundred with aggTrades.

    Candles carry taker buy volume, so the buy/sell split survives. What they
    cannot carry is tick-level realised variance, whale flow, and largest trade
    size; `PARKINSON_K` below converts the high-low range into a realised
    variance estimate calibrated against the archive.
    """
    rows, cursor = [], start
    while cursor < end:
        batch = _get("/fapi/v1/klines", symbol=symbol, interval="1m",
                     startTime=_epoch_ms(cursor), endTime=_epoch_ms(end), limit=1000)
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        cursor = datetime.fromtimestamp(last_open / 1000, tz=timezone.utc) + timedelta(minutes=1)
        if len(batch) < 1000:
            break

    if not rows:
        return pd.DataFrame()
    k = pd.DataFrame(rows).iloc[:, :11]
    k.columns = ["open_time", "open", "high", "low", "close", "volume",
                 "close_time", "quote_volume", "n_trades",
                 "taker_buy_base", "taker_buy_quote"]
    ts = pd.to_datetime(k["open_time"].astype("int64"), unit="ms")
    num = k[["open", "high", "low", "close", "volume", "quote_volume",
             "n_trades", "taker_buy_base"]].astype("float64")
    num.index = ts

    out = pd.DataFrame(index=ts)
    out["open"] = num["open"]
    out["high"] = num["high"]
    out["low"] = num["low"]
    out["close"] = num["close"]
    out["volume"] = num["volume"]
    out["quote_volume"] = num["quote_volume"]
    out["n_trades"] = num["n_trades"]
    out["buy_qty"] = num["taker_buy_base"]
    out["sell_qty"] = num["volume"] - num["taker_buy_base"]
    # trade counts are not split by side in candles; apportion by volume share
    share = (out["buy_qty"] / out["volume"].replace(0.0, np.nan)).fillna(0.5)
    out["buy_cnt"] = out["n_trades"] * share
    out["sell_cnt"] = out["n_trades"] * (1 - share)
    out["vwap"] = out["quote_volume"] / out["volume"].replace(0.0, np.nan)
    # Parkinson range estimator, rescaled to match tick-summed realised variance
    rng = np.log(out["high"] / out["low"].replace(0.0, np.nan))
    out["realized_var"] = PARKINSON_K.get(symbol, DEFAULT_PARKINSON_K) * rng ** 2
    # not observable from candles; left absent so the caller can see the gap
    out["whale_buy_qty"] = np.nan
    out["whale_sell_qty"] = np.nan
    out["max_trade_qty"] = np.nan
    out.index.name = "ts"
    return out


def fetch_metrics(symbol: str, hours: int = 48) -> pd.DataFrame:
    """Open interest and positioning ratios, published every five minutes."""
    # the endpoint rejects a non-integer limit, and `hours` arrives as a float
    limit = int(min(500, max(1, round(hours * 12))))
    oi = _get("/futures/data/openInterestHist", symbol=symbol, period="5m", limit=limit)
    tk = _get("/futures/data/takerlongshortRatio", symbol=symbol, period="5m", limit=limit)
    tp = _get("/futures/data/topLongShortPositionRatio", symbol=symbol,
              period="5m", limit=limit)
    ta = _get("/futures/data/topLongShortAccountRatio", symbol=symbol,
              period="5m", limit=limit)
    gl = _get("/futures/data/globalLongShortAccountRatio", symbol=symbol,
              period="5m", limit=limit)

    def frame(raw, tcol, cols):
        if not raw:
            return pd.DataFrame()
        d = pd.DataFrame(raw)
        d["ts"] = pd.to_datetime(d[tcol].astype("int64"), unit="ms")
        return d.set_index("ts")[list(cols)].astype("float64").rename(columns=cols)

    parts = [
        frame(oi, "timestamp", {"sumOpenInterest": "sum_open_interest",
                                "sumOpenInterestValue": "sum_open_interest_value"}),
        frame(tk, "timestamp", {"buySellRatio": "sum_taker_long_short_vol_ratio"}),
        frame(tp, "timestamp", {"longShortRatio": "sum_toptrader_long_short_ratio"}),
        frame(ta, "timestamp", {"longShortRatio": "count_toptrader_long_short_ratio"}),
        frame(gl, "timestamp", {"longShortRatio": "count_long_short_ratio"}),
    ]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1).sort_index()
    return out.resample("1min").ffill()


def depth_snapshot(symbol: str) -> pd.DataFrame:
    """
    One current book snapshot, reduced to notional at each percentage band.

    IMPORTANT LIMIT: the public endpoint returns at most the top 1000 levels per
    side. On BTCUSDT that spans roughly 0.15 % of price, so every band wider than
    that returns the same number — the whole visible book. The archive's
    bookDepth is computed server-side over the entire book and has no such limit.

    Consequence: the ±1 / 2 / 5 % depth features that exist in the historical data
    CANNOT be reproduced live from public endpoints. Only near-touch liquidity is
    genuinely observable, which is why the deployed model must not depend on them.
    """
    book = _get("/fapi/v1/depth", symbol=symbol, limit=1000)
    bids = np.array(book["bids"], dtype="float64")
    asks = np.array(book["asks"], dtype="float64")
    mid = (bids[0, 0] + asks[0, 0]) / 2.0

    # utcnow() on this pandas version returns local time once made naive, which
    # silently offset every snapshot by the machine's time zone
    now_utc = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    row = {"timestamp": now_utc.floor("s")}
    for pct in DEPTH_LEVELS_PCT:
        tag = str(pct).replace(".", "p")
        lo, hi = mid * (1 - pct / 100), mid * (1 + pct / 100)
        b = bids[bids[:, 0] >= lo]
        a = asks[asks[:, 0] <= hi]
        row[f"bid_notional_{tag}"] = float((b[:, 0] * b[:, 1]).sum())
        row[f"ask_notional_{tag}"] = float((a[:, 0] * a[:, 1]).sum())
    return pd.DataFrame([row]).set_index("timestamp")


def poll_depth(symbol: str, seconds: int, every: int = 30) -> str:
    """
    Record book snapshots to disk so that trailing depth features become
    available. The archive samples every 30 s, so this matches it.
    """
    os.makedirs(os.path.join(DEPTH_DIR, symbol), exist_ok=True)
    end = time.time() + seconds
    rows = []
    while time.time() < end:
        try:
            rows.append(depth_snapshot(symbol))
        except Exception:
            pass
        time.sleep(every)
    if not rows:
        return ""
    df = pd.concat(rows).sort_index()
    day = df.index[0].strftime("%Y-%m-%d")
    path = os.path.join(DEPTH_DIR, symbol, f"{day}.parquet")
    if os.path.exists(path):
        df = pd.concat([pd.read_parquet(path), df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    df.to_parquet(path, compression="zstd")
    return path


def _load_live_depth(symbol: str) -> pd.DataFrame:
    d = os.path.join(DEPTH_DIR, symbol)
    if not os.path.isdir(d):
        return pd.DataFrame()
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in files]).sort_index()
    return df.resample("1min").mean()


CACHE_DIR = os.path.join(ingest.ROOT, "data", "live_cache")


def _cache_path(symbol: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{symbol}.parquet")


def recent_minutes(symbol: str, hours: float = 6,
                   use_cache: bool = True) -> pd.DataFrame:
    """
    Build a minute frame for the last `hours` from live endpoints only.

    Minutes fetched on earlier runs are kept on disk, so a tool polling every
    minute re-fetches one minute instead of bridging the whole archive gap again.
    """
    # kept timezone-aware all the way to the request; the minute index that comes
    # back from pd.to_datetime(unit="ms") is naive UTC, matching the archive
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(hours=hours)

    cached = pd.DataFrame()
    path = _cache_path(symbol)
    if use_cache and os.path.exists(path):
        cached = pd.read_parquet(path)
        cached = cached[cached.index >= pd.Timestamp(start).tz_localize(None)]
        if len(cached):
            # re-fetch the last cached minute, which may have been partial
            start = cached.index.max().to_pydatetime().replace(tzinfo=timezone.utc)

    minute = cached
    # A cold start bridging a day-long gap takes minutes, during which the market
    # moves on — the first version finished the bridge only to be rejected by the
    # staleness guard. So after the long fetch, top up whatever elapsed while it
    # was running. Each top-up is one or two cheap calls and converges quickly.
    for _ in range(4):
        if start >= end:
            break
        at = fetch_agg_trades(symbol, start, end)
        if at.empty:
            break
        fresh = ingest._trades_to_minute(at)
        minute = pd.concat([minute, fresh]).sort_index() if len(minute) else fresh
        minute = minute[~minute.index.duplicated(keep="last")]

        start = end
        end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        if (end - start).total_seconds() < 120:
            break

    if minute.empty:
        return pd.DataFrame()
    if use_cache:
        minute.to_parquet(path, compression="zstd")

    depth = _load_live_depth(symbol)
    if len(depth):
        minute = minute.join(depth, how="left")

    mt = fetch_metrics(symbol, hours=max(2, hours))
    if len(mt):
        minute = minute.join(mt, how="left")
    minute.index.name = "ts"
    return minute


MAX_LIVE_HOURS = 30      # beyond this, paging REST is slower than the archive


def ensure_archive_current(symbol: str) -> pd.Timestamp:
    """
    Download any archive days that have appeared since the last run.

    Without this the archive can trail the present by days, and every rolling
    feature window then spans a hole. The features go NaN, the engine falls back
    to the newest COMPLETE row, and it silently reports on a market that closed
    days ago — which is exactly what happened before this function existed.
    """
    from datetime import timedelta as td

    d = os.path.join(ingest.MINUTE_DIR, symbol)
    stored = sorted(f[:-8] for f in os.listdir(d) if f.endswith(".parquet"))
    last = pd.Timestamp(stored[-1]).date()
    newest = (datetime.now(timezone.utc) - td(days=1)).date()
    if last < newest:
        ingest.build_range(symbol, last + td(days=1), newest, workers=3)
    return pd.Timestamp(newest)


def load_recent_archive(symbol: str, tail_days: int) -> pd.DataFrame:
    """Only the last `tail_days` stored days — loading all 730 takes minutes."""
    d = os.path.join(ingest.MINUTE_DIR, symbol)
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))[-tail_days:]
    df = pd.concat([pd.read_parquet(os.path.join(d, f)) for f in files]).sort_index()
    return df[~df.index.duplicated(keep="first")]


_FRAME_CACHE: dict[tuple, tuple[float, pd.DataFrame]] = {}
_FRAME_LOCK = threading.Lock()
FRAME_TTL_SECONDS = 90


def combined_frame_cached(symbol: str, live_hours: float = 3,
                          tail_days: int = 14) -> pd.DataFrame:
    """
    Shared, short-lived cache around combined_frame.

    Four endpoints each called combined_frame independently, so a single page
    load bridged the archive gap four times over — minutes of duplicated,
    rate-limited work for identical data. They now share one build.
    """
    key = (symbol, tail_days)
    now = time.time()
    hit = _FRAME_CACHE.get(key)
    if hit and now - hit[0] < FRAME_TTL_SECONDS:
        return hit[1]
    with _FRAME_LOCK:
        hit = _FRAME_CACHE.get(key)
        if hit and time.time() - hit[0] < FRAME_TTL_SECONDS:
            return hit[1]
        df = combined_frame(symbol, live_hours=live_hours, tail_days=tail_days)
        _FRAME_CACHE[key] = (time.time(), df)
        return df


def combined_frame(symbol: str, live_hours: float = 6,
                   update_archive: bool = True,
                   tail_days: int = 14) -> pd.DataFrame:
    """
    Archive up to its last stored day, then live REST to the present, with no
    hole in between. Returns one continuous minute frame ready for the feature
    builder.

    `tail_days` bounds how much history is loaded. The longest trailing window in
    the feature set is the 168-lag HAR term, which at a 60 minute horizon spans
    seven days, so fourteen leaves comfortable headroom while keeping the whole
    call to a couple of seconds instead of several minutes.
    """
    if update_archive:
        ensure_archive_current(symbol)
    hist = load_recent_archive(symbol, tail_days)

    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    gap_hours = (now - hist.index.max()).total_seconds() / 3600.0
    need = min(MAX_LIVE_HOURS, max(live_hours, gap_hours + 0.5))

    live = recent_minutes(symbol, hours=need)
    if live.empty:
        return hist
    live = live[live.index > hist.index.max()]
    out = pd.concat([hist, live]).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    remaining = (now - out.index.max()).total_seconds() / 60.0
    if remaining > 10:
        raise RuntimeError(
            f"{symbol}: data ends {remaining:.0f} minutes ago; the archive and "
            f"the live feed could not be joined without a hole"
        )
    return out


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    hrs = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0

    t0 = time.time()
    live = recent_minutes(sym, hours=hrs)
    print(f"{sym}: fetched {len(live)} live minutes in {time.time() - t0:.0f}s")
    if len(live):
        print(f"  range   : {live.index.min()} -> {live.index.max()}")
        print(f"  columns : {len(live.columns)}")
        print(live[["close", "volume", "buy_qty", "sell_qty",
                    "n_trades", "realized_var"]].tail(3).to_string())
        have_oi = "sum_open_interest" in live.columns
        print(f"  open interest present: {have_oi}")

    snap = depth_snapshot(sym)
    print("\n  current book snapshot:")
    print(snap.T.to_string())
