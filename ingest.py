"""
ingest.py — download Binance USD-M futures daily archives and reduce them to a
compact 1-minute microstructure table (one parquet file per symbol per day).

Raw archives are large (aggTrades ~6 MB/day zipped). We never keep them: each day
is downloaded, reduced to 1440 rows, written to parquet, and the raw bytes dropped.

Datasets used (all free, no API key):
  aggTrades  - every aggregated trade: price, qty, time, is_buyer_maker
  bookDepth  - order book notional at +/-0.2, 1, 2, 3, 4, 5 percent, every 30s
  metrics    - open interest and long/short positioning ratios, every 5 min
"""

import io
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

BASE = "https://data.binance.vision/data/futures/um/daily"
ROOT = os.path.expanduser("~/crypto-quant-lab/scalper")
MINUTE_DIR = os.path.join(ROOT, "data", "minute")

# a trade this large (in USD) is treated as institutional / whale flow
WHALE_NOTIONAL = 100_000.0

# order book levels we keep (percent away from mid price)
DEPTH_LEVELS = [0.2, 1.0, 2.0, 5.0]


# only these columns are ever used; skipping the rest roughly halves the peak
# memory of parsing a day of trades, which matters on a 3 GB machine
READ_OPTS = {
    "aggTrades": dict(
        usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
        dtype={"price": "float64", "quantity": "float64",
               "transact_time": "int64", "is_buyer_maker": "bool"},
    ),
    "bookDepth": dict(
        usecols=["timestamp", "percentage", "notional"],
        dtype={"percentage": "float32", "notional": "float64"},
    ),
    "metrics": {},
}


def _fetch_zip(dataset: str, symbol: str, day: date) -> pd.DataFrame | None:
    """Download one daily zip and return its single CSV as a DataFrame."""
    url = f"{BASE}/{dataset}/{symbol}/{symbol}-{dataset}-{day.isoformat()}.zip"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=180)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            payload = r.content
            del r
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                raw = z.read(z.namelist()[0])
            del payload
            return pd.read_csv(io.BytesIO(raw), **READ_OPTS.get(dataset, {}))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))
    return None


def _trades_to_minute(at: pd.DataFrame) -> pd.DataFrame:
    """Reduce the raw aggregated-trade tape to 1-minute order-flow features."""
    at = at.sort_values("transact_time")
    ts = pd.to_datetime(at["transact_time"], unit="ms")
    price = at["price"].to_numpy(dtype="float64")
    qty = at["quantity"].to_numpy(dtype="float64")
    notional = price * qty

    # is_buyer_maker == True  -> the buyer sat in the book, the taker was SELLING
    # is_buyer_maker == False -> the taker was BUYING
    taker_buy = ~at["is_buyer_maker"].to_numpy(dtype=bool)

    # squared tick-by-tick log returns; summed per minute this is realized variance
    logp = np.log(price)
    r2 = np.empty_like(logp)
    r2[0] = 0.0
    r2[1:] = np.diff(logp) ** 2

    whale = notional >= WHALE_NOTIONAL

    df = pd.DataFrame(
        {
            "price": price,
            "qty": qty,
            "notional": notional,
            "buy_qty": np.where(taker_buy, qty, 0.0),
            "sell_qty": np.where(taker_buy, 0.0, qty),
            "buy_cnt": taker_buy.astype("float64"),
            "sell_cnt": (~taker_buy).astype("float64"),
            "whale_buy_qty": np.where(whale & taker_buy, qty, 0.0),
            "whale_sell_qty": np.where(whale & ~taker_buy, qty, 0.0),
            "r2": r2,
        },
        index=ts,
    )

    g = df.resample("1min")
    out = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["qty"].sum(),
            "quote_volume": g["notional"].sum(),
            "n_trades": g["qty"].count().astype("float64"),
            "buy_qty": g["buy_qty"].sum(),
            "sell_qty": g["sell_qty"].sum(),
            "buy_cnt": g["buy_cnt"].sum(),
            "sell_cnt": g["sell_cnt"].sum(),
            "whale_buy_qty": g["whale_buy_qty"].sum(),
            "whale_sell_qty": g["whale_sell_qty"].sum(),
            "max_trade_qty": g["qty"].max(),
            "realized_var": g["r2"].sum(),
        }
    )
    out["vwap"] = out["quote_volume"] / out["volume"].replace(0.0, np.nan)
    return out


def _depth_to_minute(bd: pd.DataFrame) -> pd.DataFrame:
    """Reduce order-book depth snapshots to 1-minute average notional per level."""
    bd = bd.copy()
    bd["timestamp"] = pd.to_datetime(bd["timestamp"])
    wide = bd.pivot_table(
        index="timestamp", columns="percentage", values="notional", aggfunc="mean"
    )
    keep = {}
    for lv in DEPTH_LEVELS:
        if -lv in wide.columns and lv in wide.columns:
            tag = str(lv).replace(".", "p")
            keep[f"bid_notional_{tag}"] = wide[-lv]
            keep[f"ask_notional_{tag}"] = wide[lv]
    if not keep:
        return pd.DataFrame()
    return pd.DataFrame(keep).resample("1min").mean()


def _metrics_to_minute(mt: pd.DataFrame) -> pd.DataFrame:
    """Expand the 5-minute positioning metrics onto a 1-minute grid (forward fill)."""
    mt = mt.copy()
    mt["create_time"] = pd.to_datetime(mt["create_time"])
    mt = mt.set_index("create_time").sort_index()
    cols = [
        "sum_open_interest",
        "sum_open_interest_value",
        "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio",
        "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ]
    cols = [c for c in cols if c in mt.columns]
    return mt[cols].apply(pd.to_numeric, errors="coerce").resample("1min").ffill()


def build_day(symbol: str, day: date) -> str | None:
    """Download + reduce one day. Returns the parquet path, or None if unavailable."""
    out_dir = os.path.join(MINUTE_DIR, symbol)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{day.isoformat()}.parquet")
    if os.path.exists(out_path):
        return out_path

    at = _fetch_zip("aggTrades", symbol, day)
    if at is None or at.empty:
        return None
    minute = _trades_to_minute(at)
    del at

    bd = _fetch_zip("bookDepth", symbol, day)
    if bd is not None and not bd.empty:
        minute = minute.join(_depth_to_minute(bd), how="left")

    mt = _fetch_zip("metrics", symbol, day)
    if mt is not None and not mt.empty:
        minute = minute.join(_metrics_to_minute(mt), how="left")

    minute.index.name = "ts"
    minute.to_parquet(out_path, compression="zstd")
    return out_path


def build_range(symbol: str, start: date, end: date, workers: int = 3) -> None:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [
        d
        for d in days
        if not os.path.exists(
            os.path.join(MINUTE_DIR, symbol, f"{d.isoformat()}.parquet")
        )
    ]
    print(f"{symbol}: {len(days)} days requested, {len(todo)} missing", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(build_day, symbol, d): d for d in todo}
        for fut in as_completed(futs):
            d = futs[fut]
            try:
                ok = fut.result()
            except Exception as e:
                print(f"  FAIL {d}: {type(e).__name__}: {e}", flush=True)
                continue
            done += 1
            if ok is None:
                print(f"  missing on server: {d}", flush=True)
            if done % 25 == 0:
                print(f"  {done}/{len(todo)} days done", flush=True)
    print(f"{symbol}: finished", flush=True)


def load_minutes(symbol: str) -> pd.DataFrame:
    """Concatenate every stored day for a symbol into one sorted minute table."""
    d = os.path.join(MINUTE_DIR, symbol)
    files = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
    parts = [pd.read_parquet(os.path.join(d, f)) for f in files]
    df = pd.concat(parts).sort_index()
    return df[~df.index.duplicated(keep="first")]


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    ndays = int(sys.argv[2]) if len(sys.argv) > 2 else 730
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    end = date.today() - timedelta(days=2)
    build_range(symbol, end - timedelta(days=ndays - 1), end, workers=workers)
