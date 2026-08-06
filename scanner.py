"""
scanner.py — which market is worth your attention right now?

The single-symbol report answers a question the user already had. The scanner
answers the one they actually start the day with: out of everything I could
trade, where is there anything worth trading at all?

Its most valuable output is the count at the bottom — how many symbols are NOT
worth touching at this moment. A tool that says "38 of 50 are dead right now"
saves more money than one that highlights the other twelve.

Nothing here needs predictive power beyond what is already validated. It is the
same conformalised volatility model applied across symbols and ranked. The
ranking key is the expected move minus whatever round trip the user actually
pays, because a 30 bps move on an illiquid pair with a 25 bps spread is worse
than a 12 bps move on BTC.

Symbols are scored in parallel, but the shared request budget in rate_limit.py
still governs every call, so the scan cannot outrun the exchange's limits.
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import assess as A
import ingest

DEFAULT_HORIZON = 15
MAX_WORKERS = 3


def available_symbols() -> list[str]:
    """Symbols with both stored data and trained models."""
    have_data = set(os.listdir(ingest.MINUTE_DIR)) if os.path.isdir(ingest.MINUTE_DIR) else set()
    out = []
    for s in sorted(have_data):
        stem = os.path.join(A.TV.MODEL_DIR, f"vol_{s}_{DEFAULT_HORIZON}m.pkl")
        if os.path.exists(stem):
            out.append(s)
    return out


def scan_one(symbol: str, horizon: int, fees) -> dict:
    try:
        a = A.assess(symbol, horizon, fees)
    except Exception as e:
        return {"symbol": symbol, "error": f"{type(e).__name__}: {e}"}

    q = a.get("quantiles_bps") or {}
    flow = a.get("order_flow") or {}
    return {
        "symbol": symbol,
        "price": a["price"],
        "expected_bps": a["expected_move_bps"],
        "margin_bps": a["edge_bps"],
        "verdict": a["verdict"].split("—")[0].strip(),
        "vol_pctile": a["vol_percentile_30d"],
        "stop_bps": q.get("q90"),
        "target_bps": q.get("q75"),
        "regime": (a.get("regime") or {}).get("label", "").split("—")[0].strip(),
        "flow_imb": flow.get("taker_imbalance"),
        "intensity": flow.get("trade_intensity_vs_day"),
        "as_of": a["as_of_utc"],
    }


def scan(symbols: list[str] | None = None, horizon: int = DEFAULT_HORIZON,
         fees=A.DEFAULT_FEES, workers: int = MAX_WORKERS) -> pd.DataFrame:
    syms = symbols or available_symbols()
    if not syms:
        raise RuntimeError("no symbols with both data and a trained model")

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(scan_one, s, horizon, fees): s for s in syms}
        for fut in as_completed(futs):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    df.attrs["seconds"] = time.time() - t0
    df.attrs["horizon"] = horizon
    df.attrs["fees"] = fees
    ok = df[df.get("error").isna()] if "error" in df.columns else df
    if "margin_bps" in ok.columns and ok["margin_bps"].notna().any():
        ok = ok.sort_values("margin_bps", ascending=False)
    else:
        ok = ok.sort_values("expected_bps", ascending=False)
    bad = df[df["error"].notna()] if "error" in df.columns else df.iloc[0:0]
    return pd.concat([ok, bad], ignore_index=True)


def render(df: pd.DataFrame) -> str:
    horizon = df.attrs.get("horizon", DEFAULT_HORIZON)
    fees = df.attrs.get("fees")
    cost, label = A.resolve_cost(fees)

    L = ["=" * 88]
    L.append(f"  MARKET SCAN   next {horizon} minutes"
             f"   |   cost assumption: {label}")
    L.append("=" * 88)
    L.append("")
    L.append(f"  {'symbol':<10}{'price':>11}{'expected':>10}{'margin':>9}"
             f"{'stop':>8}{'target':>8}{'vol%':>6}  {'verdict':<10}{'regime':<10}")
    L.append("  " + "-" * 84)

    good = df[df.get("error").isna()] if "error" in df.columns else df
    for _, r in good.iterrows():
        margin = f"{r['margin_bps']:+8.1f}" if pd.notna(r.get("margin_bps")) else "       -"
        stop = f"{r['stop_bps']:7.1f}" if pd.notna(r.get("stop_bps")) else "      -"
        tgt = f"{r['target_bps']:7.1f}" if pd.notna(r.get("target_bps")) else "      -"
        L.append(f"  {r['symbol']:<10}{r['price']:>11,.4g}{r['expected_bps']:>10.1f}"
                 f"{margin}{stop}{tgt}{r['vol_pctile']:>6.0f}  "
                 f"{r['verdict']:<10}{r.get('regime', ''):<10}")

    if cost is not None and "margin_bps" in good.columns:
        dead = int((good["margin_bps"] <= 0).sum())
        L.append("")
        L.append(f"  {dead} of {len(good)} symbols do not cover their round trip"
                 f" right now — the useful half of this table")
    errs = df[df["error"].notna()] if "error" in df.columns else df.iloc[0:0]
    for _, r in errs.iterrows():
        L.append(f"  {r['symbol']:<10} unavailable: {r['error']}")

    L.append("")
    L.append(f"  scanned in {df.attrs.get('seconds', 0):.0f}s"
             f"   |   expected move, stop and target are in basis points")
    L.append("  Direction is not shown. It was measured and rejected; see FINDINGS.md.")
    L.append("=" * 88)
    return "\n".join(L)


if __name__ == "__main__":
    hor = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_HORIZON
    fee = sys.argv[2] if len(sys.argv) > 2 else "futures_maker"
    if fee.lower() in ("none", "-", "off"):
        fee = None
    print(render(scan(horizon=hor, fees=fee)))
