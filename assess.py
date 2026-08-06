"""
assess.py — the assessment engine. This is what the product actually shows.

It answers the question a scalper has to answer before every trade, and that no
indicator on a chart answers: is this moment worth trading at all, and if so, how
far is price likely to travel and where do the stop and target belong.

It deliberately does NOT say "buy" or "sell". Two years of testing in this repo
established that directional edge on BTC and ETH at these horizons is worth 1-4
basis points against a 4-20 basis point cost wall. A tool that emits direction
signals would be selling its users a losing position. A tool that sizes the
opportunity and tells them when to stand aside is selling them something true.

Everything below is computed from live public data with no API key, and every
number is shown next to the benchmark or cost it must beat.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import features as FE
import features2 as F2
import live_data as LD
import ta_readout as TA
import train_quantile as TQ
import train_vol as TV
import vol_model as V

# round-trip cost in basis points, both sides included
FEE_MODELS = {
    "spot_taker": 20.0,
    "spot_bnb": 15.0,
    "futures_taker": 10.5,
    "futures_maker": 4.0,
}
DEFAULT_FEES = "futures_taker"

# Cost is the user's business, not the tool's. What it costs to get in and out
# depends on the venue, the tier, whether the order rests or crosses, and how
# much size is behind it — none of which this tool can know. So the analysis is
# reported as measurements first, and the cost comparison is an optional overlay
# the user supplies. Pass a number of basis points instead of a preset name to
# use your own.
ANALYTICAL_ONLY = None

# an assessment older than this describes a market that has already moved on
MAX_AGE_MINUTES = 20

# E|r| for a zero-mean normal is sigma * sqrt(2/pi)
ABS_FACTOR = np.sqrt(2 / np.pi)


def calibrated_move(sigma_bps: float, meta: dict) -> float:
    """
    Turn a predicted volatility into the expected absolute move, using the
    monotone curve measured on held-out data rather than the textbook
    sigma * sqrt(2/pi). The measured ratio runs from about 3.0 at the calm end to
    about 1.2 at the wild end, so the textbook constant would understate quiet
    periods by roughly threefold — and quiet periods are where this tool's
    "stand aside" verdict lives.
    """
    xs = meta.get("calib_pred_bps")
    ys = meta.get("calib_actual_bps")
    if not xs or not ys:
        return sigma_bps * meta.get("move_factor", float(np.sqrt(2 / np.pi)))
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if sigma_bps <= xs[0]:
        # extend the first segment's slope rather than clamping flat
        slope = ys[0] / xs[0] if xs[0] > 0 else 1.0
        return float(sigma_bps * slope)
    if sigma_bps >= xs[-1]:
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2]) if xs[-1] > xs[-2] else 1.0
        return float(ys[-1] + (sigma_bps - xs[-1]) * slope)
    return float(np.interp(sigma_bps, xs, ys))


def load_model(symbol: str, horizon: int):
    stem = os.path.join(TV.MODEL_DIR, f"vol_{symbol}_{horizon}m")
    if not os.path.exists(stem + ".pkl"):
        raise FileNotFoundError(
            f"no trained model for {symbol} {horizon}m — run train_vol.py first"
        )
    with open(stem + ".pkl", "rb") as f:
        mdl = pickle.load(f)
    with open(stem + ".json") as f:
        meta = json.load(f)
    return mdl, meta


def build_live_features(symbol: str, horizon: int, minute: pd.DataFrame
                        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the supplied minute frame through the identical feature code used in
    training. Returns (features, gridded minute frame).

    The minute frame is passed in rather than fetched here: building it costs a
    REST round trip and a parquet load, and this function used to trigger both
    twice per assessment.
    """
    base = FE.build_features(symbol, horizon=horizon, minute=minute)
    m = FE.build_minute_frame(symbol, minute=minute)
    extra = F2.build_extra(m, ["IND", "SR", "VP", "CDL"])
    df = base.join(extra, how="left")

    # HAR lags, exactly as in training
    rv = m["realized_var"]
    for name, mult in V.HAR_LAGS.items():
        w = horizon * mult
        lag = np.sqrt(rv.rolling(w, min_periods=max(2, w // 2)).sum() / mult)
        df[f"har_{name}"] = np.log(lag.replace(0.0, np.nan))
    return df.replace([np.inf, -np.inf], np.nan), m


def order_flow_read(m: pd.DataFrame, window: int = 15) -> dict:
    """A plain-language summary of who is currently hitting the book."""
    tail = m.tail(window)
    buy, sell = tail["buy_qty"].sum(), tail["sell_qty"].sum()
    imb = (buy - sell) / (buy + sell) if (buy + sell) > 0 else 0.0
    wbuy, wsell = tail["whale_buy_qty"].sum(), tail["whale_sell_qty"].sum()
    wimb = (wbuy - wsell) / (wbuy + wsell) if (wbuy + wsell) > 0 else 0.0

    day = m.tail(1440)
    intensity = (tail["n_trades"].mean() / day["n_trades"].mean()
                 if day["n_trades"].mean() > 0 else np.nan)

    oi_chg = np.nan
    if "sum_open_interest" in m.columns and m["sum_open_interest"].notna().any():
        oi = m["sum_open_interest"].dropna()
        if len(oi) > window:
            oi_chg = float(oi.iloc[-1] / oi.iloc[-window] - 1) * 100

    ret = float(np.log(tail["close"].iloc[-1] / tail["close"].iloc[0]) * 1e4)
    return {
        "window_min": window,
        "taker_imbalance": float(imb),
        "whale_imbalance": float(wimb),
        "trade_intensity_vs_day": float(intensity),
        "open_interest_change_pct": oi_chg,
        "price_move_bps": ret,
    }


def regime_read(m: pd.DataFrame, window: int = 240) -> dict:
    """
    Describe the current regime. Deliberately NOT a prediction.

    A gradient-boosted classifier trained to predict the regime an hour ahead
    scored 48.0 % against 47.0 % for simply assuming the regime persists — one
    point, on fifteen thousand samples. That does not earn the complexity, so
    the tool measures what IS and states honestly how often it lasts.
    """
    close = m["close"]
    net = abs(close.iloc[-1] - close.iloc[-window])
    path = close.diff().abs().tail(window).sum()
    eff = float(net / path) if path > 0 else np.nan

    hist_net = (close - close.shift(window)).abs()
    hist_path = close.diff().abs().rolling(window, min_periods=window // 2).sum()
    hist_eff = (hist_net / hist_path.replace(0.0, np.nan)).dropna().tail(43200)
    pct = float((hist_eff < eff).mean() * 100) if len(hist_eff) else np.nan

    if pct >= 66:
        label = "TRENDING — price is travelling in one direction"
    elif pct <= 33:
        label = "CHOPPY — price is covering ground and going nowhere"
    else:
        label = "NEUTRAL — no clear character"
    return {"efficiency_ratio": eff, "percentile_30d": pct, "label": label,
            "persistence_1h_pct": 47.0}


def positioning_read(m: pd.DataFrame) -> dict:
    out = {}
    for col, label in [
        ("count_long_short_ratio", "retail_long_short"),
        ("sum_toptrader_long_short_ratio", "top_trader_long_short"),
        ("sum_taker_long_short_vol_ratio", "taker_buy_sell_volume"),
    ]:
        if col in m.columns and m[col].notna().any():
            s = m[col].dropna()
            out[label] = float(s.iloc[-1])
            if len(s) > 288:
                out[label + "_pctile_1d"] = float((s.tail(288) < s.iloc[-1]).mean() * 100)
    return out


def resolve_cost(fees) -> tuple[float | None, str]:
    """
    Accept a preset name, a number of basis points, or None.

    None means analysis only: report what the market is expected to do and let
    the reader apply their own cost. A number lets a user on a venue or tier this
    tool has never heard of get a correct verdict.
    """
    if fees is None:
        return None, "not applied"
    if isinstance(fees, (int, float)):
        return float(fees), f"{float(fees):.1f} bps (user supplied)"
    if fees in FEE_MODELS:
        return FEE_MODELS[fees], fees
    try:
        return float(fees), f"{float(fees):.1f} bps (user supplied)"
    except (TypeError, ValueError):
        raise ValueError(
            f"unknown fee model {fees!r}; use one of {list(FEE_MODELS)}, "
            f"a number of basis points, or none"
        )


def assess(symbol: str, horizon: int = 15, fees=DEFAULT_FEES,
           live_hours: float = 3.0) -> dict:
    mdl, meta = load_model(symbol, horizon)
    minute = LD.combined_frame_cached(symbol, live_hours=live_hours)
    df, m = build_live_features(symbol, horizon, minute)

    cols = meta["features"]
    missing = [c for c in cols if c not in df.columns]
    row = df.dropna(subset=[c for c in cols if c in df.columns]).tail(1)
    if row.empty:
        raise RuntimeError("no complete feature row available")

    # A stale assessment is worse than none: it looks entirely plausible and
    # describes a market that has already moved on. Refuse rather than mislead.
    now = pd.Timestamp(datetime.now(timezone.utc).replace(tzinfo=None))
    age_min = (now - row.index[-1]).total_seconds() / 60.0
    if age_min > MAX_AGE_MINUTES:
        raise RuntimeError(
            f"newest complete feature row is {age_min:.0f} minutes old "
            f"(limit {MAX_AGE_MINUTES}). Refusing to publish a stale assessment."
        )

    x = row[cols].to_numpy("float32") if not missing else None
    if missing:
        # keep serving rather than fail, but say so loudly in the output
        for c in missing:
            df[c] = np.nan
        x = row.reindex(columns=cols).to_numpy("float32")

    log_sigma = float(mdl.predict(x)[0])
    sigma_bps = float(np.exp(log_sigma) * 1e4)
    exp_move_bps = calibrated_move(sigma_bps, meta)

    lo = calibrated_move(float(np.exp(log_sigma + meta["resid_q10"]) * 1e4), meta)
    hi = calibrated_move(float(np.exp(log_sigma + meta["resid_q90"]) * 1e4), meta)

    cost, cost_label = resolve_cost(fees)
    edge = (exp_move_bps - cost) if cost is not None else np.nan

    price = float(m["close"].iloc[-1])

    # Technical readings, restricted to those that passed the persistence and
    # spread tests in ta_readout.py. Each carries the move that historically
    # followed readings in the same bucket, because a bare indicator value is not
    # information.
    try:
        ta = TA.readout(symbol, minute=minute)
        ta_rows = ta.head(6).to_dict("records") if len(ta) else []
    except Exception:
        ta_rows = []

    # Conformalised quantiles of the absolute move. These replace the guessed
    # multiples that used to set the stop and target: a stop belongs at a high
    # quantile of the move distribution, and a quantile is exactly that. Measured
    # coverage error on held-out data is under one percentage point on both
    # symbols and both horizons.
    quant = None
    try:
        qmodels, qmeta = TQ.load(symbol, horizon)
        quant = TQ.predict(qmodels, qmeta, row)
        quant_coverage = qmeta["verified_coverage_%"]
        quant_ci = qmeta.get("coverage_95ci_pts", {})
        quant_ref = (qmeta.get("walkforward_worst_error_pts"),
                     qmeta.get("walkforward_n"))
    except Exception as e:
        quant_coverage, quant_ci, quant_ref = {"error": str(e)}, {}, (None, None)

    # where does this sit against the last month of the model's own forecasts?
    hist_sigma = np.sqrt(
        m["realized_var"].rolling(horizon, min_periods=horizon // 2).sum()
    ).dropna()
    pctile = float((hist_sigma.tail(43200) < np.exp(log_sigma)).mean() * 100)

    return {
        "symbol": symbol,
        "horizon_min": horizon,
        "as_of_utc": str(row.index[-1]),
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "price": price,
        "expected_move_bps": exp_move_bps,
        "expected_move_usd": price * exp_move_bps / 1e4,
        "range_bps": (lo, hi),
        "sigma_bps": sigma_bps,
        "vol_percentile_30d": pctile,
        "fee_model": cost_label,
        "cost_bps": cost,
        "edge_bps": edge,
        "verdict": _verdict(edge, cost) if cost is not None
        else "ANALYSIS ONLY — apply your own round-trip cost",
        # above 100 % the number is not a target, it is a proof of impossibility:
        # the fee exceeds the whole expected move, so a flawless call still loses
        "breakeven_accuracy_pct": _breakeven(exp_move_bps, cost)
        if cost is not None else float("nan"),
        # what the expected move would have to beat, at a range of costs, so the
        # reader can locate their own venue without rerunning anything
        "cost_table": {k: round(exp_move_bps - v, 1) for k, v in FEE_MODELS.items()},
        # Both derive from the CALIBRATED expected move, not raw sigma. Deriving
        # them from sigma put the stop below the typical move, which would have
        # been hit by ordinary noise on almost every trade.
        # A stop of one expected move absorbs a normal adverse excursion; a
        # target of 1.5 leaves the trade positive after the round trip whenever
        # the move is average or better.
        # from the measured distribution when it is available, otherwise the old
        # multiples of the expected move as a fallback
        "suggested_stop_bps": quant["q90"] if quant else exp_move_bps * 1.0,
        "suggested_target_bps": quant["q75"] if quant else exp_move_bps * 1.5,
        "quantiles_bps": quant,
        "quantile_verified_coverage": quant_coverage,
        "quantile_coverage_ci": quant_ci,
        "quantile_walkforward_ref": quant_ref,
        "model_valid_r2": meta["valid_r2"],
        "benchmark_har_r2": meta["har_r2"],
        "model_trained_at": meta["trained_at"],
        "missing_features": missing,
        "order_flow": order_flow_read(m),
        "technical": ta_rows,
        "technical_rejected": len(TA.REJECTED),
        "regime": regime_read(m),
        "positioning": positioning_read(m),
    }


def _breakeven(exp_move_bps: float, cost: float) -> float:
    if exp_move_bps <= 0:
        return float("nan")
    p = (1 + cost / exp_move_bps) / 2 * 100
    return p if p <= 100 else float("inf")


def _verdict(edge: float, cost: float) -> str:
    if edge <= 0:
        return "STAND ASIDE — expected move does not cover the round trip"
    if edge < cost * 0.5:
        return "MARGINAL — move barely clears costs, no room for error"
    if edge < cost:
        return "WORKABLE — move clears costs with some room"
    return "ACTIVE — expected move is well above the cost of trading"


def render(a: dict) -> str:
    lo, hi = a["range_bps"]
    L = []
    L.append("=" * 74)
    L.append(f"  {a['symbol']}   next {a['horizon_min']} minutes"
             f"   |   price {a['price']:,.1f}")
    L.append(f"  data as of {a['as_of_utc']} UTC"
             f"   (generated {a['generated_utc']})")
    L.append("=" * 74)
    L.append("")
    L.append(f"  VERDICT   {a['verdict']}")
    L.append("")
    L.append(f"  expected move        {a['expected_move_bps']:6.1f} bps"
             f"   (${a['expected_move_usd']:,.0f})")
    if not a.get("quantiles_bps"):
        # only shown when the measured distribution is unavailable; printing both
        # put two disagreeing intervals in the same report
        L.append(f"  likely range         {lo:6.1f} - {hi:.1f} bps"
                 f"   (residual-based estimate)")
    if a["cost_bps"] is not None:
        L.append(f"  round-trip cost      {a['cost_bps']:6.1f} bps"
                 f"   [{a['fee_model']}]")
        L.append(f"  margin over cost     {a['edge_bps']:+6.1f} bps")
        be = a["breakeven_accuracy_pct"]
        if np.isinf(be):
            L.append("  accuracy needed      IMPOSSIBLE"
                     "   the fee exceeds the entire expected move")
        else:
            L.append(f"  accuracy needed      {be:6.1f} %   just to break even")
    else:
        L.append("  margin left after a round trip costing:")
        for k, v in a["cost_table"].items():
            L.append(f"      {k:<16} {v:+6.1f} bps")
    L.append("")
    L.append(f"  volatility now       {a['vol_percentile_30d']:5.0f}th percentile"
             f" of the last 30 days")
    q = a.get("quantiles_bps")
    if q:
        cov = a.get("quantile_verified_coverage") or {}
        ci = a.get("quantile_coverage_ci") or {}
        L.append("")
        L.append("  DISTRIBUTION OF THE COMING MOVE  (conformalised, verified)")
        labels = [("q10", "10 % of moves stay under"),
                  ("q25", "25 % stay under"),
                  ("q50", "median move"),
                  ("q75", "75 % stay under"),
                  ("q90", "90 % stay under")]
        for k, text in labels:
            c, e = cov.get(k), ci.get(k)
            if isinstance(c, (int, float)) and isinstance(e, (int, float)):
                extra = f"   [recent slice {c:.1f} % +/- {e:.1f}]"
            elif isinstance(c, (int, float)):
                extra = f"   [recent slice {c:.1f} %]"
            else:
                extra = ""
            L.append(f"    {text:<26} {q[k]:6.1f} bps{extra}")
        L.append("")
        L.append(f"  a stop at {q['q90']:.1f} bps survives 90 % of outcomes")
        L.append(f"  a target at {q['q75']:.1f} bps is reached by 25 % of moves")
        wf, wn = a.get("quantile_walkforward_ref", (None, None))
        if wf is not None:
            L.append(f"  coverage measured to within {wf:.1f} points over"
                     f" {wn:,} out-of-sample observations")
    else:
        L.append(f"  suggested stop       {a['suggested_stop_bps']:6.1f} bps")
        L.append(f"  suggested target     {a['suggested_target_bps']:6.1f} bps")
    L.append("")
    ta = a.get("technical") or []
    if ta:
        L.append("  TECHNICAL READINGS  (only those that passed validation)")
        L.append(f"    {'reading':<24}{'pctile':>8}{'followed':>10}"
                 f"{'usual':>8}{'x':>7}")
        for r in ta:
            L.append(f"    {r['indicator']:<24}{r['percentile']:>7.0f}%"
                     f"{r['followed_bps']:>9.1f}{r['baseline_bps']:>8.1f}"
                     f"{r['amplification']:>7.2f}")
        L.append(f"    {a.get('technical_rejected', 0)} popular readings are"
                 f" withheld: they failed the persistence test")
        L.append("")
    rg = a["regime"]
    L.append(f"  REGIME    {rg['label']}")
    L.append(f"    efficiency ratio   {rg['efficiency_ratio']:.3f}"
             f"   ({rg['percentile_30d']:.0f}th pctile of 30 days)")
    L.append(f"    this regime still holds an hour later "
             f"{rg['persistence_1h_pct']:.0f} % of the time")
    L.append("")
    f = a["order_flow"]
    L.append(f"  ORDER FLOW  (last {f['window_min']} minutes)")
    L.append(f"    taker imbalance    {f['taker_imbalance']:+.3f}"
             f"   (+1 all buying, -1 all selling)")
    L.append(f"    large-trade flow   {f['whale_imbalance']:+.3f}")
    L.append(f"    trade intensity    {f['trade_intensity_vs_day']:.2f}x"
             f" the daily average")
    if not np.isnan(f["open_interest_change_pct"]):
        L.append(f"    open interest      {f['open_interest_change_pct']:+.2f} %")
    L.append(f"    price moved        {f['price_move_bps']:+.1f} bps")
    p = a["positioning"]
    if p:
        L.append("")
        L.append("  POSITIONING")
        for k, v in p.items():
            if k.endswith("_pctile_1d"):
                continue
            pc = p.get(k + "_pctile_1d")
            extra = f"   ({pc:.0f}th pctile of today)" if pc is not None else ""
            L.append(f"    {k:<24} {v:.3f}{extra}")
    L.append("")
    L.append(f"  model out-of-sample R2 {a['model_valid_r2']:.3f}"
             f"  vs benchmark HAR {a['benchmark_har_r2']:.3f}")
    L.append(f"  model trained {a['model_trained_at']}")
    if a["missing_features"]:
        L.append(f"  WARNING: {len(a['missing_features'])} features unavailable live")
    L.append("")
    L.append("  This tool does not predict direction. Direction on liquid majors")
    L.append("  at these horizons is worth 1-4 bps against a 4-20 bps cost wall;")
    L.append("  see FINDINGS.md. It sizes the opportunity and the cost of taking it.")
    L.append("=" * 74)
    return "\n".join(L)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    hor = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    # third argument: a preset name, a number of basis points, or "none"
    fee = sys.argv[3] if len(sys.argv) > 3 else "none"
    if isinstance(fee, str) and fee.lower() in ("none", "-", "off"):
        fee = None
    print(render(assess(sym, hor, fee)))
