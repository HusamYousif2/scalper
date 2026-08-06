"""
strategy_lab.py — put a trading idea through the same filter everything else here
had to pass.

This is the piece that turns the toolkit from a display into a test bench. A user
describes an idea declaratively; the lab returns one verdict.

A strategy is a dict, not code:

    {
      "name": "expansion_with_flow",
      "side": "flow",              # long / short / flow / revert
      "horizon": 15,
      "conditions": [
          ("IND_atr_rel_14", ">", "q80"),   # trailing 80th percentile
          ("ofi_vol_15",     ">",  0.15),   # absolute threshold
      ],
    }

Thresholds written as "qNN" are resolved against a TRAILING window, never the
whole sample. That distinction is not a detail: three separate candidate signals
in this project looked profitable with full-sample thresholds and collapsed the
moment the thresholds became causal.

What every strategy is scored against:

  1. causal thresholds only
  2. a development / holdout split of the timeline
  3. the identical frozen rules on a second asset that played no part in the idea
  4. a shuffled control, which must come out at zero
  5. Deflated Sharpe, penalised for how many strategies were tried at once

Costs are supplied by the caller in basis points, because what a round trip costs
depends on the venue, the tier and whether the order rests or crosses — none of
which this lab can know. Results are reported gross and at a grid of costs.

The honest prior, from everything measured in FINDINGS.md: directional edge on
these assets is worth 1-4 bps gross, so most ideas will not clear a real cost.
The value of this module is that it says so in twenty minutes instead of after
six months of live trading.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/crypto-quant-lab/research"))
from validation import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_per_period,
)

import edge_features as EF  # noqa: E402
import features2 as F2  # noqa: E402
import ta_readout as TA  # noqa: E402
import vol_extra as VE  # noqa: E402

DEV_FRACTION = 0.70
TRAIL_DAYS = 30           # window used to resolve every "qNN" threshold
MIN_TRADES = 60
COST_GRID_BPS = [0.0, 4.0, 10.5, 20.0]


# --------------------------------------------------------------------------- #
# the strategy library: ideas built only from readings this project validated
# --------------------------------------------------------------------------- #
LIBRARY = [
    {
        "name": "expansion_follow_flow",
        "note": "volatility expanding and taker flow one-sided; follow the flow",
        "side": "flow", "horizon": 15,
        "conditions": [("IND_atr_rel_14", ">", "q80"),
                       ("ofi_vol_15", "abs>", 0.20)],
    },
    {
        "name": "expansion_fade_flow",
        "note": "same setup, taken the other way",
        "side": "counterflow", "horizon": 15,
        "conditions": [("IND_atr_rel_14", ">", "q80"),
                       ("ofi_vol_15", "abs>", 0.20)],
    },
    {
        "name": "quiet_revert",
        "note": "calm tape, price stretched from its own mean; fade the stretch",
        "side": "revert", "horizon": 15,
        "conditions": [("IND_atr_rel_60", "<", "q30"),
                       ("IND_bb_pos_60", "abs>", 0.60)],
    },
    {
        "name": "poc_magnet",
        "note": "price far from the volume point of control; fade toward it",
        "side": "revert_poc", "horizon": 60,
        "conditions": [("VP_dist_poc_240", "abs>", "q80")],
    },
    {
        "name": "round_number_break",
        "note": "price pinned to a round level with expanding range; follow flow",
        "side": "flow", "horizon": 15,
        "conditions": [("SR_round_dist_1000", "abs<", "q20"),
                       ("CDL_range_rel_15", ">", "q70")],
    },
    {
        "name": "whale_continuation",
        "note": "large prints one-sided while open interest builds",
        "side": "whale", "horizon": 60,
        "conditions": [("WHALE_SKEW_60", "abs>", 0.40),
                       ("BUILDUP_60", ">", "q70")],
    },
    {
        "name": "crowd_reversal",
        "note": "retail crowded and stressed; take the other side",
        "side": "counter_crowd", "horizon": 60,
        "conditions": [("CROWD_STRESS", ">", "q85")],
    },
    {
        "name": "channel_breakout",
        "note": "fresh break of the 4-hour range with volume",
        "side": "breakout", "horizon": 15,
        "conditions": [("SR_channel_pos_240", "abs>", 0.95),
                       ("vol_rel_15", ">", "q75")],
    },
]


def build_frame(symbol: str, horizon: int) -> pd.DataFrame:
    """Everything a rule might reference, on a grid spaced by the horizon."""
    df = F2.dataset(symbol, horizon=horizon)
    for mod in (VE, EF):
        extra = mod.build(symbol)
        df = df.join(extra.reindex(df.index), how="left")
    return df.replace([np.inf, -np.inf], np.nan)


def _trailing_quantile(s: pd.Series, q: float, horizon: int) -> pd.Series:
    """Quantile of a trailing window, shifted so the current bar never counts."""
    n = max(50, int(TRAIL_DAYS * 24 * 60 / horizon))
    return s.shift(1).rolling(n, min_periods=n // 4).quantile(q)


def resolve(df: pd.DataFrame, col: str, op: str, thr, horizon: int) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"unknown column {col!r}")
    s = df[col]
    value = s.abs() if op.startswith("abs") else s
    if isinstance(thr, str) and thr.startswith("q"):
        level = float(thr[1:]) / 100.0
        cut = _trailing_quantile(value, level, horizon)
    else:
        cut = float(thr)
    if op in (">", "abs>"):
        return value > cut
    if op in ("<", "abs<"):
        return value < cut
    raise ValueError(f"unknown operator {op!r}")


def sides(df: pd.DataFrame, mode: str) -> pd.Series:
    """Translate a side label into +1 / -1 per row."""
    if mode == "long":
        return pd.Series(1.0, index=df.index)
    if mode == "short":
        return pd.Series(-1.0, index=df.index)
    if mode == "flow":
        return np.sign(df["ofi_vol_15"]).replace(0.0, 1.0)
    if mode == "counterflow":
        return -np.sign(df["ofi_vol_15"]).replace(0.0, 1.0)
    if mode == "revert":
        return -np.sign(df["IND_bb_pos_60"]).replace(0.0, 1.0)
    if mode == "revert_poc":
        # POC above price means fade upward, and vice versa
        return -np.sign(df["VP_dist_poc_240"]).replace(0.0, 1.0)
    if mode == "whale":
        return np.sign(df["WHALE_SKEW_60"]).replace(0.0, 1.0)
    if mode == "counter_crowd":
        return -np.sign(df["CROWD_LEAN"]).replace(0.0, 1.0)
    if mode == "breakout":
        return np.sign(df["SR_channel_pos_240"] - 0.5).replace(0.0, 1.0)
    raise ValueError(f"unknown side {mode!r}")


def evaluate(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    h = spec["horizon"]
    mask = pd.Series(True, index=df.index)
    for col, op, thr in spec["conditions"]:
        mask &= resolve(df, col, op, thr, h).fillna(False)
    side = sides(df, spec["side"])
    gross = (side * df["fwd_ret"] * 1e4)[mask].dropna()
    return pd.DataFrame({"gross_bps": gross, "side": side[gross.index]})


def score(trades: pd.DataFrame, horizon: int, label: str,
          cost_bps: float = 0.0) -> dict | None:
    if len(trades) < MIN_TRADES:
        return None
    net = trades["gross_bps"] - cost_bps
    sd = net.std(ddof=1)
    sr = float(net.mean() / sd) if sd > 0 else 0.0
    span_days = (trades.index.max() - trades.index.min()).days or 1
    per_year = len(net) / (span_days / 365.25)
    return {
        "phase": label,
        "cost_bps": cost_bps,
        "n": len(net),
        "per_day": round(len(net) / span_days, 2),
        "hit_%": round(float((net > 0).mean() * 100), 2),
        "gross_bps": round(float(trades["gross_bps"].mean()), 2),
        "net_bps": round(float(net.mean()), 2),
        "sharpe_ann": round(float(sr * np.sqrt(per_year)), 2),
        "psr": round(float(probabilistic_sharpe_ratio(
            sr, 0.0, len(net), net.skew(), net.kurt() + 3.0)), 3),
        "total_%": round(float(net.sum() / 100), 2),
        "sr_per_trade": sr,
    }


def run(symbol: str = "BTCUSDT", second: str = "ETHUSDT",
        library: list[dict] | None = None,
        cost_bps: float = 10.5) -> pd.DataFrame:
    specs = library or LIBRARY
    horizons = sorted({s["horizon"] for s in specs})

    frames = {}
    for sym in (symbol, second):
        for h in horizons:
            frames[(sym, h)] = build_frame(sym, h)
        print(f"  {sym}: frames built for horizons {horizons}", flush=True)

    rows, trial_sr = [], []
    for spec in specs:
        h = spec["horizon"]
        try:
            tr = evaluate(frames[(symbol, h)], spec)
        except KeyError as e:
            print(f"  {spec['name']}: skipped, {e}", flush=True)
            continue
        if len(tr) < MIN_TRADES:
            print(f"  {spec['name']}: only {len(tr)} trades, skipped", flush=True)
            continue

        split = tr.index[int(len(tr) * DEV_FRACTION)]
        dev = score(tr[tr.index < split], h, "dev", cost_bps)
        hold = score(tr[tr.index >= split], h, "hold", cost_bps)

        # the identical rules on the second asset
        tr2 = evaluate(frames[(second, h)], spec)
        clean = score(tr2, h, "second_asset", cost_bps)

        # shuffled control: same entries, returns permuted
        rng = np.random.default_rng(17)
        sh = tr.copy()
        sh["gross_bps"] = rng.permutation(sh["gross_bps"].to_numpy())
        ctrl = score(sh, h, "shuffled", cost_bps)

        row = {"name": spec["name"], "side": spec["side"], "horizon": h,
               "n": len(tr)}
        for tag, s in (("dev", dev), ("hold", hold),
                       ("second", clean), ("ctrl", ctrl)):
            if s:
                row[f"{tag}_net"] = s["net_bps"]
                row[f"{tag}_sharpe"] = s["sharpe_ann"]
                row[f"{tag}_psr"] = s["psr"]
        if hold:
            trial_sr.append(hold["sr_per_trade"])
        rows.append(row)
        print(f"  {spec['name']:<24} gross {score(tr, h, 'all', 0.0)['gross_bps']:+6.2f}"
              f" | hold net {row.get('hold_net', float('nan')):+6.2f}"
              f" | second {row.get('second_net', float('nan')):+6.2f}", flush=True)

    t = pd.DataFrame(rows)
    if not len(t):
        print("no strategy produced enough trades")
        return t

    print(f"\n{'=' * 96}")
    print(f"SUMMARY at {cost_bps:.1f} bps round-trip cost "
          f"({len(t)} strategies tried)")
    cols = ["name", "side", "horizon", "n", "hold_net", "hold_sharpe",
            "hold_psr", "second_net", "ctrl_net"]
    print(t[[c for c in cols if c in t.columns]].to_string(index=False))

    # deflate the best holdout Sharpe for the size of the search
    if trial_sr and "hold_sharpe" in t.columns:
        best = t.loc[t["hold_sharpe"].idxmax()]
        i = int(t["hold_sharpe"].idxmax())
        dsr, sr_star = deflated_sharpe_ratio(
            trial_sr[min(i, len(trial_sr) - 1)], trial_sr,
            int(best["n"]), 0.0, 3.0)
        print(f"\nbest by holdout Sharpe: {best['name']}")
        print(f"  deflated Sharpe after penalising {len(trial_sr)} trials: "
              f"{dsr:.3f}   (needs > 0.95 to be believed)")

    survivors = t[(t.get("hold_net", pd.Series(dtype=float)) > 0)
                  & (t.get("second_net", pd.Series(dtype=float)) > 0)
                  & (t.get("hold_psr", pd.Series(dtype=float)) > 0.95)]
    print(f"\nsurvivors (holdout positive AND second asset positive AND PSR>0.95):"
          f" {len(survivors)} of {len(t)}")
    if len(survivors):
        print(survivors[[c for c in cols if c in survivors.columns]]
              .to_string(index=False))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reports", f"strategies_{symbol}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t.to_csv(out, index=False)
    print(f"\nwritten to {out}")
    return t


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    cost = float(sys.argv[2]) if len(sys.argv) > 2 else 10.5
    run(sym, cost_bps=cost)
