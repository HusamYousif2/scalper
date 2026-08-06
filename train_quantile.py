"""
train_quantile.py — fit and save the conformalised quantile models for serving.

This is the one improvement from the feature-and-model sweep that survived a
clean test. Measured coverage error on the holdout, after conformalising:

    BTCUSDT 15m  0.8 points      ETHUSDT 15m  0.8 points
    BTCUSDT 60m  0.8 points      ETHUSDT 60m  0.5 points

ETHUSDT played no part in designing the correction, so the transfer is genuine.
Before conformalising the same errors were 6.0, 9.7, 6.0 and 8.4 points — the
same sign and rough magnitude everywhere, which is what marked the problem as a
systematic bias worth correcting rather than noise to be discarded.

Each saved artifact holds one model per quantile plus the shift measured on a
calibration slice the model never saw, and the coverage that shift achieved on
data after it. Serving code applies the shift; it does not recompute it.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import quantile_vol as Q
import train_vol as TV
import vol_model as V

MODEL_DIR = TV.MODEL_DIR
WINDOW_DAYS = 180

# The walk-forward study reached a coverage error of 0.5-0.8 points, but it
# recalibrated every 7 days. A first version here measured the shift once over 45
# days and applied it to the following 30, and coverage error grew to 2.2-6.0
# points: the shift goes stale as the volatility regime drifts. So the serving
# cadence has to match the cadence that produced the number being claimed.
CALIB_DAYS = 14          # slice reserved for measuring the conformal shift
VERIFY_DAYS = 7          # slice after it, used to report honest coverage
RECALIBRATE_EVERY_DAYS = 7


def train(symbol: str, horizon: int) -> dict:
    df, cols = Q.build(symbol, horizon)
    y = df["fwd_abs_bps"].to_numpy()
    X = df[cols].to_numpy("float32")
    idx = df.index

    end = idx.max()
    fit_from = end - pd.Timedelta(days=WINDOW_DAYS)
    calib_from = end - pd.Timedelta(days=CALIB_DAYS + VERIFY_DAYS)
    verify_from = end - pd.Timedelta(days=VERIFY_DAYS)

    fit = (idx >= fit_from) & (idx < calib_from)
    cal = (idx >= calib_from) & (idx < verify_from)
    ver = idx >= verify_from
    if fit.sum() < 1000 or cal.sum() < 200 or ver.sum() < 80:
        raise RuntimeError(f"{symbol} {horizon}m: not enough data to split")

    models, shifts, coverage, cov_err = {}, {}, {}, {}
    n_ver = int(ver.sum())
    for q in Q.QUANTILES:
        mdl = LGBMRegressor(random_state=0, alpha=q, **Q.PARAMS)
        mdl.fit(X[fit], y[fit])
        resid = y[cal] - mdl.predict(X[cal])
        shift = float(np.quantile(resid, q))
        pred_ver = mdl.predict(X[ver]) + shift
        cov = float((y[ver] <= pred_ver).mean() * 100)
        key = f"q{int(q * 100)}"
        models[key] = mdl
        shifts[key] = shift
        coverage[key] = round(cov, 2)
        # A coverage figure without its sampling error invites being read as bias.
        # Seven days at the 60-minute horizon is only ~170 observations, where the
        # standard error at q90 is 2.3 points — so a measured 91.7 % is indistinct
        # from 90 %. The large-sample figure to quote is the walk-forward one in
        # quantile_*.txt, measured over ~18,000 observations at this same weekly
        # recalibration cadence.
        # 1.96 standard errors, expressed in percentage points
        cov_err[key] = round(1.96 * np.sqrt(q * (1 - q) / max(n_ver, 1)) * 100, 2)

    meta = {
        "symbol": symbol,
        "horizon": horizon,
        "quantiles": [f"q{int(q * 100)}" for q in Q.QUANTILES],
        "features": cols,
        "n_features": len(cols),
        "shifts": shifts,
        "verified_coverage_%": coverage,
        "coverage_95ci_pts": cov_err,
        "verified_on_n": n_ver,
        # the defensible large-sample figure, from the walk-forward study that
        # used this same weekly recalibration cadence
        "walkforward_worst_error_pts": 0.8,
        "walkforward_n": 17894,
        "recalibrate_every_days": RECALIBRATE_EVERY_DAYS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fit_from": str(fit_from),
        "calib_days": CALIB_DAYS,
        "verify_days": VERIFY_DAYS,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    stem = os.path.join(MODEL_DIR, f"quant_{symbol}_{horizon}m")
    with open(stem + ".pkl", "wb") as f:
        pickle.dump(models, f)
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    worst = max(abs(coverage[f"q{int(q * 100)}"] - q * 100) for q in Q.QUANTILES)
    biggest_ci = max(cov_err.values())
    print(f"{symbol} {horizon}m | {len(cols)} features | n_verify={n_ver}")
    print(f"    coverage {coverage}")
    print(f"    worst error {worst:.1f} pts, but the 95 % sampling band on this"
          f" slice is +/-{biggest_ci:.1f} pts -> saved")
    return meta


def load(symbol: str, horizon: int):
    stem = os.path.join(MODEL_DIR, f"quant_{symbol}_{horizon}m")
    if not os.path.exists(stem + ".pkl"):
        raise FileNotFoundError(
            f"no quantile model for {symbol} {horizon}m — run train_quantile.py"
        )
    with open(stem + ".pkl", "rb") as f:
        models = pickle.load(f)
    with open(stem + ".json") as f:
        meta = json.load(f)
    return models, meta


def predict(models: dict, meta: dict, row: pd.DataFrame) -> dict:
    """Conformalised quantiles for one feature row, in basis points."""
    x = row.reindex(columns=meta["features"]).to_numpy("float32")
    out = {}
    for key in meta["quantiles"]:
        out[key] = float(models[key].predict(x)[0] + meta["shifts"][key])
    # a shift per quantile can cross neighbours; keep them ordered
    keys = meta["quantiles"]
    vals = np.maximum.accumulate([out[k] for k in keys]).clip(min=0.0)
    return {k: float(v) for k, v in zip(keys, vals)}


if __name__ == "__main__":
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTCUSDT", "ETHUSDT"]
    horizons = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [15, 60]
    for s in symbols:
        for h in horizons:
            try:
                train(s, h)
            except Exception as e:
                print(f"{s} {h}m: {type(e).__name__}: {e}")
