"""
train_vol.py — fit the deployable volatility model and save it.

Deliberately trained WITHOUT order-book depth features. The measurement in
vol_nodepth.txt showed they add nothing to volatility forecasting (holdout R2
0.4261 without versus 0.4270 with), and they cannot be reproduced live from
public endpoints anyway. Dropping them means the tool works from the first
minute after install instead of needing a day of recording first.

Training uses the most recent WINDOW_DAYS only. Volatility relationships drift,
and the whole project's central lesson is that a model averaged over a long dead
period looks better than it trades.

The saved artifact carries its own feature list and its measured out-of-sample
error, so the serving code can never silently feed it the wrong columns and the
report can state honest error bars instead of a bare point estimate.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import features2 as F2
import vol_model as V

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
WINDOW_DAYS = 180
VALID_DAYS = 30          # most recent slice, held out to measure error honestly


def live_feature_cols(df: pd.DataFrame) -> list[str]:
    """Everything the live path can actually compute, in a fixed order."""
    har = [f"har_{k}" for k in V.HAR_LAGS]
    rest = [c for c in F2.feature_names(df)
            if c not in har
            and c not in V.TARGET_COLS
            and not c.startswith(V.DEPTH_PREFIXES)]
    return har + sorted(rest)


def train(symbol: str, horizon: int) -> dict:
    df = V.build(symbol, horizon)
    cols = live_feature_cols(df)

    end = df.index.max()
    train_start = end - pd.Timedelta(days=WINDOW_DAYS)
    valid_start = end - pd.Timedelta(days=VALID_DAYS)

    tr = df[(df.index >= train_start) & (df.index < valid_start)]
    va = df[df.index >= valid_start]

    mdl = LGBMRegressor(random_state=0, **V.PARAMS)
    mdl.fit(tr[cols].to_numpy("float32"), tr["y"].to_numpy())

    pred = mdl.predict(va[cols].to_numpy("float32"))
    y = va["y"].to_numpy()
    resid = y - pred
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot

    # HAR on the same slice, so the report can always show the benchmark
    har_cols = [f"har_{k}" for k in V.HAR_LAGS]
    A = np.column_stack([np.ones(len(tr)), tr[har_cols].to_numpy("float64")])
    coef, *_ = np.linalg.lstsq(A, tr["y"].to_numpy(), rcond=None)
    B = np.column_stack([np.ones(len(va)), va[har_cols].to_numpy("float64")])
    har_pred = B @ coef
    har_r2 = 1.0 - float(np.sum((y - har_pred) ** 2)) / ss_tot

    # refit on everything including the validation slice for actual serving
    final = LGBMRegressor(random_state=0, **V.PARAMS)
    keep = df[df.index >= train_start]
    final.fit(keep[cols].to_numpy("float32"), keep["y"].to_numpy())

    # --- map predicted volatility to the number the user actually reads -------
    # The textbook conversion E|r| = sigma * sqrt(2/pi) assumes the sigma being
    # predicted is the true return volatility. It is not: realised variance summed
    # tick by tick is inflated by bid-ask bounce, by a factor that differs between
    # BTC and ETH (measured at 4.7x and 6.4x the theoretical Parkinson constant in
    # calibrate_rv.py). Applying the textbook factor would bias the headline
    # number by an unknown amount, so the mapping is measured instead: regress the
    # actual absolute move on the predicted volatility, on the validation slice
    # the model never trained on.
    va_abs = va["fwd_abs_bps"].to_numpy()
    pred_sigma_bps = np.exp(pred) * 1e4
    ok = np.isfinite(va_abs) & np.isfinite(pred_sigma_bps) & (pred_sigma_bps > 0)
    theoretical = float(np.sqrt(2 / np.pi))
    move_factor = float(np.sum(va_abs[ok] * pred_sigma_bps[ok])
                        / np.sum(pred_sigma_bps[ok] ** 2))

    # A single multiplier is not enough. Measured across deciles of predicted
    # volatility, the ratio of actual to predicted move falls from about 3.0 in
    # the calmest decile to about 1.2 in the wildest: predictions are compressed
    # toward the mean, and exponentiating a mean log understates the mean. The
    # calm end is exactly where the tool says "stand aside", so a flat factor
    # would talk users out of real opportunities. A monotone curve is stored and
    # interpolated at serving time instead.
    nb = 20
    q = pd.qcut(pd.Series(pred_sigma_bps[ok]), nb, labels=False, duplicates="drop")
    cal = pd.DataFrame({"pred": pred_sigma_bps[ok], "act": va_abs[ok], "q": q})
    g = cal.groupby("q").agg(pred=("pred", "mean"), actual=("act", "mean"))
    # enforce monotonicity: a higher volatility forecast must never map to a
    # smaller expected move
    knots_x = g["pred"].to_numpy()
    knots_y = np.maximum.accumulate(g["actual"].to_numpy())
    dec = cal.assign(d=pd.qcut(cal["pred"], 10, labels=False, duplicates="drop"))
    ratio = (dec.groupby("d")["act"].mean()
             / dec.groupby("d")["pred"].mean()).round(4).tolist()

    meta = {
        "symbol": symbol,
        "horizon": horizon,
        "move_factor": move_factor,
        "move_factor_theoretical": theoretical,
        "move_factor_by_decile": ratio,
        "calib_pred_bps": knots_x.tolist(),
        "calib_actual_bps": knots_y.tolist(),
        "features": cols,
        "n_features": len(cols),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_from": str(train_start),
        "train_to": str(end),
        "n_train": int(len(keep)),
        "valid_r2": round(r2, 4),
        "har_r2": round(har_r2, 4),
        # spread of the log-volatility error, used for the prediction interval
        "resid_sd": float(resid.std(ddof=1)),
        "resid_q10": float(np.quantile(resid, 0.10)),
        "resid_q90": float(np.quantile(resid, 0.90)),
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    stem = os.path.join(MODEL_DIR, f"vol_{symbol}_{horizon}m")
    with open(stem + ".pkl", "wb") as f:
        pickle.dump(final, f)
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{symbol} {horizon}m | {len(cols)} features | "
          f"validation R2 {r2:.4f} vs HAR {har_r2:.4f} | "
          f"resid sd {meta['resid_sd']:.4f}")
    print(f"    move factor {move_factor:.4f} (textbook would be {theoretical:.4f}"
          f", off by {move_factor / theoretical:.2f}x)")
    print(f"    decile ratios actual/predicted: "
          f"{[round(x, 2) for x in ratio]}")
    return meta


if __name__ == "__main__":
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTCUSDT", "ETHUSDT"]
    horizons = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [15, 60]
    for s in symbols:
        for h in horizons:
            train(s, h)
