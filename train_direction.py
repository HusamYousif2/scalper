"""
train_direction.py — fit and save the directional model for live serving.

The model itself is the one built and measured in direction.py: gradient boosting
over the full microstructure feature set, with isotonic calibration fitted on a
slice the classifier never saw, so the probability it reports means what it says.

Training splits three ways, and the order matters:

    fit    -> the classifier
    calib  -> the isotonic correction, on predictions the classifier never saw
    verify -> reported reliability, on data neither of the above touched

Skipping the middle slice is the usual way this goes wrong: calibrate on the
classifier's own training data and the model will certify itself as perfect.

Saved alongside the model: the reliability table measured on `verify`, so the
interface can show what the stated probabilities actually delivered rather than
asserting they are trustworthy.
"""

import json
import os
import pickle
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

import direction as D
import train_vol as TV

MODEL_DIR = TV.MODEL_DIR
WINDOW_DAYS = 150
CALIB_DAYS = 30
VERIFY_DAYS = 21


def train(symbol: str, horizon: int) -> dict:
    df, cols = D.build(symbol, horizon)
    y = (df["fwd_ret"] > 0).astype(int).to_numpy()
    X = df[cols].to_numpy("float32")
    idx = df.index

    end = idx.max()
    fit_from = end - pd.Timedelta(days=WINDOW_DAYS)
    calib_from = end - pd.Timedelta(days=CALIB_DAYS + VERIFY_DAYS)
    verify_from = end - pd.Timedelta(days=VERIFY_DAYS)

    fit = (idx >= fit_from) & (idx < calib_from)
    cal = (idx >= calib_from) & (idx < verify_from)
    ver = idx >= verify_from
    if fit.sum() < 1000 or cal.sum() < 300 or ver.sum() < 200:
        raise RuntimeError(f"{symbol} {horizon}m: not enough history")

    clf = LGBMClassifier(random_state=0, **D.PARAMS)
    clf.fit(X[fit], y[fit])

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(clf.predict_proba(X[cal])[:, 1], y[cal])

    p_ver = iso.predict(clf.predict_proba(X[ver])[:, 1])
    y_ver = y[ver]

    # reliability on the untouched slice, in five buckets
    bins = pd.qcut(pd.Series(p_ver), 5, labels=False, duplicates="drop")
    rel = []
    for b in sorted(pd.Series(bins).dropna().unique()):
        m = bins == b
        rel.append({
            "said": round(float(np.mean(p_ver[m]) * 100), 1),
            "happened": round(float(np.mean(y_ver[m]) * 100), 1),
            "n": int(m.sum()),
        })

    acc = float(((p_ver > 0.5).astype(int) == y_ver).mean() * 100)
    conf = np.abs(p_ver - 0.5)
    thr = float(np.quantile(conf, 0.75))
    strong = conf >= thr
    acc_strong = float(
        ((p_ver[strong] > 0.5).astype(int) == y_ver[strong]).mean() * 100
    )

    # refit on everything for serving
    keep = idx >= fit_from
    final = LGBMClassifier(random_state=0, **D.PARAMS)
    final.fit(X[keep], y[keep])
    iso_final = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso_final.fit(final.predict_proba(X[cal])[:, 1], y[cal])

    # The isotonic clip bounds are 0.02/0.98, so a raw score outside the range
    # seen during calibration snaps straight to 98%. The reliability table shows
    # this model's buckets top out near 55%, so a served 98% would be a number it
    # has never once justified. Serving is therefore clamped to the span the
    # calibration slice actually produced, and the bound travels in the metadata.
    # Bounds must come from a model that has not seen the slice they are measured
    # on. A first version took them from `final` evaluated on `cal` — but `final`
    # is refit on a window that contains `cal`, so those predictions are in-sample
    # and wildly overconfident, which is how a 98% and then an 83% reached the
    # screen. `p_ver` below is the honest classifier scored on `verify`, which
    # neither it nor the calibrator ever touched.
    p_lo = float(np.quantile(p_ver, 0.01))
    p_hi = float(np.quantile(p_ver, 0.99))

    # Distribution of the model's own conviction, so the interface can express a
    # signal as "stronger than N% of the signals this model produces". That is a
    # different claim from "N% chance of being right" and a true one: it ranks the
    # current read against this model's own history rather than asserting an
    # outcome probability it has not earned.
    conv_ver = np.abs(p_ver - 0.5)
    conviction_grid = [round(float(np.quantile(conv_ver, x / 100)), 6)
                       for x in range(0, 101)]

    meta = {
        "symbol": symbol,
        "horizon": horizon,
        "features": cols,
        "n_features": len(cols),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(keep.sum()),
        "reliability": rel,
        "accuracy_all_%": round(acc, 2),
        "accuracy_high_conviction_%": round(acc_strong, 2),
        "high_conviction_threshold": round(thr, 4),
        "verified_on_n": int(ver.sum()),
        "prob_floor": round(p_lo, 4),
        "prob_ceiling": round(p_hi, 4),
        "conviction_grid": conviction_grid,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    stem = os.path.join(MODEL_DIR, f"dir_{symbol}_{horizon}m")
    with open(stem + ".pkl", "wb") as f:
        pickle.dump({"clf": final, "iso": iso_final}, f)
    with open(stem + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{symbol} {horizon}m | {len(cols)} features | "
          f"accuracy {acc:.1f}% all, {acc_strong:.1f}% high conviction -> saved")
    return meta


def load(symbol: str, horizon: int):
    stem = os.path.join(MODEL_DIR, f"dir_{symbol}_{horizon}m")
    if not os.path.exists(stem + ".pkl"):
        raise FileNotFoundError(f"no direction model for {symbol} {horizon}m")
    with open(stem + ".pkl", "rb") as f:
        bundle = pickle.load(f)
    with open(stem + ".json") as f:
        meta = json.load(f)
    return bundle, meta


def predict(bundle: dict, meta: dict, row: pd.DataFrame) -> dict:
    x = row.reindex(columns=meta["features"]).to_numpy("float32")
    raw = float(bundle["clf"].predict_proba(x)[0, 1])
    p = float(bundle["iso"].predict([raw])[0])
    # never report a confidence the calibration slice never supported
    p = min(max(p, meta.get("prob_floor", 0.30)), meta.get("prob_ceiling", 0.70))
    conf = abs(p - 0.5)

    # Rank this conviction against the model's own history. 87 means the current
    # read is stronger than 87 % of the reads this model has produced — a claim
    # about the signal, not a claim about the outcome.
    grid = meta.get("conviction_grid")
    strength = None
    if grid:
        strength = int(np.searchsorted(np.asarray(grid), conf, side="right"))
        strength = max(0, min(100, strength))

    return {
        "p_up": p,
        "p_down": 1 - p,
        "direction": "UP" if p > 0.5 else "DOWN",
        "bias": "LONG" if p > 0.5 else "SHORT",
        "conviction": conf,
        "strength": strength,
        "high_conviction": bool(conf >= meta.get("high_conviction_threshold", 0.05)),
    }


if __name__ == "__main__":
    symbols = sys.argv[1].split(",") if len(sys.argv) > 1 else ["BTCUSDT", "ETHUSDT"]
    horizons = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [15, 60]
    for s in symbols:
        for h in horizons:
            try:
                train(s, h)
            except Exception as e:
                print(f"{s} {h}m: {type(e).__name__}: {e}")
