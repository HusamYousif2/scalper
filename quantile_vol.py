"""
quantile_vol.py — forecast the whole distribution of the coming move, not a point.

The deployed model predicts one number and wraps an interval around it using the
spread of its own residuals. That assumes the residual spread is the same in a
calm market and a violent one, which it is not.

Quantile regression drops the assumption. A separate model is fitted for each
quantile of the actual absolute move, so the output is a real distribution:

    q10  the move is smaller than this 10 % of the time
    q50  the median move
    q90  exceeded 10 % of the time — the number a stop has to survive

Three things this fixes at once:

  1. Stops and targets stop being multiples of a guess. A stop belongs at a high
     quantile of the move distribution; that is what a quantile IS.
  2. It sidesteps the calibration problem found earlier, where the ratio of
     actual to predicted move ran from 3.0 in calm deciles to 1.2 in wild ones.
     Fitting each quantile directly needs no such correction factor.
  3. Position sizing can key off the bad case rather than the average one.

The test is coverage, and it cannot be argued with: if the q90 model is honest,
the actual move exceeds it on 10 % of out-of-sample observations. Not 4 %, not
20 %. Everything else is secondary to that number.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import edge_features as EF
import train_vol as TV
import vol_extra as VE
import vol_model as V

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
HORIZONS = [15, 60]
DEV_FRACTION = 0.70

PARAMS = dict(
    objective="quantile",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=80,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_jobs=4,
    verbose=-1,
)


def build(symbol: str, horizon: int, families: list[str] | None = None):
    """Feature matrix with the absolute forward move in basis points as target."""
    df = V.build(symbol, horizon)
    cols = TV.live_feature_cols(df)
    if families:
        extra = VE.build(symbol)
        edge = EF.build(symbol)
        df = df.join(extra.reindex(df.index), how="left")
        df = df.join(edge.reindex(df.index), how="left")
        for c in list(extra.columns):
            fam = c.split("_")[0]
            if fam in VE.FAMILIES and fam in families:
                cols.append(c)
        for c in list(edge.columns):
            if (EF.group_of(c) or "") in families:
                cols.append(c)
    df = df.dropna(subset=["fwd_abs_bps"])
    return df, [c for c in cols if c in df.columns]


CALIB_FRACTION = 0.25       # tail of each training window kept for conformalising


def walk(df: pd.DataFrame, cols: list[str], conformal: bool = True) -> pd.DataFrame:
    """
    Walk forward, optionally conformalising each quantile.

    The plain version under-disperses: measured coverage came out at 15.6 % for
    the nominal 10 % quantile and 87.0 % for the nominal 90 %, so the predicted
    distribution was too narrow at both ends. That error was stable to within a
    tenth of a point between development and holdout, which means it is a
    systematic bias and can be corrected rather than merely reported.

    Conformalised quantile regression (Romano, Patterson and Candes, 2019) does
    exactly that: a slice of each training window is held back, the model's
    residuals on it are measured, and the empirical quantile of those residuals
    shifts the prediction so coverage comes out where it was asked for. The slice
    is never seen during fitting, so the correction is honest.
    """
    y = df["fwd_abs_bps"].to_numpy()
    X = df[cols].to_numpy("float32")
    idx = df.index

    preds = {q: [] for q in QUANTILES}
    stamps = []
    cursor = idx.min() + pd.Timedelta(days=V.MIN_TRAIN_DAYS)
    while cursor < idx.max():
        stop = cursor + pd.Timedelta(days=V.RETRAIN_DAYS)
        win = (idx < cursor) & (idx >= cursor - pd.Timedelta(days=V.ROLLING_DAYS))
        te = (idx >= cursor) & (idx < stop)
        if te.sum() == 0 or win.sum() < 800:
            cursor = stop
            continue

        wi = np.flatnonzero(win)
        if conformal:
            cut = int(len(wi) * (1 - CALIB_FRACTION))
            fit_i, cal_i = wi[:cut], wi[cut:]
        else:
            fit_i, cal_i = wi, np.array([], dtype=int)

        for q in QUANTILES:
            mdl = LGBMRegressor(random_state=0, alpha=q, **PARAMS)
            mdl.fit(X[fit_i], y[fit_i])
            p_te = mdl.predict(X[te])
            if conformal and len(cal_i) > 100:
                resid = y[cal_i] - mdl.predict(X[cal_i])
                # the q-th empirical quantile of the residual is the shift that
                # makes the nominal level hold on unseen data
                shift = float(np.quantile(resid, q))
                p_te = p_te + shift
            preds[q].append(p_te)
        stamps.append(idx[te])
        cursor = stop

    index = stamps[0].append(stamps[1:])
    out = pd.DataFrame({f"q{int(q * 100)}": np.concatenate(v)
                        for q, v in preds.items()}, index=index)
    # a shift per quantile can in principle cross two neighbours; enforce order
    qcols = [f"q{int(q * 100)}" for q in QUANTILES]
    out[qcols] = np.maximum.accumulate(out[qcols].to_numpy(), axis=1)
    out[qcols] = out[qcols].clip(lower=0.0)
    out["actual"] = df.loc[index, "fwd_abs_bps"]
    return out


def coverage(out: pd.DataFrame, label: str) -> pd.DataFrame:
    """The only test that matters: does each quantile mean what it says?"""
    rows = []
    for q in QUANTILES:
        c = f"q{int(q * 100)}"
        below = float((out["actual"] <= out[c]).mean() * 100)
        # pinball loss, the proper scoring rule for a quantile forecast
        d = out["actual"] - out[c]
        pin = float(np.mean(np.maximum(q * d, (q - 1) * d)))
        rows.append({"quantile": c, "target_%": q * 100,
                     "actual_%": below, "error_pts": below - q * 100,
                     "pinball": pin, "mean_bps": float(out[c].mean())})
    t = pd.DataFrame(rows)
    print(f"\n----- {label} | {len(out):,} observations -----")
    print(t.round(3).to_string(index=False))
    bad = t["error_pts"].abs().max()
    print(f"  worst coverage error: {bad:.1f} points"
          f"   {'OK' if bad < 3 else 'MISCALIBRATED'}")
    return t


def run(symbol: str = "BTCUSDT", families: list[str] | None = None) -> None:
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "reports"), exist_ok=True)
    for h in HORIZONS:
        df, cols = build(symbol, h, families)
        print(f"\n{'=' * 78}")
        print(f"{symbol} | horizon {h}m | {len(df):,} samples | "
              f"{len(cols)} features", flush=True)
        for mode, conf in (("plain", False), ("conformal", True)):
            out = walk(df, cols, conformal=conf)
            split = out.index[int(len(out) * DEV_FRACTION)]
            coverage(out[out.index >= split], f"HOLDOUT [{mode}]")

            hold = out[out.index >= split]
            inside = float(((hold["actual"] >= hold["q10"])
                            & (hold["actual"] <= hold["q90"])).mean() * 100)
            width = float((hold["q90"] - hold["q10"]).mean())
            print(f"  80 % interval: covers {inside:.1f} % of moves"
                  f" (target 80), average width {width:.1f} bps")
            if conf:
                out.to_csv(os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "reports", f"quantile_{symbol}_{h}m.csv"))



if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    fams = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    run(sym, fams)
