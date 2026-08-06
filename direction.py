"""
direction.py — an honestly calibrated probability that price rises.

This project already established that directional edge on BTC and ETH at these
horizons is worth 1-4 basis points gross, which is less than any realistic round
trip. That finding stands and is not being relitigated here.

But "not tradeable on its own" is not the same as "worthless". The difference is
calibration. A model that says 58 % and is right 58 % of the time is real
information: it can size a position, it can be combined with a trader's own read,
and its user knows when to ignore it. A model that says 75 % and is right 53 % of
the time is actively harmful. Most tools that sell direction signals are the
second kind, and never test which they are.

So the acceptance test here is not accuracy. It is reliability of the displayed
number: gather every case where the model said 58 %, and check that price actually
rose in 58 % of them. Alongside it:

  Brier score      mean squared error of the probability. Lower is better.
  Brier skill      versus always predicting the base rate. Zero means the model
                   adds nothing over knowing the long-run drift.
  reliability      the calibration curve, in ten bins.
  resolution       does the model separate outcomes at all, or does it hedge
                   everything toward 50 %?

Probabilities are corrected with isotonic regression fitted on past data only,
because raw gradient-boosting probabilities are systematically overconfident.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

import edge_features as EF
import features2 as F2
import train_vol as TV
import vol_extra as VE
import vol_model as V

HORIZONS = [15, 60]
ROLLING_DAYS = 90
RETRAIN_DAYS = 7
MIN_TRAIN_DAYS = 90
DEV_FRACTION = 0.70
CALIB_FRACTION = 0.25      # tail of each training window reserved for isotonic

PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=15,
    min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.6, reg_lambda=5.0, n_jobs=4, verbose=-1,
)


def build(symbol: str, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    df = V.build(symbol, horizon)
    cols = TV.live_feature_cols(df)
    extra = VE.build(symbol)
    edge = EF.build(symbol)
    df = df.join(extra.reindex(df.index), how="left")
    df = df.join(edge.reindex(df.index), how="left")
    cols += [c for c in extra.columns if c.split("_")[0] in VE.FAMILIES]
    cols += [c for c in edge.columns if EF.group_of(c)]
    df = df.dropna(subset=["fwd_ret"])
    return df, [c for c in cols if c in df.columns]


def walk(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Walk forward, with a calibration slice carved out of every training window.

    The model is fitted on the older part of the window, isotonic regression is
    fitted on the newer part using predictions the model has not seen, and only
    then is the test block scored. Without that split the calibration would be
    fitted on the model's own training data and would report itself as perfect.
    """
    y = (df["fwd_ret"] > 0).astype(int).to_numpy()
    X = df[cols].to_numpy("float32")
    idx = df.index

    raw, cal, stamps = [], [], []
    cursor = idx.min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    while cursor < idx.max():
        stop = cursor + pd.Timedelta(days=RETRAIN_DAYS)
        win = (idx < cursor) & (idx >= cursor - pd.Timedelta(days=ROLLING_DAYS))
        te = (idx >= cursor) & (idx < stop)
        if te.sum() == 0 or win.sum() < 1000:
            cursor = stop
            continue

        wi = np.flatnonzero(win)
        cut = int(len(wi) * (1 - CALIB_FRACTION))
        fit_i, cal_i = wi[:cut], wi[cut:]

        clf = LGBMClassifier(random_state=0, **PARAMS)
        clf.fit(X[fit_i], y[fit_i])

        p_cal = clf.predict_proba(X[cal_i])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(p_cal, y[cal_i])

        p_raw = clf.predict_proba(X[te])[:, 1]
        raw.append(p_raw)
        cal.append(iso.predict(p_raw))
        stamps.append(idx[te])
        cursor = stop

    index = stamps[0].append(stamps[1:])
    out = pd.DataFrame({"p_raw": np.concatenate(raw),
                        "p_cal": np.concatenate(cal)}, index=index)
    out["y"] = (df.loc[index, "fwd_ret"] > 0).astype(int)
    out["fwd_ret"] = df.loc[index, "fwd_ret"]
    out["abs_bps"] = df.loc[index, "fwd_ret"].abs() * 1e4
    return out


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(out: pd.DataFrame, col: str, bins: int = 10) -> pd.DataFrame:
    q = pd.qcut(out[col], bins, labels=False, duplicates="drop")
    g = out.groupby(q)
    return pd.DataFrame({
        "n": g.size(),
        "said_%": g[col].mean() * 100,
        "happened_%": g["y"].mean() * 100,
    }).assign(gap_pts=lambda d: d["happened_%"] - d["said_%"])


def report(out: pd.DataFrame, label: str) -> dict:
    y = out["y"].to_numpy()
    base = float(y.mean())
    b_base = brier(np.full(len(y), base), y)
    res = {}
    print(f"\n----- {label} | {len(out):,} observations "
          f"| base rate up {base * 100:.2f} % -----")
    for col in ("p_raw", "p_cal"):
        p = out[col].to_numpy()
        b = brier(p, y)
        skill = (1 - b / b_base) * 100
        acc = float(((p > 0.5).astype(int) == y).mean() * 100)
        spread = float(p.std())
        res[col] = {"brier": b, "skill_%": skill, "accuracy_%": acc,
                    "prob_sd": spread}
        print(f"  {col:<6} brier {b:.5f}  skill vs base {skill:+.2f} %"
              f"  accuracy {acc:.2f} %  probability spread {spread:.4f}")

    rel = reliability(out, "p_cal")
    print("\n  reliability of the calibrated probability:")
    print(rel.round(2).to_string())
    worst = float(rel["gap_pts"].abs().max())
    print(f"  worst bin gap {worst:.1f} points"
          f"   {'RELIABLE' if worst < 5 else 'MISCALIBRATED'}")
    res["worst_gap_pts"] = worst
    return res


def run(symbol: str = "BTCUSDT") -> None:
    for h in HORIZONS:
        df, cols = build(symbol, h)
        print(f"\n{'=' * 80}")
        print(f"{symbol} | horizon {h}m | {len(df):,} samples | "
              f"{len(cols)} features", flush=True)
        out = walk(df, cols)
        split = out.index[int(len(out) * DEV_FRACTION)]
        report(out, "ALL (walk-forward)")
        hold = out[out.index >= split]
        report(hold, "HOLDOUT")

        # what a user needs to know: given their own cost, is this probability
        # ever high enough to matter?
        print("\n  gross basis points per trade if you act on the top decile"
              " of confidence:")
        conf = (hold["p_cal"] - 0.5).abs()
        thr = conf.quantile(0.9)
        sel = hold[conf >= thr]
        side = np.sign(sel["p_cal"] - 0.5)
        gross = float((side * sel["fwd_ret"]).mean() * 1e4)
        print(f"    n={len(sel)}  gross {gross:+.2f} bps"
              f"  (apply your own round-trip cost to judge)")

        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "reports"), exist_ok=True)
        out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "reports", f"direction_{symbol}_{h}m.csv"))


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
