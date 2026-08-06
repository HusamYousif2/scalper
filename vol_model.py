"""
vol_model.py — forecast how far price will move, not which way.

Direction on liquid majors is worth 1-4 bps gross against a 4-20 bps cost wall;
that is settled (see FINDINGS.md). Volatility is a different matter: it is
strongly autocorrelated and genuinely forecastable, and for a scalper it answers
the question that actually decides a trade — is the expected move over the next
N minutes big enough to pay for a round trip, and where do the target and stop
belong?

The bar to clear is NOT zero. A volatility model that does not beat HAR-RV
(Corsi 2009) adds nothing: HAR is three lagged averages and a linear regression,
and it is famously hard to beat. Four models are compared on identical data:

    naive   tomorrow looks like today
    ewma    exponentially weighted, the RiskMetrics standard
    har     heterogeneous autoregression on log realised variance
    har+ml  HAR lags plus the full microstructure feature set, gradient boosted

Metrics:
    R2_log   out-of-sample R squared on log realised volatility
    QLIKE    the loss function that penalises under-forecasting volatility much
             harder than over-forecasting it, which matches what hurts a trader
    hit      accuracy of the practical call: "will the coming move clear the
             round-trip cost?" - this is what the tool will actually display
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import features as FE
import features2 as F2

HORIZONS = [15, 60]
ROLLING_DAYS = 90
RETRAIN_DAYS = 7          # volatility drifts far more slowly than direction
MIN_TRAIN_DAYS = 90
DEV_FRACTION = 0.70

# HAR lag structure, in minutes: recent, medium, long
HAR_LAGS = {"h1": 1, "h4": 4, "h24": 24, "h168": 168}

COST_BPS = {"futures maker": 4.0, "futures taker": 10.5, "spot taker": 20.0}

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=5.0,
    n_jobs=4,
    verbose=-1,
)


def build(symbol: str, horizon: int) -> pd.DataFrame:
    """Feature matrix plus forward realised volatility, sampled every `horizon`."""
    m = FE.build_minute_frame(symbol)
    rv = m["realized_var"]

    # forward realised variance over (t, t+horizon], from the reversed rolling sum
    fwd_var = rv[::-1].rolling(horizon, min_periods=1).sum()[::-1].shift(-1)

    df = F2.dataset(symbol, horizon=horizon)
    df = df.join(np.log(np.sqrt(fwd_var).replace(0.0, np.nan)).rename("y"), how="left")

    # HAR lags: trailing realised volatility over multiples of the horizon
    for name, mult in HAR_LAGS.items():
        w = horizon * mult
        lag = np.sqrt(rv.rolling(w, min_periods=max(2, w // 2)).sum() / mult)
        df[f"har_{name}"] = np.log(lag.reindex(df.index).replace(0.0, np.nan))

    # realised absolute move over the horizon, in basis points - the quantity the
    # tool reports to the user
    df["fwd_abs_bps"] = df["fwd_ret"].abs() * 1e4
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["y"] + [f"har_{k}" for k in HAR_LAGS]
    )


def _qlike(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """QLIKE on variance. Lower is better; asymmetric against under-forecasting."""
    v_true = np.exp(2 * y_true_log)
    v_pred = np.exp(2 * y_pred_log)
    r = v_true / v_pred
    return float(np.mean(r - np.log(r) - 1.0))


# columns that describe the future and must never reach a model. features.py has
# its own exclusion list, but it knows nothing about the targets added here, so
# they are named again explicitly rather than relying on that list.
TARGET_COLS = {"y", "fwd_abs_bps"}

# Features that need a HISTORY of order-book snapshots. Binance publishes book
# depth only in the daily archive, so on a cold start a live tool has the current
# snapshot but no trailing window for it. Excluding these answers the deployment
# question: can the tool work from minute one, or must it record for a day first?
DEPTH_PREFIXES = ("depth_imb_", "depth_rel_", "BOOK_")
EXCLUDE_PREFIXES: tuple[str, ...] = ()


def walk_forward(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    har_cols = [f"har_{k}" for k in HAR_LAGS]
    ml_cols = har_cols + [c for c in F2.feature_names(df) if c not in har_cols]
    ml_cols = [c for c in ml_cols if c in df.columns and c not in TARGET_COLS]
    if EXCLUDE_PREFIXES:
        ml_cols = [c for c in ml_cols if not c.startswith(EXCLUDE_PREFIXES)]
    leaked = TARGET_COLS.intersection(ml_cols)
    assert not leaked, f"target column reached the feature set: {leaked}"

    y = df["y"].to_numpy()
    idx = df.index
    Xh = df[har_cols].to_numpy(dtype="float64")
    Xm = df[ml_cols].to_numpy(dtype="float32")

    preds = {k: [] for k in ("naive", "ewma", "har", "har_ml")}
    stamps = []
    cursor = idx.min() + pd.Timedelta(days=MIN_TRAIN_DAYS)
    while cursor < idx.max():
        stop = cursor + pd.Timedelta(days=RETRAIN_DAYS)
        tr = (idx < cursor) & (idx >= cursor - pd.Timedelta(days=ROLLING_DAYS))
        te = (idx >= cursor) & (idx < stop)
        if te.sum() == 0 or tr.sum() < 500:
            cursor = stop
            continue

        # naive: the most recent trailing volatility of the same length
        preds["naive"].append(df.loc[idx[te], "har_h1"].to_numpy())

        # ewma over the trailing series, fitted decay-free (RiskMetrics 0.94)
        ew = df["har_h1"].ewm(alpha=0.06, adjust=False).mean()
        preds["ewma"].append(ew.shift(1).loc[idx[te]].to_numpy())

        # HAR: ordinary least squares on the four lags
        A = np.column_stack([np.ones(tr.sum()), Xh[tr]])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        B = np.column_stack([np.ones(te.sum()), Xh[te]])
        preds["har"].append(B @ coef)

        # HAR plus everything else, gradient boosted
        mdl = LGBMRegressor(random_state=0, **PARAMS)
        mdl.fit(Xm[tr], y[tr])
        preds["har_ml"].append(mdl.predict(Xm[te]))

        stamps.append(idx[te])
        cursor = stop

    out = pd.DataFrame(
        {k: np.concatenate(v) for k, v in preds.items()},
        index=stamps[0].append(stamps[1:]),
    )
    out["y"] = df.loc[out.index, "y"]
    out["fwd_abs_bps"] = df.loc[out.index, "fwd_abs_bps"]
    return out


def score(out: pd.DataFrame, horizon: int, label: str) -> pd.DataFrame:
    y = out["y"].to_numpy()
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    rows = []
    for k in ("naive", "ewma", "har", "har_ml"):
        p = out[k].to_numpy()
        ok = np.isfinite(p) & np.isfinite(y)
        r2 = 1.0 - float(np.sum((y[ok] - p[ok]) ** 2)) / ss_tot
        row = {"model": k, "R2_log": r2, "QLIKE": _qlike(y[ok], p[ok])}

        # the practical call: predicted average absolute move over the horizon.
        # for a normal-ish return, E|r| = sigma * sqrt(2/pi)
        pred_bps = np.exp(p) * np.sqrt(2 / np.pi) * 1e4
        for cost_name, cost in COST_BPS.items():
            called = pred_bps > cost
            actual = out["fwd_abs_bps"].to_numpy() > cost
            row[f"hit_{cost_name.split()[-1]}"] = float(
                (called[ok] == actual[ok]).mean() * 100
            )
        rows.append(row)
    t = pd.DataFrame(rows)
    print(f"\n----- {label} | horizon {horizon}m | {len(out):,} samples -----")
    print(t.round(4).to_string(index=False))
    return t


def run(symbol: str = "BTCUSDT") -> None:
    for h in HORIZONS:
        df = build(symbol, h)
        print(f"\n{'=' * 88}")
        print(f"{symbol} | horizon {h}m | {len(df):,} samples "
              f"| {len(F2.feature_names(df))} features", flush=True)
        out = walk_forward(df, h)
        split = out.index[int(len(out) * DEV_FRACTION)]
        score(out, h, "ALL (walk-forward)")
        score(out[out.index >= split], h, "HOLDOUT")

        # how often is a scalp even viable, and does the model see it coming?
        act = out["fwd_abs_bps"]
        print(f"\n  share of {h}-minute windows whose actual move cleared:")
        for cost_name, cost in COST_BPS.items():
            print(f"     {cost_name:<15} {float((act > cost).mean() * 100):5.1f}%")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
