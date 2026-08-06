"""
decay_monitor.py — is this signal still working, or did it die?

The central discovery of this project: a directional edge existed on BTC and ETH
in late 2024, both assets simultaneously, and was consumed during the first
quarter of 2025. Every month since has been negative. Anyone measuring it on the
full two-year sample would still call it profitable today, because the dead
period is averaged with the live one.

No charting tool answers "does my method still work". That is what this does.

Method — prequential (test-then-train) evaluation:

    for each new observation, in time order:
        1. the model predicts it              -> record the error
        2. only then does the model learn it
        3. move on

Every prediction is therefore out of sample by construction; there is no
in-sample number that can flatter the result, and no holdout has to be carved
out. What comes back is not one score but a score THROUGH TIME, which is the
only form in which decay is visible.

On top of that sits a change detector (Page-Hinkley), which raises a flag when
performance degrades beyond what noise explains, rather than waiting for a
person to notice a bad quarter.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import features2 as F2
import train_vol as TV
import vol_model as V

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")

# how often the online model refreshes, and how much history it keeps
REFRESH_MIN = 1440          # rebuild once a day
MEMORY_DAYS = 90

# Page-Hinkley change detector
PH_DELTA = 0.005            # slack: ignore drift smaller than this
PH_LAMBDA = 12.0            # alarm threshold


class PageHinkley:
    """
    Flags a sustained rise in error. Tracks the running mean and the cumulative
    signed deviation from it; when the current cumulative value falls far enough
    below its own running minimum, the recent errors are consistently worse than
    history and an alarm fires.
    """

    def __init__(self, delta: float = PH_DELTA, lam: float = PH_LAMBDA):
        self.delta, self.lam = delta, lam
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.cum_min = 0.0

    def update(self, err: float) -> bool:
        self.n += 1
        self.mean += (err - self.mean) / self.n
        self.cum += err - self.mean - self.delta
        self.cum_min = min(self.cum_min, self.cum)
        return (self.cum - self.cum_min) > self.lam

    def reset(self) -> None:
        self.__init__(self.delta, self.lam)


def prequential(df: pd.DataFrame, cols: list[str],
                refresh_min: int = REFRESH_MIN,
                memory_days: int = MEMORY_DAYS) -> pd.DataFrame:
    """
    Test-then-train over the whole timeline. Returns one row per observation with
    the model's prediction, the benchmark's prediction, and both errors.
    """
    idx = df.index
    y = df["y"].to_numpy()
    X = df[cols].to_numpy("float32")
    har_cols = [f"har_{k}" for k in V.HAR_LAGS]
    Xh = df[har_cols].to_numpy("float64")

    # Refitting and predicting are separate schedules. An earlier version had a
    # single "skip until the next refresh" test at the top of the loop, which
    # skipped the PREDICTION too — it scored one observation per day instead of
    # all of them, and left the weekly table empty. Here the model is refit on a
    # cadence, and every observation in between is still predicted before being
    # learned, which is what makes the evaluation prequential.
    warmup_end = idx.min() + pd.Timedelta(days=memory_days)
    cursor = warmup_end
    parts = []

    while cursor < idx.max():
        block_end = cursor + pd.Timedelta(minutes=refresh_min)
        tr = (idx < cursor) & (idx >= cursor - pd.Timedelta(days=memory_days))
        te = (idx >= cursor) & (idx < block_end)
        if te.sum() == 0 or tr.sum() < 500:
            cursor = block_end
            continue

        mdl = LGBMRegressor(random_state=0, **V.PARAMS)
        mdl.fit(X[tr], y[tr])
        A = np.column_stack([np.ones(tr.sum()), Xh[tr]])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)

        p_ml = mdl.predict(X[te])
        p_har = np.column_stack([np.ones(te.sum()), Xh[te]]) @ coef
        parts.append(pd.DataFrame({
            "y": y[te],
            "pred_ml": p_ml,
            "pred_har": p_har,
            "err_ml": np.abs(y[te] - p_ml),
            "err_har": np.abs(y[te] - p_har),
        }, index=idx[te]))
        cursor = block_end

    out = pd.concat(parts).sort_index()
    out.index.name = "ts"
    return out


def monitor(res: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Skill through time: how much better than the benchmark, week by week."""
    g = res.groupby(pd.Grouper(freq=freq))
    out = pd.DataFrame({
        "n": g.size(),
        "mae_ml": g["err_ml"].mean(),
        "mae_har": g["err_har"].mean(),
    })
    out = out[out["n"] > 20]
    # positive means the model beats the benchmark; zero means it has no edge
    out["skill_%"] = (1 - out["mae_ml"] / out["mae_har"]) * 100

    ph = PageHinkley()
    alarms = []
    for _, r in out.iterrows():
        alarms.append(ph.update(-r["skill_%"] / 100.0))
    out["alarm"] = alarms
    return out


def run(symbol: str = "BTCUSDT", horizon: int = 15) -> pd.DataFrame:
    df = V.build(symbol, horizon)
    cols = TV.live_feature_cols(df)
    print(f"{symbol} {horizon}m | {len(df):,} observations | "
          f"{len(cols)} features | prequential evaluation", flush=True)

    res = prequential(df, cols)
    tbl = monitor(res)

    overall = (1 - res["err_ml"].mean() / res["err_har"].mean()) * 100
    recent = tbl.tail(8)
    print(f"\n  skill over the whole record : {overall:+.1f} %")
    print(f"  skill over the last 8 weeks : {recent['skill_%'].mean():+.1f} %")
    print(f"  weeks with positive skill   : "
          f"{(tbl['skill_%'] > 0).mean() * 100:.0f} %")
    print(f"  change alarms raised        : {int(tbl['alarm'].sum())}")

    print("\n  last 12 weeks:")
    print(tbl.tail(12)[["n", "mae_ml", "mae_har", "skill_%", "alarm"]]
          .round(4).to_string())

    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"decay_{symbol}_{horizon}m.csv")
    tbl.to_csv(path)
    print(f"\n  written to {path}")
    return tbl


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    hor = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    run(sym, hor)
