"""
regime.py — is the market trending, mean-reverting, or just chopping?

Knowing the expected size of the coming move (vol_model.py) tells a scalper
whether to trade at all. It does not say which kind of trade fits. A breakout
entry is right in a trending stretch and is a slow bleed in a choppy one; fading
extremes is the reverse. This layer answers that.

Measures used, all computed from past data only:

  efficiency ratio   net displacement divided by total path length. Near 1 the
                     market went somewhere in a straight line; near 0 it
                     travelled a long way and ended where it started.
  variance ratio     k-period variance over k times the one-period variance.
                     Above 1 moves reinforce each other (trend); below 1 they
                     cancel (mean reversion). The classical Lo-MacKinlay test.
  return autocorr    lag-1 autocorrelation of returns inside the window.
  directional runs   length of the current unbroken up or down streak.

The honest question is not "can we describe the current regime" — that is just
arithmetic — but "does the current regime tell us anything about the next one".
So the label is the NEXT window's regime, and the benchmark to beat is
persistence: assume tomorrow looks like today. A classifier that cannot beat
persistence has discovered nothing.
"""

import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

import features as FE
import features2 as F2

WINDOW = 240          # minutes used to characterise a regime
HORIZON = 60          # how far ahead the regime is predicted
MEMORY_DAYS = 90
REFRESH_DAYS = 7

# efficiency-ratio cut points, set from the sample's own terciles so the three
# classes are balanced by construction rather than by an arbitrary threshold
CLASSES = ["choppy", "neutral", "trending"]

PARAMS = dict(
    n_estimators=300, learning_rate=0.03, num_leaves=15,
    min_child_samples=100, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.7, reg_lambda=5.0, n_jobs=4, verbose=-1,
)


def efficiency_ratio(close: pd.Series, w: int) -> pd.Series:
    """Net move over total path length; 1 is a straight line, 0 is a round trip."""
    net = (close - close.shift(w)).abs()
    path = close.diff().abs().rolling(w, min_periods=w // 2).sum()
    return net / path.replace(0.0, np.nan)


def variance_ratio(close: pd.Series, w: int, k: int = 10) -> pd.Series:
    """Lo-MacKinlay variance ratio inside a rolling window."""
    r1 = np.log(close / close.shift(1))
    rk = np.log(close / close.shift(k))
    v1 = r1.rolling(w, min_periods=w // 2).var()
    vk = rk.rolling(w, min_periods=w // 2).var()
    return vk / (k * v1.replace(0.0, np.nan))


def rolling_acf1(r: pd.Series, w: int) -> pd.Series:
    """
    Lag-1 autocorrelation over a rolling window, computed from rolling moments.

    The obvious implementation, rolling(...).apply(autocorr), calls back into
    Python once per window — two million times over this data set, which took
    long enough to be unusable. Written as a ratio of rolling means it is one
    vectorised pass.
    """
    mp = max(2, w // 2)
    lag = r.shift(1)
    m1 = r.rolling(w, min_periods=mp).mean()
    m2 = lag.rolling(w, min_periods=mp).mean()
    cov = (r * lag).rolling(w, min_periods=mp).mean() - m1 * m2
    sd1 = r.rolling(w, min_periods=mp).std(ddof=0)
    sd2 = lag.rolling(w, min_periods=mp).std(ddof=0)
    return cov / (sd1 * sd2).replace(0.0, np.nan)


def regime_features(m: pd.DataFrame) -> pd.DataFrame:
    close = m["close"]
    r = np.log(close / close.shift(1))
    f = pd.DataFrame(index=m.index)
    for w in (60, WINDOW, 1440):
        f[f"eff_{w}"] = efficiency_ratio(close, w)
        f[f"vr_{w}"] = variance_ratio(close, w)
        f[f"acf1_{w}"] = rolling_acf1(r, w)
    sign = np.sign(close.diff()).fillna(0.0)
    grp = (sign != sign.shift(1)).cumsum()
    f["run_len"] = sign * sign.groupby(grp).cumcount().add(1)
    return f


def build(symbol: str) -> pd.DataFrame:
    m = FE.build_minute_frame(symbol)
    base = F2.dataset(symbol, horizon=HORIZON)
    reg = regime_features(m)

    df = base.join(reg, how="left")
    # forward efficiency ratio: the regime that actually follows
    fwd_eff = efficiency_ratio(m["close"], WINDOW).shift(-HORIZON)
    df["fwd_eff"] = fwd_eff.reindex(df.index)
    df["cur_eff"] = reg["eff_240"].reindex(df.index)
    return df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["fwd_eff", "cur_eff"]
    )


def to_class(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.cut(s, [-np.inf, lo, hi, np.inf], labels=[0, 1, 2]).astype("float")


def run(symbol: str = "BTCUSDT") -> pd.DataFrame:
    df = build(symbol)
    cols = [c for c in F2.feature_names(df)
            if c not in ("fwd_eff", "cur_eff", "fwd_ret", "fwd_up", "fwd_dn")]
    print(f"{symbol} | {len(df):,} samples | {len(cols)} features", flush=True)

    idx = df.index
    y_raw = df["fwd_eff"]
    cur_raw = df["cur_eff"]

    preds, truth, persist, stamps = [], [], [], []
    cursor = idx.min() + pd.Timedelta(days=MEMORY_DAYS)
    while cursor < idx.max():
        stop = cursor + pd.Timedelta(days=REFRESH_DAYS)
        tr = (idx < cursor) & (idx >= cursor - pd.Timedelta(days=MEMORY_DAYS))
        te = (idx >= cursor) & (idx < stop)
        if te.sum() == 0 or tr.sum() < 500:
            cursor = stop
            continue
        # thresholds come from the TRAINING window only
        lo, hi = y_raw[tr].quantile([1 / 3, 2 / 3])
        y_tr = to_class(y_raw[tr], lo, hi)
        y_te = to_class(y_raw[te], lo, hi)
        p_te = to_class(cur_raw[te], lo, hi)

        ok = y_tr.notna()
        clf = LGBMClassifier(random_state=0, **PARAMS)
        clf.fit(df.loc[tr, cols].to_numpy("float32")[ok.to_numpy()],
                y_tr[ok].to_numpy())
        preds.append(clf.predict(df.loc[te, cols].to_numpy("float32")))
        truth.append(y_te.to_numpy())
        persist.append(p_te.to_numpy())
        stamps.append(idx[te])
        cursor = stop

    p = np.concatenate(preds)
    t = np.concatenate(truth)
    q = np.concatenate(persist)
    ok = ~np.isnan(t) & ~np.isnan(q)

    acc_model = float((p[ok] == t[ok]).mean() * 100)
    acc_persist = float((q[ok] == t[ok]).mean() * 100)
    acc_chance = 100.0 / 3

    print(f"\n  regime {HORIZON} minutes ahead, {int(ok.sum()):,} scored")
    print(f"    always guess the middle class : {acc_chance:.1f} %")
    print(f"    persistence (same as now)     : {acc_persist:.1f} %")
    print(f"    model                         : {acc_model:.1f} %")
    print(f"    model minus persistence       : {acc_model - acc_persist:+.1f} points")

    res = pd.DataFrame({"pred": p, "true": t, "persist": q},
                       index=stamps[0].append(stamps[1:]))
    per_class = []
    for i, name in enumerate(CLASSES):
        sel = res["true"] == i
        if sel.sum() > 0:
            per_class.append({
                "regime": name,
                "share_%": float(sel.mean() * 100),
                "model_recall_%": float((res.loc[sel, "pred"] == i).mean() * 100),
                "persist_recall_%": float((res.loc[sel, "persist"] == i).mean() * 100),
            })
    print("\n" + pd.DataFrame(per_class).round(1).to_string(index=False))

    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "reports"), exist_ok=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "reports", f"regime_{symbol}.csv")
    res.to_csv(out)
    print(f"\n  written to {out}")
    return res


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
