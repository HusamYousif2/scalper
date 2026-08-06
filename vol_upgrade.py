"""
vol_upgrade.py — do the new decompositions actually forecast better?

Same walk-forward protocol as vol_model.py, same data, same benchmark. The only
thing that changes between runs is which feature families the model may use, so
any difference is attributable.

Sets tested:
    current              what the deployed model uses today
    current+SEMI         plus realised semivariance
    current+JUMP         plus the jump / continuous split
    current+SEAS         plus intraday seasonality
    current+XASS         plus the other asset's volatility
    current+ALL          all four

Read the HOLDOUT block. A family that lifts the walk-forward average but not the
holdout has fitted the past, which is the usual outcome of adding features and
the reason each one is measured separately instead of all being switched on.
"""

import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import edge_features as EF
import train_vol as TV
import vol_extra as VE
import vol_model as V

HORIZONS = [15, 60]
DEV_FRACTION = 0.70

# volatility decompositions from vol_extra, then the custom indicators from
# edge_features. Each is switched on alone so its contribution is attributable.
# Families that improved BOTH the full-sample and the holdout R2 on BTCUSDT at
# the 15-minute horizon. Naming them here is a selection made with knowledge of
# those results, so the BTCUSDT number for this set is contaminated by hindsight.
# The honest test is the same frozen set on ETHUSDT, which played no part in
# choosing it.
CURATED = ["XASS", "WHALE", "CROWD", "FLOW"]

SETS = {
    "current": [],
    "+CURATED": CURATED,
    "+SEMI": ["SEMI"],
    "+JUMP": ["JUMP"],
    "+SEAS": ["SEAS"],
    "+XASS": ["XASS"],
    "+ABSORB": ["ABSORB"],
    "+FLOW": ["FLOW"],
    "+WHALE": ["WHALE"],
    "+OIFLOW": ["OIFLOW"],
    "+CROWD": ["CROWD"],
    "+ALL_VOL": list(VE.FAMILIES),
    "+ALL_EDGE": list(EF.GROUPS),
    "+EVERYTHING": list(VE.FAMILIES) + list(EF.GROUPS),
}


def build(symbol: str, horizon: int) -> tuple[pd.DataFrame, list[str], dict]:
    df = V.build(symbol, horizon)
    base_cols = TV.live_feature_cols(df)
    extra = VE.build(symbol)
    edge = EF.build(symbol)
    df = df.join(extra.reindex(df.index), how="left")
    df = df.join(edge.reindex(df.index), how="left")

    tag = {}
    for c in extra.columns:
        fam = c.split("_")[0]
        if fam in VE.FAMILIES:
            tag[c] = fam
    for c in edge.columns:
        g = EF.group_of(c)
        if g:
            tag[c] = g
    return df, base_cols, tag


def cols_for(df: pd.DataFrame, base: list[str], fams: list[str],
             tag: dict) -> list[str]:
    return list(base) + [c for c, g in tag.items()
                         if g in fams and c in df.columns]


def walk(df: pd.DataFrame, cols: list[str], horizon: int) -> pd.DataFrame:
    y = df["y"].to_numpy()
    X = df[cols].to_numpy("float32")
    idx = df.index
    preds, stamps = [], []
    cursor = idx.min() + pd.Timedelta(days=V.MIN_TRAIN_DAYS)
    while cursor < idx.max():
        stop = cursor + pd.Timedelta(days=V.RETRAIN_DAYS)
        tr = (idx < cursor) & (idx >= cursor - pd.Timedelta(days=V.ROLLING_DAYS))
        te = (idx >= cursor) & (idx < stop)
        if te.sum() == 0 or tr.sum() < 500:
            cursor = stop
            continue
        mdl = LGBMRegressor(random_state=0, **V.PARAMS)
        mdl.fit(X[tr], y[tr])
        preds.append(mdl.predict(X[te]))
        stamps.append(idx[te])
        cursor = stop
    out = pd.DataFrame({"pred": np.concatenate(preds)},
                       index=stamps[0].append(stamps[1:]))
    out["y"] = df.loc[out.index, "y"]
    return out


def score(out: pd.DataFrame) -> dict:
    y, p = out["y"].to_numpy(), out["pred"].to_numpy()
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    r2 = 1.0 - float(np.sum((y - p) ** 2)) / float(np.sum((y - y.mean()) ** 2))
    return {"n": int(len(y)), "R2_log": r2, "QLIKE": V._qlike(y, p)}


def run(symbol: str = "BTCUSDT") -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        df, base, tag = build(symbol, h)
        print(f"\n{'=' * 78}")
        print(f"{symbol} | horizon {h}m | {len(df):,} samples", flush=True)
        for name, fams in SETS.items():
            cols = cols_for(df, base, fams, tag)
            out = walk(df, cols, h)
            split = out.index[int(len(out) * DEV_FRACTION)]
            all_s = score(out)
            hold_s = score(out[out.index >= split])
            rows.append({"horizon": h, "set": name, "n_features": len(cols),
                         "R2_all": all_s["R2_log"], "QLIKE_all": all_s["QLIKE"],
                         "R2_hold": hold_s["R2_log"], "QLIKE_hold": hold_s["QLIKE"]})
            print(f"  {name:<14} {len(cols):>3} feats | "
                  f"R2 all {all_s['R2_log']:.4f} hold {hold_s['R2_log']:.4f} | "
                  f"QLIKE all {all_s['QLIKE']:.3f} hold {hold_s['QLIKE']:.3f}",
                  flush=True)

    t = pd.DataFrame(rows)
    print(f"\n{'=' * 78}\nHOLDOUT SUMMARY — the row that decides")
    for h in HORIZONS:
        sub = t[t["horizon"] == h].copy()
        base_r2 = float(sub.loc[sub["set"] == "current", "R2_hold"].iloc[0])
        base_q = float(sub.loc[sub["set"] == "current", "QLIKE_hold"].iloc[0])
        sub["dR2"] = sub["R2_hold"] - base_r2
        sub["dQLIKE_%"] = (sub["QLIKE_hold"] / base_q - 1) * 100
        print(f"\n  horizon {h}m")
        print(sub[["set", "n_features", "R2_hold", "dR2",
                   "QLIKE_hold", "dQLIKE_%"]].round(4).to_string(index=False))
    return t


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT")
