"""
curated_test.py — test only the curated set, and test it where it cannot cheat.

The family sweep in vol_upgrade.py identified four additions that improved both
the full-sample and the holdout R2 on BTCUSDT at the 15-minute horizon:
cross-asset volatility, whale concentration, crowd positioning, and order-flow
persistence.

Choosing them BY LOOKING at those results is hindsight. The BTCUSDT figure for
this set therefore means little on its own — it is the number a researcher gets
after picking the winners. What settles it is running the identical frozen recipe
on ETHUSDT, which took no part in the selection.

Accept the curated set only if it beats the baseline on ETHUSDT too. If it does
not, the four families were noise that happened to line up on one asset, and the
deployed model stays as it is.
"""

import sys

import numpy as np
import pandas as pd

import vol_upgrade as VU

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HORIZON = 15


def run() -> pd.DataFrame:
    rows = []
    for sym in SYMBOLS:
        df, base, tag = VU.build(sym, HORIZON)
        print(f"\n{'=' * 78}")
        print(f"{sym} | horizon {HORIZON}m | {len(df):,} samples", flush=True)
        for name, fams in (("current", []), ("curated", VU.CURATED)):
            cols = VU.cols_for(df, base, fams, tag)
            out = VU.walk(df, cols, HORIZON)
            split = out.index[int(len(out) * VU.DEV_FRACTION)]
            a = VU.score(out)
            h = VU.score(out[out.index >= split])
            rows.append({"symbol": sym, "set": name, "n_features": len(cols),
                         "R2_all": a["R2_log"], "R2_hold": h["R2_log"],
                         "QLIKE_all": a["QLIKE"], "QLIKE_hold": h["QLIKE"]})
            print(f"  {name:<9} {len(cols):>3} feats | R2 all {a['R2_log']:.4f} "
                  f"hold {h['R2_log']:.4f} | QLIKE all {a['QLIKE']:.3f} "
                  f"hold {h['QLIKE']:.3f}", flush=True)

    t = pd.DataFrame(rows)
    print(f"\n{'=' * 78}\nVERDICT")
    for sym in SYMBOLS:
        s = t[t["symbol"] == sym]
        b = s[s["set"] == "current"].iloc[0]
        c = s[s["set"] == "curated"].iloc[0]
        dr2 = c["R2_hold"] - b["R2_hold"]
        dq = (c["QLIKE_hold"] / b["QLIKE_hold"] - 1) * 100
        tag = "selection-contaminated" if sym == "BTCUSDT" else "CLEAN TEST"
        ok = "improves" if (dr2 > 0 and dq < 0) else "does NOT improve"
        print(f"  {sym:<9} dR2 {dr2:+.4f}  dQLIKE {dq:+.2f} %"
              f"   -> {ok}   [{tag}]")
    return t


if __name__ == "__main__":
    run()
