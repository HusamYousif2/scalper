"""
execution.py — realistic fill simulation for limit-order entries.

The selectivity study assumed every resting limit order gets filled at the price
we wanted. That is the single most optimistic assumption in the whole study, so
it gets its own explicit model here.

How a trade is simulated:

  signal at t, side = long
    -> place a buy limit at  close_t * (1 - offset)
    -> it fills only if the market actually trades down to that price within the
       next ENTRY_WINDOW minutes (we check the real per-minute low)
    -> if it never trades there, the order is cancelled: no position, no cost
    -> the position is closed with a market order at t + horizon

Because the return is measured from the price we were ACTUALLY filled at, and
because we only hold trades that filled, adverse selection is captured directly
by the price path rather than assumed away: an order that fills because a large
seller is walking the book keeps whatever happens next.

Fees follow Binance USD-M futures standard tier.
"""

import numpy as np
import pandas as pd

MAKER_FEE = 0.0002    # resting limit order
TAKER_FEE = 0.00045   # market order
SLIPPAGE_TAKER = 0.00005

ENTRY_WINDOW = 5      # minutes the entry order is allowed to rest


def simulate(
    minute: pd.DataFrame,
    signals: pd.DataFrame,
    horizon: int,
    offset: float,
    entry_window: int = ENTRY_WINDOW,
    exit_limit: bool = False,
) -> pd.DataFrame:
    """
    Simulate limit entries for a set of directional signals.

    minute  : the 1-minute frame with high / low / close on a gap-free grid
    signals : DataFrame indexed by signal time with a 'side' column (+1 / -1)
    returns : one row per signal, with fill flag and net return
    """
    idx = minute.index
    pos_of = pd.Series(np.arange(len(idx)), index=idx)

    high = minute["high"].to_numpy()
    low = minute["low"].to_numpy()
    close = minute["close"].to_numpy()

    rows = []
    for ts, side in signals["side"].items():
        if ts not in pos_of.index:
            continue
        i = int(pos_of[ts])
        j_end = i + entry_window
        k = i + horizon
        if k >= len(idx):
            continue

        ref = close[i]
        # a buy waits below the current price, a sell waits above it
        limit_px = ref * (1 - offset) if side > 0 else ref * (1 + offset)

        # strict penetration, not a touch: our order sits at the back of the
        # queue at that price, so we only assume a fill once the market has
        # traded THROUGH the level and cleared everything resting ahead of us
        window_low = low[i + 1: j_end + 1]
        window_high = high[i + 1: j_end + 1]
        if side > 0:
            hit = np.nonzero(window_low < limit_px)[0]
        else:
            hit = np.nonzero(window_high > limit_px)[0]

        if len(hit) == 0:
            rows.append({"ts": ts, "side": side, "filled": False,
                         "entry": np.nan, "exit": np.nan, "net": 0.0})
            continue

        entry_px = limit_px
        exit_px = close[k]

        gross = side * np.log(exit_px / entry_px)
        if exit_limit:
            fees = MAKER_FEE * 2
        else:
            fees = MAKER_FEE + TAKER_FEE + SLIPPAGE_TAKER
        rows.append({"ts": ts, "side": side, "filled": True,
                     "entry": entry_px, "exit": exit_px,
                     "net": gross - fees})

    out = pd.DataFrame(rows)
    return out.set_index("ts") if len(out) else out


def score(trades: pd.DataFrame, horizon: int) -> dict:
    """Summarise a simulated trade set."""
    if len(trades) == 0:
        return {}
    filled = trades[trades["filled"]]
    n = len(filled)
    if n < 20:
        return {"n_signals": len(trades), "n_filled": n}
    net = filled["net"].to_numpy()
    sd = net.std(ddof=1)
    sr = net.mean() / sd if sd > 0 else 0.0
    # trades per year if this signal rate persisted
    per_year = n / (len(trades) * horizon / (365 * 24 * 60))
    return {
        "n_signals": len(trades),
        "n_filled": n,
        "fill_rate_%": 100.0 * n / len(trades),
        "hit_%": 100.0 * float((net > 0).mean()),
        "net_bps": float(net.mean() * 1e4),
        "total_%": float(net.sum() * 100),
        "sharpe_ann": float(sr * np.sqrt(per_year)),
        "worst_bps": float(net.min() * 1e4),
    }
