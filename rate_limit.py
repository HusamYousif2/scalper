"""
rate_limit.py — stay inside Binance's published request budget.

Written after a parallel fetcher ran six workers against /fapi/v1/aggTrades,
which carries a weight of 20 per call. That is roughly 24,000 weight per minute
against a 2,400 limit, and the exchange answered with HTTP 418: a temporary IP
ban. Bans escalate with repeat offences, and any request sent while banned
extends it, so the retry path here refuses to send rather than trying again.

Two mechanisms:

  1. a token bucket that spends request weight and blocks when the budget for
     the current minute is gone
  2. a process-wide ban gate — when any thread sees 418 or 429, every thread
     stops until the ban expires
"""

import threading
import time

# published limit is 2400 weight per minute per IP; half of it leaves room for
# other tools on the same connection and for the exchange's own accounting drift
BUDGET_PER_MIN = 1200

# request weight by endpoint. klines is the cheap bulk route: weight 5 returns
# 1000 candles, where aggTrades costs 20 for at most 1000 trades.
WEIGHTS = {
    "/fapi/v1/klines": 5,
    "/fapi/v1/aggTrades": 20,
    "/fapi/v1/depth": 20,
    "/fapi/v1/time": 1,
    "/futures/data/openInterestHist": 1,
    "/futures/data/takerlongshortRatio": 1,
    "/futures/data/topLongShortPositionRatio": 1,
    "/futures/data/topLongShortAccountRatio": 1,
    "/futures/data/globalLongShortAccountRatio": 1,
}
DEFAULT_WEIGHT = 5


class Budget:
    def __init__(self, per_min: int = BUDGET_PER_MIN):
        self.per_min = per_min
        self._lock = threading.Lock()
        self._window_start = time.monotonic()
        self._spent = 0
        self._banned_until = 0.0

    def ban_for(self, seconds: float) -> None:
        with self._lock:
            self._banned_until = max(self._banned_until, time.monotonic() + seconds)

    def banned_seconds_left(self) -> float:
        with self._lock:
            return max(0.0, self._banned_until - time.monotonic())

    def spend(self, path: str) -> None:
        """Block until this call fits in the budget, then record it."""
        weight = WEIGHTS.get(path, DEFAULT_WEIGHT)
        while True:
            with self._lock:
                left = self._banned_until - time.monotonic()
                if left > 0:
                    raise RuntimeError(
                        f"rate-limit ban active for another {left:.0f}s; "
                        f"not sending requests (sending would extend it)"
                    )
                now = time.monotonic()
                if now - self._window_start >= 60.0:
                    self._window_start = now
                    self._spent = 0
                if self._spent + weight <= self.per_min:
                    self._spent += weight
                    return
                wait = 60.0 - (now - self._window_start)
            time.sleep(min(max(wait, 0.05), 5.0))


BUDGET = Budget()
