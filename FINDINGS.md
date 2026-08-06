# Scalper Research — Findings (2026-07-28)

Question asked: can a model predict BTC/ETH direction over the next 1–4 hours
well enough to trade it profitably after costs?

## Data

Binance USD-M futures public archives, free, no API key. Two full years
(2024-07-29 → 2026-07-26), 730 days per symbol, BTCUSDT and ETHUSDT.

| dataset    | contents                                              | cadence |
|------------|-------------------------------------------------------|---------|
| aggTrades  | every aggregated trade, with taker side                | ~5/sec  |
| bookDepth  | order book notional at ±1, 2, 3, 4, 5 % (±0.2 % only from late 2025) | 30 s |
| metrics    | open interest, top-trader and account long/short ratios, taker volume ratio | 5 min |

Reduced to a 1-minute table (`ingest.py`), then to 77 features (`features.py`):
order-flow imbalance by volume / count / whale size, realized variance, activity
ratios, VWAP deviation, book depth imbalance and its change, open-interest change
and its interaction with price, positioning ratios, session clock.

## Protocol

- Non-overlapping labels (sampling step = horizon).
- Expanding-window walk forward, 180-day minimum train, 14-day test blocks.
- Development = first 70 % of predictions, holdout = last 30 %, holdout used for
  no decision anywhere.
- Realistic passive execution (`execution.py`): a limit entry fills only if price
  trades **through** the level within 5 minutes; exit is a market order.
  Fees: 0.02 % maker in, 0.045 % taker + 0.005 % slippage out.
- Shuffled-label control run through the identical pipeline.

## Results

| symbol | horizon | samples | walk-forward AUC | best holdout Sharpe | positive months |
|--------|---------|---------|------------------|---------------------|-----------------|
| BTCUSDT | 60 m  | 17 147 | 0.5232 | −2.17 | 5 %  |
| BTCUSDT | 120 m | 8 576  | 0.5263 | −0.05 | 21 % |
| BTCUSDT | 240 m | 4 297  | 0.5105 | −0.61 | 21 % |
| ETHUSDT | 60 m  | 17 142 | 0.5190 | −1.99 | 11 % |
| ETHUSDT | 120 m | 8 574  | 0.5273 | +1.47 | 32 % |
| ETHUSDT | 240 m | 4 297  | 0.5196 | +2.57 | 42 % |
| shuffled control | 60 m | 17 147 | 0.5015 | — | — |

Of 144 scored cells (2 symbols × 3 horizons × 12 configurations × 2 phases),
almost all are net negative. The few positive holdout cells (ETH 120 m / 240 m)
are contradicted by their own development phase, which is −13 to −19 bps per
trade on the identical configuration, and none reaches PSR > 0.95.

## Conclusion

**No tradeable edge.** There is a small, statistically real directional signal —
AUC 0.519–0.527 against a shuffled control of 0.5015, on 17 000 samples — but it
is worth roughly 1–2 basis points per trade, while the cheapest realistic round
trip costs about 6.5 basis points. The information exists and is three to six
times too small to pay for its own execution.

The passive-fill model was decisive. Assuming limit orders always fill turned a
losing system into an apparently profitable one; requiring the price to actually
trade through the level dropped fill rates to 30–76 % and removed the gain.
Adverse selection is not a footnote here, it is the whole result.

## Warning recorded for future work

An intermediate run on 220 days showed AUC 0.5677, a 61.8 % hit rate in the top
confidence decile, and +3 bps net with maker fees. On the full 730 days the same
pipeline gives AUC 0.5232 and negative net everywhere. The encouraging result was
a small-sample illusion. Do not act on any result from a partial download.

## Part 2 — short horizons with daily retraining (2026-07-29)

The user's hypothesis: no single model can work every day, so rebuild it daily on
a short memory. Tested at 5 / 15 / 30 minute horizons, rolling 30 and 90 day
training windows, retrained every day.

**The hypothesis was partly right.** Walk-forward AUC at 15 minutes rose from
0.5232 (fortnightly retrain, expanding window) to 0.5552 (daily retrain, 90-day
rolling window). Daily retraining on a short memory genuinely predicts better.

**But the sweep's headline numbers were an artifact.** It chose its volatility
and confidence cutoffs from the distribution of the whole two years, which is not
knowable at trade time. `validate_fast.py` redid the same configuration with
cutoffs taken from a trailing 30-day window only:

| | sweep (hindsight cutoffs) | strict (causal cutoffs), holdout |
|---|---|---|
| BTCUSDT accuracy | 67.5 % | 53.3 % |
| BTCUSDT net, taker | +4.0 bps | −8.8 bps |
| ETHUSDT accuracy | — | 57.4 % |
| ETHUSDT net, taker | — | −9.1 bps |

Deflated Sharpe on the holdout, penalised for the 36 cells inspected: **0.000**.
Shuffled control: AUC 0.5030, accuracy 49.0 %, 0 % positive months — the pipeline
is clean, and the real model does carry information above the null (gross +1.7 bps
against the control's −0.7 bps). It is simply six times too small to pay a
10.5 bps round trip.

The same illusion scaled with horizon: at 5 minutes the hindsight cutoffs
produced an implausible 83.6 % accuracy. `check_leak.py` passed (all 77 features
bit-identical when the future is truncated away), so this was selection, not
leakage.

### The finding that actually matters: the edge decayed

Month by month, both symbols independently:

| month | BTCUSDT net bps | ETHUSDT net bps |
|---|---|---|
| 2024-11 | +19.0 | +21.0 |
| 2024-12 | +12.3 | +15.2 |
| 2025-01 | +8.8 | +12.1 |
| 2025-02 | +8.0 | +8.7 |
| 2025-03 onward | negative | negative |

Two independent assets, same rise and the same collapse in the same quarter.
Noise does not synchronise like that. This was a real, tradeable edge in late
2024 that was consumed during the first quarter of 2025 and has not returned in
the twenty months since. Averaging the dead period with the live one, which is
what full-sample cutoffs silently do, is what made it look alive.

Note also ETHUSDT's holdout: 57.4 % accuracy and −9.1 bps per trade. Win rate
without average move size and cost next to it means nothing.

## Part 3 — do extra feature families help? (2026-07-29)

65 new features in five families were added on top of the original 77 and scored
through the identical strict protocol. Holdout rows only, taker fees (10.5 bps
round trip including spread):

| feature set | n | AUC | accuracy | net bps | PSR |
|---|---|---|---|---|---|
| base | 77 | 0.5566 | 53.24 % | −8.84 | 0.00 |
| base+IND indicators | 97 | 0.5571 | 55.33 % | −7.49 | 0.00 |
| base+SR support/resistance | 99 | 0.5580 | 53.93 % | **−6.75** | 0.00 |
| base+VP volume profile | 81 | 0.5567 | 54.60 % | −8.27 | 0.00 |
| base+CDL candles | 90 | 0.5566 | 54.96 % | −8.13 | 0.00 |
| base+BOOK book shape | 83 | 0.5565 | 53.09 % | −8.15 | 0.00 |
| all | 142 | 0.5598 | 53.78 % | −7.28 | 0.00 |

Three readings:

1. **Nearly doubling the feature count moved AUC by 0.003.** That is noise. The
   information ceiling of this data set was already reached by the original 77.
2. **Support/resistance is the clearest case of memorisation.** It produced the
   largest development gain of any family (56.70 % → 61.17 %, +4.5 points) and
   the largest development-to-holdout collapse (7.2 points). It is nevertheless
   the best holdout set, so it carries some real information alongside the noise
   — just 2 bps worth against a 10.5 bps wall.
3. **All families together (−7.28) is worse than support/resistance alone
   (−6.75).** They compete for splits rather than complementing each other, which
   is the standard behaviour of correlated features on a small rolling window.

Best result across every configuration ever tested: −6.75 bps per trade. Nothing
came within 6 bps of breaking even.

## Overall conclusion

Across roughly 200 scored configurations — three feature generations, six
horizons from 5 minutes to 4 hours, two retraining schemes, two symbols, four fee
models, with a passing no-lookahead test and a shuffled control pinned at the
null throughout — **no configuration is profitable after realistic costs.**

The signal is real and repeatedly measurable at roughly 1–4 bps gross. The
cheapest realistic round trip is 4 bps (futures maker) to 10.5 bps (futures taker
with spread) to 20 bps (spot). The gap is structural, not a modelling failure.

The one genuine discovery is the decay: a tradeable edge existed in late 2024 on
both BTC and ETH simultaneously and was consumed during Q1 2025. Anything that
averages across that boundary will look better than anything that can be traded
today.

## Part 4 — the improvement push (2026-07-30)

Nine new feature families and four modelling ideas were measured against the
deployed 131-feature model. Eight of nine attempts failed; the failures are as
informative as the success.

### Rejected

| attempt | BTCUSDT | ETHUSDT (clean) |
|---|---|---|
| curated winning feature set | dR2 +0.0024 | **dR2 −0.0014** |
| direction probability, 15m, top decile | +0.73 bps | +0.13 bps |
| direction probability, 60m, top decile | +4.63 bps | **−4.79 bps** |
| regime classifier | 48.0 % vs 47.0 % persistence | not run |

Every individual family (semivariance, jump split, seasonality, cross-asset,
absorption, flow persistence, whale concentration, OI cascade, crowding) gained
at most +0.0025 R2 on BTCUSDT, and combining them was consistently WORSE than the
baseline: 184 features scored 0.4226 against 0.4232 for 131. Correlated features
compete for splits rather than complementing each other.

The direction model deserves a specific note. Its probabilities are reasonably
reliable at 15 minutes (worst calibration gap 4-6 points) and badly wrong at 60
minutes (13 points: it says 63.6 % and 50.9 % happens). Brier skill against the
base rate is **−0.22 % on both assets** at 15 minutes — identical to two decimal
places, which marks it as structural rather than accidental. No direction
percentage is displayed anywhere in the product.

### Accepted: conformalised quantiles

The one clean win. Plain quantile regression under-disperses badly — measured
coverage was 15.6 % for the nominal 10 % quantile and 87.0 % for the nominal 90 %,
and the 80 % interval covered only 71.5 % of moves. That error was stable to
within a tenth of a point between development and holdout, which marked it as a
correctable bias rather than noise.

Conformalised quantile regression (Romano, Patterson, Candes 2019) fixed it:

| | worst coverage error before | after |
|---|---|---|
| BTCUSDT 15m | 6.0 pts | **0.8 pts** |
| BTCUSDT 60m | 9.7 pts | **0.8 pts** |
| ETHUSDT 15m | 6.0 pts | **0.8 pts** |
| ETHUSDT 60m | 8.4 pts | **0.5 pts** |

Pinball loss improved at every quantile as well, so correct coverage came at no
cost in sharpness — only the intervals widened, which is the bias being removed
rather than a price being paid. ETHUSDT played no part in designing the
correction.

The serving cadence matters: measuring the conformal shift once over 45 days and
applying it to the next 30 pushed coverage error back to 2.2-6.0 points. The
walk-forward figure was produced by recalibrating weekly, so `online.py`
recalibrates weekly.

### Accepted: validated technical readings

53 standard technical readings were tested for whether an amplification measured
on a trailing window survives into the next period, over twelve folds. Judged on
persistence multiplied by next-period spread, because persistence alone saturates
— rank correlation over five buckets is exactly 1.0 for any monotone
relationship, and every volatility proxy is monotone in forward volatility.

Passed on both assets (21 of 53 on BTCUSDT, 19 on ETHUSDT), and the top ten on
each are all range or volatility measures: ATR, candle range, channel width,
Bollinger bandwidth. Volume point-of-control distance and round-number distance
also qualified — the latter scoring 0.95 usefulness on ETHUSDT, stronger than
every momentum reading combined.

Failed on both assets, and withheld from display: Stochastic-14 (persistence
−0.158 on BTCUSDT, its relationship inverts between periods), candle body, candle
streaks, lower wicks, distance from a short moving average. RSI-60 is stable at
0.892 persistence but its spread is 0.292 — its extremes precede almost the same
move, so it discriminates nothing.

## What was NOT tested

1. **Sub-minute resolution.** `aggTrades` carries millisecond timestamps; we
   aggregated to 1 minute and discarded them. Microstructure alpha is generally
   strongest at second and sub-second scale, decaying within minutes.
2. **Near-touch depth (±0.2 %)**, published only from late 2025 — about 9 months
   of history, dropped here because it is missing for most of the sample.
3. **Very short horizons** (1–15 minutes). Cost per trade is unchanged, so this
   is harder, not easier, but signal decay works in its favour.
4. **Volatility / range as the target** instead of direction. A different and
   much more predictable question.
