# Scalper toolkit — state as of 2026-07-29

## What this is

A decision-support tool for intraday crypto trading. It does **not** emit buy or
sell signals. It answers the question that decides whether a scalp is worth
taking at all: how far is price likely to travel in the next N minutes, what will
the round trip cost, and is the first bigger than the second.

## Why it is built this way

Roughly 200 configurations were tested for directional edge on BTC and ETH at
horizons from 5 minutes to 4 hours — three feature generations, two retraining
schemes, four fee models, with a passing no-lookahead test and a shuffled control
pinned at the null throughout. The directional signal is real and repeatedly
measurable at **1-4 basis points gross**, against a round trip costing **4 bps**
(futures maker) to **20 bps** (spot). The gap is structural.

A tool that sold direction signals would be selling a losing position. Volatility
is a different matter: it is forecastable, it beats the published benchmark, and
unlike a directional edge it cannot be arbitraged away. Full record in
`FINDINGS.md`.

## Measured results

Volatility forecast, out-of-sample R2 on log realised volatility, versus HAR-RV
(Corsi 2009), the standard benchmark:

| | HAR-RV | this model |
|---|---|---|
| BTCUSDT 15m | 0.330 | **0.427** |
| BTCUSDT 60m | 0.233 | **0.310** |
| ETHUSDT 15m | 0.328 | **0.416** |
| ETHUSDT 60m | 0.243 | **0.308** |

Four cases out of four, ETH frozen (it played no part in any design choice).
Prequential monitoring on BTC 15m: skill **+7.7 %** over the whole record,
**+9.6 %** over the last eight weeks, **96 %** of weeks positive, zero decay
alarms. The capability is improving, not decaying.

Regime classification was tested and **rejected**: a trained classifier scored
48.0 % against 47.0 % for simply assuming the current regime persists. One point
does not earn the complexity, so the tool measures the current regime and states
honestly that it holds for another hour 47 % of the time.

## Files

    assess.py         the assessment engine — this is the product
    vol_model.py      volatility model + benchmark comparison
    train_vol.py      fits and saves the deployed models
    live_data.py      archive + REST stitched to the present minute
    rate_limit.py     request budget (an earlier version earned a 418 IP ban)
    online.py         the always-on loop: predict, score, refit, watch decay
    decay_monitor.py  prequential evaluation and change detection
    regime.py         regime measurement (classifier tested and rejected)
    cost_wall.py      break-even accuracy for any horizon and fee tier
    calibrate_rv.py   range-vs-tick variance calibration
    features.py       77 microstructure features
    features2.py      59 more: indicators, levels, volume profile, candles, book
    ingest.py         daily archive downloader
    compare_features.py / model.py / run_full.py / fast_scalp.py / validate_fast.py
                      the directional study that produced FINDINGS.md

## Running it

    cd ~/crypto-quant-lab/scalper

    # one assessment
    .venv/bin/python assess.py BTCUSDT 15 futures_taker

    # fee models: spot_taker spot_bnb futures_taker futures_maker
    .venv/bin/python assess.py ETHUSDT 60 futures_maker

    # the always-on loop; the number is minutes, 0 means a single cycle
    .venv/bin/python online.py 720

    # what would any strategy need to break even?
    .venv/bin/python cost_wall.py BTCUSDT

First run on a symbol bridges the gap between the archive and now, which takes a
few minutes and is rate-limited on purpose. It is cached, so later runs take
about 15 seconds.

## The prediction log is the asset

`state/predictions.parquet` records every forecast **before** its outcome exists,
then fills in what happened once the horizon elapses. No number in it can be
back-fitted. It cannot be faked, because it can only be built in real time — a
competitor starting today needs months to accumulate an equivalent record, while
any indicator can be copied in an hour.

`state/status.json` carries the rolling skill and the decay-alarm state.

## Nine bugs found and fixed

Each produced plausible-looking numbers rather than a crash, which is the
dangerous kind:

1. target column reached the feature set → fake R2 of 0.995
2. naive datetime converted to epoch as local time → data silently 8 hours stale
3. archive/live gap → assessment silently 2 days old
4. unthrottled parallel fetch → 418 IP ban from the exchange
5. flat move calibration → expected move understated threefold in calm markets
6. stop derived from raw sigma → tighter than the typical move, hit every time
7. refit and predict schedules conflated → one observation scored per day
8. rolling autocorrelation via Python callback → unusably slow
9. non-integer request limit → HTTP 400

What surfaced all of them was one rule: **never accept a number without a
benchmark printed next to it.**

## What is not built

- Any execution or order placement. By design.
- A web interface. The engine returns a dict; rendering is a separate concern.
- Live order-book depth beyond ±0.15 %. The public endpoint returns the top 1000
  levels only; the ±1/2/5 % bands in the archive are computed exchange-side and
  cannot be reproduced live by anyone.

## IN PROGRESS — stopped 2026-07-29, resume here

### Feature upgrade measurement (partial results saved)

`vol_upgrade.py` was interrupted part-way. The BTCUSDT 15-minute horizon finished
ten of thirteen sets; results are in `vol_upgrade_partial.txt`. Baseline holdout
R2 is 0.4222 with QLIKE 37.73.

| set | holdout R2 | holdout QLIKE | verdict |
|---|---|---|---|
| current | 0.4222 | 37.73 | baseline |
| +XASS cross-asset | **0.4245** | 36.92 | keep |
| +SEAS seasonality | 0.4243 | 36.32 | keep |
| +FLOW persistence | 0.4241 | 37.88 | R2 yes, QLIKE no |
| +CROWD positioning | 0.4236 | **35.21** | keep |
| +WHALE concentration | 0.4235 | **35.09** | keep |
| +SEMI semivariance | 0.4230 | 37.01 | marginal |
| +ABSORB absorption | 0.4223 | 38.49 | reject |
| +JUMP jump split | 0.4223 | 39.20 | reject |
| +OIFLOW cascade | 0.4220 | 39.71 | reject |

Still to run: `+ALL_VOL`, `+ALL_EDGE`, `+EVERYTHING`, and the whole 60-minute
horizon. Restart with:

    cd ~/crypto-quant-lab/scalper
    .venv/bin/python -u vol_upgrade.py BTCUSDT 2>/dev/null | tee vol_upgrade_btc.txt

Roughly 90 minutes. Nothing else heavy should run at the same time.

Reading so far: every individual gain is small (at most +0.0023 R2). The two
custom indicators that pay are WHALE concentration and CROWD positioning, and
they pay through QLIKE — a 7 % reduction — rather than through R2. QLIKE
punishes under-forecasting volatility, which is the error that actually hurts a
trader, so that is the more useful of the two metrics here. Three of the five
custom indicators failed and should be dropped.

### Quantile model — written, never run

`quantile_vol.py` is complete and untested. It fits a separate model per quantile
of the absolute move, replacing the point forecast plus assumed-constant interval.
Its acceptance test is coverage: if the q90 model is honest, the actual move
exceeds it on 10 % of holdout observations.

    .venv/bin/python -u quantile_vol.py BTCUSDT 2>/dev/null | tee quantile_btc.txt

### Online loop

`online.py 720` was running and is now stopped. `state/predictions.parquet` holds
12 logged forecasts, none scored yet (the horizon had not elapsed). Restart it
whenever; it picks up from the log and scores anything that came due meanwhile.

    setsid nohup .venv/bin/python -u online.py 720 > online_loop.txt 2>&1 &

## Honest next steps

1. Let `online.py` run for a week untouched, then check
   `actual_over_predicted_move` in the status file. Near 1.0 means the
   calibration holds on data that did not exist at training time. This is the
   only test that matters before trusting it.
2. Re-run `train_vol.py` monthly; it trains on the last 180 days by design.
3. Only after a month of clean live record is there anything worth showing a
   paying user.
