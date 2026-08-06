#!/bin/bash
# Train point and quantile models for the symbols added for the scanner.
# 15-minute horizon only: the scanner ranks on that, and these symbols have
# 200 days of history, which is enough for a 180-day training window but not
# for the longer trailing windows a 60-minute model wants.
cd /root/crypto-quant-lab/scalper || exit 1
SYMS="SOLUSDT,XRPUSDT,BNBUSDT,DOGEUSDT,ADAUSDT,LINKUSDT"
.venv/bin/python -u train_vol.py "$SYMS" 15
.venv/bin/python -u train_quantile.py "$SYMS" 15
echo "NEW_SYMBOL_TRAINING_DONE"
