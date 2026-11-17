# Stock Analyzer

Intraday (day-trading) signal app. You pick a watchlist; during market hours the engine checks
every ticker each minute on 1, 3, 5 and 15-minute bars and sends BUY / SELL (short) alerts with
entry, stop loss, two targets and a confidence level, on the website and as device notifications.

See [PLAN.md](PLAN.md) for the original design.

## Run it

Everything runs from one command. The Python environment lives outside OneDrive.

```bash
C:/Users/korad/StockAnalyzerData/venv/Scripts/python -m backend.run
```

Then open http://127.0.0.1:8000, create an account, add tickers on the Watchlist page and turn
on notifications in Settings. Leave the command running during market hours (9:30–16:00 ET).

## Other commands

| What | Command (prefix with the venv python) |
|---|---|
| Top up bar history | `python -m backend.data.alpaca_data` |
| Retrain and re-validate the models | `python -m backend.model.train` |
| Replay a past day through the live engine | `python -m backend.engine.replay 2026-11-13 AAPL NVDA` |
| Run the tests | `python -m pytest backend/tests -q` |
| Rebuild the website after frontend changes | `cd frontend && npm run build` |

## How a signal is made

1. **Bars** – 1-minute bars from Alpaca, aggregated into 3/5/15-minute bars (`backend/data`).
2. **Features** – about 45 scale-free indicators per bar (`backend/features`).
3. **Strategies** – eight setups propose trades (`backend/strategies/library.py`).
4. **Confidence** – a gradient-boosted model estimates the chance that target 1 is hit before
   the stop (`backend/model`).
5. **Gate** – only setups from a timeframe and strategy that were profitable after costs on
   unseen data are sent as validated signals (`backend/model/train.py`).
6. **Lifecycle** – each signal is tracked to target, stop or expiry and added to the track
   record (`backend/engine/core.py`).

Only signals with confidence of 50% or higher are posted or sent as alerts
(`MIN_POST_CONFIDENCE` in `backend/config.py`, or the same name in `.env`).

Stops sit beyond the level that invalidates the setup, bounded to 1.0–2.5 ATR. Targets are 1R
and 2R (a closer first target is what lets honest confidence reach 50%). Every signal is closed by 15:55 ET.

## Validation status

Run `python -m backend.model.train` to see the full report; the Track record page shows the
same numbers. A timeframe sends validated signals only if walk-forward testing found a
confidence level with positive results after costs **and** that held up on the final six months
of data, which are never used for any decision.

`ALLOW_UNVALIDATED=true` in `.env` is a test mode: it also sends the highest-confidence setups
from models that failed validation, labelled "Unvalidated" on the site. Use it to exercise the
app, not to trade.

## Data notes

- History for training is full-market (SIP) data. Live bars come from the free IEX feed, polled
  over REST a few seconds after each minute closes. Volume features are relative, so the
  difference in raw volume between the two feeds does not matter.
- Alpaca's free data is licensed for personal use. Selling subscriptions requires a commercial
  market-data agreement.

## Before going public

- Buy a commercial data licence and switch the live feed to SIP.
- Host on a server with HTTPS (device notifications need it outside localhost).
- Move from SQLite to PostgreSQL, add password reset and email verification.
- Get legal advice on selling trading signals (investment-adviser rules and disclaimers).
