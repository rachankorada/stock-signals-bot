# Stock Analyzer — Project Plan

Intraday quant signal app: users pick a watchlist, and during market hours the app sends
BUY / SELL alerts with confidence, stop loss and take profit to the website and their device.

Status: first build complete (2026-11-15). See README.md for how to run it and for the validation results.

---

## 1. What the app does

1. A user signs in and enters the tickers they want analyzed (their watchlist).
2. During US market hours the engine reads live 1-minute bars for those tickers from Alpaca
   and builds 3, 5 and 15-minute bars from them.
3. Each time a bar closes, a set of quant strategies checks for a setup.
4. A machine-learning model scores each setup. The score is the confidence level.
5. If the setup passes the filters, the app creates a signal and sends it:
   - as a live alert on the website (toast, sound, feed entry), and
   - as a push notification on the user's phone or desktop.
6. The app then tracks the signal until it hits take profit, hits stop loss or expires,
   and records the outcome in a public track record.

### What every alert contains

| Field | Example |
|---|---|
| Ticker and direction | NVDA — BUY |
| Timeframe | 5m |
| Entry price / zone | 142.30 (valid to 142.55) |
| Stop loss | 141.62 |
| Take profit 1 / 2 | 143.32 / 144.00 |
| Risk : reward | 1 : 1.5 / 1 : 2.5 |
| Confidence | 68% |
| Strategy | VWAP trend pullback |
| Reasons | Above VWAP, 15m trend up, relative volume 1.8x |
| Past results of this setup | 61% win rate over 412 backtest trades |
| Generated at / expires at | 10:42:00 ET / 10:57:00 ET |

---

## 2. Honest limits

- No set of strategies gives "optimal" or guaranteed signals. The goal is a small, measured
  edge after costs, shown honestly.
- Intraday edges are thin. On 1m and 3m bars, spread and slippage can erase most of the
  profit. Expect 5m and 15m to be the strongest timeframes; 1m and 3m only go live if they
  pass the same tests.
- A strategy/ticker/timeframe combination only sends alerts if it was profitable after costs
  on data it was never trained on (section 6). Some combinations will be switched off.
- Confidence is a calibrated probability, not a promise: of all signals shown at 70%, about
  70% should hit take profit before stop loss. We check this and show it.
- The site shows a "not financial advice" disclaimer on every page and alert.

---

## 3. Data

**Source: Alpaca Market Data API (free plan, Trading API keys).**

| Need | Source | Notes |
|---|---|---|
| Training / backtest history | Historical 1-minute bars, 2016 onward | Full-market (SIP) data, split and dividend adjusted |
| 3m, 5m, 15m bars | Built from 1-minute bars | Same code for history and live, so they always match |
| Live bars | WebSocket stream, IEX feed | Real time, but IEX is one exchange: prices are close, volume is low |
| Market calendar and hours | Alpaca calendar endpoint | Handles holidays and half days |
| Market context | SPY, QQQ and sector ETFs | Used as inputs for every ticker |

**Free-plan constraints the design has to respect**

- The live stream allows about **30 symbols at once**. That is the total across all users,
  minus 2 or 3 reserved for SPY/QQQ. Verify the current number on Alpaca's pricing page.
  Lifting it needs the paid data plan (about $99/month).
- Live volume comes from IEX only, so volume features are relative (compared with the same
  stock's usual IEX volume at that time of day), never raw share counts.
- Alpaca's free data is licensed for personal use. Showing it to the public or charging for
  the site needs a paid/commercial data agreement. Fine for personal use and testing.

**Storage**

- Bar history: Parquet files, one folder per ticker (fast, small, no database needed).
- App data (users, watchlists, signals, outcomes): SQLite to start, PostgreSQL when hosted.
- All data access goes through one `DataProvider` interface so the source can be swapped.

---

## 4. Signal engine

### 4.1 Features computed on every bar

- **Trend:** EMA 9/20/50, EMA slope, ADX, price vs. session VWAP, higher-timeframe trend.
- **Momentum:** RSI, MACD histogram, rate of change, consecutive-bar counts.
- **Mean reversion:** Bollinger z-score, distance from VWAP in ATRs, RSI(2).
- **Volatility:** ATR, realized volatility, Bollinger/Keltner squeeze, bar range vs. average.
- **Volume:** relative volume for the time of day, volume surge, OBV slope.
- **Session:** minutes since open, opening range high/low, prior day high/low/close, gap size.
- **Market context:** SPY and QQQ trend and momentum, sector ETF trend, stock's relative strength.

### 4.2 Strategies (each one proposes candidate setups)

| # | Strategy | Idea | Best in |
|---|---|---|---|
| 1 | Opening range breakout | Break of the first 15–30 minutes' range on high volume | First 90 minutes |
| 2 | VWAP trend pullback | In a trend, buy the pullback to VWAP / EMA20 | Trending days |
| 3 | VWAP mean reversion | Fade moves stretched far from VWAP | Range days, midday |
| 4 | Momentum breakout | Break of a recent high/low with volume surge | Trending, high volume |
| 5 | Volatility squeeze breakout | Bollinger bands inside Keltner channels, then expansion | After quiet periods |
| 6 | Oversold / overbought reversal | RSI(2) + Bollinger extreme with the higher-timeframe trend | Range days |
| 7 | EMA trend continuation | EMA 9/20 cross confirmed by ADX and the 15m trend | Trending days |
| 8 | Prior-day level reaction | Break or rejection at yesterday's high, low or close | Any |

Each strategy runs on 1m, 3m, 5m and 15m and outputs: direction, entry, the level that proves
it wrong (used for the stop), and the reasons.

### 4.3 Regime filter

Classifies each ticker's session as trending, ranging or chaotic (ADX, volatility, market
trend). Trend strategies are muted on range days and the reverse. No signals in the first
1–2 minutes after the open, the last 10 minutes before the close, or when spread/volatility
is abnormal.

### 4.4 Confidence model

- Model: gradient-boosted trees (LightGBM), one per timeframe.
- Input: all features at the moment of the setup, plus which strategy fired.
- Label: "triple barrier" — did price hit take profit first, stop loss first, or run out of time?
- Output: probability of hitting take profit before stop loss, calibrated so the percentage
  is trustworthy. This is the confidence level.
- When several strategies agree on the same ticker and direction, they merge into one
  signal, and the agreement is an input to the model.
- Alerts are sent only above a minimum confidence and a minimum expected value after costs.
  Users can raise the confidence threshold in settings.

### 4.5 Stop loss and take profit

- **Stop loss:** beyond the level that invalidates the setup (swing low, VWAP, range edge),
  with a floor and ceiling in ATR terms (roughly 0.8–2.0 ATR) so it is never absurdly tight or wide.
- **Take profit 1:** 1.5x the risk. **Take profit 2:** 2.5x the risk or the next key level.
- **Time stop:** each signal expires after a set number of bars (for example 12 bars).
- Signals with a risk:reward under 1.5 are discarded.
- Multipliers are tuned per timeframe in backtesting, not hand-picked.

### 4.6 Signal lifecycle

`pending → active → TP1 hit / TP2 hit / stopped / expired`

- One live signal per ticker per direction; cooldown after a signal closes.
- Follow-up alerts: "TP1 hit", "stopped out", "expired".
- Every outcome is stored and feeds the track record page.

---

## 5. Backtester

- Event-driven, bar by bar, using the same strategy code as the live engine.
- Signal decided on bar close, filled at the next bar's open. No look-ahead.
- Costs on every trade: half-spread + slippage, scaled by the stock's price and volatility.
- If stop and target are both touched in one bar, assume the stop was hit first.
- Reports per strategy, ticker and timeframe: number of trades, win rate, average win/loss,
  expectancy, profit factor, Sharpe ratio, maximum drawdown, results by time of day.

---

## 6. Validation and go-live rules

1. **Walk-forward testing:** train on a window (e.g. 2 years), test on the next 3 months,
   roll forward. Only the test periods count.
2. **Purged splits:** gap between train and test data so overlapping trades can't leak.
3. **Hold-out:** the most recent 6 months are never touched until the final check.
4. **Go-live gate:** a strategy/timeframe is enabled only if, out of sample and after costs,
   it has positive expectancy, a profit factor above about 1.2 and enough trades to trust.
5. **Calibration check:** predicted confidence vs. actual hit rate, by bucket.
6. **Paper trading:** 2–4 weeks of live signals placed automatically in the Alpaca paper
   account before trusting them. Live results are compared with the backtest.
7. **Ongoing:** weekly retrain, plus an automatic switch-off if a strategy's live results
   fall well below its backtest.

---

## 7. Website

| Page | Contents |
|---|---|
| Sign up / log in | Email and password accounts |
| Dashboard | Live signal feed, active signals with progress to TP/SL, market status, today's results |
| Watchlist | Add/remove tickers (validated against Alpaca), per-ticker on/off, slots remaining |
| Ticker detail | Live candlestick chart with VWAP/EMAs, entry/SL/TP lines, that ticker's signal history |
| Signal detail | All alert fields, the reasons, chart snapshot, outcome |
| Track record | Win rate, expectancy, equity curve, results by strategy/timeframe/confidence bucket |
| Settings | Notification on/off, minimum confidence, timeframes, strategies, long-only toggle, sound |

Works on phone and desktop. Dark theme by default.

---

## 8. Notifications

| Channel | How | Works when |
|---|---|---|
| On the website | WebSocket push → toast, sound, feed update | Site is open |
| On the device | Web Push (service worker + VAPID keys) | Site is closed; browser running on desktop, always on Android |
| iPhone | Same Web Push, but the user must first "Add to Home Screen" | iOS 16.4+ |
| Later, optional | Telegram bot or email | Anytime |

- The site is a PWA (installable), which is what makes device notifications possible without
  building native apps.
- Notification text: `BUY NVDA 142.30 · SL 141.62 · TP 143.32 · 68% · 5m`. Tapping opens the signal.
- Users only get alerts for their own watchlist and above their own confidence threshold.

---

## 9. Technology

| Part | Choice | Why |
|---|---|---|
| Engine, model, backtester | Python 3.12, pandas, numpy, LightGBM, scikit-learn | Standard quant stack |
| Alpaca access | `alpaca-py` SDK | Official |
| API server | FastAPI + WebSockets | Fast, async, same language as the engine |
| Database | SQLite → PostgreSQL | Simple now, scalable later |
| Web push | `pywebpush` | VAPID push from Python |
| Frontend | React + TypeScript + Vite | Good fit for live-updating UI |
| Charts | TradingView Lightweight Charts | Free, built for candlesticks |
| Tests | pytest | Strategy and backtester correctness |

### Structure

```
Stock Analyzer/
  backend/
    data/         Alpaca client, bar storage, resampling, calendar
    features/     indicator and feature calculations
    strategies/   one file per strategy
    model/        labelling, training, calibration, saved models
    backtest/     simulator, costs, reports
    engine/       live loop, signal lifecycle, notifier
    api/          FastAPI routes, auth, WebSocket
    tests/
  frontend/       React app + service worker
  data/           Parquet bar history (not synced, not committed)
  .env            API keys (never committed)
```

### Running it

- Three processes: live engine (market hours only), API server, frontend.
- Start on your PC. For always-on alerts it needs a small cloud server later (about $5–10/month).
- The project folder is inside OneDrive. Bar history, the Python environment and
  `node_modules` should live outside OneDrive (or the project should move), otherwise
  OneDrive will try to sync many gigabytes and thousands of files.

---

## 10. Build order

| Phase | Deliverable | Done when |
|---|---|---|
| 1. Setup | Project skeleton, `.env`, git, Alpaca connection | A script prints live account/clock info |
| 2. Data | History download, Parquet store, resampling, live stream | 5+ years of bars for a test set of tickers; live bars arrive |
| 3. Features | All indicators, tested | Live and historical features match exactly |
| 4. Strategies | The 8 strategies + regime filter | Each produces sensible setups on charts |
| 5. Backtester | Simulator, costs, reports | Per-strategy reports generated |
| 6. Model | Labelling, training, calibration, walk-forward | Go-live gate applied; calibration chart looks right |
| 7. Live engine | Real-time loop, lifecycle, outcome tracking | Signals appear in the database during market hours |
| 8. API | Auth, watchlists, signals, WebSocket | Endpoints tested |
| 9. Frontend | All pages in section 7 | Live feed updates without refresh |
| 10. Notifications | In-site alerts + Web Push | Alert arrives on a phone with the site closed |
| 11. Paper validation | Auto paper trades, live vs. backtest report | 2–4 weeks of results reviewed |
| 12. Hosting | Cloud deploy | Runs unattended through a full market day |

---

## 11. Decisions needed before coding

1. **Who uses it:** just you, or multiple users with accounts? (Plan assumes accounts, but
   the 30-symbol limit and the data license make a public launch a paid-data decision.)
2. **SELL signals:** should SELL mean "open a short trade", or only "exit a long"? (Plan
   assumes both BUY and short-SELL entries, with a long-only toggle.)
3. **Starting tickers:** which 10–25 tickers should the model be trained and tested on first?
4. **Hosting:** run on your PC during market hours at first, or go straight to a cloud server?
5. **Extra channels:** is Web Push enough, or do you also want Telegram/email alerts?
