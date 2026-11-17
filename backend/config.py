"""Central settings. Everything tunable lives here."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
BARS_DIR = DATA_DIR / "bars"
MODEL_DIR = DATA_DIR / "models"
REPORT_DIR = DATA_DIR / "reports"
DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "app.db"))
for _d in (BARS_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# 25 liquid tickers across sectors, price levels and volatility profiles.
# Used for training and backtesting; the trained model is pooled and uses
# scale-free features, so it also scores tickers outside this list.
TRAIN_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL", "NFLX",  # tech / growth
    "JPM", "BAC", "GS",                                                        # financials
    "XOM",                                                                     # energy
    "UNH", "LLY", "PFE",                                                       # health care
    "WMT", "COST",                                                             # consumer staples
    "BA", "CAT",                                                               # industrials
    "DIS",                                                                     # media
    "PLTR", "COIN", "SOFI",                                                    # high beta
    "F",                                                                       # low priced
]
CONTEXT_TICKERS = ["SPY", "QQQ"]
HISTORY_YEARS = 5   # two years to learn from, then a three-year out-of-sample backtest
BACKTEST_YEARS = 3

TIMEFRAMES = [1, 3, 5, 15]  # minutes

# Trade construction
STOP_MIN_ATR = float(os.environ.get("STOP_MIN_ATR", "1.0"))
STOP_MAX_ATR = float(os.environ.get("STOP_MAX_ATR", "2.5"))
TP1_R = float(os.environ.get("TP1_R", "1.0"))
TP2_R = float(os.environ.get("TP2_R", "2.0"))
# Bars a signal may stay open before it expires, per timeframe
MAX_HOLD_BARS = {1: 45, 3: 30, 5: 24, 15: 16}
NO_ENTRY_FIRST_MIN = 2      # no signals in the first minutes after the open
NO_ENTRY_LAST_MIN = 20      # no new signals this close to the bell
FLATTEN_BEFORE_CLOSE_MIN = 5  # day trading: everything is closed by then

# Cost per share, per side: half-spread + slippage
COST_FIXED = 0.005
COST_BPS = 0.5

# Bars are polled over REST each minute, so the 30-symbol stream limit does not apply
MAX_LIVE_SYMBOLS = 100
MAX_WATCHLIST_PER_USER = 25

# Only signals at or above this confidence are posted to the site or sent as alerts
MIN_POST_CONFIDENCE = float(os.environ.get("MIN_POST_CONFIDENCE", "0.50"))

JWT_SECRET = os.environ.get("JWT_SECRET", "")
PAPER_TRADE = os.environ.get("PAPER_TRADE", "false").lower() == "true"
PAPER_RISK_DOLLARS = float(os.environ.get("PAPER_RISK_DOLLARS", "50"))
# Send signals from setups that failed validation, flagged as unvalidated
ALLOW_UNVALIDATED = os.environ.get("ALLOW_UNVALIDATED", "false").lower() == "true"
