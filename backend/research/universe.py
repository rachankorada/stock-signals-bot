"""Pick a test universe of smaller, less-followed stocks without looking ahead.

Stocks are chosen only from what was known before the three-year backtest
window opened: listed for the two years before it, with a typical day's
trading between $5M and $50M and a price between $5 and $100. A fixed random
draw then picks the test set, so no stock is chosen for how it did later.

Known limit: only stocks still listed today are considered (survivorship).

Run:  python -m backend.research.universe          (choose and save)
      python -m backend.research.universe --download   (also fetch 1-minute bars)
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from backend import config
from backend.data.alpaca_data import client, download_history

UNIVERSE_PATH = config.DATA_DIR / "small_universe.json"
N_PICK = 40
SEED = 7
DOLLAR_VOLUME = (5e6, 50e6)   # median daily, before the backtest window
PRICE = (5.0, 100.0)
MIN_DAYS = 480                # of roughly 500 sessions in the two selection years
NOT_COMMON_STOCK = ("ETF", "FUND", "TRUST", "WARRANT", "UNIT", "RIGHT", "PREFERRED", "NOTES", "DEPOSITARY",
                    "ETN", "%", "PORTFOLIO", "INDEX", "ACQUISITION")


def candidates() -> pd.DataFrame:
    trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
    assets = trading.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
    rows = []
    for a in assets:
        name = (a.name or "").upper()
        if a.exchange.value not in ("NYSE", "NASDAQ") or not a.tradable or not a.shortable or not a.easy_to_borrow:
            continue
        if not a.symbol.isalpha() or any(word in name for word in NOT_COMMON_STOCK):
            continue
        rows.append({"symbol": a.symbol, "name": a.name, "exchange": a.exchange.value})
    return pd.DataFrame(rows)


def daily_stats(symbols: list, start: datetime, end: datetime) -> pd.DataFrame:
    """Median close, median dollar volume and day count per symbol over the selection period."""
    parts = []
    for i in range(0, len(symbols), 200):
        req = StockBarsRequest(symbol_or_symbols=symbols[i:i + 200], timeframe=TimeFrame.Day, start=start, end=end,
                               feed=DataFeed.SIP, adjustment=Adjustment.RAW)
        for attempt in range(5):
            try:
                df = client().get_stock_bars(req).df
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(5 * (attempt + 1))
        if len(df):
            df["dollars"] = df["close"] * df["volume"]
            parts.append(df.groupby(level="symbol").agg(price=("close", "median"), dollars=("dollars", "median"),
                                                        days=("close", "size")))
        print(f"  daily bars {min(i + 200, len(symbols)):,}/{len(symbols):,}", flush=True)
    return pd.concat(parts)


def choose() -> dict:
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) \
        - timedelta(days=365 * config.BACKTEST_YEARS + 3)
    start = end - timedelta(days=365 * (config.HISTORY_YEARS - config.BACKTEST_YEARS))
    cand = candidates()
    print(f"{len(cand):,} listed common stocks that can be shorted", flush=True)
    stats = daily_stats(cand["symbol"].tolist(), start, end)
    ok = stats[(stats["days"] >= MIN_DAYS) & stats["dollars"].between(*DOLLAR_VOLUME) & stats["price"].between(*PRICE)]
    ok = ok[~ok.index.isin(config.TRAIN_TICKERS + config.CONTEXT_TICKERS)].sort_index()
    picked = sorted(np.random.default_rng(SEED).choice(ok.index.to_numpy(), size=N_PICK, replace=False).tolist())
    names = cand.set_index("symbol")["name"]
    out = {
        "selected_on_data": [str(start.date()), str(end.date())], "eligible": int(len(ok)), "symbols": picked,
        "detail": {s: {"name": names[s], "price": round(float(ok.loc[s, "price"]), 2),
                       "dollar_volume_m": round(float(ok.loc[s, "dollars"]) / 1e6, 1)} for s in picked},
    }
    UNIVERSE_PATH.write_text(json.dumps(out, indent=2))
    print(f"{len(ok):,} eligible; picked {N_PICK}:")
    for s in picked:
        d = out["detail"][s]
        print(f"  {s:6s} ${d['price']:7.2f}  ${d['dollar_volume_m']:5.1f}M/day  {d['name']}")
    return out


def load_universe() -> list:
    return json.loads(UNIVERSE_PATH.read_text())["symbols"]


if __name__ == "__main__":
    symbols = load_universe() if UNIVERSE_PATH.exists() else choose()["symbols"]
    if "--download" in sys.argv:
        for sym in symbols:
            t0 = time.time()
            d = download_history(sym)
            print(f"{sym}: {len(d):,} bars {d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d} ({time.time() - t0:.0f}s)",
                  flush=True)
