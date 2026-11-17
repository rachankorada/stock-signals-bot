"""Measure the real bid-ask spread of each stock from historical quotes.

Smaller stocks cost more to trade, so the flat cost used for large caps would
flatter them. For each symbol this samples one full session per quarter of the
backtest window and takes the typical spread, in basis points of price.

Run:  python -m backend.research.spreads
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockQuotesRequest

from backend import config
from backend.data.alpaca_data import ET, client, load_bars
from backend.research.universe import load_universe

SPREADS_PATH = config.DATA_DIR / "small_spreads.json"
DAYS_PER_SYMBOL = 12


def day_spread_bps(symbol: str, day: pd.Timestamp) -> float:
    """Median of the minute-by-minute quoted spread between 09:35 and 15:55."""
    start = day.tz_localize(ET) + timedelta(hours=9, minutes=35)
    req = StockQuotesRequest(symbol_or_symbols=symbol, start=start, end=start + timedelta(hours=6, minutes=20),
                             feed=DataFeed.SIP)
    for attempt in range(4):
        try:
            q = client().get_stock_quotes(req).df
            break
        except Exception:
            if attempt == 3:
                return np.nan
            time.sleep(5 * (attempt + 1))
    if q.empty:
        return np.nan
    q = q.xs(symbol, level="symbol")
    q = q[(q["bid_price"] > 0) & (q["ask_price"] > q["bid_price"])]
    per_min = q[["bid_price", "ask_price"]].resample("1min").last().ffill()
    mid = (per_min["bid_price"] + per_min["ask_price"]) / 2
    return float(((per_min["ask_price"] - per_min["bid_price"]) / mid * 1e4).median())


def symbol_spread(symbol: str) -> dict:
    days = pd.Index(load_bars(symbol).index.normalize().unique()).tz_localize(None)
    days = days[days >= days.max() - pd.DateOffset(years=config.BACKTEST_YEARS)]
    sample = days[np.linspace(5, len(days) - 6, DAYS_PER_SYMBOL).astype(int)]
    vals = [v for v in (day_spread_bps(symbol, d) for d in sample) if np.isfinite(v)]
    out = {"spread_bps": round(float(np.median(vals)), 2) if vals else None, "days": len(vals)}
    print(f"{symbol:6s} {out}", flush=True)
    return out


def load_spreads() -> dict:
    return {k: v["spread_bps"] for k, v in json.loads(SPREADS_PATH.read_text()).items()}


if __name__ == "__main__":
    import sys
    symbols = sys.argv[1:] or load_universe()
    done = json.loads(SPREADS_PATH.read_text()) if SPREADS_PATH.exists() else {}
    todo = [s for s in symbols if s not in done]
    with ThreadPoolExecutor(4) as pool:
        for sym, res in zip(todo, pool.map(symbol_spread, todo)):
            done[sym] = res
            SPREADS_PATH.write_text(json.dumps(done, indent=2))
