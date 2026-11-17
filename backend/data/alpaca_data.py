"""Alpaca market data: historical download and the Parquet bar store."""
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from backend import config

ET = "America/New_York"
COLS = ["open", "high", "low", "close", "volume"]

_client = None


def client() -> StockHistoricalDataClient:
    global _client
    if _client is None:
        _client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    return _client


def regular_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 09:30-16:00 ET bars only."""
    if df.empty:
        return df
    minutes = df.index.hour * 60 + df.index.minute
    return df[(minutes >= 570) & (minutes < 960)]


def fetch_bars(symbol: str, start: datetime, end: datetime, feed: DataFeed = DataFeed.SIP) -> pd.DataFrame:
    """1-minute regular-hours bars, indexed by bar start time in ET."""
    req = StockBarsRequest(
        symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start, end=end,
        feed=feed, adjustment=Adjustment.ALL,
    )
    for attempt in range(5):
        try:
            raw = client().get_stock_bars(req).df
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    if raw.empty:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz=ET))
    df = raw.xs(symbol, level="symbol")[COLS]
    df.index = df.index.tz_convert(ET)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return regular_hours(df).astype("float64")


def bars_path(symbol: str):
    return config.BARS_DIR / f"{symbol}.parquet"


def load_bars(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(bars_path(symbol))


def download_history(symbol: str, years: int = config.HISTORY_YEARS) -> pd.DataFrame:
    """Download (or top up) full-market 1-minute history for one symbol."""
    # The free plan cannot query the most recent 15 minutes of SIP data
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    path = bars_path(symbol)
    old = load_bars(symbol) if path.exists() else None
    start = end - timedelta(days=365 * years)
    spans = [(start, end)]
    if old is not None and len(old):
        first, last = old.index[0].to_pydatetime(), old.index[-1].to_pydatetime()
        spans = [(last + timedelta(minutes=1), end)]            # top up to now
        if first - start > timedelta(days=7):
            spans.insert(0, (start, first))                     # backfill older history
    parts = [] if old is None else [old]
    for cursor, span_end in spans:
        while cursor < span_end:  # quarter-sized chunks keep each request manageable
            chunk_end = min(cursor + timedelta(days=90), span_end)
            parts.append(fetch_bars(symbol, cursor, chunk_end))
            cursor = chunk_end
    df = pd.concat([p for p in parts if len(p)])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_parquet(path)
    return df


if __name__ == "__main__":
    import sys

    symbols = sys.argv[1:] or config.CONTEXT_TICKERS + config.TRAIN_TICKERS
    for sym in symbols:
        t0 = time.time()
        d = download_history(sym)
        print(f"{sym}: {len(d):,} bars {d.index[0].date()} -> {d.index[-1].date()} ({time.time() - t0:.0f}s)", flush=True)
