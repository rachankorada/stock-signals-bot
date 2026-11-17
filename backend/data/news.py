"""News history from Alpaca's news feed (Benzinga), stored per ticker.

Run:  python -m backend.data.news            (all training tickers)
      python -m backend.data.news AAPL NVDA
"""
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.historical.news import NewsClient
from alpaca.data.requests import NewsRequest

from backend import config

NEWS_DIR = config.DATA_DIR / "news"
NEWS_DIR.mkdir(parents=True, exist_ok=True)
COLS = ["created_at", "headline", "summary", "source", "n_symbols"]

_client = None


def client() -> NewsClient:
    global _client
    if _client is None:
        _client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    return _client


def fetch_news(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Every article tagged with `symbol` between two times, oldest first."""
    req = NewsRequest(symbols=symbol, start=start, end=end, include_content=False, exclude_contentless=False)
    for attempt in range(5):
        try:
            raw = client().get_news(req).df
            break
        except KeyError:  # the SDK fails to build a frame when there are no articles
            return pd.DataFrame(columns=COLS)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(3 * (attempt + 1))
    if raw.empty:
        return pd.DataFrame(columns=COLS)
    raw["n_symbols"] = raw["symbols"].map(len)
    raw["created_at"] = pd.to_datetime(raw["created_at"], utc=True)
    return raw[COLS].sort_values("created_at")


def news_path(symbol: str):
    return NEWS_DIR / f"{symbol}.parquet"


def load_news(symbol: str, finbert: bool = False) -> pd.DataFrame:
    """Stored articles for a ticker. finbert=True adds the saved FinBERT score per article."""
    path = news_path(symbol)
    news = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=COLS)
    scores = NEWS_DIR / "_finbert_scores.parquet"
    if finbert and scores.exists():
        news = news.join(pd.read_parquet(scores)["finbert"])
    return news


def download_news(symbol: str, years: int = config.HISTORY_YEARS) -> pd.DataFrame:
    """Download (or top up) the news history for one ticker."""
    end = datetime.now(timezone.utc)
    old = load_news(symbol)
    start = end - timedelta(days=365 * years + 40)  # extra month so early features have a baseline
    if len(old):
        start = old["created_at"].max().to_pydatetime() + timedelta(seconds=1)
    parts = [old] if len(old) else []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=30), end)
        parts.append(fetch_news(symbol, cursor, chunk_end))
        cursor = chunk_end
    parts = [p for p in parts if len(p)]
    df = pd.concat(parts) if parts else pd.DataFrame(columns=COLS)
    df = df[~df.index.duplicated(keep="last")].sort_values("created_at")
    df.to_parquet(news_path(symbol))
    return df


if __name__ == "__main__":
    for sym in sys.argv[1:] or config.TRAIN_TICKERS:
        t0 = time.time()
        d = download_news(sym)
        span = f"{d['created_at'].min():%Y-%m-%d} -> {d['created_at'].max():%Y-%m-%d}" if len(d) else "none"
        print(f"{sym}: {len(d):,} articles {span} ({time.time() - t0:.0f}s)", flush=True)
