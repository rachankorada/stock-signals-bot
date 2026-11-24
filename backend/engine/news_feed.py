"""Always-on news scorer for the app: every minute, day and night, weekends and holidays included.

It follows the master list, every unique ticker on any user's watchlist (db.watched_symbols()).
New headlines for those tickers are read by FinBERT (or by keywords if FinBERT cannot load), stored
with their score, and sent live to every user watching one of the article's tickers. A ticker added
to the master list is caught up from the previous session's close.

A ticker's news score is worked out from the stored articles since the previous session's close, with
the same rule the trading bot uses (backend.bot.news_bot.combine).
"""
import asyncio
import logging
import time

import pandas as pd
from alpaca.data.requests import NewsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from backend import config, db
from backend.bot import news_bot
from backend.data.alpaca_data import ET
from backend.data.news import client as news_client

log = logging.getLogger("news")
POLL_S = 60              # how often to ask for new articles
OVERLAP = pd.Timedelta(minutes=10)   # re-ask a little of the last window: articles can arrive late
CHUNK = 50               # tickers per request
TOP_PER_TICKER = 5       # headlines returned per ticker with the scores


def fetch(symbols: list, start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Every article tagged with any of the symbols in the window, as plain dicts."""
    out = []
    for i in range(0, len(symbols), CHUNK):
        req = NewsRequest(symbols=",".join(symbols[i:i + CHUNK]), start=start.tz_convert("UTC").to_pydatetime(),
                          end=end.tz_convert("UTC").to_pydatetime(), include_content=False, exclude_contentless=False)
        for attempt in range(3):
            try:
                news = news_client().get_news(req)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                log.warning("news request failed, retrying: %s", str(exc)[:120])
                time.sleep(3 * (attempt + 1))
        for a in news.data.get("news", []):
            out.append({"id": int(a.id), "created_at": pd.Timestamp(a.created_at).tz_convert("UTC").isoformat(),
                        "headline": a.headline or "", "source": a.source, "url": a.url, "symbols": list(a.symbols or [])})
    return list({a["id"]: a for a in out}.values())


def store(articles: list) -> list:
    """Score and save articles not seen before; returns the new ones with their score."""
    if not articles:
        return []
    ids = [a["id"] for a in articles]
    known = {r["id"] for r in db.query(f"SELECT id FROM news WHERE id IN ({','.join('?' * len(ids))})", tuple(ids))}
    new = [a for a in articles if a["id"] not in known and a["headline"].strip()]
    if not new:
        return []
    scores = news_bot.read_articles([a["headline"] for a in new], [a["created_at"] for a in new])
    how = news_bot.method()
    for a, s in zip(new, scores):
        a["score"] = round(float(s), 4)
        db.execute("INSERT OR IGNORE INTO news (id, created_at, headline, source, url, n_symbols, score, scored_by) "
                   "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                   (a["id"], a["created_at"], a["headline"], a["source"], a["url"], len(a["symbols"]), a["score"], how))
        for sym in a["symbols"]:
            db.execute("INSERT OR IGNORE INTO news_symbols (news_id, symbol) VALUES (?, ?)", (a["id"], sym))
    return new


def ticker_scores(symbols: list, since: pd.Timestamp) -> list:
    """Each ticker's news score since `since`, with its articles count and strongest recent headlines."""
    out = []
    for sym in symbols:
        rows = db.query("SELECT n.* FROM news n JOIN news_symbols s ON s.news_id = n.id "
                        "WHERE s.symbol = ? AND n.created_at >= ? ORDER BY n.created_at DESC",
                        (sym, since.tz_convert("UTC").isoformat()))
        focused = [r for r in rows if r["n_symbols"] <= 2]
        score = news_bot.combine([r["score"] for r in focused], [r["n_symbols"] for r in focused])
        top = sorted(focused, key=lambda r: -abs(r["score"]))[:TOP_PER_TICKER]
        out.append({"symbol": sym, "score": round(score, 3), "strength": news_bot.strength(score),
                    "direction": "positive" if score > 0 else "negative" if score < 0 else "neutral",
                    "articles": len(focused), "all_articles": len(rows),
                    "headlines": [{"id": r["id"], "time": r["created_at"], "headline": r["headline"],
                                   "score": r["score"], "url": r["url"]} for r in top]})
    return out


class NewsFeed:
    def __init__(self, dispatch=None):
        self.dispatch = dispatch     # async callable(event) for live updates
        self.trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
        self.last_poll = None
        self.known = set()           # tickers already caught up
        self._since = (None, None)   # (date, previous session close)
        self.error = None
        self.current = {}            # ticker -> news score now, read by the signal engine

    def since(self) -> pd.Timestamp:
        """The previous session's close: the start of the window a news score covers."""
        today = pd.Timestamp.now(tz=ET).normalize()
        if self._since[0] != today:
            try:
                cal = self.trading.get_calendar(GetCalendarRequest(start=(today - pd.Timedelta(days=10)).date(),
                                                                   end=today.date()))
                now = pd.Timestamp.now(tz=ET)
                closes = [pd.Timestamp(c.close, tz=ET) for c in cal if pd.Timestamp(c.close, tz=ET) <= now]
                self._since = (today, closes[-1])
            except Exception as exc:
                log.warning("calendar unavailable, using the last 24 hours: %s", str(exc)[:120])
                return pd.Timestamp.now(tz=ET) - pd.Timedelta(hours=24)
        return self._since[1]

    def poll(self) -> list:
        """One round: catch up new tickers, then fetch what is new for all of them."""
        symbols = db.watched_symbols()
        if not symbols:
            return []
        now = pd.Timestamp.now(tz=ET)
        fresh = []
        new_syms = [s for s in symbols if s not in self.known]
        if new_syms:
            fresh += store(fetch(new_syms, self.since() - pd.Timedelta(hours=1), now))
            self.known |= set(new_syms)
        start = (self.last_poll - OVERLAP) if self.last_poll is not None else self.since()
        fresh += store(fetch(symbols, start, now))
        self.last_poll = now
        self.current = {t["symbol"]: t["score"] for t in ticker_scores(symbols, self.since())}
        self.known &= set(symbols)   # a ticker dropped by everyone is caught up again if it comes back
        return fresh

    async def run(self):
        loop = asyncio.get_running_loop()
        log.info("news scorer running: every %ds, around the clock", POLL_S)
        while True:
            try:
                fresh = await loop.run_in_executor(None, self.poll)
                if fresh:
                    log.info("%d new articles (%s)", len(fresh), news_bot.method())
                    if self.dispatch:
                        await self.announce(fresh)
                self.error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = str(exc)
                log.exception("news poll failed")
            await asyncio.sleep(POLL_S)

    async def announce(self, fresh: list):
        since = self.since()
        touched = sorted({s for a in fresh for s in a["symbols"]} & set(db.watched_symbols()))
        scores = {t["symbol"]: t for t in ticker_scores(touched, since)}
        for a in fresh:
            for sym in a["symbols"]:
                if sym in scores:
                    t = scores[sym]
                    await self.dispatch({"type": "news", "symbol": sym, "article": {
                        "id": a["id"], "time": a["created_at"], "headline": a["headline"], "score": a["score"],
                        "url": a["url"]}, "ticker": {k: t[k] for k in ("score", "strength", "direction", "articles")}})
