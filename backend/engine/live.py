"""Live market loop.

A few seconds after every minute closes, pull the newest 1-minute bars for all
watched tickers and hand them to the engine. Any signal it finds is dispatched
straight away.
"""
import asyncio
import logging
import time
from datetime import timedelta

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from backend import config, db
from backend.data.alpaca_data import COLS, ET, client, regular_hours
from backend.engine.core import Engine

log = logging.getLogger("live")
WARMUP_DAYS = 36     # calendar days, covers the 22 sessions the engine keeps
POLL_DELAY_S = 2     # wait after the minute closes before asking for its bar
POLL_RETRIES = 5


def fetch_many(symbols: list, start, end=None) -> dict:
    """Real-time (IEX) 1-minute bars for several symbols in one request."""
    out = {}
    for i in range(0, len(symbols), 50):
        chunk = symbols[i:i + 50]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Minute, start=start, end=end,
                               feed=DataFeed.IEX, adjustment=Adjustment.ALL)
        raw = client().get_stock_bars(req).df
        if raw.empty:
            continue
        for sym in raw.index.get_level_values("symbol").unique():
            df = raw.xs(sym, level="symbol")[COLS]
            df.index = df.index.tz_convert(ET)
            out[sym] = regular_hours(df[~df.index.duplicated(keep="last")].sort_index()).astype("float64")
    return out


class LiveRunner:
    def __init__(self, engine: Engine, dispatch):
        self.engine = engine
        self.dispatch = dispatch  # async callable(event)
        self.trading = TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)
        self.session = None       # (date, open, close) for today, or (date, None, None) when closed
        self.running = False
        self.error = None

    def load_session(self, today):
        if self.session and self.session[0] == today:
            return
        days = self.trading.get_calendar(GetCalendarRequest(start=today, end=today))
        if days and days[0].date == today:
            self.session = (today, pd.Timestamp(days[0].open, tz=ET), pd.Timestamp(days[0].close, tz=ET))
        else:
            self.session = (today, None, None)

    def status(self) -> dict:
        now = pd.Timestamp.now(tz=ET)
        is_open = bool(self.session and self.session[1] is not None and self.session[1] <= now < self.session[2])
        return {
            "is_open": is_open,
            "session_open": self.session[1].isoformat() if self.session and self.session[1] is not None else None,
            "session_close": self.session[2].isoformat() if self.session and self.session[2] is not None else None,
            "engine_running": self.running, "last_scan": self.engine.last_scan, "error": self.error,
            "symbols": len(db.watched_symbols()),
        }

    def refresh(self, symbols: list, boundary: pd.Timestamp):
        """Warm up new tickers, then pull the bar that just closed."""
        new = [s for s in symbols if s not in self.engine.bars]
        if new:
            for sym, df in fetch_many(new, boundary - timedelta(days=WARMUP_DAYS)).items():
                self.engine.add_bars(sym, df)
        want = boundary - timedelta(minutes=1)
        for attempt in range(POLL_RETRIES):
            fresh = fetch_many(symbols, boundary - timedelta(minutes=6))
            for sym, df in fresh.items():
                self.engine.add_bars(sym, df)
            spy = fresh.get(config.CONTEXT_TICKERS[0])
            if spy is not None and len(spy) and spy.index[-1] >= want:
                return
            time.sleep(1.5)

    async def run(self):
        loop = asyncio.get_running_loop()
        self.running = True
        while True:
            try:
                now = pd.Timestamp.now(tz=ET)
                boundary = now.floor("min") + pd.Timedelta(minutes=1)
                await asyncio.sleep((boundary - now).total_seconds() + POLL_DELAY_S)
                await loop.run_in_executor(None, self.load_session, boundary.date())
                _, opens, closes = self.session
                if opens is None or not (opens < boundary <= closes):
                    continue
                symbols = db.watched_symbols()
                if not symbols:
                    continue
                await loop.run_in_executor(None, self.refresh, config.CONTEXT_TICKERS + symbols, boundary)
                # On early-close days, stop looking for entries once the normal cut-off logic no longer holds
                early_close = closes.hour * 60 + closes.minute < 960
                scan_symbols = [] if early_close and boundary > closes - timedelta(minutes=60) else symbols
                events = await loop.run_in_executor(None, self.engine.on_minute, boundary, scan_symbols)
                for event in events:
                    await self.dispatch(event)
                self.error = None
            except asyncio.CancelledError:
                self.running = False
                raise
            except Exception as exc:  # keep the loop alive through API hiccups
                self.error = str(exc)
                log.exception("live loop error")
                await asyncio.sleep(5)
