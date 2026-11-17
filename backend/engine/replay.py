"""Replay a past session through the live engine, minute by minute.

Run:  python -m backend.engine.replay 2026-11-13 AAPL NVDA TSLA
      python -m backend.engine.replay 2026-11-13            (all training tickers)

Uses the downloaded bar history, writes signals to a separate replay database
(so the real track record stays clean) and prints each event. Add --live-db to
write into the app database instead, which is handy for trying out the site.
"""
import sys

import pandas as pd

from backend import config, db
from backend.data.alpaca_data import ET, load_bars
from backend.engine.core import Engine
from backend.engine.push import format_signal


def replay(date: str, symbols: list, live_db: bool = False, quiet: bool = False) -> list:
    if not live_db:
        path = config.DATA_DIR / "replay.db"
        path.unlink(missing_ok=True)
        db.connect(path)
    engine = Engine()
    day = pd.Timestamp(date, tz=ET)
    for sym in config.CONTEXT_TICKERS + symbols:
        bars = load_bars(sym)
        engine.add_bars(sym, bars[bars.index < day + pd.Timedelta(days=1)])
    events = []
    for minute in range(1, 391):
        now = day + pd.Timedelta(minutes=570 + minute)
        for event in engine.on_minute(now, symbols):
            events.append(event)
            if not quiet:
                note = format_signal(event)
                print(f"{now:%H:%M}  {note['title']}  |  {note['body']}")
    return events


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    events = replay(args[0], args[1:] or config.TRAIN_TICKERS, live_db="--live-db" in sys.argv)
    closed = [e["signal"] for e in events if e["type"] == "update" and e["signal"]["status"] in ("tp1_hit", "stopped", "expired")]
    new = sum(e["type"] == "signal" for e in events)
    total = sum(s["result_r"] for s in closed)
    print(f"\n{new} signals, {len(closed)} finished, total {total:+.2f}R")
