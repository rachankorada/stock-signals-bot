"""One process for the whole trading day: both paper-trading bots, minute by minute.

  signal follower   runs the site's live signal engine on the 25 tickers and trades every signal
                    it posts as it appears and ends (see signal_bot.py)
  stocks in play    09:35 scan and breakout orders, 09:50 cancel untriggered, trailing stops,
                    15:55 close out (see inplay_bot.py)

Stocks-in-play trades are closed five minutes before the bell; signal trades that have hit neither
level are held overnight with their stop and target working. The run can be cut in two
(`--until 12:30`, then a second run for the afternoon); open signals are kept
in a small database file that the second run picks up.

Run:  python -m backend.bot.run_day
      python -m backend.bot.run_day --until 12:30
      python -m backend.bot.run_day report
"""
import ctypes
import json
import logging
import sys
import time
from datetime import timedelta

import pandas as pd
from alpaca.trading.requests import GetCalendarRequest

from backend import config, db
from backend.bot import inplay_bot, signal_bot
from backend.data.alpaca_data import ET
from backend.engine.core import Engine, make_pool
from backend.engine.live import LiveRunner

DIR = inplay_bot.DIR
log = logging.getLogger("day")
DAILY_LOSS_LIMIT = 0.03   # no new signal trades once equity is this far below the previous close


def now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET)


def guarded(label: str, fn, *args):
    """One part failing must not stop the rest of the day."""
    try:
        return fn(*args)
    except Exception:
        log.exception("%s failed", label)


def run(until: str = None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(DIR / f"day_{now():%Y-%m-%d}.log")])
    for noisy in ("httpx", "urllib3", "alpaca"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if hasattr(ctypes, "windll"):   # keep a Windows PC from idling to sleep while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    trading = signal_bot.trading_client()
    today = now().tz_localize(None).normalize()
    cal = trading.get_calendar(GetCalendarRequest(start=today.date(), end=today.date()))
    if not cal or pd.Timestamp(cal[0].date) != today:
        log.info("market is closed today; nothing to do")
        return
    opens, closes = pd.Timestamp(cal[0].open, tz=ET), pd.Timestamp(cal[0].close, tz=ET)
    stop_at = today.tz_localize(ET) + pd.Timedelta(until + ":00") if until else None
    if now() >= closes:
        log.info("market already closed")
        return
    account = trading.get_account()
    equity = float(account.equity)
    floor = float(account.last_equity) * (1 - DAILY_LOSS_LIMIT)   # last_equity: equity at the previous close
    halted = False
    log.info("paper account equity $%.2f; session %s to %s ET", equity, f"{opens:%H:%M}", f"{closes:%H:%M}")

    # Stocks in play: the baseline must be ready before 09:35
    scan_at = opens + timedelta(minutes=5)
    prep = None
    if now() < scan_at - timedelta(minutes=1) and not inplay_bot.plan_from_orders(trading, today):
        prep = guarded("stocks-in-play baseline", inplay_bot.prepare, trading, today)

    # Signal follower: the site's own engine, with its open signals kept in a file
    db.connect(DIR / f"signals_{today:%Y-%m-%d}.db")
    engine = Engine(make_pool())
    live = LiveRunner(engine, None)
    symbols = config.BOT_TICKERS
    log.info("signal engine ready: %d tickers, models for %s", len(symbols), sorted(engine.models))

    done = {"scan": prep is None, "window": False, "cancel": False, "flatten": False}
    while True:
        boundary = now().floor("min") + pd.Timedelta(minutes=1)
        time.sleep(max((boundary - now()).total_seconds() + 2, 0))
        if boundary <= opens:
            continue

        # 09:35: stocks in play goes first, its orders are the time-critical ones
        if not done["scan"] and boundary >= scan_at:
            done["scan"] = True
            if boundary <= scan_at + timedelta(minutes=inplay_bot.LATEST_START_MIN):
                picks = guarded("stocks-in-play scan", inplay_bot.pick, prep, today)
                if picks is not None:
                    plan = guarded("stocks-in-play orders", inplay_bot.place_orders, trading, picks, today)
                    if plan:
                        (DIR / f"plan_{today:%Y-%m-%d}.json").write_text(json.dumps(plan, indent=2))
            else:
                log.info("too late for today's stocks-in-play entries")

        # Every minute: update the signal engine and act on what it says
        try:
            live.refresh(config.CONTEXT_TICKERS + symbols, boundary)
            if not halted:
                now_equity = guarded("equity check", lambda: float(trading.get_account().equity))
                if now_equity is not None and now_equity <= floor:
                    halted = True
                    log.warning("equity $%.2f is %.0f%% below the previous close: no new signal trades today "
                                "(open ones keep their stops and targets)", now_equity, DAILY_LOSS_LIMIT * 100)
            for event in engine.on_minute(boundary, symbols):
                if halted and event["type"] == "signal":
                    continue
                guarded("signal order", signal_bot.handle, trading, event, equity)
        except Exception:
            log.exception("signal engine error at %s", f"{boundary:%H:%M}")

        if not done["window"] and boundary >= scan_at + timedelta(minutes=inplay_bot.ENTRY_WINDOW_MIN):
            done["window"] = True
            guarded("stocks-in-play entry window", inplay_bot.cancel_untriggered, trading, today)
        if done["scan"] and not done["flatten"]:
            guarded("stocks-in-play stop update", inplay_bot.move_stops, trading, today)
        guarded("stale signal entries", signal_bot.cancel_stale_entries, trading, today)

        if not done["cancel"] and boundary >= closes - timedelta(minutes=30):
            done["cancel"] = True
            guarded("stocks-in-play cancel", inplay_bot.cancel_untriggered, trading, today)
        if not done["flatten"] and boundary >= closes - timedelta(minutes=5):
            done["flatten"] = True
            guarded("stocks-in-play close-out", inplay_bot.flatten, trading, inplay_bot.plan_from_orders(trading, today), today)
            guarded("signal end of day", signal_bot.end_of_day, trading, today)
        if boundary >= closes:
            break
        if stop_at is not None and boundary >= stop_at:
            log.info("handing over to the afternoon run; open trades keep their stops and targets")
            return

    time.sleep(45)
    guarded("signal end of day", signal_bot.end_of_day, trading, today)
    guarded("stocks-in-play results", inplay_bot.record, trading, inplay_bot.plan_from_orders(trading, today), today)
    posted = db.query("SELECT COUNT(*) AS n, AVG(result_r) AS r FROM signals WHERE result_r IS NOT NULL")[0]
    log.info("site's own scoring of today's signals: %s finished, average %sR", posted["n"],
             None if posted["r"] is None else round(posted["r"], 3))
    report()


def slippage(trading) -> pd.DataFrame:
    """How far real fills landed from the planned prices, every day since the bots started, in R
    (stop distances) and basis points. Positive = worse than planned."""
    first = pd.Timestamp(signal_bot.FIRST_DAY)
    today = now().tz_localize(None).normalize()
    days = [pd.Timestamp(d.date) for d in trading.get_calendar(GetCalendarRequest(start=first.date(), end=today.date()))]
    rows = []
    for day in days:
        for o in signal_bot.day_orders(trading, day):
            cid = o.client_order_id or ""
            bot = "signals" if cid.startswith("sig-") else "stocks in play" if cid.startswith("inplay-") else None
            if bot is None or not o.filled_avg_price:
                continue
            d = 1 if o.side.value == "buy" else -1
            risk = int(cid.split("-")[-1].rstrip("m")) / 10000
            fill = float(o.filled_avg_price)
            stop_leg = next((l for l in o.legs or [] if l.type.value == "stop"), None)
            if bot == "signals" and stop_leg is not None:
                planned = float(stop_leg.stop_price) + d * risk     # the signal's own entry price
            elif bot == "stocks in play":
                planned = float(o.stop_price) if o.stop_price else None   # market entries have no planned price
            else:
                planned = None
            if planned:
                rows.append({"bot": bot, "what": "entry", "R": d * (fill - planned) / risk, "bps": d * (fill - planned) / planned * 1e4})
            for leg in o.legs or []:
                if leg.type.value == "stop" and leg.filled_avg_price:
                    sp, fp = float(leg.stop_price), float(leg.filled_avg_price)
                    rows.append({"bot": bot, "what": "stop exit", "R": d * (sp - fp) / risk, "bps": d * (sp - fp) / sp * 1e4})
    return pd.DataFrame(rows)


def report():
    print("=" * 30, "STOCKS IN PLAY", "=" * 30)
    guarded("stocks-in-play report", inplay_bot.report)
    print()
    print("=" * 30, "SIGNAL FOLLOWER", "=" * 30)
    guarded("signal report", signal_bot.report, None, False)
    trading = signal_bot.trading_client()
    positions = trading.get_all_positions()
    print("\nOpen right now:" if positions else "\nOpen right now: nothing")
    for p in positions:
        print(f"  {p.symbol:6s} {'LONG ' if float(p.qty) > 0 else 'SHORT'} {abs(float(p.qty)):>6.0f} shares  "
              f"entry {float(p.avg_entry_price):.2f}  now {float(p.current_price):.2f}  P&L ${float(p.unrealized_pl):+,.2f}")
    print(f"Account equity: ${float(trading.get_account().equity):,.2f}")
    s = guarded("slippage", slippage, trading)
    if s is not None and len(s):
        print("\nSlippage, real fill vs planned price (positive = worse), every day so far:")
        t = s.groupby(["bot", "what"]).agg(fills=("R", "size"), median_R=("R", "median"), median_bps=("bps", "median"),
                                           mean_bps=("bps", "mean"))
        print(t.round(2).to_string())


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["report"]:
        logging.basicConfig(level=logging.WARNING)
        report()
    else:
        run(args[args.index("--until") + 1] if "--until" in args else None)
