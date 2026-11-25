"""Follow every signal the site posts, the way a subscriber would (Alpaca PAPER account only).

  signal posted        buy (or short) straight away with a limit order no more than a quarter of the
                       stop distance past the signal price (the backtest skips a signal whose price has
                       run further than that), with the signal's stop and first target attached as real
                       orders; an entry not filled within a minute is cancelled
  after a stop-out     no new signal on that ticker for COOLDOWN_MIN minutes
  target or stop hit   the attached order closes the trade
  signal ends another  if the site says the signal is over (expired, stopped, runner finished) and the
  way                  trade is still open, the bot cancels what is left and exits at market

Each entry order's id carries the day, the signal number, the timeframe and the
stop distance, so results can be rebuilt from the account alone.

Run:  python -m backend.bot.signal_bot report
"""
import logging
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetCalendarRequest, GetOrderByIdRequest, GetOrdersRequest, LimitOrderRequest,
                                     MarketOrderRequest, StopLossRequest, TakeProfitRequest)

from backend import config
from backend.bot import news_bot
from backend.data.alpaca_data import ET

PREFIX = "sig"
# Intended loss if a trade is stopped out, as a share of account equity (1% = $100 on a $10,000 account)
RISK_FRAC = float(os.environ.get("BOT_RISK_FRAC", "0.01"))
# Cap on one trade's size, as a share of account equity. Very tight stops would otherwise ask for huge
# positions. At 10%, about ten trades can be open together alongside the stocks-in-play orders.
MAX_POSITION_FRAC = float(os.environ.get("BOT_MAX_POSITION_FRAC", "0.10"))
FIRST_DAY = "2026-11-20"
ENTRY_CHASE_R = 0.25      # furthest past the signal price an entry may fill, in stop distances (as in the backtest)
ENTRY_WAIT_S = 60         # an entry still unfilled after this long is cancelled
COOLDOWN_MIN = 15         # after a stop-out, no new signal on that ticker for this many minutes
OPEN = ("new", "accepted", "held", "partially_filled", "pending_new", "accepted_for_bidding")

log = logging.getLogger("signals")


def trading_client() -> TradingClient:
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)  # paper account only, always


def id_stem(sig: dict) -> str:
    # The signal's time leads its number, so a restarted run (whose numbers start again at 1) cannot reuse an id
    at = pd.Timestamp(sig["created_at"]).tz_convert(ET)
    return f"{PREFIX}-{at:%Y%m%d}-{at:%H%M}{sig['id']:04d}-{sig['timeframe']}-"


def order_id(sig: dict, news_code: str = None) -> str:
    """sig-<day>-<time+number>-<timeframe>-[<news agreement>-]<risk in 1/10000ths>"""
    return id_stem(sig) + (f"{news_code}-" if news_code else "") + str(int(round(sig["risk"] * 10000)))


def parse_id(cid: str) -> dict:
    parts = cid.split("-")
    return {"signal": int(parts[2]), "timeframe": int(parts[3]), "risk": int(parts[-1]) / 10000,
            "news": news_bot.decode(parts[4]) if len(parts) == 6 else None}


def day_orders(trading: TradingClient, today: pd.Timestamp, status=QueryOrderStatus.ALL, symbol=None) -> list:
    """Every order submitted on one day (attached orders nested inside their entry), paging past the 500 limit."""
    start = today.tz_localize(ET)
    end, cursor, out, seen = start + timedelta(days=1), start, [], set()
    while True:
        page = trading.get_orders(GetOrdersRequest(status=status, after=cursor.to_pydatetime(), until=end.to_pydatetime(),
                                                   nested=True, limit=500, direction="asc",
                                                   symbols=[symbol] if symbol else None))
        fresh = [o for o in page if o.id not in seen]
        out += fresh
        seen |= {o.id for o in fresh}
        if len(page) < 500 or not fresh:
            return out
        cursor = pd.Timestamp(page[-1].submitted_at)


def shares(risk_per_share: float, price: float, equity: float, cap_frac: float, one_share_up_to: float) -> int:
    """Shares to trade: the planned risk divided by the stop distance, held to the position cap. When that
    rounds to zero (an expensive stock on a small account), one share is traded if it costs no more than
    `one_share_up_to` of equity."""
    qty = int(min(RISK_FRAC * equity / risk_per_share, cap_frac * equity / price))
    return 1 if qty < 1 and price <= one_share_up_to * equity else qty


def place(trading: TradingClient, sig: dict, equity: float, news: pd.DataFrame = None):
    """Enter a new signal at market with its stop and first target attached. The ticker's news score
    (news_bot) scales the size and adjusts the confidence shown."""
    sym, d = sig["symbol"], sig["direction"]
    if sig["timeframe"] not in config.BOT_TIMEFRAMES:
        return
    adj = news_bot.adjust(sig, news)
    base = shares(sig["risk"], sig["entry"], equity, MAX_POSITION_FRAC, one_share_up_to=0.25)
    qty = int(base * adj["size"]) if base >= 1 else base
    qty = min(qty, int(MAX_POSITION_FRAC * 2 * equity / sig["entry"])) if base >= 1 else qty   # 2x size: at most 20% of the account
    if base >= 1 and qty < 1:
        qty = 1
    if qty < 1:
        log.info("signal %s %s: size rounds to zero, skipped", sig["id"], sym)
        return
    # Leave a ticker alone while the stocks-in-play strategy is in it (nested, so this bot's own stop and
    # target orders sit inside their entry order and are not mistaken for the other strategy's)
    others = [o for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[sym], limit=100,
                                                             nested=True))
              if (o.client_order_id or "").startswith("inplay")]
    if others:
        log.info("signal %s %s: ticker in use by the other strategy, skipped", sig["id"], sym)
        return
    stopped_at = last_stop_out(trading, sym)
    if stopped_at is not None and pd.Timestamp.now(tz=ET) - stopped_at < pd.Timedelta(minutes=COOLDOWN_MIN):
        log.info("signal %s %s: stopped out at %s, cooling down, skipped", sig["id"], sym, f"{stopped_at:%H:%M}")
        return
    try:
        trading.submit_order(LimitOrderRequest(
            # Good till cancelled only as a safety net: every trade is closed before the bell (end_of_day),
            # and if that ever fails the stop and target keep protecting it overnight
            symbol=sym, qty=qty, side=OrderSide.BUY if d > 0 else OrderSide.SELL, time_in_force=TimeInForce.GTC,
            limit_price=round(sig["entry"] + d * ENTRY_CHASE_R * sig["risk"], 2),
            order_class=OrderClass.BRACKET, client_order_id=order_id(sig, news_bot.code(adj["agreement"])),
            take_profit=TakeProfitRequest(limit_price=round(sig["tp1"], 2)),
            stop_loss=StopLossRequest(stop_price=round(sig["stop"], 2))))
        log.info("signal %s: %s %d %s (%dm, %.0f%% confidence, %.0f%% with news) entry ~%.2f stop %.2f target %.2f; "
                 "news %s %+.2f, %s the trade, size %.2fx", sig["id"],
                 "BUY" if d > 0 else "SHORT", qty, sym, sig["timeframe"], sig["confidence"] * 100,
                 adj["confidence"] * 100, sig["entry"], sig["stop"], sig["tp1"], adj["strength"], adj["score"],
                 "with" if adj["agreement"] > 0 else "against" if adj["agreement"] < 0 else "no bearing on",
                 adj["size"])
    except Exception as exc:
        log.warning("signal %s %s: order rejected: %s", sig["id"], sym, str(exc)[:200])


def last_stop_out(trading: TradingClient, sym: str):
    """When this bot's most recent stop-out on a ticker happened today (None if it has not)."""
    today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize()
    times = [pd.Timestamp(leg.filled_at).tz_convert(ET)
             for o in day_orders(trading, today, symbol=sym) if (o.client_order_id or "").startswith(PREFIX + "-")
             for leg in (o.legs or []) if leg.type.value == "stop" and leg.filled_at]
    return max(times) if times else None


def cancel_stale_entries(trading: TradingClient, today: pd.Timestamp):
    """Cancel entry orders that have waited longer than ENTRY_WAIT_S without a fill: the price has moved away
    from the signal, which the backtest would have skipped."""
    now = pd.Timestamp.now(tz="UTC")
    for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500)):
        if ((o.client_order_id or "").startswith(PREFIX + "-") and float(o.filled_qty or 0) == 0
                and now - pd.Timestamp(o.submitted_at) > pd.Timedelta(seconds=ENTRY_WAIT_S)):
            try:
                trading.cancel_order_by_id(o.id)
                log.info("%s: entry not filled within %ds, cancelled", o.symbol, ENTRY_WAIT_S)
            except Exception as exc:
                log.warning("%s: could not cancel a stale entry: %s", o.symbol, str(exc)[:200])


def close_out(trading: TradingClient, parent):
    """Make sure one signal's trade is finished: cancel what is left of it and exit whatever is still held.
    Returns True when it cancelled an entry or sent an exit, False when there was nothing left to do."""
    if parent.status.value in OPEN and float(parent.filled_qty or 0) == 0:
        trading.cancel_order_by_id(parent.id)
        return True
    if float(parent.filled_qty or 0) == 0:
        return False
    if parent.status.value in OPEN:   # partly filled: stop the rest of the entry first
        try:
            trading.cancel_order_by_id(parent.id)
        except Exception:
            pass
    live = [leg for leg in (parent.legs or []) if leg.status.value in OPEN]
    for leg in live:
        try:
            trading.cancel_order_by_id(leg.id)
        except Exception:
            pass
    fresh = parent
    for _ in range(10):  # the shares are only free to sell once the cancels have gone through
        fresh = trading.get_order_by_id(parent.id, filter=GetOrderByIdRequest(nested=True))
        if not any(leg.status.value in OPEN for leg in (fresh.legs or [])):
            break
        time.sleep(0.5)
    # A stop or target can fill only part of the way, so exit exactly what is still held
    remaining = float(fresh.filled_qty) - sum(float(leg.filled_qty or 0) for leg in (fresh.legs or []))
    exit_id = parent.client_order_id.replace(PREFIX + "-", PREFIX + "x-", 1)
    try:   # an exit already sent for this trade (ids must be unique, and its shares are already out)
        sent = trading.get_order_by_client_id(exit_id)
    except Exception:
        sent = None
    if sent is not None:
        if sent.status.value in OPEN:
            return False
        remaining -= float(sent.filled_qty or 0)
        exit_id += "b"
    if remaining < 1:
        return False
    trading.submit_order(MarketOrderRequest(
        symbol=parent.symbol, qty=remaining, time_in_force=TimeInForce.DAY,
        side=OrderSide.SELL if parent.side == OrderSide.BUY else OrderSide.BUY, client_order_id=exit_id))
    return True


def finish(trading: TradingClient, sig: dict):
    """The site says this signal is over."""
    stem = id_stem(sig)
    today = pd.Timestamp(sig["created_at"]).tz_convert(ET).tz_localize(None).normalize()
    for parent in day_orders(trading, today, symbol=sig["symbol"]):
        if (parent.client_order_id or "").startswith(stem):
            try:
                close_out(trading, parent)
                log.info("signal %s %s ended (%s)", sig["id"], sig["symbol"], sig["status"])
            except Exception as exc:
                log.warning("signal %s %s: could not close out: %s", sig["id"], sig["symbol"], str(exc)[:200])
            return


def handle(trading: TradingClient, event: dict, equity: float, news: pd.DataFrame = None):
    sig = event["signal"]
    if event["type"] == "signal":
        place(trading, sig, equity, news)
    elif sig.get("closed_at"):
        finish(trading, sig)


def is_open_trade(parent) -> bool:
    """An entry still waiting, or a filled entry whose shares are not all out yet."""
    if parent.status.value in OPEN:
        return True
    filled = float(parent.filled_qty or 0)
    out = sum(float(leg.filled_qty or 0) for leg in (parent.legs or []))
    return filled > 0 and (out < filled or any(leg.status.value in OPEN for leg in (parent.legs or [])))


def end_of_day(trading: TradingClient, today: pd.Timestamp, days_back: int = 0, today_too: bool = True) -> int:
    """Day trading: before the bell every signal trade is closed at market and every waiting entry is
    cancelled. With days_back > 0 it also closes trades left from earlier days (run at startup with
    today_too=False: a market exit sent before the open fills at the open). Returns how many it closed."""
    closed = 0
    for back in range(days_back, -1 if today_too else 0, -1):
        day = today - pd.Timedelta(days=back)
        for parent in day_orders(trading, day):
            if not (parent.client_order_id or "").startswith(PREFIX + "-") or not is_open_trade(parent):
                continue
            try:
                if not close_out(trading, parent):
                    continue
                closed += 1
                log.info("%s: signal trade from %s closed out (day trading: nothing is held overnight)",
                         parent.symbol, f"{day:%Y-%m-%d}")
            except Exception as exc:
                log.warning("%s: end-of-day close-out failed: %s", parent.symbol, str(exc)[:200])
    return closed


def exits_for(trading: TradingClient, cid: str, orders: list) -> list:
    """The market exits sent for one trade: found among the day's orders, or looked up by id when the
    trade was closed on a later day."""
    xid = cid.replace(PREFIX + "-", PREFIX + "x-", 1)
    found = [o for o in orders if (o.client_order_id or "") in (xid, xid + "b")]
    if found:
        return found
    for i in (xid, xid + "b"):
        try:
            found.append(trading.get_order_by_client_id(i))
        except Exception:
            pass
    return found


def day_trades(trading: TradingClient, today: pd.Timestamp, open_too: bool = False) -> pd.DataFrame:
    """One row per followed signal entered that day, from the account's actual fills. With open_too,
    trades not yet closed are included, valued at the current price (how = "open")."""
    orders = day_orders(trading, today)
    rows, prices = [], None
    for o in orders:
        cid = o.client_order_id or ""
        if not cid.startswith(PREFIX + "-") or float(o.filled_qty or 0) == 0:
            continue
        ids = parse_id(cid)
        d = 1 if o.side == OrderSide.BUY else -1
        xs = [x for x in exits_for(trading, cid, orders) if float(x.filled_qty or 0) > 0]
        closes = [x for x in list(o.legs or []) + xs if float(x.filled_qty or 0) > 0]
        closed_qty = sum(float(x.filled_qty) for x in closes)
        if abs(closed_qty - float(o.filled_qty)) > 1e-6:
            if not open_too:
                continue  # still open, or not fully closed
            if prices is None:
                prices = {p.symbol: float(p.current_price) for p in trading.get_all_positions()}
            if o.symbol not in prices:
                continue
            # value what is still held at the current price
            left = float(o.filled_qty) - closed_qty
            closes = closes + [type("Now", (), {"filled_qty": left, "filled_avg_price": prices[o.symbol]})()]
            closed_qty = float(o.filled_qty)
            xs = []
            still_open = True
        else:
            still_open = False
        entry, qty, risk = float(o.filled_avg_price), float(o.filled_qty), ids["risk"]
        exit_px = sum(float(x.filled_qty) * float(x.filled_avg_price) for x in closes) / closed_qty
        how = "open" if still_open else "market exit" if xs else (
            "target" if d * (exit_px - entry) > 0 else "stop")
        pnl = d * (exit_px - entry) * qty
        a = ids["news"]
        # sizes up to 2x were allowed before news became protective only
        size = (news_bot.size_for(a, 2.0 if today < pd.Timestamp("2026-11-27") else news_bot.SIZE_MAX)
                if a is not None else 1.0)
        rows.append({"day": f"{today:%Y-%m-%d}", "signal": ids["signal"], "symbol": o.symbol, "timeframe": ids["timeframe"],
                     "direction": d, "qty": qty, "entry_time": pd.Timestamp(o.filled_at).tz_convert(ET).strftime("%H:%M:%S"),
                     "entry": round(entry, 4), "exit": round(exit_px, 4), "how": how, "pnl": round(pnl, 2),
                     "r": round(pnl / (qty * risk), 3), "news": a, "size": size,
                     "pnl_without_news": round(pnl / size, 2)})
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame) -> str:
    return (f"{len(t)} trades, {np.mean(t['pnl'] > 0):.0%} winners, P&L ${t['pnl'].sum():,.2f}, "
            f"average {t['r'].mean():+.3f}R")


def report(trading: TradingClient = None, show_trades: bool = True):
    trading = trading or trading_client()
    today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize()
    days = [] if today < pd.Timestamp(FIRST_DAY) else [pd.Timestamp(d.date) for d in trading.get_calendar(
        GetCalendarRequest(start=pd.Timestamp(FIRST_DAY).date(), end=today.date()))]
    parts = [p for p in (day_trades(trading, d, open_too=True) for d in days) if len(p)]
    if not parts:
        print("Signal follower: no trades yet.")
        return
    allt = pd.concat(parts, ignore_index=True)
    held = allt[allt["how"] == "open"]
    if len(held):
        print("Signal follower, still open (valued at the current price, not in the totals below):")
        for r in held.itertuples():
            print(f"  {r.day} {r.symbol:6s} {'LONG ' if r.direction > 0 else 'SHORT'} {r.qty:>6.0f} shares  "
                  f"entry {r.entry:.2f}  now {r.exit:.2f}  P&L ${r.pnl:+,.2f}  ({r.r:+.2f}R)")
    t = allt[allt["how"] != "open"]
    if not len(t):
        print("Signal follower: no completed trades yet.")
        return
    if show_trades:
        print(t.to_string(index=False))
        print()
    print("Signal follower, by day:")
    for day, g in t.groupby("day"):
        print(f"  {day}: {summary(g)}")
    print("Signal follower, by timeframe:")
    for tf, g in t.groupby("timeframe"):
        print(f"  {tf:>2}-minute: {summary(g)}")
    print("Signal follower, by how the trade ended:")
    for how, g in t.groupby("how"):
        print(f"  {how}: {summary(g)}")
    print(f"Signal follower total: {summary(t)}")
    n = t[t["news"].notna()]
    if len(n):
        group = pd.cut(n["news"], [-1.01, -0.15, 0.15, 1.01], labels=["news against", "no clear news", "news with"])
        print("Signal follower, by the news at entry:")
        for g, x in n.groupby(group, observed=True):
            print(f"  {g}: {summary(x)}")
        print(f"News adjustments: P&L ${n['pnl'].sum():,.2f} with them, ${n['pnl_without_news'].sum():,.2f} "
              f"at normal size ({len(n)} trades)")


if __name__ == "__main__":
    report(show_trades="--brief" not in sys.argv)
