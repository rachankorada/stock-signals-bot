"""Follow every signal the site posts, the way a subscriber would (Alpaca PAPER account only).

  signal posted        buy (or short) at market straight away, with the signal's stop and first
                       target attached as real orders
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
from alpaca.trading.requests import GetCalendarRequest, GetOrderByIdRequest, GetOrdersRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest

from backend import config
from backend.data.alpaca_data import ET

PREFIX = "sig"
RISK_DOLLARS = float(os.environ.get("BOT_RISK_DOLLARS", "1000"))  # intended loss if a trade is stopped out
# Cap on one trade's size, as a share of account equity. Very tight stops would otherwise ask for huge
# positions. At 10%, about ten trades can be open together alongside the stocks-in-play orders.
MAX_POSITION_FRAC = float(os.environ.get("BOT_MAX_POSITION_FRAC", "0.10"))
FIRST_DAY = "2026-11-17"
OPEN = ("new", "accepted", "held", "partially_filled", "pending_new", "accepted_for_bidding")

log = logging.getLogger("signals")


def trading_client() -> TradingClient:
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)  # paper account only, always


def order_id(sig: dict) -> str:
    # The signal's time leads its number, so a restarted run (whose numbers start again at 1) cannot reuse an id
    at = pd.Timestamp(sig["created_at"]).tz_convert(ET)
    return f"{PREFIX}-{at:%Y%m%d}-{at:%H%M}{sig['id']:04d}-{sig['timeframe']}-{int(round(sig['risk'] * 10000))}"


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


def place(trading: TradingClient, sig: dict, equity: float):
    """Enter a new signal at market with its stop and first target attached."""
    sym, d = sig["symbol"], sig["direction"]
    qty = int(min(RISK_DOLLARS / sig["risk"], MAX_POSITION_FRAC * equity / sig["entry"]))
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
    try:
        trading.submit_order(MarketOrderRequest(
            # Good till cancelled: if neither level is hit by the close, the trade is held overnight with its
            # stop and target still working
            symbol=sym, qty=qty, side=OrderSide.BUY if d > 0 else OrderSide.SELL, time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET, client_order_id=order_id(sig),
            take_profit=TakeProfitRequest(limit_price=round(sig["tp1"], 2)),
            stop_loss=StopLossRequest(stop_price=round(sig["stop"], 2))))
        log.info("signal %s: %s %d %s (%dm, %.0f%% confidence) entry ~%.2f stop %.2f target %.2f", sig["id"],
                 "BUY" if d > 0 else "SHORT", qty, sym, sig["timeframe"], sig["confidence"] * 100,
                 sig["entry"], sig["stop"], sig["tp1"])
    except Exception as exc:
        log.warning("signal %s %s: order rejected: %s", sig["id"], sym, str(exc)[:200])


def close_out(trading: TradingClient, parent):
    """Make sure one signal's trade is finished: cancel what is left of it and exit whatever is still held."""
    if parent.status.value in OPEN and float(parent.filled_qty or 0) == 0:
        trading.cancel_order_by_id(parent.id)
        return
    if float(parent.filled_qty or 0) == 0:
        return
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
    if remaining < 1:
        return
    trading.submit_order(MarketOrderRequest(
        symbol=parent.symbol, qty=remaining, time_in_force=TimeInForce.DAY,
        side=OrderSide.SELL if parent.side == OrderSide.BUY else OrderSide.BUY,
        client_order_id=parent.client_order_id.replace(PREFIX + "-", PREFIX + "x-", 1)))


def finish(trading: TradingClient, sig: dict):
    """The site says this signal is over."""
    cid = order_id(sig)
    today = pd.Timestamp(sig["created_at"]).tz_convert(ET).tz_localize(None).normalize()
    for parent in day_orders(trading, today, symbol=sig["symbol"]):
        if parent.client_order_id == cid:
            try:
                close_out(trading, parent)
                log.info("signal %s %s ended (%s)", sig["id"], sig["symbol"], sig["status"])
            except Exception as exc:
                log.warning("signal %s %s: could not close out: %s", sig["id"], sig["symbol"], str(exc)[:200])
            return


def at_day_end(sig: dict) -> bool:
    """Did the site end this signal only because the trading day was ending?"""
    closed = pd.Timestamp(sig["closed_at"]).tz_convert(ET)
    return closed.hour * 60 + closed.minute >= 16 * 60 - config.FLATTEN_BEFORE_CLOSE_MIN


def handle(trading: TradingClient, event: dict, equity: float):
    sig = event["signal"]
    if event["type"] == "signal":
        place(trading, sig, equity)
    elif sig.get("closed_at"):
        if sig["status"] == "expired" and at_day_end(sig):
            log.info("signal %s %s: no level hit by the close, held overnight with its stop and target", sig["id"], sig["symbol"])
            return
        finish(trading, sig)


def end_of_day(trading: TradingClient, today: pd.Timestamp):
    """At the close: trades still waiting on their stop or target stay open overnight. Only a position
    left with nothing protecting it (for example after a partial fill) is closed."""
    held = 0
    for parent in day_orders(trading, today):
        if not (parent.client_order_id or "").startswith(PREFIX + "-"):
            continue
        try:
            if parent.status.value in OPEN and float(parent.filled_qty or 0) == 0:
                trading.cancel_order_by_id(parent.id)
            elif any(leg.status.value in OPEN for leg in (parent.legs or [])):
                held += 1
            else:
                close_out(trading, parent)  # exits any unprotected remainder
        except Exception as exc:
            log.warning("%s: end-of-day check failed: %s", parent.symbol, str(exc)[:200])
    log.info("%d signal trades held overnight with their stop and target working", held)


def day_trades(trading: TradingClient, today: pd.Timestamp) -> pd.DataFrame:
    """One row per followed signal, from the account's actual fills."""
    orders = day_orders(trading, today)
    exits = {o.client_order_id.replace(PREFIX + "x-", PREFIX + "-", 1): o for o in orders
             if (o.client_order_id or "").startswith(PREFIX + "x-")}
    rows = []
    for o in orders:
        cid = o.client_order_id or ""
        if not cid.startswith(PREFIX + "-") or float(o.filled_qty or 0) == 0:
            continue
        _, _, sig_id, tf, code = cid.split("-")
        d = 1 if o.side == OrderSide.BUY else -1
        closes = [x for x in list(o.legs or []) + ([exits[cid]] if cid in exits else []) if float(x.filled_qty or 0) > 0]
        closed_qty = sum(float(x.filled_qty) for x in closes)
        if abs(closed_qty - float(o.filled_qty)) > 1e-6:
            continue  # still open, or not fully closed
        entry, qty, risk = float(o.filled_avg_price), float(o.filled_qty), int(code) / 10000
        exit_px = sum(float(x.filled_qty) * float(x.filled_avg_price) for x in closes) / closed_qty
        how = "market exit" if cid in exits and float(exits[cid].filled_qty or 0) > 0 else (
            "target" if d * (exit_px - entry) > 0 else "stop")
        pnl = d * (exit_px - entry) * qty
        rows.append({"day": f"{today:%Y-%m-%d}", "signal": int(sig_id), "symbol": o.symbol, "timeframe": int(tf),
                     "direction": d, "qty": qty, "entry_time": pd.Timestamp(o.filled_at).tz_convert(ET).strftime("%H:%M:%S"),
                     "entry": round(entry, 4), "exit": round(exit_px, 4), "how": how, "pnl": round(pnl, 2),
                     "r": round(pnl / (qty * risk), 3)})
    return pd.DataFrame(rows)


def summary(t: pd.DataFrame) -> str:
    return (f"{len(t)} trades, {np.mean(t['pnl'] > 0):.0%} winners, P&L ${t['pnl'].sum():,.2f}, "
            f"average {t['r'].mean():+.3f}R")


def report(trading: TradingClient = None, show_trades: bool = True):
    trading = trading or trading_client()
    today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize()
    days = [] if today < pd.Timestamp(FIRST_DAY) else [pd.Timestamp(d.date) for d in trading.get_calendar(
        GetCalendarRequest(start=pd.Timestamp(FIRST_DAY).date(), end=today.date()))]
    parts = [p for p in (day_trades(trading, d) for d in days) if len(p)]
    if not parts:
        print("Signal follower: no completed trades yet.")
        return
    t = pd.concat(parts, ignore_index=True)
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


if __name__ == "__main__":
    report(show_trades="--brief" not in sys.argv)
