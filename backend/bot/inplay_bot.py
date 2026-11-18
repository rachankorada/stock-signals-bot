"""Paper-trading bot for the stocks-in-play strategy (Alpaca PAPER account only).

Each trading day:
  before the open  build the baseline for every stock (daily range, usual opening volume)
  09:35 ET         rank stocks by opening volume against their own recent opens, keep the top 40,
                   and place a breakout entry order with an attached stop and take-profit for each
  15:30 ET         cancel entries that never triggered
  15:55 ET         close every position the bot opened
  after the close  write the day's results

Differences from the backtest, forced by the free data plan: the morning scan
uses the real-time IEX feed (a small slice of market volume) rather than the
full tape, so the stocks chosen and the breakout levels will not match exactly.

Run:  python -m backend.bot.inplay_bot              (the whole day in one process)
      python -m backend.bot.inplay_bot morning      (baseline, scan and orders, then exit)
      python -m backend.bot.inplay_bot afternoon    (cancel, close out and record, then exit)
      python -m backend.bot.inplay_bot check        (connect and build the baseline; places nothing)
      python -m backend.bot.inplay_bot report       (results so far, read from the account's fills)

The morning and afternoon halves share nothing but the paper account itself:
each entry order's id carries the day, the ticker and the stop distance, so the
afternoon half (or a restart) rebuilds the day's plan from the orders.
"""
import ctypes
import json
import logging
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (GetCalendarRequest, GetOrdersRequest, MarketOrderRequest, ReplaceOrderRequest,
                                     StopLossRequest, StopOrderRequest, TakeProfitRequest)

from backend import config
from backend.data.alpaca_data import ET, client
from backend.research.in_play import CHUNK, LOOKBACK, MIN_ATR, MIN_AVG_VOLUME, MIN_PRICE
from backend.research.universe import candidates

DIR = config.DATA_DIR / "bot"
DIR.mkdir(parents=True, exist_ok=True)
STOP_FRAC = 0.10            # stop distance as a share of the 14-day average daily range
TAKE_PROFIT_R = 3           # take-profit, in stop distances from the fill (tick replay: steadier than holding to the close)
from backend.bot.signal_bot import shares  # noqa: E402  (one sizing rule for both bots)
TOP_N = 40                  # candidates traded per day (the backtest used 20)
# 40 entry orders sit in the account at once and Alpaca sets buying power aside for each, so these stay
# small (40 x 2.5% = the account's equity) and leave the other half of buying power to the signal follower
MAX_POSITION_FRAC = 0.025
MIN_USUAL_OPEN_VOLUME = 300  # IEX shares; below this the relative-volume ratio is noise
LATEST_START_MIN = 8        # minutes after 09:35 beyond which the day's entries are skipped
PREFIX = "inplay"
FIRST_DAY = "2026-11-17"     # the report starts here
FIVE_MIN = TimeFrame(5, TimeFrameUnit.Minute)

log = logging.getLogger("bot")


def trading_client() -> TradingClient:
    return TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True)  # paper account only, always


def now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ET)


def sleep_until(when: pd.Timestamp):
    while (left := (when - now()).total_seconds()) > 0:
        time.sleep(min(left, 30))


def fetch(symbols, start, end, tf, feed) -> pd.DataFrame:
    """Bars for many symbols; an empty frame if there are none."""
    parts = []
    for i in range(0, len(symbols), CHUNK):
        req = StockBarsRequest(symbol_or_symbols=symbols[i:i + CHUNK], timeframe=tf, start=start, end=end, feed=feed,
                               adjustment=Adjustment.ALL)
        for attempt in range(6):
            try:
                df = client().get_stock_bars(req).df
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(3 * (attempt + 1))
        if len(df):
            parts.append(df.reset_index())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def opening_bars(symbols, day: pd.Timestamp, feed) -> pd.DataFrame:
    """The 09:30-09:35 bar of each symbol on one day."""
    start = day.tz_localize(ET) + timedelta(hours=9, minutes=30)
    df = fetch(symbols, start, start + timedelta(minutes=4), FIVE_MIN, feed)
    if df.empty:
        return df
    return df[df["timestamp"].dt.tz_convert(ET) == start].set_index("symbol")


def prepare(trading: TradingClient, today: pd.Timestamp) -> pd.DataFrame:
    """Per-stock baseline known before the open: daily range, average volume, usual opening volume."""
    path = DIR / f"prep_{today:%Y-%m-%d}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    symbols = candidates()["symbol"].tolist()
    log.info("preparing %d stocks", len(symbols))
    daily = fetch(symbols, today - timedelta(days=45), today.tz_localize(ET), TimeFrame.Day, DataFeed.SIP)
    daily = daily[daily["timestamp"].dt.tz_convert(ET).dt.tz_localize(None).dt.normalize() < today].sort_values(["symbol", "timestamp"])
    prev = daily.groupby("symbol")["close"].shift(1)
    daily["tr"] = np.maximum(daily["high"] - daily["low"], np.maximum((daily["high"] - prev).abs(), (daily["low"] - prev).abs()))
    last = daily.groupby("symbol").tail(LOOKBACK)
    prep = last.groupby("symbol").agg(atr=("tr", "mean"), avg_volume=("volume", "mean"), days=("tr", "count"),
                                      prev_close=("close", "last"))
    prep = prep[prep["days"] == LOOKBACK]

    sessions = [pd.Timestamp(d.date) for d in trading.get_calendar(
        GetCalendarRequest(start=(today - timedelta(days=30)).date(), end=(today - timedelta(days=1)).date()))][-LOOKBACK:]
    total = pd.Series(0.0, index=prep.index)
    for day in sessions:
        bars = opening_bars(prep.index.tolist(), day, DataFeed.IEX)
        if len(bars):
            total = total.add(bars["volume"].reindex(prep.index).fillna(0.0), fill_value=0.0)
    prep["usual_open_volume"] = total / len(sessions)
    prep.to_parquet(path)
    log.info("baseline ready for %d stocks from %d sessions", len(prep), len(sessions))
    return prep


def pick(prep: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Today's candidates from the opening five minutes."""
    for attempt in range(20):   # the bar appears a few seconds after 09:35
        if len(opening_bars(["SPY", "AAPL"], today, DataFeed.IEX)):
            break
        time.sleep(2)
    bars = opening_bars(prep.index.tolist(), today, DataFeed.IEX)
    df = prep.join(bars[["open", "high", "low", "close", "volume"]], how="inner")
    df["rel_volume"] = df["volume"] / df["usual_open_volume"]
    df = df[(df["open"] >= MIN_PRICE) & (df["avg_volume"] >= MIN_AVG_VOLUME) & (df["atr"] >= MIN_ATR)
            & (df["usual_open_volume"] >= MIN_USUAL_OPEN_VOLUME) & (df["rel_volume"] >= 1.0) & (df["close"] != df["open"])]
    return df.sort_values("rel_volume", ascending=False).head(TOP_N)


def place_orders(trading: TradingClient, picks: pd.DataFrame, today: pd.Timestamp) -> dict:
    equity = float(trading.get_account().equity)
    held = {p.symbol for p in trading.get_all_positions()}
    held |= {o.symbol for o in trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=500))}
    plan = {}
    for sym, p in picks.iterrows():
        d = 1 if p["close"] > p["open"] else -1
        level = float(p["high"] if d > 0 else p["low"])
        risk = max(STOP_FRAC * float(p["atr"]), 0.02)
        qty = shares(risk, level, equity, MAX_POSITION_FRAC, one_share_up_to=0.05)
        row = {"direction": d, "level": round(level, 2), "risk": round(risk, 4), "qty": qty, "atr": round(float(p["atr"]), 4),
               "rel_volume": round(float(p["rel_volume"]), 2), "open_volume_iex": float(p["volume"]), "status": "skipped"}
        plan[sym] = row
        if qty < 1 or sym in held:
            continue
        side = OrderSide.BUY if d > 0 else OrderSide.SELL
        entry = round(level + 0.01 * d, 2)
        common = dict(symbol=sym, qty=qty, side=side, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
                      stop_loss=StopLossRequest(stop_price=round(entry - d * risk, 2)),
                      take_profit=TakeProfitRequest(limit_price=round(entry + d * TAKE_PROFIT_R * risk, 2)))
        # The id carries the stop distance (in hundredths of a cent) so the plan can be rebuilt from the account
        order_id = f"{PREFIX}-{today:%Y%m%d}-{sym}-{int(round(risk * 10000))}"
        try:
            trading.submit_order(StopOrderRequest(stop_price=entry, client_order_id=order_id, **common))
            row["status"] = "entry order placed"
        except Exception as first:
            # Usually: price is already through the level, so a stop order there is not allowed. Enter at market.
            try:
                trading.submit_order(MarketOrderRequest(client_order_id=order_id + "m", **common))
                row["status"] = "entered at market (already through the level)"
            except Exception as second:
                row["status"] = f"rejected: {second}"[:200]
                log.warning("%s rejected: %s / %s", sym, first, second)
        log.info("%-6s %s level %.2f stop distance %.2f qty %d rel volume %.1f -> %s",
                 sym, "LONG " if d > 0 else "SHORT", level, risk, qty, p["rel_volume"], row["status"])
    return plan


def day_orders(trading: TradingClient, status: QueryOrderStatus, today: pd.Timestamp) -> list:
    from backend.bot.signal_bot import day_orders as all_orders  # pages past the 500-order limit
    return all_orders(trading, today, status)


def our_orders(trading: TradingClient, status: QueryOrderStatus, today: pd.Timestamp) -> list:
    """The bot's entry orders for one day (their attached stops are nested inside)."""
    return [o for o in day_orders(trading, status, today)
            if (o.client_order_id or "").startswith(f"{PREFIX}-{today:%Y%m%d}-")]


def plan_from_orders(trading: TradingClient, today: pd.Timestamp) -> dict:
    """The day's plan, rebuilt from the entry orders in the account."""
    plan = {}
    for o in our_orders(trading, QueryOrderStatus.ALL, today):
        code = o.client_order_id.split("-")[3].rstrip("m")
        d = 1 if o.side == OrderSide.BUY else -1
        plan[o.symbol] = {"direction": d, "risk": int(code) / 10000, "qty": float(o.qty),
                          "level": round(float(o.stop_price) - 0.01 * d, 2) if o.stop_price else None}
    return plan


def cancel_untriggered(trading: TradingClient, today: pd.Timestamp):
    for o in our_orders(trading, QueryOrderStatus.OPEN, today):
        if float(o.filled_qty or 0) == 0:
            try:
                trading.cancel_order_by_id(o.id)
            except Exception as exc:
                log.warning("could not cancel %s: %s", o.symbol, exc)
    log.info("cancelled entries that never triggered")


EXIT_PREFIX = "ipx"   # this strategy's own market exits
LIVE_LEG = ("new", "accepted", "held", "pending_new", "pending_replace", "replaced")


def latest_price(sym: str) -> float:
    from alpaca.data.requests import StockLatestTradeRequest
    return float(client().get_stock_latest_trade(StockLatestTradeRequest(symbol_or_symbols=sym, feed=DataFeed.IEX))[sym].price)


def move_stops(trading: TradingClient, today: pd.Timestamp):
    """Once an entry has filled, put its stop exactly one stop distance from the fill price and its
    take-profit TAKE_PROFIT_R stop distances from it, as in the backtest. Both are first placed from the
    breakout level, so a fill beyond the level (a market entry, or slippage on the stop entry) would
    otherwise start the trade with more than 1R at risk. Stops are only ever tightened. If the price
    is already past the new stop, the trade is closed at market, which is what the backtest does."""
    for o in our_orders(trading, QueryOrderStatus.ALL, today):
        filled = float(o.filled_qty or 0)
        if filled < 1 or not o.filled_avg_price:
            continue
        d = 1 if o.side == OrderSide.BUY else -1
        risk = int(o.client_order_id.split("-")[3].rstrip("m")) / 10000
        target = round(float(o.filled_avg_price) - d * risk, 2)
        profit = round(float(o.filled_avg_price) + d * TAKE_PROFIT_R * risk, 2)
        for leg in o.legs or []:
            if leg.status.value in LIVE_LEG and leg.limit_price is not None and abs(float(leg.limit_price) - profit) >= 0.005:
                try:
                    trading.replace_order_by_id(leg.id, ReplaceOrderRequest(limit_price=profit))
                    log.info("%s take-profit moved %.2f -> %.2f", o.symbol, float(leg.limit_price), profit)
                except Exception as exc:
                    log.warning("%s take-profit not moved, will retry: %s", o.symbol, exc)
        for leg in o.legs or []:
            if leg.status.value not in LIVE_LEG or leg.stop_price is None:
                continue
            old = float(leg.stop_price)
            if d * (target - old) < 0.005:   # already there (or the new stop would be looser)
                continue
            try:
                trading.replace_order_by_id(leg.id, ReplaceOrderRequest(stop_price=target))
                log.info("%s stop moved %.2f -> %.2f (1R from the %.2f fill)", o.symbol, old, target, float(o.filled_avg_price))
                continue
            except Exception as exc:
                problem = exc
            try:
                price = latest_price(o.symbol)
            except Exception:
                price = float("nan")
            if not np.isfinite(price) or d * (price - target) > 0:   # unknown price, or not past the stop yet
                log.warning("%s stop not moved, will retry: %s", o.symbol, problem)
                continue
            # Already through the 1R stop: close the remaining shares now
            left = filled - sum(float(x.filled_qty or 0) for x in o.legs)
            for x in o.legs:   # both legs: a live take-profit would hold the shares the exit needs
                if x.status.value in LIVE_LEG:
                    try:
                        trading.cancel_order_by_id(x.id)
                    except Exception:
                        pass
            for attempt in range(5):
                time.sleep(1)
                try:
                    trading.submit_order(MarketOrderRequest(
                        symbol=o.symbol, qty=left, side=OrderSide.SELL if d > 0 else OrderSide.BUY,
                        time_in_force=TimeInForce.DAY, client_order_id=f"{EXIT_PREFIX}-{today:%Y%m%d}-{o.symbol}-s{attempt}"))
                    log.info("%s already past its 1R stop %.2f (last %.2f); closed at market", o.symbol, target, price)
                    break
                except Exception:
                    log.exception("%s market exit attempt %d failed", o.symbol, attempt + 1)


def held_by_us(trading: TradingClient, today: pd.Timestamp) -> dict:
    """Shares this strategy still holds per ticker (signed), from its own orders only, so a signal
    trade in the same ticker is never touched."""
    held = {}
    for o in day_orders(trading, QueryOrderStatus.ALL, today):
        cid = o.client_order_id or ""
        if cid.startswith(f"{PREFIX}-{today:%Y%m%d}-"):
            d = 1 if o.side == OrderSide.BUY else -1
            out = sum(float(leg.filled_qty or 0) for leg in (o.legs or []))
            held[o.symbol] = held.get(o.symbol, 0.0) + d * (float(o.filled_qty or 0) - out)
        elif cid.startswith(f"{EXIT_PREFIX}-{today:%Y%m%d}-"):
            d = 1 if o.side == OrderSide.BUY else -1
            held[o.symbol] = held.get(o.symbol, 0.0) + d * float(o.filled_qty or 0)
    return {k: v for k, v in held.items() if abs(v) >= 1}


def flatten(trading: TradingClient, plan: dict, today: pd.Timestamp):
    """Cancel this strategy's remaining orders, then close the shares it still holds."""
    for attempt in range(5):
        try:
            for o in our_orders(trading, QueryOrderStatus.OPEN, today):
                for order in [o] + list(o.legs or []):
                    if order.status.value in ("new", "accepted", "held", "partially_filled", "pending_new"):
                        try:
                            trading.cancel_order_by_id(order.id)
                        except Exception:
                            pass
            time.sleep(2)
            open_now = held_by_us(trading, today)
            for sym, qty in open_now.items():
                trading.submit_order(MarketOrderRequest(
                    symbol=sym, qty=abs(qty), side=OrderSide.SELL if qty > 0 else OrderSide.BUY,
                    time_in_force=TimeInForce.DAY, client_order_id=f"{EXIT_PREFIX}-{today:%Y%m%d}-{sym}-{attempt}"))
            if not open_now:
                log.info("flat")
                return
            time.sleep(5)
        except Exception:
            log.exception("flatten attempt %d failed", attempt + 1)
            time.sleep(5)


def day_trades(trading: TradingClient, plan: dict, today: pd.Timestamp) -> pd.DataFrame:
    """Each trade's result for one day, worked out from the account's actual fills."""
    fills = []
    for o in day_orders(trading, QueryOrderStatus.CLOSED, today):
        cid = o.client_order_id or ""
        if not (cid.startswith(f"{PREFIX}-") or cid.startswith(f"{EXIT_PREFIX}-")):
            continue  # another strategy's order in the same ticker
        for order in [o] + list(o.legs or []):
            if order.symbol in plan and float(order.filled_qty or 0) > 0:
                fills.append({"symbol": order.symbol, "time": order.filled_at, "qty": float(order.filled_qty),
                              "price": float(order.filled_avg_price), "buy": order.side == OrderSide.BUY,
                              "entry": (order.client_order_id or "").startswith(PREFIX)})
    rows = []
    for sym, g in pd.DataFrame(fills).groupby("symbol") if fills else []:
        g = g.sort_values("time")
        bought, sold = g[g["buy"]], g[~g["buy"]]
        if abs(bought["qty"].sum() - sold["qty"].sum()) > 1e-6 or not g["entry"].any():
            log.warning("%s is not flat; left out of the results", sym)
            continue
        entry = g[g["entry"]].iloc[0]
        exits = g[~g["entry"]]
        pnl = (sold["qty"] * sold["price"]).sum() - (bought["qty"] * bought["price"]).sum()
        p = plan[sym]
        rows.append({"day": f"{today:%Y-%m-%d}", "symbol": sym, "direction": p["direction"], "qty": entry["qty"],
                     "entry_time": pd.Timestamp(entry["time"]).tz_convert(ET).strftime("%H:%M:%S"),
                     "entry": entry["price"], "level": p["level"],
                     "exit": round(float((exits["qty"] * exits["price"]).sum() / exits["qty"].sum()), 4),
                     "exit_time": pd.Timestamp(exits["time"].max()).tz_convert(ET).strftime("%H:%M:%S"),
                     "risk_per_share": p["risk"], "pnl": round(float(pnl), 2),
                     "r": round(float(pnl / (entry["qty"] * p["risk"])), 3)})
    return pd.DataFrame(rows)


def record(trading: TradingClient, plan: dict, today: pd.Timestamp):
    day = day_trades(trading, plan, today)
    if len(day):
        day.to_csv(DIR / f"trades_{today:%Y-%m-%d}.csv", index=False)
        log.info("\n%s", day.to_string(index=False))
    log.info("day finished: %d entry orders, %d trades, P&L $%.2f, average %.2fR, account equity $%s",
             len(plan), len(day), day["pnl"].sum() if len(day) else 0.0, day["r"].mean() if len(day) else 0.0,
             trading.get_account().equity)


def report():
    """Every trade so far, straight from the paper account (works from any machine)."""
    trading = trading_client()
    today = now().tz_localize(None).normalize()
    days = [] if today < pd.Timestamp(FIRST_DAY) else [pd.Timestamp(d.date) for d in trading.get_calendar(
        GetCalendarRequest(start=pd.Timestamp(FIRST_DAY).date(), end=today.date()))]
    parts =[day_trades(trading, plan_from_orders(trading, d), d) for d in days]
    parts = [p for p in parts if len(p)]
    if not parts:
        print("No completed trades yet.")
        return
    t = pd.concat(parts, ignore_index=True)
    print(t.to_string(index=False))
    print()
    by_day = t.groupby("day").agg(trades=("r", "size"), winners=("pnl", lambda s: int((s > 0).sum())),
                                  pnl=("pnl", "sum"), avg_r=("r", "mean"))
    print(by_day.round(2).to_string())
    print(f"\nTotal: {len(t)} trades, {np.mean(t['pnl'] > 0):.0%} winners, P&L ${t['pnl'].sum():,.2f}, "
          f"average {t['r'].mean():+.2f}R, account equity ${trading.get_account().equity}")


def run(mode: str = "day"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(DIR / f"bot_{now():%Y-%m-%d}.log")])
    for noisy in ("httpx", "urllib3", "alpaca"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if hasattr(ctypes, "windll"):   # keep a Windows PC from idling to sleep while this process runs
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    trading = trading_client()
    today = now().tz_localize(None).normalize()
    if mode == "check":
        account = trading.get_account()
        log.info("paper account %s, equity $%s, shorting %s", account.status.value, account.equity, account.shorting_enabled)
        log.info("baseline built for %d stocks; nothing was placed", len(prepare(trading, today)))
        return
    cal = trading.get_calendar(GetCalendarRequest(start=today.date(), end=today.date()))
    if not cal or pd.Timestamp(cal[0].date) != today:
        log.info("market is closed today; nothing to do")
        return
    opens, closes = pd.Timestamp(cal[0].open, tz=ET), pd.Timestamp(cal[0].close, tz=ET)
    scan_at = opens + timedelta(minutes=5, seconds=3)

    if mode in ("day", "morning"):
        if plan_from_orders(trading, today):
            log.info("today's entry orders are already in the account; placing nothing more")
        elif now() > scan_at + timedelta(minutes=LATEST_START_MIN):
            log.info("started too late for today's entries; nothing placed")
        else:
            prep = prepare(trading, today)
            log.info("waiting for 09:35 ET")
            sleep_until(scan_at)
            picks = pick(prep, today)
            log.info("%d candidates", len(picks))
            plan = place_orders(trading, picks, today)
            (DIR / f"plan_{today:%Y-%m-%d}.json").write_text(json.dumps(plan, indent=2))
    if mode == "morning":
        return

    sleep_until(closes - timedelta(minutes=30))
    plan = plan_from_orders(trading, today)
    if now() < closes:
        try:
            cancel_untriggered(trading, today)
        except Exception:
            log.exception("cancel failed")
        sleep_until(closes - timedelta(minutes=5))
        flatten(trading, plan, today)
    sleep_until(closes + timedelta(minutes=1))
    record(trading, plan, today)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "day"
    if mode == "report":
        report()
    else:
        try:
            run(mode)
        except Exception:
            log.exception("bot stopped on an error")
            raise
