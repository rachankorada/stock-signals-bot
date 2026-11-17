"""Optional: mirror each signal as a bracket order in the Alpaca paper account.

Off by default. Set PAPER_TRADE=true in .env to turn it on.
"""
import logging

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

from backend import config

log = logging.getLogger("paper")


def place(signal: dict):
    qty = int(config.PAPER_RISK_DOLLARS // signal["risk"])
    if qty < 1:
        return
    try:
        TradingClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY, paper=True).submit_order(MarketOrderRequest(
            symbol=signal["symbol"], qty=qty, time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
            side=OrderSide.BUY if signal["direction"] > 0 else OrderSide.SELL,
            take_profit=TakeProfitRequest(limit_price=round(signal["tp1"], 2)),
            stop_loss=StopLossRequest(stop_price=round(signal["stop"], 2)),
        ))
    except Exception:
        log.exception("paper order failed for %s", signal["symbol"])
