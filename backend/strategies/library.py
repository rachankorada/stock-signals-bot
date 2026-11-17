"""The strategy library.

Each strategy takes a feature frame and returns four Series:
(long setup, short setup, long stop reference, short stop reference).
The stop reference is the price that proves the setup wrong. Strategies are
vectorised, so the backtest and the live engine run identical code.
"""
import pandas as pd

from backend import config
from backend.data.bars import SESSION_LEN


def _cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def _cross_down(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))


def opening_range_breakout(f):
    window = (f["mins"] > 30) & (f["mins"] <= 120)
    mid = (f["or_high"] + f["or_low"]) / 2
    busy = f["vol_surge"] > 1.5
    long = window & busy & _cross_up(f["close"], f["or_high"])
    short = window & busy & _cross_down(f["close"], f["or_low"])
    return long, short, mid, mid


def vwap_trend_pullback(f):
    atr = f["atr"]
    up = (f["ema20"] > f["ema50"]) & (f["close"] > f["vwap"]) & (f["adx"] > 20)
    down = (f["ema20"] < f["ema50"]) & (f["close"] < f["vwap"]) & (f["adx"] > 20)
    support = f[["vwap", "ema20"]].max(axis=1)
    resistance = f[["vwap", "ema20"]].min(axis=1)
    long = up & (f["low"] <= support + 0.2 * atr) & (f["close"] > f["open"]) & (f["close"] > f["ema9"])
    short = down & (f["high"] >= resistance - 0.2 * atr) & (f["close"] < f["open"]) & (f["close"] < f["ema9"])
    return long, short, f[["low", "vwap"]].min(axis=1) - 0.2 * atr, f[["high", "vwap"]].max(axis=1) + 0.2 * atr


def vwap_mean_reversion(f):
    atr = f["atr"]
    calm = (f["adx"] < 25) & (f["mins"] > 45)
    long = calm & (f["vwap_z"] < -2) & (f["rsi14"] < 30) & (f["close"] > f["open"])
    short = calm & (f["vwap_z"] > 2) & (f["rsi14"] > 70) & (f["close"] < f["open"])
    return long, short, f["low"] - 0.3 * atr, f["high"] + 0.3 * atr


def momentum_breakout(f):
    active = (f["mins"] > 30) & (f["vol_surge"] > 2)
    long = active & (f["close"] > f["hh20"]) & (f["close"].shift(1) <= f["hh20"].shift(1)) & (f["close_loc_c"] > 0.2)
    short = active & (f["close"] < f["ll20"]) & (f["close"].shift(1) >= f["ll20"].shift(1)) & (f["close_loc_c"] < -0.2)
    return long, short, f["low"], f["high"]


def squeeze_breakout(f):
    coiled = f["squeeze"].shift(1).rolling(6).sum() >= 5
    long = coiled & _cross_up(f["close"], f["bb_up"])
    short = coiled & _cross_down(f["close"], f["bb_lo"])
    return long, short, f["bb_mid"], f["bb_mid"]


def rsi_reversal(f):
    atr = f["atr"]
    long = (f["rsi2"] < 10) & (f["close"] < f["bb_lo"]) & (f["htf_trend"] > 0)
    short = (f["rsi2"] > 90) & (f["close"] > f["bb_up"]) & (f["htf_trend"] < 0)
    return long, short, f["low"] - 0.5 * atr, f["high"] + 0.5 * atr


def ema_trend_continuation(f):
    strong = f["adx"] > 20
    long = strong & _cross_up(f["ema9"], f["ema20"]) & (f["htf_trend"] > 0) & (f["close"] > f["vwap"])
    short = strong & _cross_down(f["ema9"], f["ema20"]) & (f["htf_trend"] < 0) & (f["close"] < f["vwap"])
    return long, short, f["swing_lo"], f["swing_hi"]


def prior_day_level(f):
    atr = f["atr"]
    busy = f["vol_surge"] > 1.3
    long = busy & _cross_up(f["close"], f["pdh"])
    short = busy & _cross_down(f["close"], f["pdl"])
    return long, short, f["pdh"] - 0.5 * atr, f["pdl"] + 0.5 * atr


STRATEGIES = {
    "opening_range_breakout": opening_range_breakout,
    "vwap_trend_pullback": vwap_trend_pullback,
    "vwap_mean_reversion": vwap_mean_reversion,
    "momentum_breakout": momentum_breakout,
    "squeeze_breakout": squeeze_breakout,
    "rsi_reversal": rsi_reversal,
    "ema_trend_continuation": ema_trend_continuation,
    "prior_day_level": prior_day_level,
}

LABELS = {
    "opening_range_breakout": "Opening range breakout",
    "vwap_trend_pullback": "VWAP trend pullback",
    "vwap_mean_reversion": "VWAP mean reversion",
    "momentum_breakout": "Momentum breakout",
    "squeeze_breakout": "Volatility squeeze breakout",
    "rsi_reversal": "Oversold/overbought reversal",
    "ema_trend_continuation": "EMA trend continuation",
    "prior_day_level": "Prior-day level break",
}


def find_setups(f: pd.DataFrame, tf: int, last_only: bool = False) -> pd.DataFrame:
    """All setups in a feature frame: one row per (bar, strategy, direction).

    last_only=True checks just the newest bar (live use).
    """
    tradable = (
        (f["mins"] >= config.NO_ENTRY_FIRST_MIN)
        & (f["mins"] <= SESSION_LEN - config.NO_ENTRY_LAST_MIN)
        & f["atr"].notna() & f["ema50"].notna() & (f["atr"] > 0)
    )
    rows = []
    for name, fn in STRATEGIES.items():
        long, short, long_stop, short_stop = fn(f)
        for direction, mask, stop in ((1, long, long_stop), (-1, short, short_stop)):
            mask = (mask & tradable).fillna(False).to_numpy(dtype=bool)
            if last_only:
                pos = [len(f) - 1] if mask[-1] else []
            else:
                pos = mask.nonzero()[0]
            if len(pos):
                rows.append(pd.DataFrame({
                    "pos": pos, "strategy": name, "direction": direction,
                    "stop_ref": stop.to_numpy()[pos],
                }))
    if not rows:
        return pd.DataFrame(columns=["pos", "strategy", "direction", "stop_ref"])
    return pd.concat(rows, ignore_index=True)
