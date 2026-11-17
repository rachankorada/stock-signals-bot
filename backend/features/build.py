"""Indicators and model features, computed per timeframe.

Every model input is scale-free (ATR units, ratios, percentages) so one pooled
model works across tickers and price levels. Live volume comes from a single
exchange, so volume features are always relative to the same stock's own norm.
"""
import numpy as np
import pandas as pd

from backend.data.bars import session_minutes

# Features that flip sign for short setups
SIGNED = [
    "ret1", "ret3", "ret6", "ret12", "d_ema9", "d_ema20", "d_ema50", "ema20_slope",
    "htf_trend", "macd_h", "bb_z", "vwap_dist", "vwap_z", "d_pdc", "gap_pct", "day_ret",
    "bar_body", "di_diff", "rsi14_c", "rsi2_c", "day_pos_c", "close_loc_c",
    "spy_ret6", "spy_vwap_dist", "spy_day_ret", "qqq_ret6", "qqq_vwap_dist", "qqq_day_ret",
    "rel_strength",
]
# Level pairs that swap (and flip) for short setups
SWAP_PAIRS = [("d_or_high", "d_or_low"), ("d_pdh", "d_pdl"), ("d_hh20", "d_ll20"), ("d_day_high", "d_day_low")]
UNSIGNED = ["adx", "atr_pct", "bb_width", "squeeze", "rvol", "vol_surge", "day_rvol", "mins", "bar_range",
            "range_used", "dow"]
FEATURES = SIGNED + [c for pair in SWAP_PAIRS for c in pair] + UNSIGNED


def ema(s: pd.Series, span: float) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up, down = wilder(d.clip(lower=0), n), wilder(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def _same_slot_mean(s: pd.Series, day_code: np.ndarray, slot: np.ndarray) -> np.ndarray:
    """Mean of the same bar-of-day over the prior 20 sessions (at least 5)."""
    n_days, n_slots, window = int(day_code.max()) + 1, int(slot.max()) + 1, 20
    wide = np.full((n_days + 1, n_slots), np.nan)   # row 0 is padding: day d sits in row d + 1
    wide[day_code + 1, slot] = s.to_numpy()
    seen = ~np.isnan(wide)
    total = np.cumsum(np.where(seen, wide, 0.0), axis=0)
    count = np.cumsum(seen, axis=0)
    # Rows d-19..d of the padded grid are sessions d-20..d-1, the 20 before session d
    lo = np.maximum(np.arange(n_days) - window, 0)
    hi = np.arange(n_days)
    win_total, win_count = total[hi] - total[lo], count[hi] - count[lo]
    base = np.where(win_count >= 5, win_total / np.maximum(win_count, 1), np.nan)
    return base[day_code, slot]


def context_features(bars: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Market-context inputs from an index ETF (SPY or QQQ)."""
    f = compute_features(bars, tf)
    return f[["ret6", "vwap_dist", "day_ret"]]


def compute_features(bars: pd.DataFrame, tf: int, ctx: dict | None = None) -> pd.DataFrame:
    """bars: OHLCV at `tf` minutes, indexed by bar start (ET).

    ctx: {"spy": context_features(...), "qqq": ...} at the same timeframe.
    Each row uses only data up to and including that bar.
    """
    f = bars.copy()
    o, h, l, c, v = f["open"], f["high"], f["low"], f["close"], f["volume"]
    day = pd.Series(f.index.normalize(), index=f.index)
    mins = pd.Series(session_minutes(f.index), index=f.index)
    f["mins"] = mins + tf  # minutes since the open when this bar completes

    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = wilder(tr, 14)
    f["atr"] = atr
    f["atr_pct"] = atr / c * 100

    for n in (1, 3, 6, 12):
        f[f"ret{n}"] = (c - c.shift(n)) / atr
    for n in (9, 20, 50):
        f[f"ema{n}"] = ema(c, n)
        f[f"d_ema{n}"] = (c - f[f"ema{n}"]) / atr
    f["ema20_slope"] = (f["ema20"] - f["ema20"].shift(3)) / atr
    # Higher-timeframe trend: an EMA spanning 20 fifteen-minute bars
    f["htf_trend"] = (c - ema(c, 20 * 15 / tf)) / atr

    up_move, down_move = h.diff(), -l.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di, minus_di = 100 * wilder(plus_dm, 14) / atr, 100 * wilder(minus_dm, 14) / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    f["adx"] = wilder(dx.fillna(0), 14)
    f["di_diff"] = plus_di - minus_di

    f["rsi14"], f["rsi2"] = rsi(c, 14), rsi(c, 2)
    f["rsi14_c"], f["rsi2_c"] = f["rsi14"] - 50, f["rsi2"] - 50
    macd = ema(c, 12) - ema(c, 26)
    f["macd_h"] = (macd - ema(macd, 9)) / atr

    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_mid"], f["bb_up"], f["bb_lo"] = mid, mid + 2 * sd, mid - 2 * sd
    f["bb_z"] = (c - mid) / sd.replace(0, np.nan)
    f["bb_width"] = 4 * sd / atr
    f["squeeze"] = ((f["bb_up"] < f["ema20"] + 1.5 * atr) & (f["bb_lo"] > f["ema20"] - 1.5 * atr)).astype(float)

    # Session VWAP with volume-weighted deviation
    tp = (h + l + c) / 3
    vol = v.where(v > 0, 1e-9)
    cum_v = vol.groupby(day).cumsum()
    vwap = (tp * vol).groupby(day).cumsum() / cum_v
    var = ((tp ** 2) * vol).groupby(day).cumsum() / cum_v - vwap ** 2
    f["vwap"] = vwap
    f["vwap_dist"] = (c - vwap) / atr
    f["vwap_z"] = (c - vwap) / np.maximum(np.sqrt(var.clip(lower=0)), 0.1 * atr)

    # Opening range: the first 30 minutes, known only once they are over
    in_or = mins < 30
    f["or_high"] = h.where(in_or).groupby(day).transform("max").where(f["mins"] > 30)
    f["or_low"] = l.where(in_or).groupby(day).transform("min").where(f["mins"] > 30)
    f["d_or_high"] = (c - f["or_high"]) / atr
    f["d_or_low"] = (c - f["or_low"]) / atr

    daily = f.groupby(day).agg(dh=("high", "max"), dl=("low", "min"), dc=("close", "last")).shift(1)
    for k, src in (("pdh", "dh"), ("pdl", "dl"), ("pdc", "dc")):
        f[k] = day.map(daily[src])
        f[f"d_{k}"] = (c - f[k]) / atr
    day_open = o.groupby(day).transform("first")
    day_high, day_low = h.groupby(day).cummax(), l.groupby(day).cummin()
    f["gap_pct"] = (day_open - f["pdc"]) / f["pdc"] * 100
    f["day_ret"] = (c - day_open) / day_open * 100
    f["day_pos_c"] = (c - day_low) / (day_high - day_low).replace(0, np.nan) - 0.5
    f["d_day_high"], f["d_day_low"] = (c - day_high) / atr, (c - day_low) / atr
    # How much of a normal day's range has already been travelled (exhaustion)
    span = f.groupby(day).agg(dh=("high", "max"), dl=("low", "min"))
    usual_range = (span["dh"] - span["dl"]).rolling(14, min_periods=5).mean().shift(1)
    f["range_used"] = (day_high - day_low) / day.map(usual_range)
    f["dow"] = f.index.dayofweek.astype(float)

    rng = (h - l).replace(0, np.nan)
    f["bar_body"] = (c - o) / atr
    f["bar_range"] = (h - l) / atr
    f["close_loc_c"] = (c - l) / rng - 0.5

    f["hh20"], f["ll20"] = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    f["d_hh20"], f["d_ll20"] = (c - f["hh20"]) / atr, (c - f["ll20"]) / atr
    f["swing_lo"], f["swing_hi"] = l.rolling(10).min(), h.rolling(10).max()

    # Relative volume: against the same bar-of-day over the prior 20 sessions
    slot = (mins // tf).to_numpy()
    day_code = pd.factorize(day)[0]
    f["rvol"] = v / pd.Series(_same_slot_mean(v, day_code, slot), index=f.index).replace(0, np.nan)
    cum_day_v = v.groupby(day).cumsum()
    f["day_rvol"] = cum_day_v / pd.Series(_same_slot_mean(cum_day_v, day_code, slot), index=f.index).replace(0, np.nan)
    f["vol_surge"] = v / v.rolling(20).mean().shift(1).replace(0, np.nan)

    for name in ("spy", "qqq"):
        cf = ctx[name].reindex(f.index).ffill(limit=3) if ctx and name in ctx else None
        for col in ("ret6", "vwap_dist", "day_ret"):
            f[f"{name}_{col}"] = cf[col] if cf is not None else np.nan
    f["rel_strength"] = f["day_ret"] - f["spy_day_ret"]
    return f


def directional(X: pd.DataFrame, direction: np.ndarray) -> pd.DataFrame:
    """Mirror features for short setups so the model learns one symmetric pattern."""
    X = X.copy()
    short = direction < 0
    for col in SIGNED:
        X[col] = np.where(short, -X[col], X[col])
    for a, b in SWAP_PAIRS:
        va, vb = X[a].to_numpy(), X[b].to_numpy()
        X[a] = np.where(short, -vb, va)
        X[b] = np.where(short, -va, vb)
    return X
