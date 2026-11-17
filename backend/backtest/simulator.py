"""Trade construction and outcome simulation.

A setup is decided on a bar's close and filled at the next bar's open, so no
future data leaks in. Each trade ends at take profit, stop loss, the time
limit or the end-of-day flatten, whichever comes first (triple barrier).
"""
import numpy as np
import pandas as pd

from backend import config
from backend.data.alpaca_data import load_bars
from backend.data.bars import SESSION_LEN, resample
from backend.features.build import FEATURES, compute_features, context_features
from backend.strategies.library import find_setups


def make_levels(entry, stop_ref, atr, direction):
    """Stop, targets and risk per share. Works on scalars or arrays."""
    raw = direction * (entry - stop_ref)
    raw = np.where(np.isfinite(raw) & (raw > 0), raw, config.STOP_MIN_ATR * atr)
    risk = np.clip(raw, config.STOP_MIN_ATR * atr, config.STOP_MAX_ATR * atr)
    stop = entry - direction * risk
    return stop, entry + direction * config.TP1_R * risk, entry + direction * config.TP2_R * risk, risk


def side_cost(price):
    """Half-spread plus slippage, per share, per side."""
    return config.COST_FIXED + config.COST_BPS / 1e4 * price


def simulate(f: pd.DataFrame, setups: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Outcome of every setup in `setups` against the bars in `f`."""
    if setups.empty:
        return pd.DataFrame()
    n = len(f)
    o, h, l, c = (f[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr_all, mins = f["atr"].to_numpy(), f["mins"].to_numpy()
    idx = np.arange(n)
    day = f.index.normalize()
    # Last bar of each session that still closes before the end-of-day flatten
    in_time = mins <= SESSION_LEN - config.FLATTEN_BEFORE_CLOSE_MIN
    last_ok = pd.Series(np.where(in_time, idx, -1)).groupby(day.to_numpy()).transform("max").to_numpy()

    pos = setups["pos"].to_numpy()
    d = setups["direction"].to_numpy()
    ref, atr = c[pos], atr_all[pos]
    stop, tp1, tp2, risk = make_levels(ref, setups["stop_ref"].to_numpy(), atr, d)

    nxt = np.minimum(pos + 1, n - 1)
    fill = o[nxt]
    valid = (pos + 1 <= last_ok[pos]) & (pos + 1 < n)
    valid &= d * (fill - ref) <= 0.25 * risk   # price ran away before entry
    valid &= d * (fill - stop) > 0

    hold = config.MAX_HOLD_BARS[tf]
    win = nxt[:, None] + np.arange(hold)[None, :]
    inside = win <= last_ok[pos][:, None]
    wc = np.minimum(win, n - 1)
    hi = np.where(inside, h[wc], -np.inf)
    lo = np.where(inside, l[wc], np.inf)
    long = (d > 0)[:, None]
    stop_hit = np.where(long, lo <= stop[:, None], hi >= stop[:, None])
    tp_hit = np.where(long, hi >= tp1[:, None], lo <= tp1[:, None])
    first_stop = np.where(stop_hit.any(1), stop_hit.argmax(1), hold)
    first_tp = np.where(tp_hit.any(1), tp_hit.argmax(1), hold)

    # If both are touched in the same bar, assume the stop came first
    is_tp = first_tp < first_stop
    is_stop = ~is_tp & (first_stop < hold)
    n_in = np.maximum(inside.sum(1), 1)
    exit_k = np.where(is_tp, first_tp, np.where(is_stop, first_stop, n_in - 1))
    exit_pos = np.minimum(nxt + exit_k, n - 1)
    gap_open = o[exit_pos]
    stop_px = np.where(d > 0, np.minimum(stop, gap_open), np.maximum(stop, gap_open))
    exit_px = np.where(is_tp, tp1, np.where(is_stop, stop_px, c[exit_pos]))

    cost = side_cost(fill)
    r = (d * (exit_px - fill) - 2 * cost) / risk

    out = pd.DataFrame({
        "time": f.index[pos] + pd.Timedelta(minutes=tf),
        "exit_time": f.index[exit_pos] + pd.Timedelta(minutes=tf),
        "pos": pos, "strategy": setups["strategy"].to_numpy(), "direction": d,
        "entry": fill, "stop": stop, "tp1": tp1, "risk": risk,
        "stop_atr": risk / atr, "cost_r": 2 * cost / risk,
        "win": is_tp.astype(int), "outcome": np.where(is_tp, "tp", np.where(is_stop, "stop", "time")),
        "r": r, "bars_held": exit_k + 1,
    })
    out["n_agree"] = out.groupby(["pos", "direction"])["strategy"].transform("count")
    feats = f[FEATURES].iloc[pos].reset_index(drop=True)
    out = pd.concat([out, feats], axis=1)[valid]
    return out.reset_index(drop=True)


def load_context(tf: int) -> dict:
    return {
        name.lower(): context_features(resample(load_bars(name), tf), tf)
        for name in config.CONTEXT_TICKERS
    }


def build_dataset(tf: int, symbols=None) -> pd.DataFrame:
    """Every setup with its outcome for all symbols at one timeframe."""
    ctx = load_context(tf)
    parts = []
    for sym in symbols or config.TRAIN_TICKERS:
        f = compute_features(resample(load_bars(sym), tf), tf, ctx)
        trades = simulate(f, find_setups(f, tf), tf)
        trades.insert(0, "symbol", sym)
        parts.append(trades)
    return pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)


def no_overlap(trades: pd.DataFrame) -> pd.DataFrame:
    """Keep trades as a user would see them: one open signal per ticker at a time."""
    keep, busy_until = [], {}
    for row in trades.sort_values("time").itertuples():
        if row.time >= busy_until.get(row.symbol, pd.Timestamp.min.tz_localize(row.time.tz)):
            keep.append(row.Index)
            busy_until[row.symbol] = row.exit_time
    return trades.loc[keep]


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["r"]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": int(len(r)),
        "win_rate": round(float(trades["win"].mean()), 4),
        "expectancy_r": round(float(r.mean()), 4),
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
        "avg_cost_r": round(float(trades["cost_r"].mean()), 3),
    }
