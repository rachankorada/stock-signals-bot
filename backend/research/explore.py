"""Research harness for new strategy ideas.

Everything here runs on the development period only. The final six months are
never loaded, so they stay a clean test for whatever is chosen.

Run:  python -m backend.research.explore
"""
import numpy as np
import pandas as pd

from backend import config
from backend.backtest.simulator import side_cost
from backend.data.alpaca_data import load_bars
from backend.data.bars import resample

HOLDOUT_MONTHS = 6
BARS_PER_DAY = 78  # 5-minute bars


def day_matrices(symbol: str, holdout: bool = False) -> dict:
    """5-minute OHLCV as (days x 78) arrays. Development period unless holdout=True."""
    bars = resample(load_bars(symbol), 5)
    cut = bars.index.max().normalize() - pd.DateOffset(months=HOLDOUT_MONTHS)
    if not holdout:
        bars = bars[bars.index < cut]
    day = bars.index.normalize()
    slot = (bars.index.hour * 60 + bars.index.minute - 570) // 5
    days = pd.Index(sorted(set(day)))
    row = days.get_indexer(day)
    out = {"days": days, "cut": cut}
    for col in ("open", "high", "low", "close", "volume"):
        m = np.full((len(days), BARS_PER_DAY), np.nan)
        m[row, slot] = bars[col].to_numpy()
        out[col] = m
    # Fill gaps so a missing bar repeats the last price with zero volume
    close = pd.DataFrame(out["close"]).ffill(axis=1).to_numpy()
    for col in ("open", "high", "low"):
        out[col] = np.where(np.isnan(out[col]), close, out[col])
    out["close"] = close
    out["volume"] = np.nan_to_num(out["volume"])
    full = ~np.isnan(out["close"][:, 0]) & ~np.isnan(out["close"][:, -1])
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = out[col][full]
    out["days"] = days[full]
    return out


def daily_atr(m: dict, n: int = 14) -> np.ndarray:
    """Daily ATR known before each session (uses prior days only)."""
    hi, lo, cl = m["high"].max(1), m["low"].min(1), m["close"][:, -1]
    prev = np.r_[np.nan, cl[:-1]]
    tr = np.nanmax(np.c_[hi - lo, np.abs(hi - prev), np.abs(lo - prev)], axis=1)
    return pd.Series(tr).rolling(n).mean().shift(1).to_numpy()


def prior_mean(x: np.ndarray, n: int = 14) -> np.ndarray:
    return pd.DataFrame(x).rolling(n, min_periods=n).mean().shift(1).to_numpy()


def orb_in_play(symbol: str, stop_frac: float = 0.10, holdout: bool = False) -> pd.DataFrame:
    """5-minute opening range breakout, graded by opening relative volume.

    Trade the direction of the first 5-minute candle with a stop order at its
    extreme, a stop loss of `stop_frac` of the daily ATR, exit at the close.
    """
    m = day_matrices(symbol, holdout)
    O, H, L, C, V = m["open"], m["high"], m["low"], m["close"], m["volume"]
    atr = daily_atr(m)
    rvol = V[:, 0] / prior_mean(V[:, :1])[:, 0]
    rows = []
    for d in range(len(O)):
        if not np.isfinite(atr[d]) or not np.isfinite(rvol[d]) or C[d, 0] == O[d, 0]:
            continue
        side = 1 if C[d, 0] > O[d, 0] else -1
        level = H[d, 0] if side > 0 else L[d, 0]
        risk = stop_frac * atr[d]
        hit = (H[d, 1:] > level) if side > 0 else (L[d, 1:] < level)
        if not hit.any():
            continue
        j = 1 + int(hit.argmax())
        entry = max(level, O[d, j]) if side > 0 else min(level, O[d, j])
        stop = entry - side * risk
        # Conservative: the entry bar can also stop the trade out
        stopped = (L[d, j:] <= stop) if side > 0 else (H[d, j:] >= stop)
        exit_px = stop if stopped.any() else C[d, -1]
        cost = 2 * side_cost(entry)
        rows.append({
            "symbol": symbol, "day": m["days"][d], "side": side, "rvol": rvol[d],
            "r": (side * (exit_px - entry) - cost) / risk, "ret_bps": (side * (exit_px - entry) - cost) / entry * 1e4,
            "stopped": bool(stopped.any()), "cost_r": cost / risk,
            "gap_pct": (O[d, 0] / C[d - 1, -1] - 1) * 100 if d else np.nan,
            "atr_pct": atr[d] / O[d, 0] * 100,
        })
    return pd.DataFrame(rows)


def noise_breakout(symbol: str, lookback: int = 14, band: float = 1.0, every: int = 6,
                   holdout: bool = False) -> pd.DataFrame:
    """Trend entries when price leaves its usual range for that time of day.

    The "noise" band is the average absolute move from the open at each time
    of day over the prior `lookback` sessions. Checked every `every` bars
    (6 = half-hourly). Exit when price closes back inside the band or VWAP,
    or at the close.
    """
    m = day_matrices(symbol, holdout)
    O, H, L, C, V = m["open"], m["high"], m["low"], m["close"], m["volume"]
    move = np.abs(C / O[:, :1] - 1)
    sigma = prior_mean(move, lookback) * band
    prev_close = np.r_[np.nan, C[:-1, -1]]
    upper = np.maximum(O[:, 0], prev_close)[:, None] * (1 + sigma)
    lower = np.minimum(O[:, 0], prev_close)[:, None] * (1 - sigma)
    tp = (H + L + C) / 3
    vwap = np.cumsum(tp * V, 1) / np.maximum(np.cumsum(V, 1), 1e-9)
    atr = daily_atr(m)
    checks = np.arange(every - 1, BARS_PER_DAY - 1, every)  # bar closes at :00 and :30
    rows = []
    for d in range(len(O)):
        if not np.isfinite(sigma[d, -1]) or not np.isfinite(prev_close[d]) or not np.isfinite(atr[d]):
            continue
        pos, entry, entry_j = 0, 0.0, 0
        for j in checks:
            px = C[d, j]
            if pos == 0:
                want = 1 if px > upper[d, j] else -1 if px < lower[d, j] else 0
                if want:
                    pos, entry_j = want, j
                    entry = O[d, j + 1]  # filled at the next bar's open
            else:
                trail = max(upper[d, j], vwap[d, j]) if pos > 0 else min(lower[d, j], vwap[d, j])
                if (pos > 0 and px < trail) or (pos < 0 and px > trail):
                    rows.append(_trade(symbol, m["days"][d], pos, entry, O[d, j + 1], entry_j, j, atr[d], sigma[d, entry_j]))
                    pos = 0
        if pos:
            rows.append(_trade(symbol, m["days"][d], pos, entry, C[d, -1], entry_j, BARS_PER_DAY - 1, atr[d], sigma[d, entry_j]))
    return pd.DataFrame(rows)


def _trade(symbol, day, side, entry, exit_px, entry_j, exit_j, atr, sigma):
    cost = 2 * side_cost(entry)
    pnl = side * (exit_px - entry) - cost
    return {"symbol": symbol, "day": day, "side": side, "entry_slot": int(entry_j), "bars": int(exit_j - entry_j),
            "ret_bps": pnl / entry * 1e4, "r_atr": pnl / atr, "sigma_pct": sigma * 100}


def report(trades: pd.DataFrame, value: str, label: str):
    """Mean result with a t-statistic clustered by day (trades on one day move together)."""
    if trades.empty:
        print(f"{label}: no trades")
        return
    daily = trades.groupby("day")[value].sum()
    t = daily.mean() / daily.std() * np.sqrt(len(daily)) if len(daily) > 2 and daily.std() > 0 else np.nan
    print(f"{label:38s} n={len(trades):6d} mean={trades[value].mean():8.3f} win={np.mean(trades[value] > 0):.3f} "
          f"days={len(daily):4d} t={t:5.2f}")


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    symbols = config.TRAIN_TICKERS

    print("=== Opening range breakout by opening relative volume (result in R) ===")
    orb = pd.concat([orb_in_play(s) for s in symbols], ignore_index=True)
    report(orb, "r", "all days")
    for lo, hi in ((0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 99)):
        report(orb[(orb["rvol"] >= lo) & (orb["rvol"] < hi)], "r", f"rvol {lo}-{hi}")
    for year, g in orb[orb["rvol"] >= 2].groupby(orb["day"].dt.year):
        report(g, "r", f"rvol>=2 in {year}")
    print(orb[orb["rvol"] >= 2].groupby("symbol")["r"].agg(["size", "mean"]).round(3).T.to_string())

    print("\n=== Noise-boundary breakout (result in basis points per trade) ===")
    noise = pd.concat([noise_breakout(s) for s in symbols + config.CONTEXT_TICKERS], ignore_index=True)
    report(noise, "ret_bps", "all")
    report(noise[noise["symbol"].isin(config.CONTEXT_TICKERS)], "ret_bps", "SPY and QQQ only")
    for side, name in ((1, "long"), (-1, "short")):
        report(noise[noise["side"] == side], "ret_bps", name)
    for year, g in noise.groupby(noise["day"].dt.year):
        report(g, "ret_bps", f"year {year}")
    print(noise.groupby("symbol")["ret_bps"].agg(["size", "mean"]).round(1).T.to_string())
