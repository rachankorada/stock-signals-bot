"""Stocks in play: scan the whole market each morning and trade only the unusual ones.

Every earlier test ran a fixed list of tickers. This one picks, each day, the
stocks whose first five minutes traded the most volume relative to their own
recent opens (usually a sign of news), then trades a break of that opening
range in the direction of the first five-minute bar, with a tight stop, out by
the close.

Stages (each saves its work, so they can be rerun or resumed):
  python -m backend.research.in_play scan      opening 5-minute bar of every stock, every day
  python -m backend.research.in_play minutes   1-minute bars for each day's top candidates
  python -m backend.research.in_play test      the backtest (development period; add --final for the last six months)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from backend import config
from backend.data.alpaca_data import ET, client, load_bars
from backend.research.universe import candidates

DIR = config.DATA_DIR / "inplay"
DIR.mkdir(exist_ok=True)
TOP_N = 20              # candidates kept per day
LOOKBACK = 14           # days for average opening volume, average volume and ATR
MIN_PRICE, MIN_AVG_VOLUME, MIN_ATR = 5.0, 1_000_000, 0.50
HOLDOUT_MONTHS = 6
CHUNK = 700             # symbols per request
NOT_LAST_SALE = set("479BCGHMNPQRTUVWZ")


def fetch(symbols, start, end, tf) -> pd.DataFrame:
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start, end=end, feed=DataFeed.SIP,
                           adjustment=Adjustment.ALL)
    for attempt in range(6):
        try:
            return client().get_stock_bars(req).df
        except Exception:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))


def trading_days() -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(load_bars("SPY").index.normalize().unique()).tz_localize(None)
    start = days.max() - pd.DateOffset(years=config.BACKTEST_YEARS) - pd.Timedelta(days=30)
    return days[(days >= start) & (days < days.max())]


def scan():
    """Opening five-minute bar and daily bars for every candidate stock."""
    symbols = candidates()["symbol"].tolist()
    days = trading_days()
    chunks = [symbols[i:i + CHUNK] for i in range(0, len(symbols), CHUNK)]

    daily_path = DIR / "daily.parquet"
    if not daily_path.exists():
        parts = []
        for i, chunk in enumerate(chunks):
            parts.append(fetch(chunk, days[0] - pd.Timedelta(days=40), days[-1] + pd.Timedelta(days=1), TimeFrame.Day))
            print(f"daily bars {i + 1}/{len(chunks)}", flush=True)
        pd.concat(parts)[["open", "high", "low", "close", "volume"]].to_parquet(daily_path)

    open_path = DIR / "opens.parquet"
    done = pd.read_parquet(open_path) if open_path.exists() else None
    have = set(done["day"].unique()) if done is not None else set()
    todo = [d for d in days if d not in have]
    parts = [] if done is None else [done]

    def one_day(day):
        start = day.tz_localize(ET) + timedelta(hours=9, minutes=30)
        got = [fetch(c, start, start + timedelta(minutes=4), TimeFrame(5, TimeFrameUnit.Minute)) for c in chunks]
        df = pd.concat([g for g in got if len(g)]).reset_index()
        df = df[df["timestamp"].dt.tz_convert(ET) == start]
        df["day"] = day
        return df[["day", "symbol", "open", "high", "low", "close", "volume"]]

    with ThreadPoolExecutor(3) as pool:
        for i, df in enumerate(pool.map(one_day, todo)):
            parts.append(df)
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                pd.concat(parts, ignore_index=True).to_parquet(open_path)
                print(f"opening bars {i + 1}/{len(todo)} days", flush=True)


def ranked() -> pd.DataFrame:
    """Each day's eligible stocks with relative opening volume, using only earlier days as the baseline."""
    opens = pd.read_parquet(DIR / "opens.parquet").sort_values(["symbol", "day"])
    daily = pd.read_parquet(DIR / "daily.parquet").reset_index()
    daily["day"] = daily["timestamp"].dt.tz_convert(ET).dt.normalize().dt.tz_localize(None)
    daily = daily.sort_values(["symbol", "day"])
    g = daily.groupby("symbol")
    prev_close = g["close"].shift(1)
    tr = np.maximum(daily["high"] - daily["low"],
                    np.maximum((daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()))
    daily["atr"] = tr.groupby(daily["symbol"]).transform(lambda s: s.rolling(LOOKBACK).mean().shift(1))
    daily["avg_volume"] = g["volume"].transform(lambda s: s.rolling(LOOKBACK).mean().shift(1))
    daily["prev_close"] = prev_close
    opens["usual_open_volume"] = opens.groupby("symbol")["volume"].transform(
        lambda s: s.rolling(LOOKBACK, min_periods=LOOKBACK).mean().shift(1))
    df = opens.merge(daily[["symbol", "day", "atr", "avg_volume", "prev_close"]], on=["symbol", "day"])
    df["rel_volume"] = df["volume"] / df["usual_open_volume"]
    df["gap"] = df["open"] / df["prev_close"] - 1
    df = df[(df["open"] >= MIN_PRICE) & (df["avg_volume"] >= MIN_AVG_VOLUME) & (df["atr"] >= MIN_ATR)
            & (df["rel_volume"] >= 1.0) & (df["close"] != df["open"])]
    df["rank"] = df.groupby("day")["rel_volume"].rank(ascending=False, method="first")
    return df[df["rank"] <= TOP_N].sort_values(["day", "rank"]).reset_index(drop=True)


def minutes():
    """One-minute bars for each day's top candidates."""
    picks = ranked()
    path = DIR / "minutes.parquet"
    done = pd.read_parquet(path) if path.exists() else None
    have = set(done["day"].unique()) if done is not None else set()
    todo = [(d, g["symbol"].tolist()) for d, g in picks.groupby("day") if d not in have]
    parts = [] if done is None else [done]

    def one_day(item):
        day, symbols = item
        start = day.tz_localize(ET) + timedelta(hours=9, minutes=30)
        df = fetch(symbols, start, start + timedelta(hours=6, minutes=29), TimeFrame.Minute).reset_index()
        df["day"] = day
        return df[["day", "symbol", "timestamp", "open", "high", "low", "close", "volume"]]

    with ThreadPoolExecutor(3) as pool:
        for i, df in enumerate(pool.map(one_day, todo)):
            parts.append(df)
            if (i + 1) % 50 == 0 or i + 1 == len(todo):
                pd.concat(parts, ignore_index=True).to_parquet(path)
                print(f"minute bars {i + 1}/{len(todo)} days", flush=True)


def simulate(stop_frac: float = 0.10) -> pd.DataFrame:
    """One opening-range-breakout trade per candidate per day. Result in R (R = the stop distance)."""
    picks = ranked().set_index(["day", "symbol"])
    bars = pd.read_parquet(DIR / "minutes.parquet")
    bars["mins"] = (bars["timestamp"].dt.tz_convert(ET).dt.hour * 60 + bars["timestamp"].dt.tz_convert(ET).dt.minute) - 570
    rows = []
    for (day, sym), b in bars.sort_values("timestamp").groupby(["day", "symbol"], sort=False):
        if (day, sym) not in picks.index:
            continue
        p = picks.loc[(day, sym)]
        first, rest = b[b["mins"] < 5], b[b["mins"] >= 5]
        if first.empty or len(rest) < 30:
            continue
        d = 1 if p["close"] > p["open"] else -1
        level = first["high"].max() if d > 0 else first["low"].min()
        o, h, l, c = (rest[k].to_numpy() for k in ("open", "high", "low", "close"))
        broke = np.flatnonzero(h > level) if d > 0 else np.flatnonzero(l < level)
        if not len(broke) or rest["mins"].iloc[broke[0]] > 360:
            continue
        k = broke[0]
        fill = max(level, o[k]) if d > 0 else min(level, o[k])
        risk = stop_frac * p["atr"]
        stop = fill - d * risk
        # Adverse side of every bar from the entry bar on; the entry bar counts (cautious)
        adverse = l[k:] <= stop if d > 0 else h[k:] >= stop
        if adverse.any():
            j = k + int(adverse.argmax())
            gap = o[j] if j > k else stop
            exit_px = min(stop, gap) if d > 0 else max(stop, gap)
            stopped_in_entry_bar = j == k
        else:
            exit_px, stopped_in_entry_bar = c[-1], False
        # Kinder assumption for the entry bar: stopped there only if it also closed beyond the stop
        if stopped_in_entry_bar and d * (c[k] - stop) > 0:
            later = l[k + 1:] <= stop if d > 0 else h[k + 1:] >= stop
            if later.any():
                j = k + 1 + int(later.argmax())
                kind_exit = min(stop, o[j]) if d > 0 else max(stop, o[j])
            else:
                kind_exit = c[-1]
        else:
            kind_exit = exit_px
        rows.append({"day": day, "symbol": sym, "direction": d, "entry": fill, "risk": risk, "rank": p["rank"],
                     "rel_volume": p["rel_volume"], "gap": p["gap"], "entry_min": rest["mins"].iloc[k],
                     "r": d * (exit_px - fill) / risk, "r_kind": d * (kind_exit - fill) / risk,
                     "stopped": bool(adverse.any())})
    return pd.DataFrame(rows)


def report(label: str, t: pd.DataFrame, col: str = "r", cost_bps: float = 0.0):
    if t.empty:
        print(f"  {label:44s} n=0")
        return
    r = t[col] - 2 * cost_bps / 1e4 * t["entry"] / t["risk"]
    daily = r.groupby(t["day"]).mean()
    tstat = daily.mean() / daily.std() * np.sqrt(len(daily)) if len(daily) > 2 else np.nan
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    print(f"  {label:44s} n={len(r):6d} win={np.mean(r > 0):.3f} avg={r.mean():+.3f}R "
          f"PF={gains / losses:.2f} days={len(daily)} t={tstat:+.2f}")


def test(final: bool = False):
    cut = trading_days().max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    for stop_frac in (0.10, 0.25, 0.50):
        t = simulate(stop_frac)
        t = t[t["day"] >= cut] if final else t[t["day"] < cut]
        print(f"\n=== stop = {stop_frac:.0%} of the daily range; {'FINAL SIX MONTHS' if final else 'development period'} "
              f"{t['day'].min():%Y-%m-%d} to {t['day'].max():%Y-%m-%d} ===")
        report("all candidates, before costs (cautious)", t)
        report("all candidates, before costs (kind entry bar)", t, "r_kind")
        for bps in (2, 5, 10):
            report(f"  after {bps} bps per side", t, "r", bps)
        report("top 5 by relative volume", t[t["rank"] <= 5])
        report("  after 5 bps per side", t[t["rank"] <= 5], "r", 5)
        report("relative volume 5x or more", t[t["rel_volume"] >= 5])
        report("  after 5 bps per side", t[t["rel_volume"] >= 5], "r", 5)
        report("longs", t[t["direction"] > 0])
        report("shorts", t[t["direction"] < 0])
        report("entered in the first 15 minutes", t[t["entry_min"] < 15])
        for y, g in t.groupby(t["day"].dt.year):
            report(f"  {y} after 5 bps per side", g, "r", 5)


def tick_check(n: int = 1500, seed: int = 7):
    """Settle the entry-minute question with real trade-by-trade data.

    With a tight stop, one-minute bars cannot say whether price hit the stop
    before or after the breakout within the entry minute. This replays the
    actual trades of that minute for a random sample (development period only):
    the order triggers on the first round-lot trade through the level, fills at
    the next trade, and is stopped if any later trade in the minute reaches the stop.
    """
    from alpaca.data.requests import StockTradesRequest

    picks = ranked().set_index(["day", "symbol"])
    bars = pd.read_parquet(DIR / "minutes.parquet").sort_values("timestamp")
    bars["mins"] = (bars["timestamp"].dt.tz_convert(ET).dt.hour * 60 + bars["timestamp"].dt.tz_convert(ET).dt.minute) - 570
    cut = trading_days().max() - pd.DateOffset(months=HOLDOUT_MONTHS)
    groups = {k: b for k, b in bars[bars["day"] < cut].groupby(["day", "symbol"], sort=False) if k in picks.index}
    keys = sorted(groups)
    keys = [keys[i] for i in np.random.default_rng(seed).permutation(len(keys))[:n]]

    def one(key):
        p, b = picks.loc[key], groups[key]
        first, rest = b[b["mins"] < 5], b[b["mins"] >= 5]
        if first.empty or len(rest) < 30:
            return None
        d = 1 if p["close"] > p["open"] else -1
        level = first["high"].max() if d > 0 else first["low"].min()
        o, h, l, c = (rest[k].to_numpy() for k in ("open", "high", "low", "close"))
        broke = np.flatnonzero(h > level) if d > 0 else np.flatnonzero(l < level)
        if not len(broke) or rest["mins"].iloc[broke[0]] > 360:
            return None
        k = broke[0]
        t0 = rest["timestamp"].iloc[k]
        req = StockTradesRequest(symbol_or_symbols=key[1], start=t0, end=t0 + timedelta(seconds=59.999), feed=DataFeed.SIP)
        for attempt in range(5):
            try:
                ticks = client().get_stock_trades(req).df
                break
            except Exception:
                if attempt == 4:
                    return None
                time.sleep(5 * (attempt + 1))
        if ticks.empty:
            return None
        # Round-lot trades that count towards the last sale (no late, average-price or off-hours prints)
        regular = ticks["conditions"].map(lambda c: not (set(c or []) & NOT_LAST_SALE))
        px = ticks[(ticks["size"] >= 100) & regular]["price"].to_numpy()
        if not len(px):
            return None
        # Bars are adjusted for splits and dividends, trades are not: put trades on the bar scale
        scale_hi, scale_lo = h[k] / px.max(), l[k] / px.min()
        if abs(scale_hi / scale_lo - 1) > 0.002:
            return {"skipped": True}
        px = px * (scale_hi + scale_lo) / 2
        through = np.flatnonzero(d * (px - level) > 0)
        if not len(through):
            return {"skipped": True}
        i = through[0]
        fill = px[min(i + 1, len(px) - 1)]
        after = px[i + 1:]
        out = {"day": key[0], "symbol": key[1], "entry": fill, "slip_bps": d * (fill - level) / level * 1e4}
        for frac in (0.10, 0.25, 0.50):
            risk = frac * p["atr"]
            stop = fill - d * risk
            if (d * (after - stop) <= 0).any():
                exit_px = stop
            else:
                adverse = l[k + 1:] <= stop if d > 0 else h[k + 1:] >= stop
                if adverse.any():
                    j = k + 1 + int(adverse.argmax())
                    exit_px = min(stop, o[j]) if d > 0 else max(stop, o[j])
                else:
                    exit_px = c[-1]
            out[f"r_{frac}"], out[f"risk_{frac}"] = d * (exit_px - fill) / risk, risk
        return out

    # Work in batches and save each one, so a stopped run resumes where it left off
    part_path = DIR / f"tick_parts_{n}.parquet"
    res = pd.read_parquet(part_path).to_dict("records") if part_path.exists() else []
    seen = {(r["day"], r["symbol"]) for r in res}
    todo = [k for k in keys if k not in seen]
    with ThreadPoolExecutor(3) as pool:
        for i in range(0, len(todo), 300):
            batch = todo[i:i + 300]
            for key, r in zip(batch, pool.map(one, batch)):
                res.append({"day": key[0], "symbol": key[1], "skipped": True, **(r or {})} if not r or r.get("skipped")
                           else {**r, "skipped": False})
            pd.DataFrame(res).to_parquet(part_path)
            print(f"replayed {i + len(batch):,}/{len(todo):,}", flush=True)
    res = [r for r in res if "entry" in r and r["entry"] == r["entry"] or r.get("skipped")]
    skipped = sum(1 for r in res if r and r.get("skipped"))
    t = pd.DataFrame([r for r in res if r and not r.get("skipped")])
    t.to_parquet(DIR / "tick_check.parquet")
    print(f"sampled {len(keys)}, replayed {len(t)}, could not be matched to the bars {skipped}")
    print(f"slippage from the breakout level to the fill: median {t['slip_bps'].median():.1f} bps, mean {t['slip_bps'].mean():.1f} bps")
    for frac in (0.10, 0.25, 0.50):
        print(f"stop = {frac:.0%} of the daily range (entry at the real next trade)")
        view = t.rename(columns={f"r_{frac}": "r", f"risk_{frac}": "risk"})
        report("exit costs nothing", view)
        for bps in (2, 5):
            report(f"  {bps} bps on the way out", view, "r", bps / 2)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "test"
    {"scan": scan, "minutes": minutes, "ticks": lambda: tick_check(int(sys.argv[2]) if len(sys.argv) > 2 else 1500), "test": lambda: test("--final" in sys.argv)}[stage]()

