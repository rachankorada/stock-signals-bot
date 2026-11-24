"""Training table for the news model: one row per headline focused on one or two tickers, with what the
stock was doing when it appeared and what it did after the bot could first act on it.

Entry (no look-ahead):
  intraday  headline 09:30-15:30 ET: the open of the first minute starting 60 s or more after the headline
  overnight headline after a close and before the next open: the 09:31 open of the next session
Labels are abnormal returns in basis points: the stock's return minus beta x SPY's over the same minutes
(beta from the previous 60 sessions' daily returns).
  intraday   y15, y60, yclose (entry -> 15 / 60 minutes later, or 15:59)
  overnight  y60 (09:31 -> 10:31), yclose (09:31 -> 15:59)

Run:  python -m backend.research.news_ml_data
"""
import glob
import os

import numpy as np
import pandas as pd

from backend import config
from backend.data.alpaca_data import ET, load_bars

OUT = config.DATA_DIR / "newsml"
OUT.mkdir(exist_ok=True)
NEWS = config.DATA_DIR / "news"


def session_bars(sym):
    b = load_bars(sym)
    t = b.index.hour * 60 + b.index.minute
    return b[(t >= 570) & (t < 960)]


def main():
    spy = session_bars("SPY")
    spy_daily = spy["close"].groupby(spy.index.normalize()).last().pct_change()
    fb = pd.read_parquet(NEWS / "_finbert_scores.parquet")["finbert"]
    bar_syms = {os.path.basename(f)[:-8] for f in glob.glob(str(config.BARS_DIR / "*.parquet"))}
    rows = []
    for f in sorted(glob.glob(str(NEWS / "*.parquet"))):
        sym = os.path.basename(f)[:-8]
        if sym.startswith("_") or sym not in bar_syms or sym in ("SPY", "QQQ"):
            continue
        news = pd.read_parquet(f)
        news = news[(news["n_symbols"] <= 2) & news["headline"].notna()].copy()
        if news.empty:
            continue
        b = session_bars(sym)
        s = spy.reindex(b.index).ffill()
        idx = b.index.as_unit("ns").asi8
        day = b.index.normalize()
        daily = b["close"].groupby(day).last()
        dret = daily.pct_change()
        beta = (dret.rolling(60).cov(spy_daily.reindex(dret.index)) /
                spy_daily.reindex(dret.index).rolling(60).var()).shift(1).clip(0.3, 3).fillna(1.0)
        o, c, so, sc = b["open"].to_numpy(), b["close"].to_numpy(), s["open"].to_numpy(), s["close"].to_numpy()
        vol = b["volume"].to_numpy()
        # first and last bar of each session
        first = pd.Series(np.arange(len(b)), index=day).groupby(level=0).first()
        last = pd.Series(np.arange(len(b)), index=day).groupby(level=0).last()
        sessions = first.index
        news["t"] = pd.to_datetime(news["created_at"], utc=True).dt.tz_convert(ET)
        for aid, a in news.iterrows():
            t = a["t"]
            mins = t.hour * 60 + t.minute
            today = t.normalize()
            if today in first.index and 570 <= mins < 930:
                kind = "intraday"
                i = int(np.searchsorted(idx, (t + pd.Timedelta(seconds=60)).value, side="left"))
                if i >= len(b) or day[i] != today:
                    continue
            else:
                kind = "overnight"
                k = sessions.searchsorted(today if mins < 570 else today + pd.Timedelta(days=1))
                if k >= len(sessions):
                    continue
                today = sessions[k]
                i = int(first[today]) + 1     # 09:31
                if i > last[today]:
                    continue
            end = int(last[today])
            if (pd.Timestamp(idx[i]) - t.tz_convert("UTC").tz_localize(None)) > pd.Timedelta(days=4):
                continue   # data gap
            bt = float(beta.get(today, 1.0))
            def ab(j):
                j = min(j, end)
                return ((c[j] / o[i] - 1) - bt * (sc[j] / so[i] - 1)) * 1e4
            f0 = int(first[today])
            prev_close_i = f0 - 1 if f0 > 0 else None
            pc = c[prev_close_i] if prev_close_i is not None else np.nan
            spc = sc[prev_close_i] if prev_close_i is not None else np.nan
            r = {"id": aid, "symbol": sym, "time": t, "session": today, "kind": kind, "headline": a["headline"],
                 "source": a.get("source"), "n_symbols": int(a["n_symbols"]), "finbert": fb.get(aid, np.nan),
                 "minute": (pd.Timestamp(idx[i]).tz_localize("UTC").tz_convert(ET).hour * 60
                            + pd.Timestamp(idx[i]).tz_localize("UTC").tz_convert(ET).minute),
                 "beta": bt, "price": o[i],
                 # context at entry (all known by then)
                 "gap": (o[f0] / pc - 1) * 1e4 if pc == pc else np.nan,
                 "spy_gap": (so[f0] / spc - 1) * 1e4 if spc == spc else np.nan,
                 "pre5": (o[i] / o[max(i - 5, f0)] - 1) * 1e4,
                 "pre30": (o[i] / o[max(i - 30, f0)] - 1) * 1e4,
                 "day_so_far": (o[i] / o[f0] - 1) * 1e4,
                 "spy_pre30": (so[i] / so[max(i - 30, f0)] - 1) * 1e4,
                 "spy_day": (so[i] / so[f0] - 1) * 1e4,
                 "rvol30": vol[max(i - 30, f0):i].sum() / max(vol[max(i - 30 - 390 * 5, 0):max(i - 30, 1)].mean() * 30, 1),
                 "y15": ab(i + 15), "y60": ab(i + 60), "yclose": ab(end)}
            rows.append(r)
        print(sym, len(news), flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "articles.parquet")
    print("rows", len(d), d["kind"].value_counts().to_dict())


if __name__ == "__main__":
    main()
