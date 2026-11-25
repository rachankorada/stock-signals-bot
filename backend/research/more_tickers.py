"""Pick more tickers for the signal bot, spread across sectors so they do not all move together.

Step 1  (python -m backend.research.more_tickers pick)
    Daily returns over the last two years for the current 30 tickers and a pool of liquid large caps
    from every sector. Greedy pick: each round adds the candidate whose strongest correlation with
    anything already in the list is the lowest, at most MAX_PER_SECTOR per sector, liquid enough
    (median daily dollar volume) to fill paper orders cleanly. Stops once the best remaining candidate
    correlates above MAX_CORR with a ticker already in.

Step 2  (python -m backend.research.more_tickers check)
    Minute bars for the picks (IEX, what the live engine reads, and full market for outcomes); keeps
    picks whose IEX feed covers enough minutes for the engine to build clean 15-minute bars; then runs
    the bot's 15-minute model over the picks and the current 30 alike, so the new names can be
    compared with the old ones on the same footing. (The model was trained on the current 30 only, so
    the new names are out of sample by ticker, but not by period.)
"""
import json
import sys
import time
from datetime import datetime, timedelta, timezone

import joblib
import numpy as np
import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from backend import config
from backend.data.alpaca_data import client, download_history, load_bars

OUT = config.DATA_DIR / "more_tickers"
OUT.mkdir(exist_ok=True)

POOL = {
    "tech": ["ORCL", "CRM", "ADBE", "CSCO", "QCOM", "MU", "INTC", "IBM", "TXN", "AMAT", "NOW", "UBER", "SHOP"],
    "communication": ["T", "VZ", "TMUS", "CMCSA"],
    "financials": ["MS", "WFC", "SCHW", "AXP", "C", "BLK", "PGR", "CB"],
    "health": ["JNJ", "MRK", "ABBV", "TMO", "ISRG", "ABT", "CVS", "AMGN", "GILD", "BMY"],
    "staples": ["PG", "PEP", "MO", "PM", "CL", "MDLZ", "KR"],
    "discretionary": ["MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "CMG", "GM", "ABNB"],
    "industrials": ["GE", "UNP", "UPS", "LMT", "RTX", "DE", "HON", "DAL", "FDX", "ETN"],
    "energy": ["COP", "SLB", "OXY", "EOG", "KMI"],
    "utilities": ["NEE", "DUK", "SO", "AEP"],
    "real_estate": ["PLD", "AMT", "O", "SPG"],
    "materials": ["LIN", "FCX", "NEM", "DOW", "NUE"],
}
SECTOR = {s: k for k, v in POOL.items() for s in v}
CURRENT_SECTOR = {**{s: "tech" for s in ["AAPL", "MSFT", "NVDA", "AMD", "AVGO", "PLTR"]},
                  **{s: "communication" for s in ["META", "GOOGL", "NFLX", "DIS"]},
                  **{s: "discretionary" for s in ["TSLA", "AMZN", "HD", "F"]},
                  **{s: "financials" for s in ["JPM", "BAC", "GS", "V", "COIN", "SOFI"]},
                  **{s: "energy" for s in ["XOM", "CVX"]},
                  **{s: "health" for s in ["UNH", "LLY", "PFE"]},
                  **{s: "staples" for s in ["WMT", "COST", "KO"]},
                  **{s: "industrials" for s in ["BA", "CAT"]}}
N_NEW = 30
MAX_PER_SECTOR = 6          # counting the current tickers: no sector may dominate
MIN_DOLLAR_VOL = 300e6      # median daily dollar volume
MIN_IEX_COVER = 0.85        # share of regular-hours minutes with an IEX bar
MIN_PRICE = 10
MAX_CORR = 0.6             # stop once every remaining candidate moves this closely with one already in


def daily(symbols, years=2):
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=end - timedelta(days=365 * years),
                           end=end, feed=DataFeed.SIP, adjustment=Adjustment.ALL)
    df = client().get_stock_bars(req).df.reset_index()
    df["day"] = df["timestamp"].dt.tz_convert("America/New_York").dt.normalize()
    return df


def pick():
    cands = [s for v in POOL.values() for s in v]
    d = daily(config.TRAIN_TICKERS + cands + ["SPY"])
    close = d.pivot(index="day", columns="symbol", values="close")
    dv = (d["close"] * d["volume"]).groupby(d["symbol"]).median()
    price = close.iloc[-1]
    ret = np.log(close).diff().iloc[1:]
    corr = ret.corr()
    ok = [s for s in cands if s in corr and dv.get(s, 0) >= MIN_DOLLAR_VOL and price.get(s, 0) >= MIN_PRICE]
    chosen = list(config.TRAIN_TICKERS)
    count = pd.Series(CURRENT_SECTOR).value_counts().to_dict()
    order = []
    while len(order) < N_NEW:
        left = [s for s in ok if s not in chosen and count.get(SECTOR[s], 0) < MAX_PER_SECTOR]
        if not left:
            break
        worst = {s: corr.loc[s, chosen].abs().max() for s in left}
        s = min(worst, key=worst.get)
        if worst[s] > MAX_CORR:
            break
        order.append({"symbol": s, "sector": SECTOR[s], "max_corr": round(float(worst[s]), 3),
                      "closest": corr.loc[s, chosen].abs().idxmax(), "corr_spy": round(float(corr.loc[s, "SPY"]), 3),
                      "dollar_vol_m": round(dv[s] / 1e6)})
        chosen.append(s)
        count[SECTOR[s]] = count.get(SECTOR[s], 0) + 1
    t = pd.DataFrame(order)
    old = corr.loc[config.TRAIN_TICKERS, config.TRAIN_TICKERS].to_numpy()
    new = corr.loc[chosen, chosen].to_numpy()
    off = lambda m: m[~np.eye(len(m), dtype=bool)]
    print(t.to_string(index=False))
    print(f"\naverage pairwise correlation: current 30 {off(old).mean():.3f}, with the picks {off(new).mean():.3f}")
    print("dropped for liquidity or price:", sorted(set(cands) - set(ok)))
    t.to_csv(OUT / "picks.csv", index=False)
    return t


def coverage(sym):
    iex, sip = load_bars(sym, "iex"), load_bars(sym)
    since = sip.index[-1] - pd.Timedelta(days=365)
    return len(iex[iex.index >= since]) / max(len(sip[sip.index >= since]), 1)


def check():
    from backend.backtest.simulator import build_dataset, no_overlap, summarize
    picks = pd.read_csv(OUT / "picks.csv")["symbol"].tolist()
    for feed in ("iex", "sip"):
        for s in picks:
            t0 = time.time()
            df = download_history(s, years=3, feed=feed)
            print(f"{s} {feed}: {len(df):,} bars ({time.time() - t0:.0f}s)", flush=True)
    cover = {s: round(coverage(s), 3) for s in picks}
    usable = [s for s in picks if cover[s] >= MIN_IEX_COVER]
    print("IEX coverage:", cover)
    print("too thin on IEX for the live engine:", [s for s in picks if s not in usable])

    model = joblib.load(config.ROOT / "backend" / "bot" / "models" / "tf15.joblib")
    since = pd.Timestamp.now(tz="America/New_York") - pd.DateOffset(years=2)
    res = {}
    for name, syms in (("current 30", config.TRAIN_TICKERS), ("new picks", usable)):
        ds = build_dataset(15, syms)
        ds = ds[ds["time"] >= since].copy()
        ds["conf"] = model.predict(ds)
        posted = ds[ds["conf"] >= config.MIN_POST_CONFIDENCE]
        one = no_overlap(posted)
        days = ds["time"].dt.normalize().nunique()
        res[name] = {"tickers": len(syms), "posted": summarize(posted), "one_per_ticker": summarize(one),
                     "per_day": round(len(one) / days, 2),
                     "by_ticker": {s: summarize(g) for s, g in one.groupby("symbol")}}
        print(f"\n{name}: {len(syms)} tickers, {res[name]['per_day']} signals a day (one per ticker at a time)")
        print("   posted:", res[name]["posted"])
        print("   one per ticker:", res[name]["one_per_ticker"])
    (OUT / "check.json").write_text(json.dumps({"coverage": cover, "usable": usable, **res}, indent=2, default=str))


if __name__ == "__main__":
    {"pick": pick, "check": check}[sys.argv[1]]()
