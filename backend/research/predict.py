"""Is anything predictable at all? A strategy-free test.

Instead of hand-written setups, train a model on every bar to predict the
return over the next hour and to the close, then look at how the bars it
ranks highest and lowest actually did, out of sample and after costs.
Development period only.

Run:  python -m backend.research.predict 15
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from backend import config
from backend.backtest.simulator import load_context, side_cost
from backend.data.alpaca_data import load_bars
from backend.data.bars import SESSION_LEN, resample
from backend.features.build import FEATURES, compute_features

HOLDOUT_MONTHS = 6


def bar_dataset(tf: int, symbols=None, horizons=(60,)) -> pd.DataFrame:
    """Every tradable bar with its features and forward returns (in basis points)."""
    ctx = load_context(tf)
    parts = []
    for sym in symbols or config.TRAIN_TICKERS:
        f = compute_features(resample(load_bars(sym), tf), tf, ctx)
        day = pd.Series(f.index.normalize(), index=f.index)
        nxt_open = f["open"].shift(-1)
        same_day_next = day.shift(-1) == day
        out = f[FEATURES].copy()
        out["symbol"], out["day"] = sym, day
        out["entry"] = nxt_open.where(same_day_next)
        day_close = f["close"].groupby(day).transform("last")
        out["fwd_eod"] = (day_close / out["entry"] - 1) * 1e4
        for h in horizons:
            k = h // tf
            later = f["close"].shift(-k)
            ok = day.shift(-k) == day
            out[f"fwd_{h}"] = (later.where(ok) / out["entry"] - 1) * 1e4
        out["cost_bps"] = 2 * side_cost(out["entry"]) / out["entry"] * 1e4
        keep = (f["mins"] >= 15) & (f["mins"] <= SESSION_LEN - 30) & f["ema50"].notna() & out["entry"].notna()
        parts.append(out[keep])
    ds = pd.concat(parts).sort_index()
    ds["time"] = ds.index
    return ds.reset_index(drop=True)


def walk_forward(ds: pd.DataFrame, target: str, first_months=12, step_months=3) -> pd.DataFrame:
    """Out-of-sample predictions of a volatility-scaled forward return."""
    ds = ds.dropna(subset=[target]).copy()
    ds["y"] = (ds[target] / (ds["atr_pct"] * 100)).clip(-8, 8)
    start = ds["time"].min() + pd.DateOffset(months=first_months)
    parts = []
    while start < ds["time"].max():
        stop = start + pd.DateOffset(months=step_months)
        train = ds[ds["time"] < start - pd.Timedelta(days=1)]
        test = ds[(ds["time"] >= start) & (ds["time"] < stop)].copy()
        if len(test):
            model = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=500, subsample=0.7,
                subsample_freq=1, colsample_bytree=0.6, reg_lambda=10.0, verbose=-1, n_jobs=-1, random_state=7,
            ).fit(train[FEATURES].astype("float32"), train["y"])
            test["pred"] = model.predict(test[FEATURES].astype("float32"))
            parts.append(test)
        start = stop
    return pd.concat(parts)


def evaluate(oos: pd.DataFrame, target: str, label: str):
    ic = oos.groupby("day").apply(lambda g: g["pred"].corr(g[target], method="spearman")).mean()
    print(f"\n{label}: {len(oos):,} bars, average daily rank correlation {ic:.4f}")
    for frac in (0.10, 0.05, 0.02, 0.01):
        hi, lo = oos["pred"].quantile(1 - frac), oos["pred"].quantile(frac)
        longs, shorts = oos[oos["pred"] >= hi], oos[oos["pred"] <= lo]
        trades = pd.concat([
            longs.assign(net=longs[target] - longs["cost_bps"]),
            shorts.assign(net=-shorts[target] - shorts["cost_bps"]),
        ])
        daily = trades.groupby("day")["net"].mean()
        t = daily.mean() / daily.std() * np.sqrt(len(daily))
        print(f"  top/bottom {frac:4.0%}: long gross {longs[target].mean():6.2f}  short gross {-shorts[target].mean():6.2f}"
              f"  cost {trades['cost_bps'].mean():4.2f}  net {trades['net'].mean():6.2f} bps  win {np.mean(trades['net'] > 0):.3f}"
              f"  days {len(daily)}  t {t:5.2f}")


if __name__ == "__main__":
    tf = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    ds = bar_dataset(tf)
    cut = ds["time"].max().normalize() - pd.DateOffset(months=HOLDOUT_MONTHS)
    dev = ds[ds["time"] < cut]
    print(f"{tf}-minute bars, development period: {len(dev):,} rows")
    for target in ("fwd_60", "fwd_eod"):
        evaluate(walk_forward(dev, target), target, target)
