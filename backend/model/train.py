"""Confidence model: training, the three-year backtest and the validation check.

Run:  python -m backend.model.train            (all timeframes)
      python -m backend.model.train 5 15       (chosen timeframes)
      add --cached to reuse the setups built by the previous run
      add --dry to print the report without saving models

Steps per timeframe
1. Build every setup and its outcome for the training tickers over the full history.
2. Walk forward through the last three years in three-month steps: train on
   everything earlier, score the next three months. Every backtest trade is
   therefore scored by a model that had never seen it.
3. Calibrate confidence against how earlier out-of-sample scores really did.
4. Backtest the posting rule (confidence at or above the site minimum).
5. A timeframe is validated only if that backtest is profitable after costs
   both before and during the final six months.
6. Train the production model on all data.
"""
import json
import sys

import joblib
import numpy as np
import pandas as pd

from backend import config
from backend.backtest.simulator import build_dataset, no_overlap, summarize
from backend.model.confidence import ConfidenceModel, calibrator
from backend.strategies.library import STRATEGIES

CHECK_MONTHS = 6          # final stretch used only to confirm the result
TEST_MONTHS = 3
MIN_CALIBRATION_ROWS = 5000
THRESHOLDS = np.round(np.arange(0.40, 0.701, 0.05), 2)
MIN_TRADES = 200
MIN_TRADES_FOR_STRATEGY = 40
MIN_EXPECTANCY_R = 0.03
MIN_PROFIT_FACTOR = 1.1


def walk_forward(ds: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample raw score and confidence for every setup in the backtest window."""
    end = ds["time"].max()
    start = end.normalize() - pd.DateOffset(years=config.BACKTEST_YEARS)
    parts, past_raw, past_won = [], [], []
    while start < end:
        stop = start + pd.DateOffset(months=TEST_MONTHS)
        # One-day embargo so trades straddling the boundary cannot leak
        train = ds[ds["exit_time"] < start - pd.Timedelta(days=1)]
        test = ds[(ds["time"] >= start) & (ds["time"] < stop)].copy()
        if len(test) and len(train) > 5000:
            model = ConfidenceModel().fit(train)
            test["raw"] = model.raw(test)
            if sum(map(len, past_raw)) >= MIN_CALIBRATION_ROWS:
                # Calibrate on how earlier out-of-sample scores turned out
                test["conf"] = calibrator(np.concatenate(past_raw), np.concatenate(past_won)).predict(test["raw"])
            else:
                test["conf"] = model.iso.predict(test["raw"])
            past_raw.append(test["raw"].to_numpy())
            past_won.append(test["win"].to_numpy())
            parts.append(test)
        start = stop
    return pd.concat(parts)


def passes(stats: dict, min_trades: int) -> bool:
    return (
        stats["trades"] >= min_trades
        and stats["expectancy_r"] >= MIN_EXPECTANCY_R
        and (stats["profit_factor"] or 0) >= MIN_PROFIT_FACTOR
    )


def calibration_table(ds: pd.DataFrame) -> list:
    cut = pd.cut(ds["conf"], np.arange(0.0, 1.01, 0.1), include_lowest=True)
    table = ds.groupby(cut, observed=True).agg(n=("win", "size"), predicted=("conf", "mean"), actual=("win", "mean"))
    return [{"bucket": str(k), "n": int(v.n), "predicted": round(v.predicted, 3), "actual": round(v.actual, 3)}
            for k, v in table.iterrows()]


def train_timeframe(tf: int, save: bool = True) -> dict:
    print(f"\n===== {tf}-minute timeframe =====", flush=True)
    cache = config.DATA_DIR / "datasets"
    cache.mkdir(exist_ok=True)
    name = f"tf{tf}.parquet" if config.TRAIN_FEED == "sip" else f"tf{tf}_{config.TRAIN_FEED}.parquet"
    print(f"inputs from {config.TRAIN_FEED.upper()} bars, {len(config.TRAIN_TICKERS)} tickers", flush=True)
    if "--cached" in sys.argv and (cache / name).exists():
        ds = pd.read_parquet(cache / name)  # skip rebuilding setups (strategies unchanged)
    else:
        ds = build_dataset(tf)
        if save:
            ds.to_parquet(cache / name)

    oos = walk_forward(ds)
    days = oos["time"].dt.normalize().nunique()
    print(f"setups: {len(ds):,} from {ds['time'].min():%Y-%m-%d}; backtest window {oos['time'].min():%Y-%m-%d} "
          f"to {oos['time'].max():%Y-%m-%d} ({days} trading days, {len(oos):,} setups)")
    print(f"every setup, no model: {summarize(oos)}", flush=True)

    sweep = [{"threshold": float(t), **summarize(oos[oos["conf"] >= t])} for t in THRESHOLDS]
    for row in sweep:
        print("  ", row)

    # The site posts everything at or above its minimum confidence
    posted = oos[oos["conf"] >= config.MIN_POST_CONFIDENCE]
    check_start = oos["time"].max() - pd.DateOffset(months=CHECK_MONTHS)
    earlier, final = posted[posted["time"] < check_start], posted[posted["time"] >= check_start]
    backtest = {
        "all": summarize(posted), "one_per_ticker": summarize(no_overlap(posted)) if len(posted) else {"trades": 0},
        "earlier": summarize(earlier), "final_six_months": summarize(final),
        "per_day": round(len(no_overlap(posted)) / days, 2) if len(posted) else 0.0,
        "by_year": [{"year": int(y), **summarize(g)} for y, g in posted.groupby(posted["time"].dt.year)],
        "start": str(oos["time"].min().date()), "end": str(oos["time"].max().date()),
    }
    validated = bool(passes(backtest["earlier"], MIN_TRADES) and final["r"].mean() > 0 and len(final) >= 30)
    strategies = {}
    for name in STRATEGIES:
        stats = summarize(posted[posted["strategy"] == name])
        strategies[name] = {**stats, "enabled": bool(validated and passes(stats, MIN_TRADES_FOR_STRATEGY))}

    print(f"THREE-YEAR BACKTEST of posted signals (confidence >= {config.MIN_POST_CONFIDENCE:.0%}): {backtest['all']}")
    print(f"   one signal per ticker at a time: {backtest['one_per_ticker']}  ({backtest['per_day']} per day)")
    for row in backtest["by_year"]:
        print("   ", row)
    print(f"   before the final six months: {backtest['earlier']}")
    print(f"   final six months:            {backtest['final_six_months']}")
    for name, v in strategies.items():
        print(f"   {name:26s} {v}")
    print(f"timeframe validated: {validated}", flush=True)

    meta = {
        "timeframe": tf, "trained_through": str(ds["time"].max()), "validated": validated,
        "p_min": config.MIN_POST_CONFIDENCE, "strategies": strategies, "sweep": sweep, "backtest": backtest,
        "history": {name: {k: v for k, v in s.items() if k != "enabled"} for name, s in strategies.items()},
        "calibration": calibration_table(oos), "raw": summarize(oos),
        "geometry": {"tp1_r": config.TP1_R, "tp2_r": config.TP2_R,
                     "stop_min_atr": config.STOP_MIN_ATR, "stop_max_atr": config.STOP_MAX_ATR},
    }
    if save:
        # Production model: trees on all data, calibration from every out-of-sample score
        model = ConfidenceModel().fit(ds, holdback=0)
        model.iso = calibrator(oos["raw"].to_numpy(), oos["win"].to_numpy())
        joblib.dump(model, config.MODEL_DIR / f"tf{tf}.joblib")
        (config.MODEL_DIR / f"meta_tf{tf}.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    for tf in [int(a) for a in sys.argv[1:] if a.isdigit()] or config.TIMEFRAMES:
        train_timeframe(tf, save="--dry" not in sys.argv)
