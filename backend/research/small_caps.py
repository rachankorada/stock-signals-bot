"""The full signal pipeline on smaller, less-followed stocks.

Same setups, same confidence model and the same three-year walk-forward as the
main backtest, but on the universe chosen by `universe.py` and charged each
stock's measured bid-ask spread (`spreads.py`) instead of the flat large-cap cost.

Run:  python -m backend.research.small_caps 15      (add --cached to reuse built setups)
"""
import sys

import numpy as np
import pandas as pd

from backend import config
from backend.backtest.simulator import build_dataset, no_overlap, summarize
from backend.model.train import CHECK_MONTHS, walk_forward
from backend.research.spreads import load_spreads
from backend.research.universe import load_universe


def with_costs(ds: pd.DataFrame, spreads: dict, mult: float = 1.0) -> pd.DataFrame:
    """Re-price every trade: half the measured spread on the way in and again on the way out."""
    out = ds.copy()
    gross = out["r"] + out["cost_r"]
    half = out["symbol"].map(spreads).astype(float) / 2 / 1e4 * out["entry"]
    out["cost_r"] = 2 * half * mult / out["risk"]
    out["r"] = gross - out["cost_r"]
    return out


def line(label: str, rows: pd.DataFrame):
    s = summarize(rows)
    if not s["trades"]:
        print(f"  {label:46s} n=      0")
        return
    gross = (rows["r"] + rows["cost_r"]).mean()
    print(f"  {label:46s} n={s['trades']:7d} win={s['win_rate']:.3f} before costs={gross:+.3f}R "
          f"after={s['expectancy_r']:+.3f}R (cost {s['avg_cost_r']:.3f}R)")


if __name__ == "__main__":
    tf = int(next((a for a in sys.argv[1:] if a.isdigit()), 15))
    symbols, spreads = load_universe(), load_spreads()
    symbols = [s for s in symbols if spreads.get(s)]
    cache = config.DATA_DIR / "datasets" / f"small_tf{tf}.parquet"
    if "--cached" in sys.argv and cache.exists():
        raw = pd.read_parquet(cache)
    else:
        raw = build_dataset(tf, symbols, feed="sip")
        raw.to_parquet(cache)

    ds = with_costs(raw, spreads)
    oos = walk_forward(ds)
    days = oos["time"].dt.normalize().nunique()
    check = oos["time"].max() - pd.DateOffset(months=CHECK_MONTHS)
    print(f"===== {tf}-minute, {len(symbols)} small stocks, {oos['time'].min():%Y-%m-%d} to {oos['time'].max():%Y-%m-%d} "
          f"({days} days) =====")
    print(f"typical spread: median {np.median([spreads[s] for s in symbols]):.1f} bps "
          f"(range {min(spreads[s] for s in symbols):.1f} to {max(spreads[s] for s in symbols):.1f})")

    line("every setup, no model", oos)
    line("  longs", oos[oos["direction"] > 0])
    line("  shorts", oos[oos["direction"] < 0])
    for name, g in oos.groupby("strategy"):
        line(f"  {name}", g)

    print("by confidence:")
    for t in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        line(f"confidence >= {t:.0%}", oos[oos["conf"] >= t])
    print("by model rank:")
    for frac in (0.20, 0.10, 0.05, 0.02):
        line(f"top {frac:.0%} by score", oos[oos["raw"] >= oos["raw"].quantile(1 - frac)])

    posted = oos[oos["conf"] >= config.MIN_POST_CONFIDENCE]
    one = no_overlap(posted) if len(posted) else posted
    print(f"posted signals (confidence >= {config.MIN_POST_CONFIDENCE:.0%}):")
    line("all", posted)
    line(f"one per ticker at a time ({len(one) / days:.1f}/day)", one)
    line("before the final six months", posted[posted["time"] < check])
    line("final six months", posted[posted["time"] >= check])
    for y, g in posted.groupby(posted["time"].dt.year):
        line(f"  {y}", g)
    tight = pd.Series(spreads).reindex(symbols).rank(pct=True)
    for label, lo, hi in (("tightest-spread third", 0, 1 / 3), ("middle third", 1 / 3, 2 / 3), ("widest third", 2 / 3, 1.01)):
        line(f"  {label}", posted[posted["symbol"].map(tight).between(lo, hi, inclusive="right")])

    print("posted signals if costs were different (model retrained each time):")
    for mult in (0.0, 0.5):
        alt = walk_forward(with_costs(raw, spreads, mult))
        line(f"  {mult:.0%} of the measured spread", alt[alt["conf"] >= config.MIN_POST_CONFIDENCE])
        line(f"    top 5% by score", alt[alt["raw"] >= alt["raw"].quantile(0.95)])
