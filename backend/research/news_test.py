"""Does news improve the signals? Three out-of-sample checks over the backtest window.

1. Raw setups split by news conditions (earnings day, heavy news, fresh news,
   sentiment agreeing or disagreeing with the trade).
2. The confidence model with and without news inputs, walked forward.
3. A setup-free event study: what price does after a fresh article with a
   clear positive or negative headline.

Run:  python -m backend.research.news_test 5      (add --small for the small-stock universe)
"""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend import config
from backend.backtest.simulator import summarize
from backend.data.news import load_news
from backend.features.news import FINBERT_SIGNED, NEWS_FEATURES, NEWS_SIGNED, news_features
from backend.model.confidence import model_matrix

TEST_MONTHS = 3


def add_news(ds: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """Attach news features to rows that have a symbol and a bar-completion time."""
    parts = []
    for sym, g in ds.groupby("symbol"):
        feats = news_features(load_news(sym, finbert=True), pd.DatetimeIndex(g[time_col]))
        feats.index = g.index
        parts.append(feats)
    return ds.join(pd.concat(parts))


SETS = {
    "without news": [],
    "keyword news": NEWS_FEATURES,
    "FinBERT news": NEWS_FEATURES + FINBERT_SIGNED,
}


def matrix(ds: pd.DataFrame, news_cols: list) -> pd.DataFrame:
    X = model_matrix(ds)
    for col in news_cols:
        flip = ds["direction"].to_numpy() if col in NEWS_SIGNED + FINBERT_SIGNED else 1
        X[col] = (ds[col].to_numpy() * flip).astype("float32")
    return X


def walk_forward(ds: pd.DataFrame, news_cols: list) -> pd.DataFrame:
    end = ds["time"].max()
    start = end.normalize() - pd.DateOffset(years=config.BACKTEST_YEARS)
    parts = []
    while start < end:
        stop = start + pd.DateOffset(months=TEST_MONTHS)
        train = ds[ds["exit_time"] < start - pd.Timedelta(days=1)]
        test = ds[(ds["time"] >= start) & (ds["time"] < stop)].copy()
        if len(test):
            model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=300, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1, n_jobs=-1, random_state=7,
            ).fit(matrix(train, news_cols), train["win"])
            test["score"] = model.predict_proba(matrix(test, news_cols))[:, 1]
            parts.append(test)
        start = stop
    return pd.concat(parts)


def line(label: str, rows: pd.DataFrame):
    s = summarize(rows)
    if s["trades"]:
        print(f"  {label:44s} n={s['trades']:7d} win={s['win_rate']:.3f} avg={s['expectancy_r']:+.3f}R cost={s['avg_cost_r']:.3f}")
    else:
        print(f"  {label:44s} n=      0")


if __name__ == "__main__":
    tf = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    small = "--small" in sys.argv   # the small-stock universe, charged its measured spreads
    ds = pd.read_parquet(config.DATA_DIR / "datasets" / f"{'small_' if small else ''}tf{tf}.parquet")
    if small:
        from backend.research.small_caps import with_costs
        from backend.research.spreads import load_spreads
        ds = with_costs(ds, load_spreads())
    ds = add_news(ds)
    window = ds[ds["time"] >= ds["time"].max().normalize() - pd.DateOffset(years=config.BACKTEST_YEARS)]
    aligned = window["news_sent_24h"] * window["direction"]
    fresh = window["news_sent_30m"] * window["direction"]

    print(f"=== {tf}-minute setups in the three-year window, no model ===")
    line("all setups", window)
    line("earnings news in last 24h", window[window["news_earnings_24h"] > 0])
    line("no earnings news", window[window["news_earnings_24h"] == 0])
    line("news volume 2x normal or more", window[window["news_ratio"] >= 2])
    line("news volume below normal", window[window["news_ratio"] < 1])
    line("article in the last 30 minutes", window[window["news_mins_since"] <= 30])
    line("24h sentiment agrees with trade", window[aligned > 0])
    line("24h sentiment against trade", window[aligned < 0])
    line("fresh (30m) sentiment agrees with trade", window[fresh > 0])
    line("fresh (30m) sentiment against trade", window[fresh < 0])
    fb_day, fb_fresh = window["news_fb_24h"] * window["direction"], window["news_fb_peak_2h"] * window["direction"]
    line("FinBERT 24h reading agrees (strongly)", window[fb_day >= 1])
    line("FinBERT 24h reading against (strongly)", window[fb_day <= -1])
    line("FinBERT clear headline <2h, agrees", window[fb_fresh > 0])
    line("FinBERT clear headline <2h, against", window[fb_fresh < 0])
    line("FinBERT 30m reading agrees", window[window["news_fb_30m"] * window["direction"] >= 0.5])
    line("FinBERT 30m reading against", window[window["news_fb_30m"] * window["direction"] <= -0.5])

    print()
    print("=== Confidence model with different news inputs (walk-forward) ===")
    for name, cols in SETS.items():
        oos = walk_forward(ds, cols)
        print(f" {name}: ranking quality (AUC) {roc_auc_score(oos['win'], oos['score']):.4f}")
        for frac in (0.10, 0.05, 0.02):
            line(f"top {frac:.0%} by score", oos[oos["score"] >= oos["score"].quantile(1 - frac)])
