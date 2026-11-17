"""News features: what was published about a ticker before each bar closed.

Every value at a bar uses only articles time-stamped at or before that bar's
completion, so nothing leaks from the future.
"""
import re

import numpy as np
import pandas as pd

NEWS_SIGNED = ["news_sent_30m", "news_sent_24h", "news_sent_last"]
# The same three, read by FinBERT instead of keywords, plus its strongest recent reading
FINBERT_SIGNED = ["news_fb_30m", "news_fb_24h", "news_fb_last", "news_fb_peak_2h"]
NEWS_UNSIGNED = ["news_1h", "news_24h", "news_ratio", "news_focus_24h", "news_mins_since", "news_earnings_24h",
                 "news_overnight"]
NEWS_FEATURES = NEWS_SIGNED + NEWS_UNSIGNED

POSITIVE = re.compile(
    r"\b(beats?|tops?|topped|raises?|raised|upgrad\w+|surg\w+|jumps?|soars?|rall\w+|record|strong\w*|outperform|"
    r"wins?|gains?|boosts?|approv\w+|bullish|higher|exceed\w*|accelerat\w+|buyback|breakthrough)\b", re.I)
NEGATIVE = re.compile(
    r"\b(miss\w*|cuts?|downgrad\w+|falls?|fell|drops?|plung\w+|slump\w*|sinks?|tumbl\w+|weak\w*|lawsuit|sued|probe|"
    r"investigat\w+|recall\w*|warn\w*|lower|layoffs?|loss\w*|underperform|bearish|halt\w*|delay\w*|declin\w+|"
    r"slash\w*|fraud|bankrupt\w*)\b", re.I)
EARNINGS = re.compile(r"\b(earnings|EPS|Q[1-4]\b|quarterly results|guidance|revenue (beat|miss)|reports? (first|second|third|fourth)"
                      r"[- ]quarter)", re.I)
FOCUS_MAX_SYMBOLS = 2   # an article tagged with many tickers is rarely about this one
MINUTE = 60 * 10 ** 9


def score_headlines(news: pd.DataFrame) -> pd.DataFrame:
    """Per-article sentiment (-1, 0, +1), earnings flag and focus flag."""
    text = news["headline"].fillna("")
    pos = text.str.count(POSITIVE.pattern, flags=re.I)
    neg = text.str.count(NEGATIVE.pattern, flags=re.I)
    out = pd.DataFrame({
        "t": news["created_at"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy(),
        "focus": (news["n_symbols"] <= FOCUS_MAX_SYMBOLS).to_numpy(dtype=float),
        "earnings": text.str.contains(EARNINGS).to_numpy(dtype=float),
    })
    out["sent"] = np.sign(pos - neg).to_numpy(dtype=float) * out["focus"]
    # FinBERT score in [-1, 1]; absent for unscored or unfocused articles
    out["fb"] = news["finbert"].fillna(0.0).to_numpy(dtype=float) if "finbert" in news else 0.0
    return out.sort_values("t").reset_index(drop=True)


def _window_sum(times: np.ndarray, cum: np.ndarray, at: np.ndarray, minutes: int) -> np.ndarray:
    """Sum of a per-article value over the `minutes` before each time in `at`."""
    hi = np.searchsorted(times, at, side="right")
    lo = np.searchsorted(times, at - minutes * MINUTE, side="right")
    return cum[hi] - cum[lo]


def news_features(news: pd.DataFrame, bar_end: pd.DatetimeIndex) -> pd.DataFrame:
    """News features for bars completing at `bar_end` (timezone-aware)."""
    at = bar_end.tz_convert("UTC").tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
    out = pd.DataFrame(index=bar_end, columns=NEWS_FEATURES + FINBERT_SIGNED, dtype=float)
    if news is None or news.empty:
        return out
    a = score_headlines(news)
    times = a["t"].to_numpy()
    cum = {k: np.r_[0.0, np.cumsum(v)] for k, v in
           (("n", np.ones(len(a))), ("focus", a["focus"]), ("earn", a["earnings"]), ("sent", a["sent"]),
            ("fb", a["fb"]))}

    out["news_1h"] = _window_sum(times, cum["n"], at, 60)
    out["news_24h"] = _window_sum(times, cum["n"], at, 1440)
    usual = _window_sum(times, cum["n"], at - 1440 * MINUTE, 30 * 1440) / 30  # prior 30 days, per day
    out["news_ratio"] = out["news_24h"] / np.where(usual > 0, usual, np.nan)
    out["news_focus_24h"] = _window_sum(times, cum["focus"], at, 1440)
    out["news_earnings_24h"] = (_window_sum(times, cum["earn"], at, 1440) > 0).astype(float)
    out["news_sent_30m"] = _window_sum(times, cum["sent"], at, 30)
    out["news_sent_24h"] = _window_sum(times, cum["sent"], at, 1440)

    last = np.searchsorted(times, at, side="right") - 1
    seen = last >= 0
    out["news_mins_since"] = np.where(seen, np.minimum((at - times[np.maximum(last, 0)]) / MINUTE, 1440), 1440)
    # Sentiment of the most recent article that had any
    toned = a[a["sent"] != 0]
    if len(toned):
        k = np.searchsorted(toned["t"].to_numpy(), at, side="right") - 1
        out["news_sent_last"] = np.where(k >= 0, toned["sent"].to_numpy()[np.maximum(k, 0)], 0.0)
    else:
        out["news_sent_last"] = 0.0

    out["news_fb_30m"] = _window_sum(times, cum["fb"], at, 30)
    out["news_fb_24h"] = _window_sum(times, cum["fb"], at, 1440)
    strong = a[a["fb"].abs() >= 0.5]  # headlines FinBERT reads as clearly good or bad
    if len(strong):
        st, sv = strong["t"].to_numpy(), strong["fb"].to_numpy()
        k = np.searchsorted(st, at, side="right") - 1
        latest = np.where(k >= 0, sv[np.maximum(k, 0)], 0.0)
        age = np.where(k >= 0, (at - st[np.maximum(k, 0)]) / MINUTE, np.inf)
        out["news_fb_last"] = latest
        out["news_fb_peak_2h"] = np.where(age <= 120, latest, 0.0)
    else:
        out["news_fb_last"], out["news_fb_peak_2h"] = 0.0, 0.0

    # Articles between the previous 16:00 close and today's 09:30 open
    local = bar_end.tz_convert("America/New_York")
    open_ns = (local.normalize() + pd.Timedelta(minutes=570)).tz_convert("UTC").tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()
    since_close = (17 * 60 + 30)  # 16:00 to 09:30
    weekend = np.where(local.dayofweek == 0, 2 * 1440, 0)  # Monday also covers the weekend
    hi = np.searchsorted(times, open_ns, side="right")
    lo = np.searchsorted(times, open_ns - (since_close + weekend) * MINUTE, side="right")
    out["news_overnight"] = cum["n"][hi] - cum["n"][lo]
    return out
