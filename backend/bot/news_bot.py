"""News scores for the signal follower: which tickers have news, which way it leans and how strongly.

Every headline tagged with a ticker since the previous session's close is read by FinBERT (a free
model trained on financial text; a simple keyword reading is used if FinBERT cannot load). Articles
tagged with one ticker count fully, two tickers half, more than two not at all. The ticker's score
runs from -1 (clearly bad news) to +1 (clearly good) and is pulled toward zero when there are only one
or two articles:

    score = sum(weight * article score) / (sum(weight) + 1)

How a score steers a signal (a = direction x score, from -1 against to +1 with the trade):
    signal strength   confidence + 0.05 * a
    trade size        1 + 0.5 * a, between 0.5x and 1.5x
    caution           strong news against the trade (a <= -0.6): 0.25x, almost sitting it out

The backtest found news did not improve the signals (2026-11-23), so the effect is kept moderate and
every trade carries its news reading in its order id; the report shows results with and without
the news adjustments, to settle it on live trades.

Run:  python -m backend.bot.news_bot        (today's scores, as the bot sees them now)
"""
import logging

import numpy as np
import pandas as pd
from alpaca.trading.requests import GetCalendarRequest

from backend import config
from backend.data.alpaca_data import ET
from backend.data.news import fetch_news
from backend.features.news import NEGATIVE, POSITIVE

log = logging.getLogger("news")

CONF_STEP = 0.05        # confidence change at full agreement
SIZE_STEP = 0.5         # size change at full agreement
SIZE_MIN, SIZE_MAX = 0.5, 1.5
CAUTION_AT = -0.6       # this much against the trade: cut to CAUTION_SIZE
CAUTION_SIZE = 0.25
LEVELS = ((0.6, "strong"), (0.35, "moderate"), (0.15, "weak"))

_finbert = None          # False once it has failed to load
_seen = {}               # article id -> score, so each headline is read once per run


def strength(score: float) -> str:
    return next((name for cut, name in LEVELS if abs(score) >= cut), "none")


def method() -> str:
    """Which reader scores headlines: 'finbert' or 'keywords' (after the first call)."""
    return "finbert" if _finbert else "keywords"


def read_headlines(texts: list) -> np.ndarray:
    """A score in [-1, 1] per headline: FinBERT, or a keyword reading if FinBERT cannot load."""
    return _score_texts(texts)


def combine(article_scores, n_symbols) -> float:
    """A ticker's score from its articles: one-ticker articles count fully, two-ticker articles half,
    others not at all; pulled toward zero when there are few articles."""
    a = np.asarray(article_scores, dtype=float); n = np.asarray(n_symbols)
    w = np.where(n == 1, 1.0, np.where(n == 2, 0.5, 0.0))
    return float((w * a).sum() / (w.sum() + 1)) if len(a) else 0.0


def _score_texts(texts: list) -> np.ndarray:
    global _finbert
    if _finbert is None:
        try:
            from backend.data.news_sentiment import score_texts
            score_texts(["warm up"])
            _finbert = score_texts
            log.info("FinBERT loaded")
        except Exception as exc:
            _finbert = False
            log.warning("FinBERT unavailable (%s); using keyword reading", str(exc)[:120])
    if _finbert:
        return _finbert(texts)
    return np.array([float(np.sign(len(POSITIVE.findall(t)) - len(NEGATIVE.findall(t)))) for t in texts])


def window_start(trading, today: pd.Timestamp) -> pd.Timestamp:
    """The previous session's close."""
    cal = trading.get_calendar(GetCalendarRequest(start=(today - pd.Timedelta(days=10)).date(),
                                                  end=(today - pd.Timedelta(days=1)).date()))
    last = cal[-1]
    return pd.Timestamp(last.close, tz=ET)


def scores(symbols: list, since: pd.Timestamp, until: pd.Timestamp = None) -> pd.DataFrame:
    """One row per ticker: articles counted, score, strength and the latest headline."""
    until = until or pd.Timestamp.now(tz=ET)
    rows = []
    for sym in symbols:
        try:
            news = fetch_news(sym, since.tz_convert("UTC").to_pydatetime(), until.tz_convert("UTC").to_pydatetime())
        except Exception as exc:
            log.warning("%s: news fetch failed: %s", sym, str(exc)[:120])
            news = pd.DataFrame()
        news = news[(news["n_symbols"] <= 2) & news["headline"].notna()] if len(news) else news
        if len(news):
            new = [i for i in news.index if i not in _seen]
            if new:
                for i, s in zip(new, _score_texts(news.loc[new, "headline"].tolist())):
                    _seen[i] = float(s)
            fb = np.array([_seen[i] for i in news.index])
            score = combine(fb, news["n_symbols"])
            top = news.iloc[int(np.argmax(np.abs(fb)))]["headline"]
        else:
            score, top = 0.0, ""
        rows.append({"symbol": sym, "articles": len(news), "score": round(score, 3), "strength": strength(score),
                     "headline": str(top)[:110]})
    return pd.DataFrame(rows).set_index("symbol")


def adjust(sig: dict, table: pd.DataFrame) -> dict:
    """How the news steers one signal."""
    score = float(table["score"].get(sig["symbol"], 0.0)) if table is not None and len(table) else 0.0
    a = sig["direction"] * score
    return {"score": score, "agreement": a, "strength": strength(score), "size": size_for(a),
            "confidence": float(sig.get("confidence", 0)) + CONF_STEP * a}


def code(a: float) -> str:
    """Agreement for an order id ('-' separates the id's parts): p42 = +0.42, m30 = -0.30."""
    v = int(round(abs(a) * 100))
    return f"{'p' if a >= 0 else 'm'}{min(v, 99):02d}"


def decode(c: str) -> float:
    return (1 if c[0] == "p" else -1) * int(c[1:]) / 100


def size_for(a: float) -> float:
    return CAUTION_SIZE if a <= CAUTION_AT else float(np.clip(1 + SIZE_STEP * a, SIZE_MIN, SIZE_MAX))


def show(table: pd.DataFrame, title: str):
    hot = table[table["strength"] != "none"].sort_values("score", key=abs, ascending=False)
    log.info("%s: %d of %d tickers have news worth noting", title, len(hot), len(table))
    for sym, r in hot.iterrows():
        log.info("  %-5s %-8s %-8s %+.2f  (%d articles)  %s", sym, r["strength"],
                 "positive" if r["score"] > 0 else "negative", r["score"], r["articles"], r["headline"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "urllib3", "alpaca", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from backend.bot.signal_bot import trading_client
    t = trading_client()
    today = pd.Timestamp.now(tz=ET).tz_localize(None).normalize()
    show(scores(config.BOT_TICKERS, window_start(t, today)), "news since the last close")
