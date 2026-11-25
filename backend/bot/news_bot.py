"""News scores for the signal follower: which tickers have news, which way it leans and how strongly.

Every headline tagged with a ticker since the previous session's close is read twice and the two
readings are averaged:
  FinBERT      a free model trained on financial text (tone); a keyword reading if it cannot load
  news model   our own: a small local language model turns the headline into a meaning vector, and a
               model trained on 3 years of price reactions maps it (with event flags) to an expected
               move (backend/research/news_ml_*; models/news_text.joblib). Its strongest 5% of readings
               count as "strong". On held-out data it did not predict direction (2026-11-23). Articles
tagged with one ticker count fully, two tickers half, more than two not at all. The ticker's score
runs from -1 (clearly bad news) to +1 (clearly good) and is pulled toward zero when there are only one
or two articles:

    score = sum(weight * article score) / (sum(weight) + 1)

How a score steers a signal (a = direction x score, from -1 against to +1 with the trade). News can
only take risk away (from 2026-11-27):
    signal strength   confidence + 0.15 * min(a, 0)
    trade size        1 + min(a, 0), at least 0.25x; never above 1x
    caution           news against the trade (a <= -0.4): 0.25x, almost sitting it out

The backtest found news did not improve the signals (2026-11-23). Every trade carries its news
reading in its order id; the report shows results with and without the news adjustments.

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

# Protective only, from 2026-11-27: news against a trade lowers its confidence and cuts its size; news
# with a trade changes nothing. Our news model has not shown it can tell good news from bad (held-out
# tests, and on 2026-11-25 it read "insider stock sales" as positive and made a losing trade 1.26x),
# so it may only take risk away, never add it.
CONF_STEP = 0.15        # confidence cut at full disagreement: decides which setups are posted
SIZE_STEP = 1.0         # size cut per unit of disagreement
SIZE_MIN, SIZE_MAX = 0.25, 1.0
CAUTION_AT = -0.4       # this much against the trade: cut to CAUTION_SIZE
CAUTION_SIZE = 0.25
LEVELS = ((0.6, "strong"), (0.35, "moderate"), (0.15, "weak"))

_finbert = None          # False once it has failed to load
_ml = None               # our news model; False once it has failed to load
ML_WEIGHT = 1.0          # share of the news model in each headline's score (the rest is FinBERT).
                         # 1.0 from 2026-11-25: our model only, FinBERT just a fallback (user's trial)
_seen = {}               # article id -> score, so each headline is read once per run


def strength(score: float) -> str:
    return next((name for cut, name in LEVELS if abs(score) >= cut), "none")


def method() -> str:
    """Which readers score headlines (after the first call)."""
    if _ml and ML_WEIGHT >= 1:
        return "news-model"
    return ("finbert" if _finbert else "keywords") + ("+news-model" if _ml else "")


def _load_ml():
    global _ml
    if _ml is None:
        try:
            import joblib
            from backend.research.news_ml_text import embed, event_flags
            _ml = (joblib.load(config.ROOT / "backend" / "bot" / "models" / "news_text.joblib"), embed, event_flags)
            embed(["warm up"])
            log.info("news model loaded")
        except Exception as exc:
            _ml = False
            log.warning("news model unavailable (%s); FinBERT only", str(exc)[:120])
    return _ml


def model_scores(texts: list, times: list) -> np.ndarray:
    """Our news model's reading of each headline in [-1, 1] (0 if it is unavailable). Headlines from
    the session use the intraday model, the rest the overnight one."""
    ml = _load_ml()
    if not ml or not texts:
        return np.zeros(len(texts))
    models, embed, event_flags = ml
    X = np.hstack([embed(texts), event_flags(pd.Series(texts)).to_numpy(np.float32)])
    out = np.zeros(len(texts))
    for i, t in enumerate(times):
        t = pd.Timestamp(t).tz_convert(ET)
        kind = "intraday" if t.weekday() < 5 and 570 <= t.hour * 60 + t.minute < 930 else "overnight"
        m = models[kind]
        out[i] = np.clip(m["text"].predict(X[i:i + 1])[0] / m["scale_bps"] * 0.6, -1, 1)
    return out


def read_articles(texts: list, times: list) -> np.ndarray:
    """Each headline's score: our news model and FinBERT, mixed by ML_WEIGHT. FinBERT is only loaded
    when it has a share, or when the news model cannot load."""
    ml = model_scores(texts, times)
    if _ml and ML_WEIGHT >= 1:
        return ml
    fb = _score_texts(texts)
    return fb if not _ml else (1 - ML_WEIGHT) * fb + ML_WEIGHT * ml


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
                for i, s in zip(new, read_articles(news.loc[new, "headline"].tolist(), news.loc[new, "created_at"].tolist())):
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
    # the engine has already moved the signal's confidence by the news (Engine.news_score)
    return {"score": score, "agreement": a, "strength": strength(score), "size": size_for(a),
            "confidence": float(sig.get("confidence", 0))}


def code(a: float) -> str:
    """Agreement for an order id ('-' separates the id's parts): p42 = +0.42, m30 = -0.30."""
    v = int(round(abs(a) * 100))
    return f"{'p' if a >= 0 else 'm'}{min(v, 99):02d}"


def decode(c: str) -> float:
    return (1 if c[0] == "p" else -1) * int(c[1:]) / 100


def size_for(a: float, size_max: float = None) -> float:
    top = SIZE_MAX if size_max is None else size_max
    return CAUTION_SIZE if a <= CAUTION_AT else float(np.clip(1 + SIZE_STEP * a, SIZE_MIN, top))


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
