"""Score news headlines with FinBERT, a free model trained on financial text.

Each headline gets a score from -1 (clearly bad for the stock) to +1 (clearly
good): the model's probability of "positive" minus "negative". Scores are
saved by article id, so each headline is only ever scored once.

Run:  python -m backend.data.news_sentiment
"""
import time

import numpy as np
import pandas as pd

from backend import config
from backend.data.news import NEWS_DIR, load_news
from backend.features.news import FOCUS_MAX_SYMBOLS

MODEL_NAME = "ProsusAI/finbert"
SCORES_PATH = NEWS_DIR / "_finbert_scores.parquet"
BATCH = 64
MAX_TOKENS = 64

_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
        labels = {v.lower(): k for k, v in model.config.id2label.items()}
        _pipeline = (torch, tokenizer, model, labels["positive"], labels["negative"])
    return _pipeline


def score_texts(texts: list) -> np.ndarray:
    """FinBERT score in [-1, 1] for each headline."""
    torch, tokenizer, model, pos, neg = _load()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), BATCH):
            enc = tokenizer(texts[i:i + BATCH], padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            prob = torch.softmax(model(**enc).logits, dim=-1)
            out.append((prob[:, pos] - prob[:, neg]).numpy())
    return np.concatenate(out) if out else np.array([])


def load_scores() -> pd.Series:
    if SCORES_PATH.exists():
        return pd.read_parquet(SCORES_PATH)["finbert"]
    return pd.Series(dtype="float64", name="finbert")


def score_all(symbols=None, log_every: int = 50) -> pd.Series:
    """Score every not-yet-scored headline that is focused on one or two tickers."""
    news = pd.concat([load_news(s) for s in symbols or config.TRAIN_TICKERS])
    news = news[(news["n_symbols"] <= FOCUS_MAX_SYMBOLS) & news["headline"].notna()]
    news = news[~news.index.duplicated()]
    scores = load_scores()
    todo = news[~news.index.isin(scores.index)]
    print(f"{len(news):,} focused headlines, {len(todo):,} to score", flush=True)
    t0, chunk = time.time(), BATCH * log_every
    for i in range(0, len(todo), chunk):
        part = todo.iloc[i:i + chunk]
        new = pd.Series(score_texts(part["headline"].tolist()), index=part.index, name="finbert")
        scores = pd.concat([scores, new])
        scores.to_frame().to_parquet(SCORES_PATH)  # save as we go so a stop loses little
        done = i + len(part)
        print(f"  {done:,}/{len(todo):,}  {done / (time.time() - t0):.0f} headlines/s", flush=True)
    return scores


if __name__ == "__main__":
    score_all()
