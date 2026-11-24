"""The news model's text reader: turns headlines into numbers a model can learn from.

  meaning    a 384-number summary of each headline from a small free language model
             (sentence-transformers/all-MiniLM-L6-v2, runs locally), so "CEO reinstated" lands near
             "founder returns as chief executive" even with no shared words
  events     flags for event types (CEO change, guidance, earnings beat/miss, FDA, deal, ...)
  novelty    how much the headline repeats the ticker's headlines of the previous 24 hours
"""
import re

import numpy as np
import pandas as pd

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EVENTS = {
    "ceo": r"\b(ceo|chief executive|president and ceo|founder|steps? down|resign\w*|reinstat\w*|ousted|appoint\w*|names? .* (ceo|cfo))\b",
    "guidance": r"\b(guidance|outlook|forecast|raises? .*(view|outlook|forecast)|cuts? .*(view|outlook|forecast)|reaffirm\w*)\b",
    "beat": r"\b(beats?|tops?|above (estimates|expectations)|better[- ]than[- ]expected|record (revenue|quarter|sales))\b",
    "miss": r"\b(miss(es|ed)?|below (estimates|expectations)|worse[- ]than[- ]expected|shortfall)\b",
    "earnings": r"\b(earnings|eps|q[1-4]\b|quarter(ly)? results|revenue)\b",
    "upgrade": r"\b(upgrad\w+|raises? (price )?target|outperform|overweight|buy rating|initiat\w+ .*buy)\b",
    "downgrade": r"\b(downgrad\w+|cuts? (price )?target|lowers? (price )?target|underperform|underweight|sell rating)\b",
    "fda": r"\b(fda|approv\w+|clearance|phase [123i]+|trial (results|data)|breakthrough designation)\b",
    "deal": r"\b(acquir\w+|acquisition|merger|to buy|takeover|buyout|deal|stake)\b",
    "legal": r"\b(lawsuit|sued|probe|investigat\w+|sec |doj|antitrust|settle\w*|fine[ds]?|subpoena|fraud)\b",
    "capital": r"\b(buyback|repurchase|dividend|offering|stock split|share sale|convertible|notes)\b",
    "contract": r"\b(contract|order|partnership|partners? with|collaborat\w+|wins?|award\w*)\b",
    "macro": r"\b(tariff\w*|fed\b|rates?|inflation|export|ban|sanction\w*|government|trump|china)\b",
    "move": r"\b(shares? (rise|fall|jump|drop|surge|slide|tumble|gain)\w*|stock (rises|falls|jumps|drops|surges)|trading (higher|lower)|movers?)\b",
}
_rx = {k: re.compile(v, re.I) for k, v in EVENTS.items()}


def event_flags(headlines: pd.Series) -> pd.DataFrame:
    h = headlines.fillna("")
    return pd.DataFrame({f"ev_{k}": h.str.contains(rx).astype(np.int8) for k, rx in _rx.items()}, index=headlines.index)


_enc = None


def embed(texts: list, batch: int = 256) -> np.ndarray:
    """Unit-length meaning vectors (mean of the model's token vectors)."""
    global _enc
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _enc is None:
        _enc = (AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval())
    tok, model = _enc
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=48, return_tensors="pt")
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(torch.nn.functional.normalize(v, dim=1).numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def novelty(symbols: np.ndarray, times: np.ndarray, vecs: np.ndarray, hours: int = 24) -> tuple:
    """For each headline: highest similarity to the same ticker's headlines of the previous `hours`, and
    how many there were. Rows must be sorted by time."""
    sim = np.zeros(len(vecs), np.float32); cnt = np.zeros(len(vecs), np.int32)
    win = np.timedelta64(hours, "h")
    for sym in np.unique(symbols):
        ix = np.where(symbols == sym)[0]
        t = times[ix]
        lo = np.searchsorted(t, t - win, side="left")
        for k, j in enumerate(ix):
            prev = ix[lo[k]:k]
            if len(prev):
                sim[j] = float((vecs[prev] @ vecs[j]).max()); cnt[j] = len(prev)
    return sim, cnt
