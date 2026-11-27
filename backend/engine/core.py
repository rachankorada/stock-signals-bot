"""The signal engine.

`on_minute(now)` is called once per completed minute, by the live loop or by a
replay. It first updates open signals against the newest bars, then scans each
timeframe whose bar has just closed. It returns the events to notify.
"""
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import joblib
import numpy as np
import pandas as pd

from backend import config, db
from backend.backtest.simulator import make_levels, side_cost
from backend.data.bars import SESSION_LEN, SESSION_OPEN_MIN, resample
from backend.features.build import FEATURES, compute_features, context_features
from backend.model.confidence import ConfidenceModel  # noqa: F401  (needed to load saved models)
from backend.strategies.library import LABELS, find_setups

log = logging.getLogger("engine")
WARMUP_SESSIONS = 22  # enough for the 20-session relative-volume baseline
UTC = "UTC"


def iso(ts: pd.Timestamp) -> str:
    return ts.tz_convert(UTC).isoformat()


def describe(row: pd.Series, direction: int) -> list:
    """Plain-language facts behind a signal."""
    out = []
    if np.isfinite(row["vwap_dist"]):
        out.append("Price above VWAP" if row["vwap_dist"] > 0 else "Price below VWAP")
    if np.isfinite(row["htf_trend"]):
        out.append("Higher-timeframe trend up" if row["htf_trend"] > 0 else "Higher-timeframe trend down")
    if np.isfinite(row["rvol"]):
        out.append(f"Relative volume {row['rvol']:.1f}x")
    if np.isfinite(row["adx"]):
        out.append(f"Trend strength (ADX) {row['adx']:.0f}")
    if np.isfinite(row["rsi14"]):
        out.append(f"RSI {row['rsi14']:.0f}")
    if np.isfinite(row["spy_vwap_dist"]):
        out.append("SPY above its VWAP" if row["spy_vwap_dist"] > 0 else "SPY below its VWAP")
    return out


def context_task(hist: pd.DataFrame, tf: int) -> pd.DataFrame:
    return context_features(resample(hist, tf), tf)


def find_task(sym: str, timeframes: list, hist: pd.DataFrame, ctx: dict, now: pd.Timestamp) -> list:
    """Features and setups for one ticker's newest bars. Runs in a worker process when live."""
    hits = []
    for tf in timeframes:
        bars = resample(hist, tf)
        if bars.index[-1] != now - pd.Timedelta(minutes=tf):  # no fresh bar for this ticker
            continue
        f = compute_features(bars, tf, ctx[tf])
        setups = find_setups(f, tf, last_only=True)
        if not setups.empty:
            hits.append((sym, tf, f.iloc[-1], setups))
    return hits


def _warm(_):
    return os.getpid()  # importing this module in the worker is the point


def make_pool():
    """Worker processes so a full watchlist is scanned within a second or two of the bar closing."""
    workers = min(6, (os.cpu_count() or 2) - 1)
    if workers < 2:
        return None
    pool = ProcessPoolExecutor(max_workers=workers)
    list(pool.map(_warm, range(workers * 4)))  # start the workers now, not on the first signal
    return pool


def news_reason(news: float, direction: int) -> list:
    from backend.bot.news_bot import strength
    level = strength(news)
    if level == "none":
        return []
    return [f"{level.capitalize()} {'positive' if news > 0 else 'negative'} news "
            f"({'with' if news * direction > 0 else 'against'} this trade)"]


class Engine:
    def __init__(self, pool=None):
        self.pool = pool  # optional process pool; scans run inline without one
        self.news_score = None   # optional callable(symbol) -> news score in [-1, 1]; moves confidence
        self.bars: dict[str, pd.DataFrame] = {}
        self.models: dict[int, tuple] = {}
        self.last_scan: str | None = None
        self.load_models()

    def load_models(self):
        self.models = {}
        for tf in config.TIMEFRAMES:
            model_path = config.MODEL_DIR / f"tf{tf}.joblib"
            meta_path = config.MODEL_DIR / f"meta_tf{tf}.json"
            if model_path.exists() and meta_path.exists():
                try:
                    self.models[tf] = (joblib.load(model_path), json.loads(meta_path.read_text()))
                except Exception:  # a broken model file must not stop the app
                    log.exception("could not load the %s-minute model", tf)

    def model_summary(self) -> list:
        return [meta for _, meta in self.models.values()]

    def add_bars(self, symbol: str, df: pd.DataFrame):
        """Merge new 1-minute bars into the in-memory history."""
        if df is None or df.empty:
            return
        old = self.bars.get(symbol)
        merged = df if old is None else pd.concat([old, df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        self.bars[symbol] = merged.iloc[-(WARMUP_SESSIONS + 2) * SESSION_LEN:]

    def history(self, symbol: str, now: pd.Timestamp) -> pd.DataFrame:
        """Completed 1-minute bars before `now`."""
        df = self.bars.get(symbol)
        if df is None:
            return pd.DataFrame()
        return df[df.index < now].iloc[-WARMUP_SESSIONS * SESSION_LEN:]

    # ---- main entry point -------------------------------------------------

    def on_minute(self, now: pd.Timestamp, symbols: list) -> list:
        events = self.update_open_signals(now)
        k = now.hour * 60 + now.minute - SESSION_OPEN_MIN  # minutes since the open
        if 0 < k <= SESSION_LEN:
            due = [tf for tf in config.TIMEFRAMES if k % tf == 0 and tf in self.models]
            events += self.scan(now, due, symbols)
        self.last_scan = iso(now)
        return events

    # ---- signal detection -------------------------------------------------

    def _run(self, fn, *columns) -> list:
        if not columns[0]:
            return []
        return list(self.pool.map(fn, *columns) if self.pool else map(fn, *columns))

    def scan(self, now: pd.Timestamp, timeframes: list, symbols: list) -> list:
        """Check every ticker on every timeframe whose bar just closed."""
        names = [n for n in config.CONTEXT_TICKERS if len(self.history(n, now))]
        jobs = [(name, tf) for tf in timeframes for name in names]
        frames = self._run(context_task, [self.history(n, now) for n, _ in jobs], [tf for _, tf in jobs])
        ctx = {tf: {} for tf in timeframes}
        for (name, tf), frame in zip(jobs, frames):
            ctx[tf][name.lower()] = frame

        work = [(sym, hist) for sym in symbols if len(hist := self.history(sym, now))]
        n = len(work)
        found = self._run(find_task, [w[0] for w in work], [timeframes] * n, [w[1] for w in work], [ctx] * n, [now] * n)
        events = []
        for sym, tf, last, setups in (hit for hits in found for hit in hits):
            sig = self.build_signal(sym, tf, now, last, setups, *self.models[tf])
            if sig:
                events.append({"type": "signal", "signal": db.insert_signal(sig)})
        return events

    def build_signal(self, sym, tf, now, last, setups, model, meta):
        entry, atr = float(last["close"]), float(last["atr"])
        d = setups["direction"].to_numpy()
        stop, tp1, tp2, risk = make_levels(entry, setups["stop_ref"].to_numpy(dtype=float), atr, d)
        cost_r = 2 * side_cost(entry) / risk
        ds = pd.DataFrame([last[FEATURES].to_dict()] * len(setups))
        ds["strategy"], ds["direction"] = setups["strategy"].to_numpy(), d
        ds["stop_atr"], ds["cost_r"] = risk / atr, cost_r
        ds["n_agree"] = setups.groupby("direction")["strategy"].transform("count").to_numpy()
        conf = model.predict(ds)
        news = float(self.news_score(sym) or 0.0) if self.news_score else 0.0
        if news:   # news against a setup lowers its confidence; news with it changes nothing (news_bot)
            from backend.bot.news_bot import CONF_STEP, acting
            news = acting(news)
            conf = conf + CONF_STEP * np.minimum(d * news, 0.0)

        # A setup is validated when its timeframe and strategy passed the go-live gate
        p_min = meta.get("p_min")
        enabled = np.array([meta["strategies"].get(s, {}).get("enabled", False) for s in ds["strategy"]])
        postable = conf >= config.MIN_POST_CONFIDENCE  # site rule: nothing below this is posted
        validated = postable & enabled & bool(meta.get("validated")) & (conf >= (p_min or 1.0))
        if validated.any():
            pick = int(np.argmax(np.where(validated, conf, -1)))
        elif config.ALLOW_UNVALIDATED and postable.any():
            pick = int(np.argmax(conf))  # test mode: posted, but labelled unvalidated
        else:
            return None

        direction = int(d[pick])
        open_now = db.open_signals(sym)
        if any(s["direction"] != direction for s in open_now):
            return None  # never opposing signals on one ticker
        if sum(s["timeframe"] == tf for s in open_now) >= config.MAX_STACKED_SIGNALS:
            return None  # one open signal per ticker and timeframe (more when stacking is allowed)

        strategy = ds["strategy"].iloc[pick]
        agreeing = [LABELS[s] for s, dd in zip(ds["strategy"], d) if dd == direction and s != strategy]
        flatten = now.normalize() + pd.Timedelta(minutes=SESSION_OPEN_MIN + SESSION_LEN - config.FLATTEN_BEFORE_CLOSE_MIN)
        expires = min(now + pd.Timedelta(minutes=tf * config.MAX_HOLD_BARS[tf]), flatten)
        r = float(risk[pick])
        return {
            "symbol": sym, "timeframe": tf, "direction": direction, "strategy": strategy,
            "agreeing": agreeing, "entry": round(entry, 4),
            "entry_max": round(entry + direction * 0.25 * r, 4),
            "stop": round(float(stop[pick]), 4), "tp1": round(float(tp1[pick]), 4),
            "tp2": round(float(tp2[pick]), 4), "risk": round(r, 4), "cost_r": round(float(cost_r[pick]), 4),
            "confidence": round(float(conf[pick]), 4),
            "reasons": describe(last, direction) + news_reason(news, direction),
            "history": meta.get("history", {}).get(strategy, {}), "validated": int(bool(validated[pick])),
            "status": "active", "created_at": iso(now), "expires_at": iso(expires), "checked_until": iso(now),
        }

    # ---- signal lifecycle -------------------------------------------------

    def update_open_signals(self, now: pd.Timestamp) -> list:
        events = []
        for sig in db.open_signals():
            df = self.bars.get(sig["symbol"])
            if df is None:
                continue
            since = pd.Timestamp(sig["checked_until"]).tz_convert(now.tz)
            new_bars = df[(df.index >= since) & (df.index < now)]
            before = sig["status"]
            fields = self.advance(sig, new_bars, now)
            fields["checked_until"] = iso(now)
            updated = db.update_signal(sig["id"], **fields)
            if updated["status"] != before:
                events.append({"type": "update", "signal": updated})
        return events

    def advance(self, sig: dict, bars: pd.DataFrame, now: pd.Timestamp) -> dict:
        """Walk an open signal through new bars. Returns the fields that changed."""
        d, status, out = sig["direction"], sig["status"], {}
        entry, risk = sig["entry"], sig["risk"]
        for ts, bar in bars.iterrows():
            closed_at = iso(ts + pd.Timedelta(minutes=1))
            hit = (lambda level: bar["high"] >= level) if d > 0 else (lambda level: bar["low"] <= level)
            lost = (lambda level: bar["low"] <= level) if d > 0 else (lambda level: bar["high"] >= level)
            if status == "active":
                if lost(sig["stop"]):  # stop first when both are touched in one bar
                    return {"status": "stopped", "closed_at": closed_at, "exit_price": sig["stop"],
                            "result_r": round(-1 - sig["cost_r"], 3)}
                if hit(sig["tp1"]):
                    status = "tp1_hit"
                    out = {"status": status, "exit_price": sig["tp1"],
                           "result_r": round(config.TP1_R - sig["cost_r"], 3)}
                    if hit(sig["tp2"]):
                        return {**out, "status": "tp2_hit", "closed_at": closed_at}
                    continue
            if status == "tp1_hit":
                # The result is booked at the first target; this only tracks the runner
                if hit(sig["tp2"]):
                    return {**out, "status": "tp2_hit", "closed_at": closed_at}
                if lost(entry):
                    return {**out, "status": "tp1_closed", "closed_at": closed_at}
        if now >= pd.Timestamp(sig["expires_at"]).tz_convert(now.tz):
            if status == "tp1_hit":
                return {**out, "status": "tp1_closed", "closed_at": iso(now)}
            df = self.bars[sig["symbol"]]
            past = df[df.index < now]
            last_px = float(past["close"].iloc[-1]) if len(past) else entry
            return {"status": "expired", "closed_at": iso(now), "exit_price": last_px,
                    "result_r": round(d * (last_px - entry) / risk - sig["cost_r"], 3)}
        return out
