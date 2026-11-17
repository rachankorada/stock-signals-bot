"""Web API, live alert socket and static site. Also hosts the live engine."""
import asyncio
import json
import logging
import os
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import config, db
from backend.data.alpaca_data import ET
from backend.data.bars import resample
from backend.engine import paper, push
from backend.engine.core import Engine, make_pool
from backend.engine.live import LiveRunner, fetch_many
from backend.strategies.library import LABELS

log = logging.getLogger("api")
DIST = config.ROOT / "frontend" / "dist"
TOKEN_DAYS = 30
NOTIFY_STATUSES = {"tp1_hit", "tp2_hit", "stopped", "expired"}


def jwt_secret() -> str:
    secret = config.JWT_SECRET or db.kv_get("jwt_secret")
    if not secret:
        secret = secrets.token_urlsafe(48)
        db.kv_set("jwt_secret", secret)
    return secret


# ---- live alert fan-out ----------------------------------------------------

def wants(settings: dict, sig: dict) -> bool:
    """Does this user's settings allow this signal?"""
    if sig["timeframe"] not in settings["timeframes"]:
        return False
    if settings["long_only"] and sig["direction"] < 0:
        return False
    return sig["confidence"] * 100 >= max(settings["min_confidence"], config.MIN_POST_CONFIDENCE * 100)


class Hub:
    def __init__(self):
        self.sockets: dict[int, set] = {}

    def add(self, user_id: int, ws: WebSocket):
        self.sockets.setdefault(user_id, set()).add(ws)

    def remove(self, user_id: int, ws: WebSocket):
        self.sockets.get(user_id, set()).discard(ws)

    async def dispatch(self, event: dict):
        """Send an engine event to every user watching that ticker."""
        sig = event["signal"]
        if event["type"] == "signal" and config.PAPER_TRADE:
            await asyncio.get_running_loop().run_in_executor(None, paper.place, sig)
        users = db.query(
            "SELECT u.* FROM users u JOIN watchlist w ON w.user_id = u.id WHERE w.symbol = ? AND w.enabled = 1",
            (sig["symbol"],))
        for user in users:
            settings = db.user_settings(user)
            if not wants(settings, sig):
                continue
            for ws in list(self.sockets.get(user["id"], ())):
                try:
                    await ws.send_json(event)
                except Exception:
                    self.remove(user["id"], ws)
            if settings["push"] and (event["type"] == "signal" or sig["status"] in NOTIFY_STATUSES):
                asyncio.get_running_loop().run_in_executor(None, push.send, user["id"], push.format_signal(event))


hub = Hub()
engine: Engine = None
runner: LiveRunner = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, runner
    db.connect()
    live = os.environ.get("ENGINE", "on") != "off"
    pool = make_pool() if live else None
    engine = Engine(pool)
    runner = LiveRunner(engine, hub.dispatch)
    task = asyncio.create_task(runner.run()) if live else None
    yield
    if task:
        task.cancel()
    if pool:
        pool.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="Stock Analyzer", lifespan=lifespan)


# ---- auth ------------------------------------------------------------------

class Credentials(BaseModel):
    email: str
    password: str


def make_token(user_id: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=TOKEN_DAYS)
    return jwt.encode({"sub": str(user_id), "exp": exp}, jwt_secret(), algorithm="HS256")


def user_from_token(token: str):
    try:
        user_id = int(jwt.decode(token, jwt_secret(), algorithms=["HS256"])["sub"])
    except Exception:
        return None
    rows = db.query("SELECT * FROM users WHERE id = ?", (user_id,))
    return rows[0] if rows else None


def current_user(authorization: str = Header(default="")):
    user = user_from_token(authorization.removeprefix("Bearer ").strip())
    if not user:
        raise HTTPException(401, "Please sign in again.")
    return user


def public_user(user: dict) -> dict:
    return {"id": user["id"], "email": user["email"], "settings": db.user_settings(user)}


@app.post("/api/auth/register")
def register(body: Credentials):
    email = body.email.strip().lower()
    if "@" not in email or len(email) < 5:
        raise HTTPException(400, "Enter a valid email address.")
    if len(body.password) < 8:
        raise HTTPException(400, "Use a password of at least 8 characters.")
    if db.query("SELECT 1 FROM users WHERE email = ?", (email,)):
        raise HTTPException(409, "An account with this email already exists. Sign in instead.")
    pw_hash = bcrypt.hashpw(body.password.encode()[:72], bcrypt.gensalt()).decode()
    user_id = db.execute(
        "INSERT INTO users (email, pw_hash, settings, created_at) VALUES (?, ?, ?, ?)",
        (email, pw_hash, json.dumps(db.DEFAULT_SETTINGS), datetime.now(timezone.utc).isoformat()))
    return {"token": make_token(user_id)}


@app.post("/api/auth/login")
def login(body: Credentials):
    rows = db.query("SELECT * FROM users WHERE email = ?", (body.email.strip().lower(),))
    if not rows or not bcrypt.checkpw(body.password.encode()[:72], rows[0]["pw_hash"].encode()):
        raise HTTPException(401, "Email or password is incorrect.")
    return {"token": make_token(rows[0]["id"])}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return public_user(user)


class Settings(BaseModel):
    min_confidence: float = 0
    timeframes: list[int] = [1, 3, 5, 15]
    long_only: bool = False
    push: bool = True
    sound: bool = True


@app.put("/api/settings")
def save_settings(body: Settings, user=Depends(current_user)):
    data = body.model_dump()
    data["timeframes"] = [t for t in data["timeframes"] if t in config.TIMEFRAMES]
    data["min_confidence"] = max(0, min(95, data["min_confidence"]))
    db.execute("UPDATE users SET settings = ? WHERE id = ?", (json.dumps(data), user["id"]))
    return data


# ---- watchlist -------------------------------------------------------------

class SymbolBody(BaseModel):
    symbol: str


class EnabledBody(BaseModel):
    enabled: bool


def user_symbols(user_id: int, enabled_only=False) -> list:
    sql = "SELECT symbol, enabled FROM watchlist WHERE user_id = ?" + (" AND enabled = 1" if enabled_only else "")
    return db.query(sql + " ORDER BY symbol", (user_id,))


@app.get("/api/watchlist")
def get_watchlist(user=Depends(current_user)):
    return {"items": user_symbols(user["id"]), "max": config.MAX_WATCHLIST_PER_USER}


@app.post("/api/watchlist")
def add_symbol(body: SymbolBody, user=Depends(current_user)):
    symbol = body.symbol.strip().upper()
    if not symbol or not symbol.replace(".", "").isalnum() or len(symbol) > 8:
        raise HTTPException(400, "Enter a ticker symbol such as AAPL.")
    mine = [r["symbol"] for r in user_symbols(user["id"])]
    if symbol in mine:
        raise HTTPException(409, f"{symbol} is already on your watchlist.")
    if len(mine) >= config.MAX_WATCHLIST_PER_USER:
        raise HTTPException(400, f"Your watchlist is full ({config.MAX_WATCHLIST_PER_USER} tickers). Remove one first.")
    watched = db.watched_symbols()
    if symbol not in watched and len(watched) >= config.MAX_LIVE_SYMBOLS:
        raise HTTPException(400, "The service is at its ticker limit. Try again later.")
    try:
        asset = runner.trading.get_asset(symbol)
    except Exception:
        raise HTTPException(404, f"{symbol} was not found. Check the ticker symbol.")
    if not asset.tradable or str(asset.asset_class.value) != "us_equity":
        raise HTTPException(400, f"{symbol} is not a tradable US stock or ETF.")
    db.execute("INSERT INTO watchlist (user_id, symbol, enabled) VALUES (?, ?, 1)", (user["id"], symbol))
    return {"symbol": symbol, "enabled": 1, "name": asset.name}


@app.patch("/api/watchlist/{symbol}")
def toggle_symbol(symbol: str, body: EnabledBody, user=Depends(current_user)):
    db.execute("UPDATE watchlist SET enabled = ? WHERE user_id = ? AND symbol = ?",
               (int(body.enabled), user["id"], symbol.upper()))
    return {"symbol": symbol.upper(), "enabled": int(body.enabled)}


@app.delete("/api/watchlist/{symbol}")
def remove_symbol(symbol: str, user=Depends(current_user)):
    db.execute("DELETE FROM watchlist WHERE user_id = ? AND symbol = ?", (user["id"], symbol.upper()))
    return {"ok": True}


# ---- signals ---------------------------------------------------------------

def visible_signals(user: dict, where: str = "", params: tuple = (), limit: int = 200) -> list:
    """Signals for the user's own tickers that pass their settings."""
    symbols = [r["symbol"] for r in user_symbols(user["id"])]
    if not symbols:
        return []
    marks = ", ".join("?" for _ in symbols)
    rows = db.query(
        f"SELECT * FROM signals WHERE symbol IN ({marks}) {where} ORDER BY created_at DESC LIMIT ?",
        (*symbols, *params, limit))
    settings = db.user_settings(user)
    return [s for s in map(db.signal_row, rows) if wants(settings, s)]


def with_label(sig: dict) -> dict:
    sig["strategy_label"] = LABELS.get(sig["strategy"], sig["strategy"])
    return sig


@app.get("/api/signals")
def list_signals(status: str = "all", symbol: str = "", limit: int = 100, user=Depends(current_user)):
    where, params = "", ()
    if status == "open":
        where += " AND status IN ('active', 'tp1_hit')"
    elif status == "closed":
        where += " AND status NOT IN ('active', 'tp1_hit')"
    if symbol:
        where, params = where + " AND symbol = ?", (symbol.upper(),)
    return [with_label(s) for s in visible_signals(user, where, params, min(limit, 500))]


@app.get("/api/signals/{signal_id}")
def one_signal(signal_id: int, user=Depends(current_user)):
    sig = db.get_signal(signal_id)
    if not sig or sig["symbol"] not in [r["symbol"] for r in user_symbols(user["id"])]:
        raise HTTPException(404, "Signal not found.")
    return with_label(sig)


def _group(df: pd.DataFrame, key) -> list:
    out = []
    for name, g in df.groupby(key, observed=True):
        out.append({"key": str(name), "trades": int(len(g)), "win_rate": round(float((g["result_r"] > 0).mean()), 3),
                    "expectancy_r": round(float(g["result_r"].mean()), 3), "total_r": round(float(g["result_r"].sum()), 2)})
    return out


@app.get("/api/stats")
def stats(user=Depends(current_user)):
    """Track record of every finished signal on the user's tickers."""
    rows = [s for s in visible_signals(user, " AND result_r IS NOT NULL AND status != 'active'", (), 5000)]
    if not rows:
        return {"trades": 0}
    df = pd.DataFrame(rows).sort_values("created_at")
    df["strategy"] = df["strategy"].map(lambda s: LABELS.get(s, s))
    df["bucket"] = pd.cut(df["confidence"] * 100, [0, 40, 50, 60, 70, 80, 100],
                          labels=["under 40%", "40-50%", "50-60%", "60-70%", "70-80%", "80%+"])
    r = df["result_r"]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "trades": int(len(df)), "win_rate": round(float((r > 0).mean()), 3),
        "expectancy_r": round(float(r.mean()), 3), "total_r": round(float(r.sum()), 2),
        "profit_factor": round(float(gains / losses), 2) if losses > 0 else None,
        "by_strategy": _group(df, "strategy"), "by_timeframe": _group(df, "timeframe"),
        "by_confidence": _group(df, "bucket"), "by_side": _group(df, "side"),
        "equity": [{"time": t, "cum_r": round(float(v), 2)} for t, v in zip(df["created_at"], r.cumsum())],
    }


# ---- market data and status ------------------------------------------------

def wall_epoch(ts: pd.Timestamp) -> int:
    """Exchange wall-clock time as an epoch, so charts show New York time."""
    return int(ts.tz_convert(ET).tz_localize(None).timestamp())


@app.get("/api/bars/{symbol}")
def bars(symbol: str, tf: int = 5, user=Depends(current_user)):
    symbol = symbol.upper()
    if tf not in config.TIMEFRAMES:
        raise HTTPException(400, "Unsupported timeframe.")
    df = engine.bars.get(symbol)
    if df is None or df.empty:
        df = fetch_many([symbol], datetime.now(timezone.utc) - timedelta(days=7)).get(symbol)
    if df is None or df.empty:
        return {"bars": [], "signals": []}
    sessions = {1: 2, 3: 3, 5: 5, 15: 10}[tf]
    days = sorted(set(df.index.normalize()))[-sessions:]
    out = resample(df[df.index >= days[0]], tf)
    candles = [{"time": wall_epoch(ts), "open": row.open, "high": row.high, "low": row.low,
                "close": row.close, "volume": row.volume} for ts, row in out.iterrows()]
    marks = [{**with_label(s), "time": wall_epoch(pd.Timestamp(s["created_at"]))}
             for s in visible_signals(user, " AND symbol = ?", (symbol,), 100)]
    return {"bars": candles, "signals": marks}


@app.get("/api/market")
def market(user=Depends(current_user)):
    return {**runner.status(), "models": [
        {"timeframe": m["timeframe"], "validated": m["validated"], "p_min": m["p_min"],
         "trained_through": m["trained_through"], "backtest": m["backtest"], "geometry": m["geometry"],
         "strategies": {LABELS[k]: v for k, v in m["strategies"].items()},
         "calibration": m["calibration"], "raw": m["raw"]}
        for m in engine.model_summary()],
        "allow_unvalidated": config.ALLOW_UNVALIDATED, "min_post_confidence": config.MIN_POST_CONFIDENCE}


# ---- push notifications ----------------------------------------------------

@app.get("/api/push/key")
def push_key():
    return {"key": push.vapid_keys()[1]}


@app.post("/api/push/subscribe")
def push_subscribe(sub: dict, user=Depends(current_user)):
    if not str(sub.get("endpoint", "")).startswith("https://"):
        raise HTTPException(400, "Invalid push subscription.")
    db.execute(
        "INSERT INTO push_subs (user_id, endpoint, sub) VALUES (?, ?, ?) "
        "ON CONFLICT(endpoint) DO UPDATE SET user_id = excluded.user_id, sub = excluded.sub",
        (user["id"], sub["endpoint"], json.dumps(sub)))
    return {"ok": True}


@app.post("/api/push/test")
def push_test(user=Depends(current_user)):
    sent = push.send(user["id"], {"title": "Stock Analyzer", "body": "Device notifications are working.",
                                  "tag": "test", "url": "/"})
    return {"sent": sent}


# ---- live socket -----------------------------------------------------------

@app.websocket("/ws")
async def socket(ws: WebSocket, token: str = ""):
    user = user_from_token(token)
    if not user:
        await ws.close(code=4401)
        return
    await ws.accept()
    hub.add(user["id"], ws)
    try:
        while True:
            await ws.receive_text()  # keep-alive pings from the page
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(user["id"], ws)


# ---- static site -----------------------------------------------------------

if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        file = DIST / path
        if path and file.is_file() and DIST in file.resolve().parents:
            return FileResponse(file)
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})
