"""SQLite storage for users, watchlists, signals and push subscriptions."""
import json
import sqlite3
import threading
from contextlib import contextmanager

from backend import config

_lock = threading.Lock()
_conn = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL,
    settings TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS watchlist (
    user_id INTEGER NOT NULL, symbol TEXT NOT NULL, enabled INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (user_id, symbol));
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY, symbol TEXT NOT NULL, timeframe INTEGER NOT NULL,
    direction INTEGER NOT NULL, strategy TEXT NOT NULL, agreeing TEXT NOT NULL,
    entry REAL NOT NULL, entry_max REAL NOT NULL, stop REAL NOT NULL,
    tp1 REAL NOT NULL, tp2 REAL NOT NULL, risk REAL NOT NULL, cost_r REAL NOT NULL,
    confidence REAL NOT NULL, reasons TEXT NOT NULL, history TEXT NOT NULL,
    validated INTEGER NOT NULL, status TEXT NOT NULL,
    created_at TEXT NOT NULL, expires_at TEXT NOT NULL, checked_until TEXT NOT NULL,
    closed_at TEXT, exit_price REAL, result_r REAL);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol, created_at);
CREATE INDEX IF NOT EXISTS idx_signals_status ON signals (status);
CREATE TABLE IF NOT EXISTS push_subs (
    id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, endpoint TEXT UNIQUE NOT NULL, sub TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

DEFAULT_SETTINGS = {
    "min_confidence": 0,          # 0 = the site minimum (config.MIN_POST_CONFIDENCE)
    "timeframes": [1, 3, 5, 15],
    "long_only": False,
    "push": True,
    "sound": True,
}
OPEN_STATUSES = ("active", "tp1_hit")
JSON_FIELDS = ("agreeing", "reasons", "history")


def connect(path=None):
    global _conn
    _conn = sqlite3.connect(str(path or config.DB_PATH), check_same_thread=False)
    _conn.row_factory = sqlite3.Row
    _conn.executescript(SCHEMA)
    return _conn


@contextmanager
def cursor():
    if _conn is None:
        connect()
    with _lock:
        cur = _conn.cursor()
        try:
            yield cur
            _conn.commit()
        finally:
            cur.close()


def query(sql, params=()):
    with cursor() as cur:
        return [dict(r) for r in cur.execute(sql, params).fetchall()]


def execute(sql, params=()):
    with cursor() as cur:
        cur.execute(sql, params)
        return cur.lastrowid


def kv_get(key, default=None):
    rows = query("SELECT value FROM kv WHERE key = ?", (key,))
    return rows[0]["value"] if rows else default


def kv_set(key, value):
    execute("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))


def signal_row(row: dict) -> dict:
    """Decode a signals row for the API and the engine."""
    for k in JSON_FIELDS:
        row[k] = json.loads(row[k])
    row["side"] = "BUY" if row["direction"] > 0 else "SELL"
    return row


def insert_signal(sig: dict) -> dict:
    data = {k: (json.dumps(v) if k in JSON_FIELDS else v) for k, v in sig.items()}
    cols = ", ".join(data)
    marks = ", ".join("?" for _ in data)
    sid = execute(f"INSERT INTO signals ({cols}) VALUES ({marks})", tuple(data.values()))
    return get_signal(sid)


def get_signal(sid: int):
    rows = query("SELECT * FROM signals WHERE id = ?", (sid,))
    return signal_row(rows[0]) if rows else None


def open_signals(symbol=None):
    sql = "SELECT * FROM signals WHERE status IN ('active', 'tp1_hit')"
    params = ()
    if symbol:
        sql, params = sql + " AND symbol = ?", (symbol,)
    return [signal_row(r) for r in query(sql, params)]


def update_signal(sid: int, **fields):
    sets = ", ".join(f"{k} = ?" for k in fields)
    execute(f"UPDATE signals SET {sets} WHERE id = ?", (*fields.values(), sid))
    return get_signal(sid)


def watched_symbols() -> list:
    """Every enabled ticker on any user's watchlist."""
    return [r["symbol"] for r in query("SELECT DISTINCT symbol FROM watchlist WHERE enabled = 1 ORDER BY symbol")]


def user_settings(user: dict) -> dict:
    return {**DEFAULT_SETTINGS, **json.loads(user["settings"])}
