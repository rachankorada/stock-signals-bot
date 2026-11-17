"""API checks: accounts, per-user signal filtering and the live alert socket."""
import os

import pytest
from fastapi.testclient import TestClient

os.environ["ENGINE"] = "off"

from backend import config, db  # noqa: E402
from backend.api import main  # noqa: E402

SIGNAL = {
    "symbol": "NVDA", "timeframe": 5, "direction": -1, "strategy": "momentum_breakout", "agreeing": [],
    "entry": 100.0, "entry_max": 99.8, "stop": 101.0, "tp1": 98.5, "tp2": 97.5, "risk": 1.0, "cost_r": 0.03,
    "confidence": 0.65, "reasons": ["Price below VWAP"], "history": {}, "validated": 1, "status": "active",
    "created_at": "2026-11-13T15:00:00+00:00", "expires_at": "2026-11-13T17:00:00+00:00",
    "checked_until": "2026-11-13T15:00:00+00:00",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "api.db")
    with TestClient(main.app) as c:
        yield c


def sign_up(client, email) -> dict:
    res = client.post("/api/auth/register", json={"email": email, "password": "local-test-pass"})
    assert res.status_code == 200
    return {"Authorization": "Bearer " + res.json()["token"]}


def watch(email, symbol):
    user_id = db.query("SELECT id FROM users WHERE email = ?", (email,))[0]["id"]
    db.execute("INSERT INTO watchlist (user_id, symbol, enabled) VALUES (?, ?, 1)", (user_id, symbol))


def test_accounts(client):
    auth = sign_up(client, "a@example.test")
    assert client.get("/api/me", headers=auth).json()["email"] == "a@example.test"
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/auth/register", json={"email": "a@example.test", "password": "local-test-pass"}).status_code == 409
    assert client.post("/api/auth/login", json={"email": "a@example.test", "password": "wrong-password"}).status_code == 401
    assert client.post("/api/auth/register", json={"email": "b@example.test", "password": "short"}).status_code == 400


def test_signals_only_for_own_watchlist_and_settings(client):
    nvda_user, other_user = sign_up(client, "nvda@example.test"), sign_up(client, "other@example.test")
    watch("nvda@example.test", "NVDA")
    watch("other@example.test", "AAPL")
    sig = db.insert_signal(SIGNAL)

    assert [s["id"] for s in client.get("/api/signals", headers=nvda_user).json()] == [sig["id"]]
    assert client.get("/api/signals", headers=other_user).json() == []
    assert client.get(f"/api/signals/{sig['id']}", headers=other_user).status_code == 404

    # The signal is a short at 65% confidence on the 5-minute chart
    base = {"min_confidence": 0, "timeframes": [1, 3, 5, 15], "long_only": False, "push": True, "sound": True}
    for change in ({"long_only": True}, {"min_confidence": 70}, {"timeframes": [1, 15]}):
        client.put("/api/settings", headers=nvda_user, json={**base, **change})
        assert client.get("/api/signals", headers=nvda_user).json() == []
    client.put("/api/settings", headers=nvda_user, json=base)
    assert len(client.get("/api/signals", headers=nvda_user).json()) == 1


def test_signals_below_site_minimum_are_never_shown(client):
    auth = sign_up(client, "nvda@example.test")
    watch("nvda@example.test", "NVDA")
    db.insert_signal({**SIGNAL, "confidence": 0.49})
    assert client.get("/api/signals", headers=auth).json() == []
    db.insert_signal({**SIGNAL, "confidence": 0.50})
    assert [s["confidence"] for s in client.get("/api/signals", headers=auth).json()] == [0.50]


def test_live_socket_delivers_only_to_watchers(client):
    nvda_user, other_user = sign_up(client, "nvda@example.test"), sign_up(client, "other@example.test")
    watch("nvda@example.test", "NVDA")
    watch("other@example.test", "AAPL")
    token = lambda auth: auth["Authorization"].removeprefix("Bearer ")
    with client.websocket_connect(f"/ws?token={token(nvda_user)}") as mine, \
            client.websocket_connect(f"/ws?token={token(other_user)}") as theirs:
        sig = db.insert_signal(SIGNAL)
        client.portal.call(main.hub.dispatch, {"type": "signal", "signal": sig})
        event = mine.receive_json()
        assert event["type"] == "signal" and event["signal"]["symbol"] == "NVDA" and event["signal"]["side"] == "SELL"
        # The other user must get nothing: send them a marker event and check it arrives first
        aapl = db.insert_signal({**SIGNAL, "symbol": "AAPL"})
        client.portal.call(main.hub.dispatch, {"type": "signal", "signal": aapl})
        assert theirs.receive_json()["signal"]["symbol"] == "AAPL"


def test_stats_after_results(client):
    auth = sign_up(client, "nvda@example.test")
    watch("nvda@example.test", "NVDA")
    assert client.get("/api/stats", headers=auth).json() == {"trades": 0}
    for status, result in (("tp1_hit", 1.47), ("stopped", -1.03)):
        sig = db.insert_signal(SIGNAL)
        db.update_signal(sig["id"], status=status, result_r=result)
    stats = client.get("/api/stats", headers=auth).json()
    assert stats["trades"] == 2 and stats["win_rate"] == 0.5 and stats["total_r"] == 0.44
