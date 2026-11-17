"""Web Push: device notifications through the browser's push service."""
import base64
import json
import logging
import os

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from pywebpush import WebPushException, webpush

from backend import db

log = logging.getLogger("push")
CONTACT = os.environ.get("VAPID_CONTACT", "mailto:admin@example.com")


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def vapid_keys() -> tuple:
    """(private, public) application server keys, created on first use."""
    private, public = db.kv_get("vapid_private"), db.kv_get("vapid_public")
    if not private:
        key = ec.generate_private_key(ec.SECP256R1())
        private = _b64(key.private_numbers().private_value.to_bytes(32, "big"))
        public = _b64(key.public_key().public_bytes(
            serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))
        db.kv_set("vapid_private", private)
        db.kv_set("vapid_public", public)
    return private, public


def send(user_id: int, payload: dict) -> int:
    """Push to every device a user has registered. Returns how many were delivered."""
    private, _ = vapid_keys()
    sent = 0
    for row in db.query("SELECT id, sub FROM push_subs WHERE user_id = ?", (user_id,)):
        try:
            webpush(json.loads(row["sub"]), json.dumps(payload), vapid_private_key=private,
                    vapid_claims={"sub": CONTACT}, ttl=300)
            sent += 1
        except WebPushException as exc:
            code = getattr(exc.response, "status_code", None)
            if code in (404, 410):  # the device unsubscribed
                db.execute("DELETE FROM push_subs WHERE id = ?", (row["id"],))
            else:
                log.warning("push failed: %s", exc)
    return sent


def format_signal(event: dict) -> dict:
    """Notification title and body for a signal event."""
    s = event["signal"]
    tag = f"{s['side']} {s['symbol']}"
    if event["type"] == "signal":
        title = f"{tag} @ {s['entry']:.2f} · {s['confidence'] * 100:.0f}%"
        body = f"SL {s['stop']:.2f} · TP {s['tp1']:.2f} / {s['tp2']:.2f} · {s['timeframe']}m"
    else:
        label = {"tp1_hit": "Target 1 hit", "tp2_hit": "Target 2 hit", "stopped": "Stopped out",
                 "expired": "Expired", "tp1_closed": "Closed after target 1"}.get(s["status"], s["status"])
        title = f"{tag}: {label}"
        body = f"{s['timeframe']}m signal from {s['entry']:.2f}"
        if s.get("result_r") is not None:
            body += f" · {s['result_r']:+.2f}R"
    return {"title": title, "body": body, "tag": f"signal-{s['id']}", "url": f"/signal/{s['id']}"}
