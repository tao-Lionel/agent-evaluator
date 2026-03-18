"""Feishu (Lark) bot integration — receives messages via event subscription,
replies via Feishu Open API, using the same Zhipu backend as server.py.

Environment variables needed:
  ZHIPU_API_KEY        — for the chat backend
  FEISHU_APP_ID        — from Feishu open platform
  FEISHU_APP_SECRET    — from Feishu open platform
  FEISHU_VERIFY_TOKEN  — event subscription verification token (optional if using encrypt)
  FEISHU_ENCRYPT_KEY   — event subscription encrypt key (optional)

Start:
  uvicorn feishu_bot:app --port 8101
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from server import generate_reply

load_dotenv()

logger = logging.getLogger("feishu_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

# ── Feishu config ──
APP_ID = os.getenv("FEISHU_APP_ID", "")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
VERIFY_TOKEN = os.getenv("FEISHU_VERIFY_TOKEN", "")
ENCRYPT_KEY = os.getenv("FEISHU_ENCRYPT_KEY", "")

# ── Token cache ──
_tenant_access_token: str = ""
_token_expires_at: float = 0


def get_tenant_access_token() -> str:
    """Get or refresh the tenant access token."""
    global _tenant_access_token, _token_expires_at

    if _tenant_access_token and time.time() < _token_expires_at - 60:
        return _tenant_access_token

    resp = httpx.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": APP_ID, "app_secret": APP_SECRET},
    )
    data = resp.json()
    if data.get("code") != 0:
        logger.error("Failed to get tenant_access_token: %s", data)
        raise RuntimeError(f"Feishu auth failed: {data.get('msg')}")

    _tenant_access_token = data["tenant_access_token"]
    _token_expires_at = time.time() + data.get("expire", 7200)
    logger.info("Refreshed tenant_access_token, expires in %ds", data.get("expire", 7200))
    return _tenant_access_token


def send_feishu_reply(message_id: str, text: str) -> None:
    """Reply to a Feishu message using the reply API."""
    token = get_tenant_access_token()
    resp = httpx.post(
        f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "content": json.dumps({"text": text}),
            "msg_type": "text",
        },
    )
    data = resp.json()
    if data.get("code") != 0:
        logger.error("Failed to send reply: %s", data)
    else:
        logger.info("Replied to message %s", message_id)


# ── Dedup: ignore repeated events (Feishu may retry) ──
_seen_event_ids: dict[str, float] = {}
MAX_SEEN = 500


def _is_duplicate(event_id: str) -> bool:
    now = time.time()
    # Clean old entries
    if len(_seen_event_ids) > MAX_SEEN:
        cutoff = now - 300
        to_remove = [k for k, v in _seen_event_ids.items() if v < cutoff]
        for k in to_remove:
            del _seen_event_ids[k]

    if event_id in _seen_event_ids:
        return True
    _seen_event_ids[event_id] = now
    return False


def _extract_text(event: dict[str, Any]) -> tuple[str | None, str | None]:
    """Extract plain text and message_id from a Feishu message event.

    Returns (text, message_id) or (None, None) if not a text message.
    """
    try:
        msg = event["event"]["message"]
        message_id = msg["message_id"]
        msg_type = msg.get("message_type", "")
        if msg_type != "text":
            return None, None
        content = json.loads(msg["content"])
        text = content.get("text", "")
        return text, message_id
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning("Failed to extract text from event: %s", e)
        return None, None


@app.post("/feishu/event")
async def feishu_event(request: Request):
    """Handle Feishu event callback.

    Handles:
    1. URL verification challenge (setup phase)
    2. im.message.receive_v1 events (user messages)
    """
    body = await request.json()
    logger.info("Raw event body: %s", json.dumps(body, ensure_ascii=False)[:500])

    # ── URL verification challenge ──
    if "challenge" in body:
        logger.info("Received URL verification challenge")
        return {"challenge": body["challenge"]}

    # ── Schema v2 event ──
    header = body.get("header", {})
    event_id = header.get("event_id", "")
    event_type = header.get("event_type", "")

    # Verify token if configured
    if VERIFY_TOKEN and header.get("token") != VERIFY_TOKEN:
        logger.warning("Invalid verify token, ignoring event")
        return {"code": 0}

    # Dedup
    if event_id and _is_duplicate(event_id):
        logger.info("Duplicate event %s, skipping", event_id)
        return {"code": 0}

    # Only handle message events
    if event_type != "im.message.receive_v1":
        logger.info("Ignoring event type: %s", event_type)
        return {"code": 0}

    # Extract text
    text, message_id = _extract_text(body)
    if not text or not message_id:
        logger.info("Non-text message or parse error, skipping")
        return {"code": 0}

    logger.info("Received message: %s (id=%s)", text[:100], message_id)

    # Generate reply via Zhipu
    try:
        reply = generate_reply(text)
    except Exception as e:
        logger.error("Zhipu API error: %s", e)
        reply = "抱歉，我暂时无法处理您的请求，请稍后再试。"

    # Send reply
    try:
        send_feishu_reply(message_id, reply)
    except Exception as e:
        logger.error("Failed to reply via Feishu: %s", e)

    return {"code": 0}


@app.get("/health")
def health():
    return {"status": "ok", "service": "feishu_bot"}
