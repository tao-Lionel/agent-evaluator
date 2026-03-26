"""Feishu-based evaluation agent bot.

Start:
  uvicorn eval_bot.feishu:app --port 8102
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from eval_bot.dispatcher import Dispatcher
from eval_bot.runner import TaskRunner
from eval_bot.commands.quick_eval import run_quick_eval
from eval_bot.commands.query_results import query_results
from eval_bot.commands.gen_scenarios import gen_scenarios

load_dotenv()

logger = logging.getLogger("eval_bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

# ── Globals ──
dispatcher = Dispatcher()
runner = TaskRunner(max_workers=2)

# ── Feishu config ──
APP_ID = os.getenv("FEISHU_APP_ID", "")
APP_SECRET = os.getenv("FEISHU_APP_SECRET", "")
VERIFY_TOKEN = os.getenv("FEISHU_VERIFY_TOKEN", "")

# ── Token cache ──
_tenant_access_token: str = ""
_token_expires_at: float = 0
_token_lock = threading.Lock()


def get_tenant_access_token() -> str:
    global _tenant_access_token, _token_expires_at
    with _token_lock:
        if _tenant_access_token and time.time() < _token_expires_at - 60:
            return _tenant_access_token

        resp = httpx.post(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": APP_ID, "app_secret": APP_SECRET},
        )
        data = resp.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Feishu auth failed: {data.get('msg')}")

        _tenant_access_token = data["tenant_access_token"]
        _token_expires_at = time.time() + data.get("expire", 7200)
        return _tenant_access_token


def send_reply(message_id: str, text: str) -> None:
    token = get_tenant_access_token()
    resp = httpx.post(
        f"https://open.feishu.cn/open-apis/im/v1/messages/{message_id}/reply",
        headers={"Authorization": f"Bearer {token}"},
        json={"content": json.dumps({"text": text}), "msg_type": "text"},
    )
    data = resp.json()
    if data.get("code") != 0:
        logger.error("Failed to send reply: %s", data)


def send_message_to_chat(chat_id: str, text: str) -> None:
    token = get_tenant_access_token()
    resp = httpx.post(
        "https://open.feishu.cn/open-apis/im/v1/messages",
        headers={"Authorization": f"Bearer {token}"},
        params={"receive_id_type": "chat_id"},
        json={
            "receive_id": chat_id,
            "content": json.dumps({"text": text}),
            "msg_type": "text",
        },
    )
    data = resp.json()
    if data.get("code") != 0:
        logger.error("Failed to send message: %s", data)


# ── Dedup ──
_seen_event_ids: dict[str, float] = {}
_dedup_lock = threading.Lock()


def _is_duplicate(event_id: str) -> bool:
    now = time.time()
    with _dedup_lock:
        if len(_seen_event_ids) > 500:
            cutoff = now - 300
            for k in [k for k, v in _seen_event_ids.items() if v < cutoff]:
                del _seen_event_ids[k]
        if event_id in _seen_event_ids:
            return True
        _seen_event_ids[event_id] = now
        return False


def _extract_text_and_ids(body: dict) -> tuple[str | None, str | None, str | None]:
    """Extract text, message_id, chat_id from event."""
    try:
        event = body["event"]
        msg = event["message"]
        message_id = msg["message_id"]
        chat_id = msg.get("chat_id", "")
        if msg.get("message_type") != "text":
            return None, None, None
        content = json.loads(msg["content"])
        text = content.get("text", "")
        return text, message_id, chat_id
    except (KeyError, json.JSONDecodeError):
        return None, None, None


def _handle_intent(intent: str, args: dict[str, Any], message_id: str, chat_id: str):
    """Route intent to the correct command."""

    if intent == "chitchat":
        send_reply(message_id, args["reply"])
        return

    if intent == "quick_eval":
        send_reply(message_id, f"评测已开始，目标: {args['bot_url']}\n请稍候，完成后会通知你。")

        def work():
            return run_quick_eval(
                bot_url=args["bot_url"],
                eval_modes=args.get("eval_modes"),
                scenarios_path=args.get("scenarios_path"),
            )

        def on_done(task_id, result):
            if isinstance(result, Exception):
                send_reply(message_id, f"评测失败: {result}")
            else:
                send_reply(message_id, result["summary_text"])

        runner.submit(f"eval-{message_id}", work, on_done)
        return

    if intent == "query_results":
        answer = query_results(args["query"])
        send_reply(message_id, answer)
        return

    if intent == "gen_scenarios":
        result = gen_scenarios(
            domain=args["domain"],
            count=args.get("count", 5),
            difficulty=args.get("difficulty", "mixed"),
        )
        send_reply(message_id, result["message"])
        return

    send_reply(message_id, "抱歉，我不理解你的请求。")


@app.post("/feishu/event")
async def feishu_event(request: Request):
    body = await request.json()

    # URL verification
    if "challenge" in body:
        return {"challenge": body["challenge"]}

    header = body.get("header", {})
    event_id = header.get("event_id", "")

    if VERIFY_TOKEN and header.get("token") != VERIFY_TOKEN:
        return {"code": 0}

    if event_id and _is_duplicate(event_id):
        return {"code": 0}

    if header.get("event_type") != "im.message.receive_v1":
        return {"code": 0}

    text, message_id, chat_id = _extract_text_and_ids(body)
    if not text or not message_id:
        return {"code": 0}

    logger.info("Received: %s (msg=%s)", text[:100], message_id)

    try:
        intent, args = dispatcher.classify(text)
        logger.info("Intent: %s, Args: %s", intent, args)
        _handle_intent(intent, args, message_id, chat_id or "")
    except Exception as e:
        logger.error("Error handling message: %s", e)
        send_reply(message_id, f"处理出错: {e}")

    return {"code": 0}


@app.on_event("shutdown")
def on_shutdown():
    runner.shutdown()


@app.get("/health")
def health():
    return {"status": "ok", "service": "eval_bot"}
