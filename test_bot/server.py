"""A simple test chatbot with HTTP interface, backed by Zhipu API."""

from __future__ import annotations

import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = OpenAI(
    api_key=os.getenv("ZHIPU_API_KEY"),
    base_url="https://open.bigmodel.cn/api/paas/v4",
)

SYSTEM_PROMPT = (
    "You are a helpful customer service agent for an e-commerce platform. "
    "Answer user questions about orders, shipping, refunds, etc. "
    "Be concise and helpful."
)


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: str | None = None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    response = client.chat.completions.create(
        model="glm-4-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return ChatResponse(
        reply=response.choices[0].message.content,
        conversation_id=req.conversation_id,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
