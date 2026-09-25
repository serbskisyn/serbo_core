"""
llm_client.py — central LiteLLM (OpenAI-compatible) client (serbo_core).

One place for base URL + key + model aliases, so every bot routes through the
Atolls LiteLLM proxy. Web-search enrichment uses Gemini googleSearch grounding
(chat_grounded).
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx

from serbo_core.config import (
    LITELLM_API_KEY, LITELLM_BASE_URL, LLM_GROUNDED_MODEL, LLM_MAX_CONCURRENCY,
)

logger = logging.getLogger(__name__)

# Process-wide guard for ALL outgoing LLM calls (handlers AND scheduled jobs).
llm_semaphore = asyncio.Semaphore(max(1, LLM_MAX_CONCURRENCY))


def _url() -> str:
    return f"{LITELLM_BASE_URL.rstrip('/')}/chat/completions"


async def chat(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 600,
    tools: list[dict] | None = None,
    timeout: float = 30.0,
    extra: dict | None = None,
) -> str:
    """Single LiteLLM chat completion. Returns the assistant content string.
    Raises httpx errors so callers keep their existing try/except handling.
    `extra` merges additional payload keys (z. B. reasoning_effort='none' um das
    Thinking-Token-Budget von gemini-2.5-flash abzuschalten)."""
    if not LITELLM_BASE_URL or not LITELLM_API_KEY:
        raise RuntimeError("LiteLLM nicht konfiguriert (LITELLM_BASE_URL/API_KEY fehlt)")
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
    if extra:
        payload.update(extra)
    headers = {
        "Authorization": f"Bearer {LITELLM_API_KEY}",
        "Content-Type": "application/json",
    }
    async with llm_semaphore:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(_url(), json=payload, headers=headers)
            # Some models (e.g. Claude Opus 4.8 on Vertex) reject the temperature
            # param ("temperature is deprecated for this model"). Retry without it.
            if r.status_code == 400 and "temperature" in r.text.lower():
                payload.pop("temperature", None)
                r = await client.post(_url(), json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    usage = data.get("usage") or {}
    logger.info(
        "[LLM-CALL] model=%s total_tokens=%s prompt=%s completion=%s",
        model, usage.get("total_tokens", "?"),
        usage.get("prompt_tokens", "?"), usage.get("completion_tokens", "?"),
    )
    return (data["choices"][0]["message"].get("content") or "")


async def chat_stream(
    messages: list[dict],
    *,
    model: str,
    temperature: float = 0.3,
    max_tokens: int = 4000,
    timeout: float = 600.0,
    extra: dict | None = None,
) -> str:
    """Wie chat(), aber per SSE-Streaming — für lange Antworten. Der LiteLLM-Gateway bricht
    nicht-gestreamte Requests nach ~60 s mit 504 ab; gestreamt bleibt die Verbindung aktiv."""
    if not LITELLM_BASE_URL or not LITELLM_API_KEY:
        raise RuntimeError("LiteLLM nicht konfiguriert (LITELLM_BASE_URL/API_KEY fehlt)")
    payload: dict = {"model": model, "messages": messages, "temperature": temperature,
                     "max_tokens": max_tokens, "stream": True}
    if extra:
        payload.update(extra)
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
    async with llm_semaphore:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=180.0)) as client:
            for attempt in range(2):
                parts: list[str] = []
                finish = None
                async with client.stream("POST", _url(), json=payload, headers=headers) as r:
                    if r.status_code == 400 and attempt == 0:
                        body = (await r.aread()).decode(errors="replace").lower()
                        if "temperature" in body:        # Opus 4.8 lehnt temperature ab
                            payload.pop("temperature", None)
                            continue
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            ch = json.loads(data)["choices"][0]
                        except (ValueError, KeyError, IndexError):
                            continue
                        finish = ch.get("finish_reason") or finish
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                out = "".join(parts)
                logger.info("[LLM-CALL] model=%s stream=1 chars=%d finish=%s", model, len(out), finish)
                if finish == "length":
                    logger.warning("chat_stream: max_tokens=%d erreicht → Ausgabe abgeschnitten", max_tokens)
                return out
    raise RuntimeError("chat_stream: kein Ergebnis")


async def chat_grounded(
    messages: list[dict],
    *,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout: float = 60.0,
) -> str:
    """Chat with Gemini Google-Search grounding (live web research)."""
    return await chat(
        messages,
        model=model or LLM_GROUNDED_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=[{"googleSearch": {}}],
        timeout=timeout,
    )
