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
import re
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass

import httpx

from serbo_core.config import (
    LITELLM_API_KEY, LITELLM_BASE_URL, LLM_GROUNDED_MODEL, LLM_MAX_CONCURRENCY,
)

logger = logging.getLogger(__name__)

# Process-wide guard for ALL outgoing LLM calls (handlers AND scheduled jobs).
llm_semaphore = asyncio.Semaphore(max(1, LLM_MAX_CONCURRENCY))


def _url() -> str:
    return f"{LITELLM_BASE_URL.rstrip('/')}/chat/completions"


# ── Signale neben der Antwort ────────────────────────────────────────────────
# Ein Guardrail/Filter/Fallback zeigt sich selten als Fehler, sondern als leere,
# abgeschnittene oder ausweichende Antwort — die Aufrufer fallen dann still auf
# Vorlagen zurück. Jeder Call bekommt darum eine Klassifikation (kind), die in der
# [LLM-CALL]-Logzeile steht (Health-Check zählt sie) und per `last_signal` für den
# Aufrufer lesbar ist. kind: ok | filtered | refused | empty | truncated | downgraded
_FILTER_FINISH = {"content_filter", "safety", "refusal", "blocklist", "prohibited_content",
                  "spii", "recitation", "image_safety"}
# Nur der Anfang der Antwort: Ablehnungen stehen vorn, ein Witz mit „I can't…" mitten drin nicht.
_REFUSAL_RX = re.compile(
    r"^\W{0,3}(?:I(?:'m| am) sorry,? but|I can(?:'|no)t (?:help|assist|write|create|do|provide)"
    r"|I(?:'m| am) (?:unable|not able) to|I won'?t (?:write|create|help)"
    r"|(?:Es )?[Tt]ut mir leid,? (?:aber )?ich kann|Ich kann (?:dir )?(?:da |dabei |hier )?leider nicht"
    r"|Ich kann (?:dir |dabei |hier )?(?:keine?[nrs]? |das |so etwas |solche )[^.!?]{0,60}?(?:nicht )?"
    r"(?:helfen|schreiben|erstellen|verfassen|unterstützen|formulieren)|Das kann ich (?:leider )?nicht)", re.I)
_CLIENT_MODULES = ("serbo_core.llm_client", "serbo_core.llm_compat",
                   "app.services.llm_client", "app.services.llm_compat")


@dataclass
class CallSignal:
    caller: str
    model: str          # angefragtes Modell/Alias
    model_used: str     # laut Antwort (weicht ab → LiteLLM-Fallback)
    finish: str | None
    latency_ms: int
    chars: int
    kind: str


# Signal des letzten chat()-Calls im aktuellen Task. Ein awaited Coroutine-Call läuft im
# selben Context → der Aufrufer sieht den Wert direkt nach `await chat(...)`.
last_signal: ContextVar[CallSignal | None] = ContextVar("llm_last_signal", default=None)


def _caller() -> str:
    """Erstes Modul.Funktion außerhalb des LLM-Clients (für die Zählung pro Aufrufstelle)."""
    f = sys._getframe(1)
    while f is not None:
        mod = f.f_globals.get("__name__", "")
        if not mod.startswith(_CLIENT_MODULES):
            return f"{mod}.{f.f_code.co_name}"
        f = f.f_back
    return "?"


def _model_base(name: str) -> str:
    return (name or "").rsplit("/", 1)[-1].lower()


def classify(content: str, finish: str | None, model: str, model_used: str,
             fallbacks: str | None = None) -> str:
    finish_l = (finish or "").lower()
    if finish_l in _FILTER_FINISH:
        return "filtered"
    if model_used and _model_base(model_used) != _model_base(model) or (fallbacks or "0") not in ("", "0"):
        return "downgraded"
    if not (content or "").strip():
        return "empty"
    if _REFUSAL_RX.match(content.strip()[:200]):
        return "refused"
    if finish_l == "length":
        return "truncated"
    return "ok"


def _record(caller: str, model: str, model_used: str, finish: str | None, t0: float,
            content: str, usage: str, fallbacks: str | None = None) -> None:
    kind = classify(content, finish, model, model_used, fallbacks)
    sig = CallSignal(caller=caller, model=model, model_used=model_used or model, finish=finish,
                     latency_ms=int((time.monotonic() - t0) * 1000), chars=len(content or ""), kind=kind)
    last_signal.set(sig)
    used = "" if _model_base(sig.model_used) == _model_base(model) else f" used={sig.model_used}"
    logger.log(logging.INFO if kind == "ok" else logging.WARNING,
               "[LLM-CALL] caller=%s model=%s%s kind=%s finish=%s ms=%d chars=%d %s",
               caller, model, used, kind, finish, sig.latency_ms, sig.chars, usage)


# Zweiter Versuch: leer/abgeschnitten → mit doppeltem max_tokens (Reasoning-Modelle
# verbrauchen das Budget oft fürs Denken); Verbindungsfehler/Timeout/429/5xx → kurz warten
# und nochmal. Filter/Ablehnung werden NICHT wiederholt (gleiche Anfrage, gleiches Ergebnis).
_RETRY_KINDS = ("empty", "truncated")
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_TOKENS_CAP = 16384
_RETRY_DELAY_S = 2.0


def _retryable_error(exc: Exception) -> bool:
    if isinstance(exc, httpx.TransportError):            # Timeout, Verbindung, DNS
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code in _RETRY_STATUS


def _log_retry(caller: str, model: str, why: str, max_tokens: int, new_tokens: int) -> None:
    logger.info("[LLM-RETRY] caller=%s model=%s %s → 2. Versuch (max_tokens %d→%d)",
                caller, model, why, max_tokens, new_tokens)


async def _post_once(payload: dict, headers: dict, timeout: float):
    async with llm_semaphore:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(_url(), json=payload, headers=headers)
            # Some models (e.g. Claude Opus 4.8 on Vertex) reject the temperature
            # param ("temperature is deprecated for this model"). Retry without it.
            if r.status_code == 400 and "temperature" in r.text.lower():
                payload.pop("temperature", None)
                r = await client.post(_url(), json=payload, headers=headers)
            r.raise_for_status()
            return r, r.json()


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
    Thinking-Token-Budget von gemini-2.5-flash abzuschalten).
    Leere/abgeschnittene Antworten und Verbindungsfehler werden einmal wiederholt."""
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
    caller = _caller()
    last_signal.set(None)
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            r, data = await _post_once(payload, headers, timeout)
        except Exception as exc:
            if attempt == 1 and _retryable_error(exc):
                _log_retry(caller, model, f"Fehler {type(exc).__name__}", payload["max_tokens"],
                           payload["max_tokens"])
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            raise
        usage = data.get("usage") or {}
        tokens = (f"total_tokens={usage.get('total_tokens', '?')} prompt={usage.get('prompt_tokens', '?')} "
                  f"completion={usage.get('completion_tokens', '?')}")
        choices = data.get("choices") or []
        # Gemini 2.5 ohne choices: max_tokens komplett fürs Reasoning verbraucht → keine Antwort
        choice = choices[0] if choices else {}
        content = (choice.get("message") or {}).get("content") or ""
        finish = choice.get("finish_reason") or ("no_choices" if not choices else None)
        model_used = str(data.get("model") or "")
        fallbacks = (getattr(r, "headers", None) or {}).get("x-litellm-attempted-fallbacks")
        kind = classify(content, finish, model, model_used, fallbacks)
        if attempt == 1 and kind in _RETRY_KINDS:
            new_tokens = min(payload["max_tokens"] * 2, _MAX_TOKENS_CAP)
            _log_retry(caller, model, f"kind={kind} finish={finish}", payload["max_tokens"], new_tokens)
            payload["max_tokens"] = new_tokens
            continue
        _record(caller, model, model_used, finish, t0, content,
                tokens + f" max_tokens={payload['max_tokens']}" + (" attempt=2" if attempt == 2 else ""),
                fallbacks)
        return content
    raise RuntimeError("chat: kein Ergebnis")


async def _stream_once(payload: dict, headers: dict, timeout: float):
    async with llm_semaphore:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, read=180.0)) as client:
            for attempt in range(2):
                parts: list[str] = []
                finish = None
                model_used = ""
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
                            chunk = json.loads(data)
                            ch = chunk["choices"][0]
                        except (ValueError, KeyError, IndexError):
                            continue
                        model_used = chunk.get("model") or model_used
                        finish = ch.get("finish_reason") or finish
                        delta = ch.get("delta") or {}
                        if delta.get("content"):
                            parts.append(delta["content"])
                    fallbacks = r.headers.get("x-litellm-attempted-fallbacks")
                return "".join(parts), finish, model_used, fallbacks
    raise RuntimeError("chat_stream: kein Ergebnis")


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
    nicht-gestreamte Requests nach ~60 s mit 504 ab; gestreamt bleibt die Verbindung aktiv.
    Gleiche Wiederholungsregeln wie chat()."""
    if not LITELLM_BASE_URL or not LITELLM_API_KEY:
        raise RuntimeError("LiteLLM nicht konfiguriert (LITELLM_BASE_URL/API_KEY fehlt)")
    payload: dict = {"model": model, "messages": messages, "temperature": temperature,
                     "max_tokens": max_tokens, "stream": True}
    if extra:
        payload.update(extra)
    headers = {"Authorization": f"Bearer {LITELLM_API_KEY}", "Content-Type": "application/json"}
    caller = _caller()
    last_signal.set(None)
    for attempt in (1, 2):
        t0 = time.monotonic()
        try:
            out, finish, model_used, fallbacks = await _stream_once(payload, headers, timeout)
        except Exception as exc:
            if attempt == 1 and _retryable_error(exc):
                _log_retry(caller, model, f"Fehler {type(exc).__name__}", payload["max_tokens"],
                           payload["max_tokens"])
                await asyncio.sleep(_RETRY_DELAY_S)
                continue
            raise
        kind = classify(out, finish, model, model_used, fallbacks)
        if attempt == 1 and kind in _RETRY_KINDS:
            new_tokens = min(payload["max_tokens"] * 2, _MAX_TOKENS_CAP)
            _log_retry(caller, model, f"kind={kind} finish={finish}", payload["max_tokens"], new_tokens)
            payload["max_tokens"] = new_tokens
            continue
        _record(caller, model, model_used, finish, t0, out,
                f"stream=1 max_tokens={payload['max_tokens']}" + (" attempt=2" if attempt == 2 else ""),
                fallbacks)
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
