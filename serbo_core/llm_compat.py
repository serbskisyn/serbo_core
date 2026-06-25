"""
llm_compat.py — resilient compatibility wrapper around llm_client.chat (serbo_core).

ask_llm() adds a default system prompt, optional history, and turns
HTTP/timeout/quota errors into friendly German strings (whereas llm_client.chat
raises). Many callers rely on that resilience.
"""
import httpx
import logging
from serbo_core.config import OPENROUTER_API_KEY, OPENROUTER_MODEL  # noqa: F401

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
EXTRACTOR_MODEL = "openai/gpt-4o-mini"


async def ask_llm(
    user_text: str,
    history: list[dict] = None,
    system_prompt: str = "Du bist ein hilfreicher Assistent. Antworte auf Deutsch.",
    model: str | None = None,
) -> str:
    from serbo_core.llm_client import chat as _llm_chat
    from serbo_core.config import LLM_CHEAP_MODEL

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    try:
        return await _llm_chat(
            messages, model=model or LLM_CHEAP_MODEL,
            temperature=0.7, max_tokens=1024, timeout=60.0,
        )
    except httpx.TimeoutException:
        logger.error("LLM Timeout")
        return "Die Anfrage hat zu lange gedauert. Bitte versuche es erneut."
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:300]
        logger.error("LLM Fehler %s: %s", e.response.status_code, body[:200])
        body_lc = body.lower()
        if "limit exceeded" in body_lc or "insufficient_quota" in body_lc:
            return "⚠️ LLM-Konto-Limit erreicht — bitte Credits/Key prüfen."
        if e.response.status_code == 401:
            return "⚠️ LLM API-Key abgelehnt (401) — Key in .env prüfen."
        if e.response.status_code == 429:
            return "⏳ LLM Rate-Limit aktiv — kurz warten und nochmal."
        return f"Problem mit der KI-Verbindung (HTTP {e.response.status_code}). Bitte später erneut."
    except Exception as e:
        logger.error(f"Unbekannter Fehler: {e}", exc_info=True)
        return "Ein unerwarteter Fehler ist aufgetreten."


async def extract_facts(user_text: str, assistant_response: str) -> dict:
    """Deprecated no-op (real extraction lives in the private bot's profile_learner)."""
    return {"direct": {}, "indirect": []}
