"""
serbo_core.config — GETEILTER Konfigurations-Kern aller Serbo-Bots.

Nur Infrastruktur, die jeder Bot braucht: Telegram-Token, LLM/LiteLLM, Grok,
Such-Keys, Whitelist, Rate-Limit, Logging. Bot-spezifische Keys (News, Kicktipp,
Lead, Goldkind, Birthday, …) leben in der jeweiligen Bot-eigenen config, die
`from serbo_core.config import *` macht und ihre Keys ergänzt.

load_dotenv() lädt das .env aus dem Arbeitsverzeichnis des jeweiligen Bots.
"""
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL   = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

# ── LiteLLM (OpenAI-kompatibler Proxy, löst OpenRouter ab) ────────────────────
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY", "")
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "")
# Model aliases on the LiteLLM proxy
LLM_CHEAP_MODEL    = os.getenv("LLM_CHEAP_MODEL", "gemini-2.5-flash")
LLM_STRONG_MODEL   = os.getenv("LLM_STRONG_MODEL", "vertex_ai/claude-opus-4-8")
LLM_GROUNDED_MODEL = os.getenv("LLM_GROUNDED_MODEL", "gemini-2.5-flash")
# Stronger grounding model for structured company-fact extraction (validate_company)
LLM_GROUNDED_STRONG_MODEL = os.getenv("LLM_GROUNDED_STRONG_MODEL", "gemini-3-1-pro-preview")
LLM_EMBED_MODEL    = os.getenv("LLM_EMBED_MODEL", "gemini-embedding-2")  # 3072-dim
# Hard cap on the on-disk embedding cache (LRU). ~12.3 KB/entry at 3072 dims →
# 5000 ≈ 62 MB. Prevents unbounded growth on the Pi's SD card.
EMBED_CACHE_MAX_ENTRIES: int = int(os.getenv("EMBED_CACHE_MAX_ENTRIES", "5000"))
# Process-wide cap on concurrent outgoing LLM calls (handlers + scheduled jobs
# share it) so parallel jobs can't hammer the proxy. See serbo_core.llm_client.
LLM_MAX_CONCURRENCY: int = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
# Der LiteLLM-Key muss alle N Tage erneuert werden → Reminder-Job
LITELLM_KEY_RENEW_DAYS: int = int(os.getenv("LITELLM_KEY_RENEW_DAYS", "7"))
LITELLM_KEY_REMINDER_HOUR: int = int(os.getenv("LITELLM_KEY_REMINDER_HOUR", "9"))
LITELLM_KEY_REMINDER_MINUTE: int = int(os.getenv("LITELLM_KEY_REMINDER_MINUTE", "0"))

# ── Grok (xAI) für X.com Live-Search ──────────────────────────────────────────
GROK_MODEL    = os.getenv("GROK_MODEL", "x-ai/grok-4.3")
GROK_API_KEY  = os.getenv("GROK_API_KEY", "")
GROK_BASE_URL = os.getenv("GROK_BASE_URL", "https://api.x.ai/v1")
BOT_NAME           = os.getenv("BOT_NAME", "MeinAgent")
LOG_LEVEL          = os.getenv("LOG_LEVEL", "INFO")
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY")
GNEWS_API_KEY      = os.getenv("GNEWS_API_KEY", "")
BRAVE_API_KEY      = os.getenv("BRAVE_API_KEY", "")

ALLOWED_USER_IDS: set[int] = set(
    int(x) for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()
)

# Rate Limiting
RATE_LIMIT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", 10))
RATE_LIMIT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

# Admin-Chat für System-Alerts/Reports (geteilt). Fallback: kleinste Whitelist-ID.
_admin_raw = os.getenv("ADMIN_CHAT_ID", "")
if _admin_raw.strip():
    ADMIN_CHAT_ID: int | None = int(_admin_raw.strip())
else:
    ADMIN_CHAT_ID: int | None = min(ALLOWED_USER_IDS) if ALLOWED_USER_IDS else None
