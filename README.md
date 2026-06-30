# serbo_core

Geteilter Kern der **Serbo-Bots** — die wiederverwendbare Library hinter dem 3-Repo-Split
(`serbo_core` + `goldkind_bot` + `atolls_bot`, komponiert im Privat-`Serbo_bot`). Enthält
ausschließlich **domänenneutrales Plumbing**, keine geschäftsspezifische Logik.

> **Öffentliches Repo** (github.com/serbskisyn/serbo_core) — daher **niemals Secrets** hier.
> Alle Keys/Tokens leben in den `.env`-Dateien der konsumierenden Bots (gitignored).

## Module
| Modul | Zweck |
|---|---|
| `config.py` | Geteilte Env-Konfig (Telegram/LLM/Grok/Whitelist/Rate-Limit/ADMIN_CHAT_ID). **Single source** — `os.getenv()` nur hier. |
| `security/whitelist.py` | `require_whitelist` / `guarded` Decorators (erlaubte User-IDs). |
| `security/rate_limiter.py` | Sliding-Window-Rate-Limit pro User. |
| `security/injection_guard.py` | Zweistufiger Prompt-Injection-Guard (Pattern + LLM, `is_injection_async`). |
| `llm_client.py` | Async LLM-Client gegen den **LiteLLM-Proxy** (+ Concurrency-Semaphore, Grounding). |
| `llm_compat.py` | `ask_llm`-Kompat-Shim (routet über `llm_client.chat`). |
| `gspread.py` | Google-Sheets-Auth (`_get_client(env_var=…)` — optional ZWEITER Service-Account, z. B. GPM). |
| `semantic.py` | sqlite-vec Semantik-Store (Pfad via `SEMANTIC_DB_PATH`). |
| `embeddings.py` | Gemini-Embeddings via LiteLLM + Cache (`EMBEDDING_CACHE_PATH`). |
| `web_search.py` | Tavily-Websuche. |
| `bot_context.py` | Geteilter Bot-Handle (`set_bot`/`get_bot`). |
| `utils/telegram.py` | `split_message` (Telegram-4096-Limit). |
| `utils/logging_setup.py` | Einheitliches Logging-Setup. |

## Nutzung
**Als Dependency** (in den Bot-Repos via `pyproject.toml`):
```toml
dependencies = ["serbo_core @ git+https://github.com/serbskisyn/serbo_core.git@main", ...]
```
```python
from serbo_core.config import TELEGRAM_BOT_TOKEN, ADMIN_CHAT_ID
from serbo_core.security.whitelist import require_whitelist
from serbo_core.llm_client import chat
from serbo_core.gspread import _get_client
```

**Lokale Entwicklung (Pi):** editable installieren, damit lokale Änderungen sofort greifen:
```bash
pip install -e /home/pi/serbo_core --no-deps
```
> ⚠️ **Editable-Falle:** Ein `pip install -e` der Bot-Repos zieht über ihre
> `serbo_core @ git+…`-Dependency die GIT-Version nach site-packages und überschreibt das
> lokale editable. Danach immer `pip install -e /home/pi/serbo_core --no-deps` als **letztes**
> ausführen, damit das lokale serbo_core gewinnt.

## Daten-Pfade (Env-Override)
Modul-relative DB-Pfade brechen, wenn serbo_core als Paket installiert ist → die Hosts setzen
absolute Pfade in ihrer `.env`:
```
SEMANTIC_DB_PATH=/home/pi/Serbo_bot/app/data/semantic.db
EMBEDDING_CACHE_PATH=/home/pi/Serbo_bot/app/data/embedding_cache.bin
```

## Shims
Alte `app/`-Pfade im Privat-Bot sind Re-Export-Shims auf serbo_core (z. B.
`app.services.llm_client`, `app.security.whitelist`) — teils explizite Re-Exporte, teils
`sys.modules`-Aliase (für mutablen State/Interna-Tests).

## Tests / CI
Wird in den Bot-Repos mitgetestet. CI zieht `serbo_core` als **public** Git-Dependency (kein
Token nötig); `requirements-ci.txt` der Hosts lässt schwere Libs (gspread/langgraph) weg →
betroffene Tests mit `pytest.importorskip` guarden.
