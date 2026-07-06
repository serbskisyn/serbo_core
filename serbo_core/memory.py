"""
memory.py — Geteilte User-Memory über alle komponierten Bots hinweg.

Alle Bots (serbo, atolls, goldkind) laufen im selben Prozess und sollen dieselbe
strukturierte User-Memory (profile.yaml) lesen UND schreiben können. Früher las
nur der serbo-Bot das private Profil; die Arbeits-Bots hatten es bewusst
ausgespart (Trennung Privat/Arbeit). Diese Fassade hebt die Trennung auf:
main.py registriert das Profil-Backend, jeder Bot greift über serbo_core.memory
darauf zu — Sprach- und Text-Fakten fließen so in eine gemeinsame Memory.

Fail-soft: ohne registriertes Backend (z.B. atolls standalone / in CI) liefern
die Reads neutrale Werte und apply_ops ist ein No-op — nichts crasht.

Verwendung:
    # In serbo_bot/app/main.py beim Start:
    from serbo_core import memory as shared_memory
    from app.bot import profile
    shared_memory.register_backend(profile)

    # In irgendeinem Bot:
    from serbo_core import memory as shared_memory
    ctx   = shared_memory.get_memory_prompt(user_id)   # Fakten als Prompt-Block
    facts = shared_memory.get_confirmed(user_id)        # {key: value}
    await shared_memory.apply_ops(user_id, [...])        # schreiben
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Duck-typed Backend: ein Modul/Objekt mit as_prompt, as_flat_confirmed, apply_ops.
_backend: Any = None


def register_backend(backend: Any) -> None:
    """Registriert das Profil-Backend (z.B. app.bot.profile). Idempotent."""
    global _backend
    _backend = backend
    name = getattr(backend, "__name__", type(backend).__name__)
    logger.info("shared memory: Backend registriert (%s)", name)


def is_available() -> bool:
    """True, sobald ein Profil-Backend registriert wurde."""
    return _backend is not None


def get_memory_prompt(user_id: int) -> str:
    """Private User-Fakten als deutscher Prompt-Kontextblock ('' ohne Backend)."""
    if _backend is None:
        return ""
    try:
        return _backend.as_prompt(user_id) or ""
    except Exception as e:  # pragma: no cover - defensiv
        logger.warning("shared memory get_memory_prompt: %s", e)
        return ""


def get_confirmed(user_id: int) -> dict:
    """Bestätigte Fakten als flaches key/value-Dict ({} ohne Backend)."""
    if _backend is None:
        return {}
    try:
        return _backend.as_flat_confirmed(user_id) or {}
    except Exception as e:  # pragma: no cover - defensiv
        logger.warning("shared memory get_confirmed: %s", e)
        return {}


async def apply_ops(user_id: int, ops: list[dict]) -> dict:
    """Schreibt strukturierte Update-Ops ins geteilte Profil ({} ohne Backend)."""
    if _backend is None:
        logger.debug("shared memory apply_ops: kein Backend — no-op")
        return {}
    try:
        return await _backend.apply_ops(user_id, ops) or {}
    except Exception as e:  # pragma: no cover - defensiv
        logger.warning("shared memory apply_ops: %s", e)
        return {}
