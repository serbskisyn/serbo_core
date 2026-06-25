"""
telegram.py — shared Telegram helpers (serbo_core).

split_message: single implementation for chunking long text under Telegram's
4096-char message limit.
"""
from __future__ import annotations

# Stay safely below Telegram's hard 4096-char cap (Markdown overhead).
DEFAULT_LIMIT = 4000


def split_message(text: str, limit: int = DEFAULT_LIMIT) -> list[str]:
    """Split text into Telegram-safe chunks at line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        if current_len + len(line) + 1 > limit:
            chunks.append("\n".join(current))
            current, current_len = [line], len(line)
        else:
            current.append(line)
            current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks
