"""
embeddings.py — Async wrapper around the LiteLLM embedding model + cache.

Uses gemini-embedding-2 (3072-dim) via the LiteLLM proxy. NOTE: switching the
embedding model invalidates any previously-stored vectors (different model =
different vector space), so semantic.db + the cache must be rebuilt on switch.

A tiny on-disk cache (SHA-256-of-text → vector) prevents re-embedding
the same exact string. Cache lives next to semantic.db.

Returns lists of floats so callers can pass them straight to sqlite-vec.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import struct
from collections import OrderedDict
from pathlib import Path

import httpx

from serbo_core.config import (
    LITELLM_API_KEY, LITELLM_BASE_URL, LLM_EMBED_MODEL, EMBED_CACHE_MAX_ENTRIES,
)

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = LLM_EMBED_MODEL
EMBEDDING_DIM = 3072

CACHE_FILE = Path(os.getenv("EMBEDDING_CACHE_PATH") or (Path(__file__).parent.parent / "data" / "embedding_cache.bin"))

# LRU bound: keep at most MAX entries in memory + on disk. The on-disk file is
# append-only for speed; it is compacted (rewritten from the live LRU set) once
# it grows past COMPACT_THRESHOLD records, so it can't grow without bound.
_MAX_ENTRIES = max(100, EMBED_CACHE_MAX_ENTRIES)
_COMPACT_THRESHOLD = _MAX_ENTRIES * 2

_lock = asyncio.Lock()
_cache: "OrderedDict[str, list[float]]" = OrderedDict()  # insertion/LRU order
_cache_loaded = False
_disk_records = 0  # records currently appended to CACHE_FILE


def _digest(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _load_cache() -> None:
    """Binary cache format: 32B SHA-256 + N*4B float32 floats per record."""
    global _cache_loaded, _disk_records
    if _cache_loaded:
        return
    if not CACHE_FILE.exists():
        _cache_loaded = True
        return
    try:
        data = CACHE_FILE.read_bytes()
        offset = 0
        record_size = 32 + EMBEDDING_DIM * 4
        records = 0
        while offset + record_size <= len(data):
            key = data[offset:offset + 32].hex()
            vec_bytes = data[offset + 32:offset + record_size]
            vec = list(struct.unpack(f"{EMBEDDING_DIM}f", vec_bytes))
            # Re-inserting moves the key to the end → file order == LRU order.
            _cache[key] = vec
            _cache.move_to_end(key)
            offset += record_size
            records += 1
        _disk_records = records
        logger.info("embeddings: loaded %d cached vectors from %s", len(_cache), CACHE_FILE.name)
    except Exception as exc:
        logger.warning("embeddings: cache load failed: %s", exc)
    _cache_loaded = True
    # Trim to the cap (oldest first) + compact the file if it's overgrown.
    _evict_if_needed()
    if _disk_records > _COMPACT_THRESHOLD or _disk_records > len(_cache):
        _compact()


def _evict_if_needed() -> None:
    """Drop oldest entries until within _MAX_ENTRIES (in memory)."""
    while len(_cache) > _MAX_ENTRIES:
        _cache.popitem(last=False)


def _compact() -> None:
    """Rewrite CACHE_FILE from the live (≤MAX) LRU set. Bounds the file size."""
    global _disk_records
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".bin.tmp")
        with tmp.open("wb") as f:
            for key, vec in _cache.items():
                f.write(bytes.fromhex(key))
                f.write(struct.pack(f"{EMBEDDING_DIM}f", *vec))
        tmp.replace(CACHE_FILE)
        _disk_records = len(_cache)
        logger.info("embeddings: cache compacted to %d entries", _disk_records)
    except Exception as exc:
        logger.warning("embeddings: cache compaction failed: %s", exc)


def _append_cache(text: str, vec: list[float]) -> None:
    """Append-only cache write — avoids rewriting the whole file each time."""
    global _disk_records
    if len(vec) != EMBEDDING_DIM:
        return
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        key_bytes = bytes.fromhex(_digest(text))
        with CACHE_FILE.open("ab") as f:
            f.write(key_bytes)
            f.write(struct.pack(f"{EMBEDDING_DIM}f", *vec))
        _disk_records += 1
    except Exception as exc:
        logger.warning("embeddings: cache append failed: %s", exc)
        return
    # Amortised compaction: once the append-only file is well over the cap.
    if _disk_records > _COMPACT_THRESHOLD:
        _compact()


async def embed(text: str) -> list[float] | None:
    """Return the embedding vector for `text`, or None on failure.

    Hits the on-disk cache first. Updates the cache after a fresh API call.
    """
    text = (text or "").strip()
    if not text:
        return None
    # No proxy configured (e.g. CI) → skip the network call, degrade gracefully.
    if not LITELLM_BASE_URL or not LITELLM_API_KEY:
        return None

    _load_cache()
    key = _digest(text)

    async with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)  # mark most-recently-used
    if cached:
        return cached

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                f"{LITELLM_BASE_URL.rstrip('/')}/embeddings",
                json={"model": EMBEDDING_MODEL, "input": text},
                headers={
                    "Authorization": f"Bearer {LITELLM_API_KEY}",
                    "Content-Type": "application/json",
                },
            )
            r.raise_for_status()
            payload = r.json()
    except Exception as exc:
        logger.warning("embeddings: API call failed for %r: %s", text[:60], exc)
        return None

    try:
        vec = payload["data"][0]["embedding"]
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("embeddings: unexpected payload shape: %s", exc)
        return None

    if not isinstance(vec, list) or len(vec) != EMBEDDING_DIM:
        logger.warning("embeddings: unexpected dim=%s for %r", len(vec) if isinstance(vec, list) else "?", text[:60])
        return None

    async with _lock:
        _cache[key] = vec
        _cache.move_to_end(key)
        _evict_if_needed()
    _append_cache(text, vec)
    return vec
