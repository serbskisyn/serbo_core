import time
from collections import deque
from serbo_core import config

# Sliding window per user_id. Plain dict (not defaultdict) so we control
# insertion; each window is a bounded deque, and stale users are swept
# periodically so _windows can't grow without bound.
_windows: dict[int, deque] = {}

_calls_since_cleanup = 0
_CLEANUP_EVERY = 500


def _new_window() -> deque:
    return deque(maxlen=max(1, config.RATE_LIMIT_MAX_REQUESTS))


def _cleanup(now: float) -> None:
    """Drop users whose window is empty or fully expired."""
    cutoff = now - config.RATE_LIMIT_WINDOW_SECONDS
    stale = [uid for uid, w in _windows.items() if not w or w[-1] < cutoff]
    for uid in stale:
        del _windows[uid]


def is_rate_limited(user_id: int) -> tuple[bool, int]:
    """
    Sliding window rate limiter.
    Returns (limited: bool, retry_after_seconds: int).
    """
    global _calls_since_cleanup
    now = time.time()
    window = _windows.get(user_id)
    if window is None:
        window = _windows[user_id] = _new_window()

    # Alte Einträge außerhalb des Fensters entfernen
    while window and window[0] < now - config.RATE_LIMIT_WINDOW_SECONDS:
        window.popleft()

    # Periodischer Sweep gegen Memory-Leak durch nie wiederkehrende User.
    _calls_since_cleanup += 1
    if _calls_since_cleanup >= _CLEANUP_EVERY:
        _calls_since_cleanup = 0
        _cleanup(now)

    if len(window) >= config.RATE_LIMIT_MAX_REQUESTS:
        retry_after = int(config.RATE_LIMIT_WINDOW_SECONDS - (now - window[0])) + 1
        return True, retry_after

    window.append(now)
    return False, 0
