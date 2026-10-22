"""
logging_setup.py — shared logging config (serbo_core).

LOG_DIR + log file name are per-bot (CWD-relative / parametrisierbar), damit
jeder Bot in seine eigene Logdatei schreibt.
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from serbo_core.config import LOG_LEVEL


def setup_logging(log_name: str = "bot.log", log_dir: str | None = None):
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    LOG_DIR = Path(log_dir or os.getenv("LOG_DIR", "logs"))
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        LOG_DIR / log_name, maxBytes=5 * 1024 * 1024, backupCount=7, encoding="utf-8",
    )
    file_handler.setFormatter(logging.Formatter(fmt))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt))

    # force=True: falls ein importiertes Modul schon basicConfig gesetzt hat (kudos_bot tat das),
    # gewinnt trotzdem dieses Setup — sonst greifen Datei-Log + Rotation still nicht.
    logging.basicConfig(level=level, handlers=[console_handler, file_handler], force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.WARNING)
