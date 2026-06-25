"""
serbo_core.gspread — generischer Google-Sheets-Client (Auth).

Geteilt: Goldkind (Dienstplan) + Atolls (Lead-Sheets) nutzen denselben Auth-Pfad.
Primär GOOGLE_SERVICE_ACCOUNT_JSON (env), Fallback credentials.json (CWD bzw.
GOOGLE_CREDENTIALS_FILE). Domänen-spezifische Sheet-Logik liegt im jeweiligen Bot.
"""
from __future__ import annotations

import json
import logging
import os

import gspread
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


def _get_client() -> gspread.Client:
    json_str = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    if json_str:
        try:
            info = json.loads(json_str)
            if "private_key" in info:
                info["private_key"] = info["private_key"].replace("\\n", "\n")
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
            return gspread.authorize(creds)
        except Exception as e:
            logger.warning("GOOGLE_SERVICE_ACCOUNT_JSON ungueltig (%s), versuche credentials.json", e)

    creds_path = os.path.abspath(os.getenv("GOOGLE_CREDENTIALS_FILE", "credentials.json"))
    if os.path.exists(creds_path):
        creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
        return gspread.authorize(creds)

    raise EnvironmentError(
        "Kein Google-Credential gefunden. "
        "Setze GOOGLE_SERVICE_ACCOUNT_JSON oder lege credentials.json an."
    )
