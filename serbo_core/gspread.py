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

# Least-Privilege: alle Aufrufer öffnen Sheets per open_by_key (kein Öffnen per
# Titel, kein Anlegen, kein Drive-Listing) → der volle Drive-Scope ist unnötig.
# Nur der Spreadsheets-Scope, damit ein geleaktes Credential nicht die ganze
# Drive des Accounts erreicht.
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]


def _get_client(env_var: str = "GOOGLE_SERVICE_ACCOUNT_JSON",
                creds_file_env: str = "GOOGLE_CREDENTIALS_FILE") -> gspread.Client:
    """Service-Account-Client. Default-Credential = GOOGLE_SERVICE_ACCOUNT_JSON
    (Goldkind/Atolls-Bestand). Über env_var kann ein ZWEITER, isolierter Account
    genutzt werden — z. B. GPM_SERVICE_ACCOUNT_JSON fürs Atolls-SF-Sheet."""
    json_str = os.environ.get(env_var, "").strip()
    if json_str:
        try:
            info = json.loads(json_str)
            if "private_key" in info:
                info["private_key"] = info["private_key"].replace("\\n", "\n")
            creds = Credentials.from_service_account_info(info, scopes=SCOPES)
            return gspread.authorize(creds)
        except Exception as e:
            logger.warning("%s ungueltig (%s), versuche credentials.json", env_var, e)

    creds_path = os.path.abspath(os.getenv(creds_file_env, "credentials.json"))
    if os.path.exists(creds_path):
        creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
        return gspread.authorize(creds)

    raise EnvironmentError(
        "Kein Google-Credential gefunden. "
        "Setze GOOGLE_SERVICE_ACCOUNT_JSON oder lege credentials.json an."
    )
