from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from utils.logger import logger

_TOKEN_FILE = Path(__file__).resolve().parent.parent / ".butlr_tokens"
_KEY_FILE = Path(__file__).resolve().parent.parent / ".butlr_key"
_lock = threading.Lock()
_refresh_token: str = ""
_access_token: str = ""


def _get_fernet() -> Fernet:
    """Return a Fernet cipher using BUTLR_TOKEN_KEY env var, key file, or a newly generated key."""
    raw_key = os.getenv("BUTLR_TOKEN_KEY", "").strip().encode()
    if raw_key:
        return Fernet(raw_key)

    if _KEY_FILE.exists():
        return Fernet(_KEY_FILE.read_bytes().strip())

    key = Fernet.generate_key()
    try:
        _KEY_FILE.write_bytes(key)
        _KEY_FILE.chmod(0o600)
        logger.info("Generated new token encryption key at %s", _KEY_FILE)
    except Exception as exc:
        logger.warning("Could not persist encryption key to disk: %s", exc)
    return Fernet(key)


def _load_from_disk() -> None:
    global _refresh_token
    if not _TOKEN_FILE.exists():
        return
    try:
        f = _get_fernet()
        encrypted = _TOKEN_FILE.read_bytes()
        plaintext = f.decrypt(encrypted)
        data = json.loads(plaintext)
        _refresh_token = data.get("refresh_token", "")
        return
    except InvalidToken:
        logger.warning(
            "Token file could not be decrypted (wrong key or corrupted). "
            "Attempting to read as legacy plaintext."
        )
    except Exception as exc:
        logger.warning("Could not load encrypted tokens from disk: %s", exc)
        return

    # Migration path: try legacy plaintext JSON format
    try:
        data = json.loads(_TOKEN_FILE.read_text())
        _refresh_token = data.get("refresh_token", "")
        if _refresh_token:
            logger.info("Migrated legacy plaintext token to encrypted storage")
            _save_to_disk()
    except Exception as exc:
        logger.warning("Could not read legacy token file: %s", exc)


def _save_to_disk() -> None:
    """Encrypt and persist the current refresh token. Called while holding _lock."""
    try:
        f = _get_fernet()
        plaintext = json.dumps({"refresh_token": _refresh_token}).encode()
        encrypted = f.encrypt(plaintext)
        _TOKEN_FILE.write_bytes(encrypted)
        _TOKEN_FILE.chmod(0o600)
    except Exception as exc:
        logger.warning("Could not persist tokens to disk: %s", exc)


# Load persisted token immediately at import time so the service starts
# authenticated without requiring a browser flow on every restart.
_load_from_disk()


def set_tokens(refresh_token: str, access_token: str = "") -> None:
    global _refresh_token, _access_token
    with _lock:
        _refresh_token = refresh_token
        _access_token = access_token
        _save_to_disk()


def get_refresh_token() -> str:
    with _lock:
        return _refresh_token


def is_authenticated() -> bool:
    with _lock:
        return bool(_refresh_token)


def clear() -> None:
    global _refresh_token, _access_token
    with _lock:
        _refresh_token = ""
        _access_token = ""
        try:
            _TOKEN_FILE.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not remove token file: %s", exc)
