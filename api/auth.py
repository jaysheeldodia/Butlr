from __future__ import annotations

import secrets
import threading
import time
from urllib.parse import quote_plus, urlencode

import requests
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse

from api import token_store
from config import settings
from utils.logger import logger

router = APIRouter(prefix="/auth", tags=["auth"])

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_SCOPES = " ".join([
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
])

# ── OAuth state token store (CSRF protection) ─────────────────────────────────
# Maps state_token -> expiry monotonic timestamp.  Single-process safe.
_pending_states: dict[str, float] = {}
_states_lock = threading.Lock()
_STATE_TTL_SECONDS = 600  # 10 minutes


def _generate_state() -> str:
    token = secrets.token_urlsafe(32)
    with _states_lock:
        # Prune expired entries to avoid unbounded growth
        now = time.monotonic()
        expired = [k for k, exp in list(_pending_states.items()) if exp < now]
        for k in expired:
            del _pending_states[k]
        _pending_states[token] = now + _STATE_TTL_SECONDS
    return token


def _verify_and_consume_state(token: str) -> bool:
    """Return True and remove the token if it is valid and not yet expired."""
    if not token:
        return False
    with _states_lock:
        expiry = _pending_states.pop(token, None)
    return expiry is not None and time.monotonic() < expiry


def _frontend_redirect_url(request: Request, *, auth_error: str | None = None) -> str:
    base_url = str(request.base_url).rstrip("/")
    if auth_error:
        return f"{base_url}/?auth_error={quote_plus(auth_error)}"
    return f"{base_url}/"


@router.get("/login")
def login() -> RedirectResponse:
    state = _generate_state()
    params = {
        "client_id": settings.GMAIL_OAUTH_CLIENT_ID,
        "redirect_uri": settings.GMAIL_OAUTH_REDIRECT_URI,
        "response_type": "code",
        "scope": _SCOPES,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return RedirectResponse(_GOOGLE_AUTH_URL + "?" + urlencode(params))


@router.get("/callback")
def callback(
    request: Request,
    code: str = "",
    error: str = "",
    state: str = "",
) -> RedirectResponse:
    if error:
        logger.error("OAuth consent denied: %s", error)
        return RedirectResponse(_frontend_redirect_url(request, auth_error=error))

    if not _verify_and_consume_state(state):
        logger.error(
            "OAuth callback: missing or invalid state parameter — possible CSRF attempt"
        )
        return RedirectResponse(
            _frontend_redirect_url(request, auth_error="invalid_state")
        )

    response = requests.post(
        _GOOGLE_TOKEN_URL,
        data={
            "client_id": settings.GMAIL_OAUTH_CLIENT_ID,
            "client_secret": settings.GMAIL_OAUTH_CLIENT_SECRET,
            "redirect_uri": settings.GMAIL_OAUTH_REDIRECT_URI,
            "grant_type": "authorization_code",
            "code": code,
        },
        timeout=30,
    )

    if not response.ok:
        try:
            detail = response.json()
            err = f"{detail.get('error')}: {detail.get('error_description')}"
        except Exception:
            err = response.text
        logger.error("OAuth token exchange failed: %s", err)
        return RedirectResponse(_frontend_redirect_url(request, auth_error=err))

    tokens = response.json()
    refresh_token = tokens.get("refresh_token", "")
    access_token = tokens.get("access_token", "")

    if not refresh_token:
        logger.error("Google did not return a refresh_token")
        return RedirectResponse(
            _frontend_redirect_url(request, auth_error="no_refresh_token")
        )

    token_store.set_tokens(refresh_token=refresh_token, access_token=access_token)
    logger.info("OAuth callback: Gmail connected successfully")
    return RedirectResponse(_frontend_redirect_url(request))


@router.get("/status")
def status() -> dict:
    return {"authenticated": token_store.is_authenticated()}


@router.post("/logout")
def logout() -> dict:
    token_store.clear()
    logger.info("Gmail disconnected")
    return {"authenticated": False}
