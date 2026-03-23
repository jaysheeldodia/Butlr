from __future__ import annotations

import os

import pytest

# Remove any real BUTLR_TOKEN_KEY from the environment so tests use auto-generated
# key files scoped to tmp_path rather than a potentially invalid env value.
os.environ.pop("BUTLR_TOKEN_KEY", None)


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    """Redirect all persistent files and reset all module-level token/auth state
    before each test so tests are fully isolated from each other and from any
    real credentials on the developer's machine."""
    import api.token_store as ts
    import workers.processing_service as ps
    from config import settings

    # Redirect runtime files to a per-test temporary directory
    monkeypatch.setattr(ts, "_TOKEN_FILE", tmp_path / ".butlr_tokens")
    monkeypatch.setattr(ts, "_KEY_FILE", tmp_path / ".butlr_key")
    monkeypatch.setattr(ps, "_PROCESSED_IDS_FILE", tmp_path / ".butlr_processed")

    # Reset in-memory token state so tests start unauthenticated
    monkeypatch.setattr(ts, "_refresh_token", "")
    monkeypatch.setattr(ts, "_access_token", "")

    # Prevent the FastAPI lifespan from auto-loading a real refresh token from .env
    monkeypatch.setattr(settings, "GMAIL_OAUTH_REFRESH_TOKEN", "")

    yield
