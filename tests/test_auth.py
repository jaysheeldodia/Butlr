"""Tests for OAuth 2.0 authentication flow (state/CSRF, token exchange, callbacks)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from api.auth import _generate_state, _verify_and_consume_state
from api.main import app


@pytest.fixture
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ── State token helpers ────────────────────────────────────────────────────────


def test_generate_state_is_unique():
    s1 = _generate_state()
    s2 = _generate_state()
    assert s1 != s2


def test_verify_state_valid():
    token = _generate_state()
    assert _verify_and_consume_state(token) is True


def test_verify_state_consumed_once():
    token = _generate_state()
    assert _verify_and_consume_state(token) is True
    # Second use of the same token must fail (replay protection)
    assert _verify_and_consume_state(token) is False


def test_verify_state_unknown_token():
    assert _verify_and_consume_state("not-a-real-token") is False


def test_verify_state_empty_string():
    assert _verify_and_consume_state("") is False


# ── /auth/login ────────────────────────────────────────────────────────────────


def test_login_redirects_to_google(client):
    response = client.get("/auth/login", follow_redirects=False)
    assert response.status_code in (302, 307)
    location = response.headers["location"]
    assert "accounts.google.com" in location


def test_login_includes_state_param(client):
    response = client.get("/auth/login", follow_redirects=False)
    location = response.headers["location"]
    parsed = urlparse(location)
    params = parse_qs(parsed.query)
    assert "state" in params
    assert len(params["state"][0]) > 10


def test_login_state_verifiable(client):
    """The state token embedded in the login URL must pass verification."""
    response = client.get("/auth/login", follow_redirects=False)
    location = response.headers["location"]
    parsed = urlparse(location)
    params = parse_qs(parsed.query)
    state = params["state"][0]
    assert _verify_and_consume_state(state) is True


# ── /auth/callback ─────────────────────────────────────────────────────────────


def test_callback_error_param_redirects_home(client):
    response = client.get(
        "/auth/callback?error=access_denied&state=whatever",
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert "auth_error=access_denied" in response.headers["location"]


def test_callback_invalid_state_redirects_with_error(client):
    response = client.get(
        "/auth/callback?code=mycode&state=bogus-state",
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert "invalid_state" in response.headers["location"]


def test_callback_missing_state_redirects_with_error(client):
    response = client.get(
        "/auth/callback?code=mycode",
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert "invalid_state" in response.headers["location"]


def test_callback_replayed_state_redirects_with_error(client):
    """Using a state token twice must fail on the second attempt."""
    state = _generate_state()
    _verify_and_consume_state(state)  # consume it

    response = client.get(
        f"/auth/callback?code=mycode&state={state}",
        follow_redirects=False,
    )
    assert response.status_code in (302, 307)
    assert "invalid_state" in response.headers["location"]


def test_callback_successful_exchange_stores_tokens(client, monkeypatch):
    """A valid state + successful token exchange stores the refresh token."""
    import api.token_store as ts

    state = _generate_state()

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {
        "refresh_token": "rt-abc123",
        "access_token": "at-xyz789",
    }

    with patch("api.auth.requests.post", return_value=mock_response):
        response = client.get(
            f"/auth/callback?code=authcode&state={state}",
            follow_redirects=False,
        )

    assert response.status_code in (302, 307)
    assert "auth_error" not in response.headers["location"]
    assert ts.get_refresh_token() == "rt-abc123"


def test_callback_token_exchange_failure_redirects_with_error(client):
    state = _generate_state()

    mock_response = MagicMock()
    mock_response.ok = False
    mock_response.json.return_value = {
        "error": "invalid_grant",
        "error_description": "Code expired",
    }

    with patch("api.auth.requests.post", return_value=mock_response):
        response = client.get(
            f"/auth/callback?code=badcode&state={state}",
            follow_redirects=False,
        )

    assert response.status_code in (302, 307)
    assert "auth_error" in response.headers["location"]


def test_callback_no_refresh_token_redirects_with_error(client):
    state = _generate_state()

    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"access_token": "at-only"}  # no refresh_token

    with patch("api.auth.requests.post", return_value=mock_response):
        response = client.get(
            f"/auth/callback?code=authcode&state={state}",
            follow_redirects=False,
        )

    assert response.status_code in (302, 307)
    assert "no_refresh_token" in response.headers["location"]


# ── /auth/status & /auth/logout ────────────────────────────────────────────────


def test_status_unauthenticated(client):
    response = client.get("/auth/status")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}


def test_status_authenticated(client):
    import api.token_store as ts
    ts.set_tokens(refresh_token="some-token")
    response = client.get("/auth/status")
    assert response.json()["authenticated"] is True


def test_logout_clears_token(client):
    import api.token_store as ts
    ts.set_tokens(refresh_token="some-token")
    response = client.post("/auth/logout")
    assert response.status_code == 200
    assert response.json() == {"authenticated": False}
    assert not ts.is_authenticated()
