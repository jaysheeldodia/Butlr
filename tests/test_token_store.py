"""Tests for encrypted token storage and persistence."""
from __future__ import annotations

import json

import pytest


def test_set_and_get_refresh_token():
    import api.token_store as ts
    ts.set_tokens(refresh_token="my-refresh-token", access_token="my-access-token")
    assert ts.get_refresh_token() == "my-refresh-token"


def test_is_authenticated_false_by_default():
    import api.token_store as ts
    assert not ts.is_authenticated()


def test_is_authenticated_true_after_set():
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")
    assert ts.is_authenticated()


def test_clear_removes_tokens():
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt", access_token="at")
    ts.clear()
    assert ts.get_refresh_token() == ""
    assert not ts.is_authenticated()


def test_clear_removes_token_file(tmp_path, monkeypatch):
    import api.token_store as ts
    token_file = tmp_path / ".butlr_tokens"
    monkeypatch.setattr(ts, "_TOKEN_FILE", token_file)
    ts.set_tokens(refresh_token="rt")
    assert token_file.exists()
    ts.clear()
    assert not token_file.exists()


def test_token_file_is_encrypted(tmp_path, monkeypatch):
    """The token file must not contain the refresh token in plaintext."""
    import api.token_store as ts
    token_file = tmp_path / ".butlr_tokens"
    monkeypatch.setattr(ts, "_TOKEN_FILE", token_file)
    ts.set_tokens(refresh_token="super-secret-token")
    raw_bytes = token_file.read_bytes()
    assert b"super-secret-token" not in raw_bytes


def test_token_persists_across_load(tmp_path, monkeypatch):
    """A token set in one session must be recoverable after _load_from_disk."""
    import api.token_store as ts
    token_file = tmp_path / ".butlr_tokens"
    key_file = tmp_path / ".butlr_key"
    monkeypatch.setattr(ts, "_TOKEN_FILE", token_file)
    monkeypatch.setattr(ts, "_KEY_FILE", key_file)

    ts.set_tokens(refresh_token="persistent-token")

    # Simulate a fresh load by resetting state and calling _load_from_disk
    monkeypatch.setattr(ts, "_refresh_token", "")
    ts._load_from_disk()

    assert ts.get_refresh_token() == "persistent-token"


def test_load_from_disk_handles_missing_file(tmp_path, monkeypatch):
    """Loading when no file exists should not raise."""
    import api.token_store as ts
    monkeypatch.setattr(ts, "_TOKEN_FILE", tmp_path / ".nonexistent")
    ts._load_from_disk()  # must not raise
    assert ts.get_refresh_token() == ""


def test_load_from_disk_handles_corrupted_file(tmp_path, monkeypatch):
    """Corrupted token file should log a warning and leave the token empty."""
    import api.token_store as ts
    token_file = tmp_path / ".butlr_tokens"
    token_file.write_bytes(b"this is not valid fernet ciphertext and not json either")
    monkeypatch.setattr(ts, "_TOKEN_FILE", token_file)
    ts._load_from_disk()  # must not raise
    assert ts.get_refresh_token() == ""


def test_legacy_plaintext_migration(tmp_path, monkeypatch):
    """A legacy plaintext JSON token file should be migrated to encrypted format."""
    import api.token_store as ts
    token_file = tmp_path / ".butlr_tokens"
    key_file = tmp_path / ".butlr_key"
    monkeypatch.setattr(ts, "_TOKEN_FILE", token_file)
    monkeypatch.setattr(ts, "_KEY_FILE", key_file)

    # Write plaintext JSON (legacy format)
    token_file.write_text(json.dumps({"refresh_token": "legacy-rt"}))
    monkeypatch.setattr(ts, "_refresh_token", "")
    ts._load_from_disk()

    assert ts.get_refresh_token() == "legacy-rt"

    # After migration the file must now be encrypted (not plaintext JSON)
    raw = token_file.read_bytes()
    assert b"legacy-rt" not in raw
