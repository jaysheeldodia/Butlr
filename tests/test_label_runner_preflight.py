"""Tests for the label_runner endpoint preflight.

The bug this guards: with no llama-server running, every row raised
ConnectError, so a run printed one full traceback per email and wrote an empty
output file - which looks like a code bug but is really "no model server".
"""
from __future__ import annotations

import httpx

import scripts.label_runner as lr

URL = "http://127.0.0.1:8080/v1/chat/completions"


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


def test_unreachable_reports_error(monkeypatch):
    def boom(url, timeout=None):
        raise httpx.ConnectError("[Errno 61] Connection refused")

    monkeypatch.setattr(httpx, "get", boom)
    ok, detail = lr.preflight(URL)
    assert ok is False
    assert "ConnectError" in detail


def test_reachable_when_server_answers(monkeypatch):
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Response(200))
    ok, detail = lr.preflight(URL)
    assert ok is True
    assert "200" in detail


def test_loading_server_counts_as_reachable(monkeypatch):
    # 503 means the server is up and still loading the model - not a
    # connection problem, and not a reason to tell Jay to start the server.
    monkeypatch.setattr(httpx, "get", lambda url, timeout=None: _Response(503))
    ok, _detail = lr.preflight(URL)
    assert ok is True


def test_probe_targets_the_host_root_not_the_chat_path(monkeypatch):
    seen: list[str] = []

    def record(url, timeout=None):
        seen.append(url)
        return _Response(200)

    monkeypatch.setattr(httpx, "get", record)
    lr.preflight(URL)
    assert seen[0] == "http://127.0.0.1:8080/health"


def test_falls_back_to_models_endpoint(monkeypatch):
    seen: list[str] = []

    def flaky(url, timeout=None):
        seen.append(url)
        if url.endswith("/health"):
            raise httpx.ConnectError("refused")
        return _Response(200)

    monkeypatch.setattr(httpx, "get", flaky)
    ok, _detail = lr.preflight(URL)
    assert ok is True
    assert seen == ["http://127.0.0.1:8080/health", "http://127.0.0.1:8080/v1/models"]
