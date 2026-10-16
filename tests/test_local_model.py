"""Tests for the local-model changes: local provider, embeddings swap,
vector store dims, dropped-email counting, and the category field."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import Email, ExtractedTask


def _make_email(id: str = "e1") -> Email:
    return Email(
        id=id,
        subject="Test",
        sender="a@b.com",
        body="Reply please.",
        timestamp="2026-10-15T00:00:00Z",
    )


# ── Category field ────────────────────────────────────────────────────────────

def test_extracted_task_category_defaults_to_other():
    task = ExtractedTask(
        id="t1", title="Reply", description="d", source="email",
        priority="low", action_required=True,
    )
    assert task.category == "other"


def test_extracted_task_rejects_unknown_category():
    with pytest.raises(Exception):
        ExtractedTask(
            id="t1", title="Reply", description="d", source="email",
            priority="low", category="weird", action_required=True,
        )


def test_extractor_prompt_contains_category_rules():
    from agents.extractor import EXTRACTOR_SYSTEM_PROMPT

    for word in ("reply", "application", "interview", "admin", "other"):
        assert f'"{word}"' in EXTRACTOR_SYSTEM_PROMPT


# ── Local provider branch ────────────────────────────────────────────────────

def test_local_provider_posts_to_local_url(monkeypatch):
    import utils.llm_client as lc

    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "[]"}}]}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("utils.llm_client.LLM_PROVIDER", "local")
    monkeypatch.setattr("httpx.post", fake_post)
    out = lc.chat("system", "user")
    assert out == "[]"
    assert captured["url"].startswith("http://127.0.0.1")
    assert captured["payload"]["messages"][0]["role"] == "system"


def test_local_provider_sends_bearer_and_extra_body(monkeypatch):
    import utils.llm_client as lc

    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "[]"}}]}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["url"] = url
        captured["payload"] = json
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("utils.llm_client.LLM_PROVIDER", "local")
    monkeypatch.setattr("config.settings.LOCAL_LLM_API_KEY", "sk-test-123")
    monkeypatch.setattr(
        "config.settings.LOCAL_LLM_EXTRA_BODY",
        '{"provider": {"zdr": true, "data_collection": "deny"}}',
    )
    monkeypatch.setattr("httpx.post", fake_post)
    lc.chat("s", "u")
    assert captured["headers"]["Authorization"] == "Bearer sk-test-123"
    assert captured["payload"]["provider"] == {"zdr": True, "data_collection": "deny"}


def test_local_provider_omits_bearer_when_no_key(monkeypatch):
    import utils.llm_client as lc

    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "x"}}]}

    def fake_post(url, json=None, timeout=None, headers=None):
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("utils.llm_client.LLM_PROVIDER", "local")
    monkeypatch.setattr("config.settings.LOCAL_LLM_API_KEY", "")
    monkeypatch.setattr("httpx.post", fake_post)
    lc.chat("s", "u")
    assert captured["headers"] == {}


def test_label_runner_forces_local_provider(monkeypatch):
    """label_runner must route to LOCAL_LLM_URL even when .env says openai."""
    import scripts.label_runner as lr
    import utils.llm_client as lc

    assert lc.LLM_PROVIDER == "local"  # set at label_runner import time
    assert lr.settings.LLM_PROVIDER == "local"


# ── Embeddings: no cloud call ────────────────────────────────────────────────

def test_embed_text_uses_local_model_not_openai():
    import memory.embeddings as emb

    fake = MagicMock()
    fake.encode.return_value = [0.1, 0.2]
    with patch("memory.embeddings._get_model", return_value=fake), \
         patch("openai.OpenAI") as mock_openai:
        emb._model = fake  # reset the cache to the fake
        result = emb.embed_text("hello")
        emb._model = None  # restore for other tests
    assert result == [0.1, 0.2]
    mock_openai.assert_not_called()


# ── Vector store dimension guard ─────────────────────────────────────────────

def test_vector_store_rebuilds_on_dim_mismatch(tmp_path, monkeypatch):
    import faiss

    import memory.vector_store as vs

    monkeypatch.setattr(vs, "_STORE_DIR", tmp_path)
    monkeypatch.setattr(vs, "_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(vs, "_META_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(vs, "EMBEDDING_DIM", 384)

    old_index = faiss.IndexFlatL2(1536)
    faiss.write_index(old_index, str(tmp_path / "faiss.index"))
    (tmp_path / "metadata.json").write_text(json.dumps([]))

    store = vs.VectorStore()
    assert store.dim == 384
    assert store.index.d == 384
    assert store.index.ntotal == 0  # rebuilt empty, old vectors not mixed in


# ── Dropped-email counting ───────────────────────────────────────────────────

def test_run_pipeline_reports_dropped_emails(monkeypatch):
    import agents.supervisor as sup

    emails = [_make_email("e1")]
    fake_store = MagicMock()
    fake_store.search.return_value = []

    monkeypatch.setattr(sup, "classify_email", MagicMock(side_effect=Exception("boom")))
    monkeypatch.setattr(sup, "extract_tasks", MagicMock())
    monkeypatch.setattr(sup, "generate_draft", MagicMock())

    stats: dict = {}
    sup.run_pipeline(emails, fake_store, stats=stats)
    assert stats["dropped_email_count"] == 1


def test_run_pipeline_no_drops_on_success(monkeypatch):
    import agents.supervisor as sup

    emails = [_make_email("e1")]
    fake_store = MagicMock()
    fake_store.search.return_value = []

    monkeypatch.setattr(
        sup, "classify_email",
        MagicMock(return_value=MagicMock(is_noise=True, category="newsletter")),
    )

    stats: dict = {}
    sup.run_pipeline(emails, fake_store, stats=stats)
    assert stats["dropped_email_count"] == 0
