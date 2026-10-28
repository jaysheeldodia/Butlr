"""Tests for the ProcessingService: happy path, error path, and deduplication."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from schemas import CalendarEvent, Email, ExtractedTask, PipelineResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_email(id: str = "email-1") -> Email:
    return Email(
        id=id,
        subject="Test subject",
        sender="sender@example.com",
        body="Please review and reply.",
        timestamp="2026-03-23T10:00:00Z",
    )


def _make_task(id: str = "task-1", email_id: str = "email-1") -> ExtractedTask:
    return ExtractedTask(
        id=id,
        title="Review document",
        description="Sender asked for a review.",
        source="email",
        priority="medium",
        action_required=True,
        deadline=None,
        related_email_id=email_id,
    )


def _make_pipeline_result(tasks=None) -> PipelineResult:
    return PipelineResult(
        tasks=tasks or [_make_task()],
        drafts={},
    )


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A fresh ProcessingService with mocked I/O-heavy collaborators."""
    import workers.processing_service as ps

    monkeypatch.setattr(ps, "_PROCESSED_IDS_FILE", tmp_path / ".butlr_processed")

    with (
        patch("workers.processing_service.GmailTool", autospec=True),
        patch("workers.processing_service.CalendarTool", autospec=True),
        patch("workers.processing_service.VectorStore", autospec=True),
    ):
        svc = ps.ProcessingService()
        yield svc


# ── Authentication guard ───────────────────────────────────────────────────────

def test_run_once_skips_when_not_authenticated(service):
    import api.token_store as ts
    assert not ts.is_authenticated()
    snapshot = service.run_once()
    assert snapshot.result is None
    assert snapshot.next_poll_at is None


# ── Happy path ─────────────────────────────────────────────────────────────────

def test_run_once_processes_new_emails(service):
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    email = _make_email("e1")
    service.gmail_tool.run.return_value = [email]
    service.calendar_tool.run.return_value = []

    result = _make_pipeline_result([_make_task("t1", "e1")])
    with patch("workers.processing_service.run_pipeline", return_value=result):
        snapshot = service.run_once()

    assert snapshot.result is not None
    assert len(snapshot.result.tasks) == 1
    assert snapshot.processed_email_count == 1
    assert snapshot.last_error is None


def test_run_once_stores_processed_email_ids(service):
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    email = _make_email("e1")
    service.gmail_tool.run.return_value = [email]
    service.calendar_tool.run.return_value = []

    with patch("workers.processing_service.run_pipeline", return_value=_make_pipeline_result()):
        service.run_once()

    assert "e1" in service._processed_email_ids


def test_run_once_persists_processed_ids_to_disk(service, tmp_path, monkeypatch):
    import api.token_store as ts
    import workers.processing_service as ps

    ids_file = tmp_path / ".butlr_processed"
    monkeypatch.setattr(ps, "_PROCESSED_IDS_FILE", ids_file)
    ts.set_tokens(refresh_token="rt")

    service.gmail_tool.run.return_value = [_make_email("e1")]
    service.calendar_tool.run.return_value = []

    with patch("workers.processing_service.run_pipeline", return_value=_make_pipeline_result()):
        service.run_once()

    data = json.loads(ids_file.read_text())
    assert "e1" in data["email_ids"]


# ── Deduplication ──────────────────────────────────────────────────────────────

def test_run_once_skips_already_processed_emails(service):
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    email = _make_email("e1")
    service.gmail_tool.run.return_value = [email]
    service.calendar_tool.run.return_value = []

    # First run — populate snapshot
    with patch("workers.processing_service.run_pipeline", return_value=_make_pipeline_result()):
        service.run_once()

    # Second run — same email, no force flag
    with patch("workers.processing_service.run_pipeline") as mock_pipeline:
        snapshot = service.run_once()
        mock_pipeline.assert_not_called()

    # Snapshot from first run is preserved
    assert snapshot.result is not None


def test_run_once_force_reprocesses_seen_emails(service):
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    email = _make_email("e1")
    service.gmail_tool.run.return_value = [email]
    service.calendar_tool.run.return_value = []

    result1 = _make_pipeline_result([_make_task("t1", "e1")])
    result2 = _make_pipeline_result([_make_task("t2", "e1")])

    with patch("workers.processing_service.run_pipeline", return_value=result1):
        service.run_once()

    with patch("workers.processing_service.run_pipeline", return_value=result2) as mock_pipeline:
        snapshot = service.run_once(force=True)
        mock_pipeline.assert_called_once()

    assert snapshot.result.tasks[0].id == "t2"


def test_processed_ids_loaded_from_disk_on_init(tmp_path, monkeypatch):
    """IDs persisted in a previous session are loaded on service start."""
    import workers.processing_service as ps
    import api.token_store as ts

    ids_file = tmp_path / ".butlr_processed"
    ids_file.write_text(json.dumps({"email_ids": ["old-e1", "old-e2"], "calendar_event_ids": []}))
    monkeypatch.setattr(ps, "_PROCESSED_IDS_FILE", ids_file)

    with (
        patch("workers.processing_service.GmailTool", autospec=True),
        patch("workers.processing_service.CalendarTool", autospec=True),
        patch("workers.processing_service.VectorStore", autospec=True),
    ):
        svc = ps.ProcessingService()

    assert "old-e1" in svc._processed_email_ids
    assert "old-e2" in svc._processed_email_ids


# ── Calendar error isolation ───────────────────────────────────────────────────

def test_calendar_failure_does_not_abort_pipeline(service):
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    service.gmail_tool.run.return_value = [_make_email("e1")]
    service.calendar_tool.run.side_effect = Exception("Calendar API down")

    with patch("workers.processing_service.run_pipeline", return_value=_make_pipeline_result()) as mock_pipeline:
        snapshot = service.run_once()

    # pipeline should still have been called with empty calendar events
    assert mock_pipeline.called
    _, kwargs = mock_pipeline.call_args
    assert kwargs.get("calendar_events", []) == []
    assert snapshot.result is not None
    assert snapshot.last_error is None


# ── Task persistence (merge) ───────────────────────────────────────────────────

def test_tasks_from_previous_run_persist_when_new_run_yields_no_tasks(service):
    """Tasks extracted in run 1 must still be visible after run 2 produces 0 tasks."""
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    # Run 1 – email e1 yields task t1
    service.gmail_tool.run.return_value = [_make_email("e1")]
    service.calendar_tool.run.return_value = []
    result1 = _make_pipeline_result([_make_task("t1", "e1")])
    with patch("workers.processing_service.run_pipeline", return_value=result1):
        service.run_once()

    # Run 2 – new email e2 yields zero tasks (all noise/spam)
    service.gmail_tool.run.return_value = [_make_email("e1"), _make_email("e2")]
    result2 = PipelineResult(tasks=[], drafts={})
    with patch("workers.processing_service.run_pipeline", return_value=result2):
        snapshot = service.run_once()

    task_ids = {t.id for t in snapshot.result.tasks}
    assert "t1" in task_ids, "Task from run 1 must be preserved after run 2"


def test_new_task_with_same_id_replaces_old_task(service):
    """If a task ID appears in both runs, the new version takes precedence."""
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    service.gmail_tool.run.return_value = [_make_email("e1")]
    service.calendar_tool.run.return_value = []

    old_task = _make_task("shared-id", "e1")
    old_task = old_task.model_copy(update={"title": "Old title"})
    with patch("workers.processing_service.run_pipeline", return_value=PipelineResult(tasks=[old_task], drafts={})):
        service.run_once()

    service.gmail_tool.run.return_value = [_make_email("e1"), _make_email("e2")]
    new_task = _make_task("shared-id", "e2")
    new_task = new_task.model_copy(update={"title": "New title"})
    with patch("workers.processing_service.run_pipeline", return_value=PipelineResult(tasks=[new_task], drafts={})):
        snapshot = service.run_once(force=True)

    matching = [t for t in snapshot.result.tasks if t.id == "shared-id"]
    assert len(matching) == 1
    assert matching[0].title == "New title"


# ── Error path ─────────────────────────────────────────────────────────────────

def test_run_once_records_error_on_gmail_failure(service):
    import api.token_store as ts
    ts.set_tokens(refresh_token="rt")

    service.gmail_tool.run.side_effect = Exception("Gmail auth failed")

    snapshot = service.run_once()
    assert snapshot.result is None
    assert snapshot.last_error == "Gmail auth failed"
    assert snapshot.next_poll_at is not None  # retry is still scheduled
