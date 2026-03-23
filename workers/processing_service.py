from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agents.supervisor import run_pipeline
from api import token_store
from config import settings
from memory.vector_store import VectorStore
from models.schemas import PipelineResult
from tools.calendar import CalendarTool
from tools.gmail import GmailTool
from utils.logger import logger

_PROCESSED_IDS_FILE = Path(__file__).resolve().parent.parent / ".butlr_processed"


def _merge_results(
    previous: PipelineResult | None,
    incoming: PipelineResult,
) -> PipelineResult:
    """Return a PipelineResult that combines *previous* and *incoming* tasks.

    Tasks from *incoming* take precedence when their IDs collide with existing
    ones.  Tasks from *previous* that have no match in *incoming* are preserved,
    so cards from earlier runs never disappear just because a new run produces
    fewer (or zero) actionable tasks.
    """
    if previous is None:
        return incoming

    incoming_ids = {t.id for t in incoming.tasks}
    retained = [t for t in previous.tasks if t.id not in incoming_ids]
    merged_tasks = incoming.tasks + retained
    merged_drafts = {**previous.drafts, **incoming.drafts}
    return PipelineResult(tasks=merged_tasks, drafts=merged_drafts)


@dataclass
class ProcessingSnapshot:
    result: PipelineResult | None = None
    last_run_at: str | None = None
    next_poll_at: str | None = None
    last_error: str | None = None
    processed_email_count: int = 0


class ProcessingService:
    def __init__(self) -> None:
        self.gmail_tool = GmailTool()
        self.calendar_tool = CalendarTool()
        self._snapshot = ProcessingSnapshot()
        self._processed_email_ids: set[str] = set()
        self._processed_calendar_event_ids: set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._load_processed_ids()

    # ── Processed ID persistence ───────────────────────────────────────────────

    def _load_processed_ids(self) -> None:
        try:
            if _PROCESSED_IDS_FILE.exists():
                data = json.loads(_PROCESSED_IDS_FILE.read_text())
                self._processed_email_ids = set(data.get("email_ids", []))
                self._processed_calendar_event_ids = set(data.get("calendar_event_ids", []))
                logger.info(
                    "Loaded %d processed email IDs and %d calendar event IDs from disk",
                    len(self._processed_email_ids),
                    len(self._processed_calendar_event_ids),
                )
        except Exception as exc:
            logger.warning("Could not load processed IDs from disk: %s", exc)

    def _save_processed_ids(self) -> None:
        """Persist processed IDs so deduplication survives restarts. Called while holding _lock."""
        try:
            data = {
                "email_ids": list(self._processed_email_ids),
                "calendar_event_ids": list(self._processed_calendar_event_ids),
            }
            _PROCESSED_IDS_FILE.write_text(json.dumps(data))
        except Exception as exc:
            logger.warning("Could not save processed IDs to disk: %s", exc)

    # ── Scheduling helpers ─────────────────────────────────────────────────────

    def _next_poll_time(self, *, from_time: datetime | None = None) -> str:
        base = from_time or datetime.now(tz=UTC)
        return (base + timedelta(seconds=max(settings.GMAIL_POLL_INTERVAL_SECONDS, 15))).isoformat()

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="gmail-poller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    # ── Core processing ────────────────────────────────────────────────────────

    def run_once(self, *, force: bool = False) -> ProcessingSnapshot:
        if not token_store.is_authenticated():
            logger.info("Skipping processing run: not authenticated")
            with self._lock:
                self._snapshot.next_poll_at = None
            return self.get_snapshot()

        logger.info("Starting Butlr processing run")
        completed_at = datetime.now(tz=UTC)
        next_poll_at = self._next_poll_time(from_time=completed_at)

        try:
            emails = self.gmail_tool.run()

            with self._lock:
                new_emails = [
                    e for e in emails if force or e.id not in self._processed_email_ids
                ]
                if not new_emails and self._snapshot.result is not None and not force:
                    logger.info("No new emails since last run — skipping pipeline")
                    self._snapshot.last_error = None
                    self._snapshot.next_poll_at = next_poll_at
                    return self._copy_snapshot()

            # Embed only genuinely new emails (force flag does not bypass deduplication
            # for embeddings to avoid duplicating vectors in the store).
            vector_store = VectorStore()
            emails_to_embed = [e for e in emails if e.id not in self._processed_email_ids]
            for email in emails_to_embed:
                vector_store.add(
                    email.body,
                    {"source": "email", "title": email.subject, "content": email.body, "id": email.id},
                )
            if emails_to_embed:
                vector_store.save()

            try:
                calendar_events = self.calendar_tool.run()
            except Exception as exc:
                logger.warning("Calendar sync unavailable, continuing without events: %s", exc)
                calendar_events = []

            with self._lock:
                new_calendar_events = [
                    e for e in calendar_events
                    if force or e.id not in self._processed_calendar_event_ids
                ]

            result = run_pipeline(new_emails, vector_store, calendar_events=new_calendar_events)

            with self._lock:
                self._processed_email_ids.update(e.id for e in emails)
                self._processed_calendar_event_ids.update(e.id for e in calendar_events)
                self._save_processed_ids()

                # Merge new tasks with those from previous runs so cards are never
                # wiped when a run produces fewer (or zero) new actionable tasks.
                # New tasks win on ID collision; old tasks without a matching new ID persist.
                merged_result = _merge_results(self._snapshot.result, result)

                self._snapshot = ProcessingSnapshot(
                    result=merged_result,
                    last_run_at=completed_at.isoformat(),
                    next_poll_at=next_poll_at,
                    last_error=None,
                    processed_email_count=len(new_emails),
                )
                return self._copy_snapshot()

        except Exception as exc:
            logger.exception("Butlr processing run failed: %s", exc)
            with self._lock:
                self._snapshot.last_error = str(exc)
                self._snapshot.next_poll_at = next_poll_at
                return self._copy_snapshot()

    def get_snapshot(self) -> ProcessingSnapshot:
        with self._lock:
            return self._copy_snapshot()

    def _copy_snapshot(self) -> ProcessingSnapshot:
        return ProcessingSnapshot(
            result=self._snapshot.result,
            last_run_at=self._snapshot.last_run_at,
            next_poll_at=self._snapshot.next_poll_at,
            last_error=self._snapshot.last_error,
            processed_email_count=self._snapshot.processed_email_count,
        )

    def _poll_loop(self) -> None:
        self.run_once()
        interval = max(settings.GMAIL_POLL_INTERVAL_SECONDS, 15)
        while not self._stop_event.wait(interval):
            self.run_once()


processing_service = ProcessingService()
