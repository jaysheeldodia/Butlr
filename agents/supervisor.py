from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.calendar_agent import extract_calendar_tasks
from agents.classifier import classify_email
from agents.drafter import generate_draft
from agents.extractor import extract_tasks
from agents.rag import retrieve_context
from memory.vector_store import VectorStore
from schemas import CalendarEvent, DraftResponse, Email, ExtractedTask, PipelineResult
from utils.logger import logger


def _process_email(
    email: Email,
    vector_store: VectorStore,
) -> tuple[list[ExtractedTask], dict[str, DraftResponse], bool]:
    """Classify, extract tasks, and optionally draft a reply for a single email.

    Returns a (tasks, drafts, failed) tuple. Errors are caught per-email so one
    bad message never aborts the entire pipeline run. The third element reports
    whether this email was dropped by an internal failure, so callers can
    surface silent drops instead of hiding them.
    """
    try:
        classification = classify_email(email)
        if classification.is_noise:
            logger.info("Skipping noisy email %s (%s)", email.id, classification.category)
            return [], {}, False

        context = retrieve_context(f"{email.subject} {email.body}", vector_store)
        tasks = extract_tasks(email, context)

        drafts: dict[str, DraftResponse] = {}
        if classification.category == "requires_reply":
            drafts[email.id] = generate_draft(email)

        return tasks, drafts, False
    except Exception:
        logger.exception("Failed to process email %s — skipping", email.id)
        return [], {}, True


def run_pipeline(
    emails: list[Email],
    vector_store: VectorStore,
    calendar_events: list[CalendarEvent] | None = None,
    stats: dict | None = None,
) -> PipelineResult:
    """Run the full pipeline. If *stats* is a dict, it receives a
    "dropped_email_count" key with the number of emails dropped by internal
    failures, so callers can surface silent drops."""
    logger.info(
        "Running Butlr pipeline: %d email(s), %d calendar event(s)",
        len(emails),
        len(calendar_events) if calendar_events else 0,
    )

    unified_tasks: list[ExtractedTask] = []
    drafts: dict[str, DraftResponse] = {}
    dropped = 0

    # Process emails concurrently — each email is independent.
    with ThreadPoolExecutor(max_workers=min(len(emails), 8) or 1) as pool:
        futures = {pool.submit(_process_email, email, vector_store): email for email in emails}
        for future in as_completed(futures):
            tasks, email_drafts, failed = future.result()
            if failed:
                dropped += 1
            unified_tasks.extend(tasks)
            drafts.update(email_drafts)

    if stats is not None:
        stats["dropped_email_count"] = dropped

    # Process calendar events if provided.
    if calendar_events:
        try:
            calendar_tasks = extract_calendar_tasks(calendar_events)
            unified_tasks.extend(calendar_tasks)
        except Exception:
            logger.exception("Failed to extract calendar tasks — skipping")

    # Drop low-priority, non-actionable tasks to keep the dashboard focused.
    filtered_tasks = [
        task for task in unified_tasks
        if not (task.priority == "low" and not task.action_required)
    ]

    logger.info(
        "Pipeline complete: %d task(s) after filtering (%d draft(s))",
        len(filtered_tasks),
        len(drafts),
    )
    return PipelineResult(tasks=filtered_tasks, drafts=drafts)
