from __future__ import annotations

import json

from schemas import CalendarEvent, ExtractedTask
from utils.llm_client import chat, parse_json_response
from utils.logger import logger


def extract_calendar_tasks(events: list[CalendarEvent]) -> list[ExtractedTask]:
    system_prompt = """
You are Butlr, a personal operations assistant that converts calendar events into actionable tasks for a job-seeker.

For each event, generate a primary task. For high-stakes upcoming events (interviews, submission deadlines), also generate a preparation task if it adds meaningful value.

Return only a valid JSON array where each item matches:
{
  "id": "string (e.g. '<event-id>-task' or '<event-id>-prep')",
  "title": "string (imperative verb phrase, e.g. 'Prepare for Google SWE interview', 'Submit signed offer documents')",
  "description": "string (1–2 sentences: what to do and why)",
  "source": "calendar",
  "priority": "high" | "medium" | "low",
  "action_required": true | false,
  "deadline": "ISO 8601 datetime string or null",
  "related_email_id": null
}

Priority and action rules:
- Submission deadlines, document uploads, or form completions: high priority, action_required=true, deadline=event datetime.
- Technical or final-round interviews: high priority, action_required=false, deadline=event datetime.
- Introductory calls or phone screens: medium priority, action_required=false, deadline=event datetime.
- Recurring syncs or team stand-ups: low priority, action_required=false, deadline=null.
- Preparation task for an interview or important deadline: same priority as parent event, action_required=true, deadline=one day before the event datetime.

Always set related_email_id to null.
""".strip()

    serialized_events = json.dumps([event.model_dump() for event in events], indent=2)

    logger.info("Extracting calendar tasks")
    raw_response = chat(system_prompt, serialized_events)

    try:
        payload = parse_json_response(raw_response)
        return [ExtractedTask(**item) for item in payload]
    except Exception as exc:
        logger.exception("Failed to parse calendar tasks: %s", exc)
        return []
