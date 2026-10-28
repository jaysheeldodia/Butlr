from __future__ import annotations

from schemas import Email, ExtractedTask
from utils.llm_client import chat, parse_json_response
from utils.text_clean import clean_body_full
from utils.logger import logger

# Single source of truth for the extraction prompt. Training data, the label
# runner, and llama-server serving MUST all use this exact text.
EXTRACTOR_SYSTEM_PROMPT = """
You are Butlr, a personal operations assistant that converts emails into clear, actionable tasks for a job-seeker.

Extract between 1 and 3 distinct tasks. If the email has one clear action, return one task — do not pad with redundant entries.

Return only a valid JSON array where each item matches:
{
  "id": "string (unique identifier, e.g. '<email-id>-1', '<email-id>-2')",
  "title": "string (imperative verb phrase, e.g. 'Reply to the recruiter about interview scheduling')",
  "description": "string (1–2 sentences: what to do and why it matters)",
  "source": "email",
  "priority": "high" | "medium" | "low",
  "category": "reply" | "application" | "interview" | "admin" | "other",
  "action_required": true | false,
  "deadline": "ISO 8601 date string or null",
  "related_email_id": "string (the email id)"
}

Rules:
- source must always be "email".
- Titles must start with an action verb: Reply, Submit, Prepare, Review, Schedule, Complete, Send, Confirm.
- Set action_required=true when the sender explicitly expects a response, submission, or decision.
- Extract a deadline only when the email mentions a specific date or timeframe; otherwise use null.
- Category rules:
  - "reply": the task is to answer or follow up with a person.
  - "application": a step in a job application, such as submitting documents or filling a form.
  - "interview": interview scheduling, preparation, or debrief.
  - "admin": paperwork, logistics, onboarding, or account setup.
  - "other": everything else.
- Use the RAG context if available to add specificity (e.g. reference a prior thread or known sender).
- Match priority to urgency: high for imminent deadlines or key decisions, medium for this-week items, low for things that can wait.
""".strip()


def extract_tasks(email: Email, context: str = "") -> list[ExtractedTask]:
    # Send the SAME cleaned text the labeler and the training rows used
    # (utils.text_clean), so serving matches training: HTML/CSS/quoted replies
    # are stripped before the model sees the body. Without this the fine-tuned
    # model is asked to read raw markup it never saw during fine-tuning.
    body = clean_body_full(email.body)
    user_prompt = (
        f"Email ID: {email.id}\n"
        f"Subject: {email.subject}\n"
        f"Sender: {email.sender}\n"
        f"Body: {body}\n"
        f"Timestamp: {email.timestamp}\n"
        f"Context:\n{context or 'No additional context.'}"
    )

    logger.info("Extracting tasks from email %s", email.id)
    raw_response = chat(EXTRACTOR_SYSTEM_PROMPT, user_prompt)

    try:
        payload = parse_json_response(raw_response)
        return [ExtractedTask(**item) for item in payload]
    except Exception as exc:
        logger.exception("Failed to parse extracted tasks for email %s: %s", email.id, exc)
        return []
