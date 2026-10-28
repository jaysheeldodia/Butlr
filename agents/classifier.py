from __future__ import annotations

from schemas import ClassificationResult, Email
from utils.llm_client import chat, parse_json_response
from utils.logger import logger


def classify_email(email: Email) -> ClassificationResult:
    system_prompt = """
You are Butlr, a personal operations assistant that filters and prioritises incoming emails for a job-seeker.

Return only valid JSON matching this schema:
{
  "category": "requires_reply" | "informational" | "newsletter" | "spam",
  "priority": "high" | "medium" | "low",
  "is_noise": true | false
}

Category rules:
- "requires_reply": A real person expects a response or action. Includes recruiter outreach, interview invitations, offer letters, scheduling requests, document requests, and any direct question from a human sender.
- "informational": A useful update the user should know about but need not reply to. Includes application status updates, GitHub notifications, system alerts, and automated confirmations.
- "newsletter": Bulk communications not personally addressed to the user. Includes product digests, marketing emails, and mailing-list traffic.
- "spam": Irrelevant, unsolicited, or suspicious content with no professional value.

Priority rules:
- "high": Time-sensitive or career-critical. Interview invitations, offer letters, signing deadlines, or any message with an explicit due date within 7 days.
- "medium": Important but not urgent. Recruiter outreach worth responding to, active application updates, and direct informational messages from known contacts.
- "low": Low stakes or no deadline. General newsletters, passive system updates, and non-urgent automated notifications.

Noise rules:
- Set is_noise=true only for "newsletter" and "spam".
- Set is_noise=false for "requires_reply" and "informational".

When uncertain between two categories, prefer the one that is less likely to cause the user to miss something important.
""".strip()

    user_prompt = (
        f"Subject: {email.subject}\n"
        f"Sender: {email.sender}\n"
        f"Body: {email.body}\n"
        f"Timestamp: {email.timestamp}"
    )

    logger.info("Classifying email %s", email.id)
    raw_response = chat(system_prompt, user_prompt)

    try:
        payload = parse_json_response(raw_response)
        return ClassificationResult(**payload)
    except Exception as exc:
        logger.exception("Failed to parse classification for email %s: %s", email.id, exc)
        return ClassificationResult(
            category="informational",
            priority="low",
            is_noise=True,
        )
