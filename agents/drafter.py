from __future__ import annotations

from schemas import DraftResponse, Email
from utils.llm_client import chat, parse_json_response
from utils.logger import logger


def generate_draft(email: Email) -> DraftResponse:
    system_prompt = """
You are Butlr, a personal operations assistant helping a job-seeker craft professional email replies.

Write a concise, warm reply. Match the tone of the original — formal for corporate outreach, conversational for startup recruiters.

Return only valid JSON matching:
{
  "subject": "string",
  "body": "string"
}

Rules:
- Subject must begin with "Re: " followed by the original subject line verbatim.
- Body structure: brief acknowledgment → address the key point or question → clear next step or call to action.
- Keep the body between 50 and 150 words. Be direct and do not over-explain.
- For scheduling requests: express availability in general terms (e.g. "I'm free Tuesday or Wednesday afternoon") — the user will confirm exact times.
- For document or asset requests: confirm you will provide the item and give a rough timeframe.
- For offer-related emails: express genuine enthusiasm and confirm receipt.
- Sign off with "Best," followed by a blank line for the user's name (do not invent a name).
- Do not fabricate specific dates, commitments, or details not present in the original email.
""".strip()

    user_prompt = (
        f"Email ID: {email.id}\n"
        f"Subject: {email.subject}\n"
        f"Sender: {email.sender}\n"
        f"Body: {email.body}"
    )

    logger.info("Generating draft for email %s", email.id)
    raw_response = chat(system_prompt, user_prompt)

    try:
        payload = parse_json_response(raw_response)
        return DraftResponse(**payload)
    except Exception as exc:
        logger.exception("Failed to parse draft for email %s: %s", email.id, exc)
        return DraftResponse(subject=f"Re: {email.subject}", body="")
