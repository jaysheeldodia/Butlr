from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000").strip()

GMAIL_OAUTH_CLIENT_ID: str = os.getenv("GMAIL_OAUTH_CLIENT_ID", "").strip()
GMAIL_OAUTH_CLIENT_SECRET: str = os.getenv("GMAIL_OAUTH_CLIENT_SECRET", "").strip()
GMAIL_OAUTH_REDIRECT_URI: str = os.getenv(
    "GMAIL_OAUTH_REDIRECT_URI", "http://localhost:8000/auth/callback"
).strip()
GMAIL_OAUTH_REFRESH_TOKEN: str = os.getenv("GMAIL_OAUTH_REFRESH_TOKEN", "").strip()

GMAIL_POLL_INTERVAL_SECONDS: int = int(
    os.getenv("GMAIL_POLL_INTERVAL_SECONDS", "300").strip()
)
GMAIL_MAX_RESULTS: int = int(os.getenv("GMAIL_MAX_RESULTS", "10").strip())
GMAIL_QUERY: str = os.getenv("GMAIL_QUERY", "in:inbox").strip()
GMAIL_LABEL_IDS: List[str] = [
    value.strip()
    for value in os.getenv("GMAIL_LABEL_IDS", "INBOX").split(",")
    if value.strip()
]
CALENDAR_LOOKAHEAD_DAYS: int = int(os.getenv("CALENDAR_LOOKAHEAD_DAYS", "14").strip())
