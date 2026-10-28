from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import requests

from api import token_store
from config import settings
from schemas import CalendarEvent
from tools.base import BaseTool
from utils.logger import logger

CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3"
CALENDAR_TOKEN_URL = "https://oauth2.googleapis.com/token"


class CalendarTool(BaseTool):
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        lookahead_days: int | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.lookahead_days = lookahead_days if lookahead_days is not None else settings.CALENDAR_LOOKAHEAD_DAYS

    def run(self) -> list[CalendarEvent]:
        if not token_store.is_authenticated():
            logger.info("CalendarTool: not authenticated, returning empty list")
            return []

        access_token = self._refresh_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        return self._fetch_events(headers)

    def _refresh_access_token(self) -> str:
        refresh_token = token_store.get_refresh_token() or settings.GMAIL_OAUTH_REFRESH_TOKEN

        required = {
            "GMAIL_OAUTH_CLIENT_ID": settings.GMAIL_OAUTH_CLIENT_ID,
            "GMAIL_OAUTH_CLIENT_SECRET": settings.GMAIL_OAUTH_CLIENT_SECRET,
            "GMAIL_OAUTH_REFRESH_TOKEN": refresh_token,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(f"Missing Google OAuth configuration: {', '.join(sorted(missing))}")

        response = self.session.post(
            CALENDAR_TOKEN_URL,
            data={
                "client_id": settings.GMAIL_OAUTH_CLIENT_ID,
                "client_secret": settings.GMAIL_OAUTH_CLIENT_SECRET,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=30,
        )
        if not response.ok:
            try:
                detail = response.json()
                error = detail.get("error", response.status_code)
                description = detail.get("error_description", "")
            except Exception:
                error, description = response.status_code, response.text
            raise RuntimeError(f"Google token refresh failed ({error}): {description}")

        access_token = response.json().get("access_token")
        if not access_token:
            raise RuntimeError("Google OAuth token response did not include access_token")
        return access_token

    def _fetch_events(self, headers: dict[str, str]) -> list[CalendarEvent]:
        now = datetime.now(tz=UTC)
        time_max = now + timedelta(days=self.lookahead_days)

        response = self.session.get(
            f"{CALENDAR_API_BASE}/calendars/primary/events",
            headers=headers,
            params={
                "timeMin": now.isoformat(),
                "timeMax": time_max.isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": 25,
            },
            timeout=30,
        )
        response.raise_for_status()

        events: list[CalendarEvent] = []
        for item in response.json().get("items", []):
            event = self._item_to_event(item)
            if event is not None:
                events.append(event)

        logger.info("CalendarTool: fetched %d event(s)", len(events))
        return events

    def _item_to_event(self, item: dict[str, Any]) -> CalendarEvent | None:
        event_id = item.get("id", "")
        title = item.get("summary", "(No title)").strip()
        description = (item.get("description") or "").strip()
        location = item.get("location") or None

        start = item.get("start", {})
        dt_str = start.get("dateTime") or start.get("date", "")
        if not dt_str:
            return None

        return CalendarEvent(
            id=event_id,
            title=title,
            description=description,
            datetime=dt_str,
            location=location,
        )
