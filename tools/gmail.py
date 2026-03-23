from __future__ import annotations

import base64
import html
import re
from datetime import UTC, datetime
from typing import Any, Iterable

import requests

from api import token_store
from config import settings
from models.schemas import Email
from tools.base import BaseTool

GMAIL_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API_BASE_URL = "https://gmail.googleapis.com/gmail/v1/users/me"


class GmailTool(BaseTool):
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        max_results: int | None = None,
        query: str | None = None,
        label_ids: Iterable[str] | None = None,
    ) -> None:
        self.session = session or requests.Session()
        self.max_results = max_results or settings.GMAIL_MAX_RESULTS
        self.query = query if query is not None else settings.GMAIL_QUERY
        self.label_ids = list(label_ids) if label_ids is not None else settings.GMAIL_LABEL_IDS

    def run(self) -> list[Email]:
        access_token = self._refresh_access_token()
        headers = {"Authorization": f"Bearer {access_token}"}
        message_ids = self._list_message_ids(headers)

        emails: list[Email] = []
        for message_id in message_ids:
            payload = self._get_message(headers, message_id)
            emails.append(self._message_to_email(payload))

        emails.sort(key=lambda email: email.timestamp, reverse=True)
        return emails

    def _refresh_access_token(self) -> str:
        refresh_token = token_store.get_refresh_token() or settings.GMAIL_OAUTH_REFRESH_TOKEN

        required = {
            "GMAIL_OAUTH_CLIENT_ID": settings.GMAIL_OAUTH_CLIENT_ID,
            "GMAIL_OAUTH_CLIENT_SECRET": settings.GMAIL_OAUTH_CLIENT_SECRET,
            "GMAIL_OAUTH_REFRESH_TOKEN": refresh_token,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise RuntimeError(f"Missing Gmail OAuth configuration: {missing_str}")

        response = self.session.post(
            GMAIL_TOKEN_URL,
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
            raise RuntimeError(
                f"Google token refresh failed ({error}): {description}"
            )

        token_payload = response.json()
        access_token = token_payload.get("access_token")
        if not access_token:
            raise RuntimeError("Google OAuth token response did not include access_token")
        return access_token

    def _list_message_ids(self, headers: dict[str, str]) -> list[str]:
        params: dict[str, Any] = {
            "maxResults": self.max_results,
            "q": self.query,
        }
        if self.label_ids:
            params["labelIds"] = self.label_ids

        response = self.session.get(
            f"{GMAIL_API_BASE_URL}/messages",
            headers=headers,
            params=params,
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()
        return [message["id"] for message in payload.get("messages", []) if message.get("id")]

    def _get_message(self, headers: dict[str, str], message_id: str) -> dict[str, Any]:
        response = self.session.get(
            f"{GMAIL_API_BASE_URL}/messages/{message_id}",
            headers=headers,
            params={"format": "full"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _message_to_email(self, message: dict[str, Any]) -> Email:
        payload = message.get("payload", {})
        headers = self._headers_to_dict(payload.get("headers", []))
        body = self._extract_body(payload) or message.get("snippet", "").strip()

        internal_date = message.get("internalDate")
        timestamp = datetime.now(tz=UTC).isoformat()
        if internal_date:
            timestamp = datetime.fromtimestamp(
                int(internal_date) / 1000,
                tz=UTC,
            ).isoformat()

        return Email(
            id=message["id"],
            subject=headers.get("subject", "(no subject)"),
            sender=headers.get("from", "unknown"),
            body=body or "(empty body)",
            timestamp=timestamp,
        )

    def _headers_to_dict(self, headers: list[dict[str, str]]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for item in headers:
            name = item.get("name", "").strip().lower()
            value = item.get("value", "").strip()
            if name and value:
                normalized[name] = value
        return normalized

    def _extract_body(self, payload: dict[str, Any]) -> str:
        plain_bodies = list(self._walk_bodies(payload, mime_type="text/plain"))
        if plain_bodies:
            return self._normalize_text("\n".join(plain_bodies))

        html_bodies = list(self._walk_bodies(payload, mime_type="text/html"))
        if html_bodies:
            return self._normalize_text(self._strip_html("\n".join(html_bodies)))

        body_data = payload.get("body", {}).get("data")
        if body_data:
            return self._normalize_text(self._decode_base64(body_data))
        return ""

    def _walk_bodies(self, payload: dict[str, Any], *, mime_type: str) -> Iterable[str]:
        if payload.get("mimeType") == mime_type:
            body_data = payload.get("body", {}).get("data")
            if body_data:
                yield self._decode_base64(body_data)

        for part in payload.get("parts", []) or []:
            yield from self._walk_bodies(part, mime_type=mime_type)

    def _decode_base64(self, value: str) -> str:
        padding = "=" * (-len(value) % 4)
        decoded = base64.urlsafe_b64decode(value + padding)
        return decoded.decode("utf-8", errors="replace")

    def _strip_html(self, value: str) -> str:
        without_tags = re.sub(r"<[^>]+>", " ", value)
        return html.unescape(without_tags)

    def _normalize_text(self, value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()
