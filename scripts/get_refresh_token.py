#!/usr/bin/env python3
"""
One-time script to obtain a Gmail OAuth refresh token.

Usage:
    source .venv/bin/activate
    python scripts/get_refresh_token.py

Reads GMAIL_OAUTH_CLIENT_ID, GMAIL_OAUTH_CLIENT_SECRET, and
GMAIL_OAUTH_REDIRECT_URI from .env. Opens a browser for consent,
listens on the registered redirect URI, exchanges the code, and
prints the refresh token to paste into .env.
"""
from __future__ import annotations

import os
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.getenv("GMAIL_OAUTH_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("GMAIL_OAUTH_CLIENT_SECRET", "").strip()
REDIRECT_URI = os.getenv(
    "GMAIL_OAUTH_REDIRECT_URI", "http://localhost:8080/oauth2callback"
).strip()
SCOPES = " ".join([
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
])

missing = [name for name, val in [
    ("GMAIL_OAUTH_CLIENT_ID", CLIENT_ID),
    ("GMAIL_OAUTH_CLIENT_SECRET", CLIENT_SECRET),
] if not val]
if missing:
    sys.exit(f"Missing in .env: {', '.join(missing)}")

parsed_redirect = urlparse(REDIRECT_URI)
callback_path = parsed_redirect.path or "/"
port = parsed_redirect.port or 8080

auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode({
    "client_id": CLIENT_ID,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "scope": SCOPES,
    "access_type": "offline",
    "prompt": "consent",
})

received: dict[str, str] = {}


class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        req = urlparse(self.path)
        if req.path != callback_path:
            self._respond(404, b"Not found")
            return
        params = parse_qs(req.query)
        if "error" in params:
            received["error"] = params["error"][0]
            self._respond(400, b"Authorization denied. You can close this tab.")
        elif "code" in params:
            received["code"] = params["code"][0]
            self._respond(200, b"Authorization successful! You can close this tab.")
        else:
            self._respond(400, b"Missing code parameter.")

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        pass


print(f"Opening browser for Gmail authorization...")
print(f"If the browser does not open, visit:\n  {auth_url}\n")
webbrowser.open(auth_url)

server = HTTPServer(("localhost", port), CallbackHandler)
print(f"Waiting for Google to redirect to {REDIRECT_URI} ...")
server.handle_request()

if "error" in received:
    sys.exit(f"Authorization denied: {received['error']}")
if "code" not in received:
    sys.exit("No authorization code received.")

response = requests.post(
    "https://oauth2.googleapis.com/token",
    data={
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
        "code": received["code"],
    },
    timeout=30,
)
if not response.ok:
    try:
        detail = response.json()
        err = f"{detail.get('error')}: {detail.get('error_description')}"
    except Exception:
        err = response.text
    sys.exit(f"Token exchange failed: {err}")

tokens = response.json()
refresh_token = tokens.get("refresh_token")
if not refresh_token:
    sys.exit(
        "Google did not return a refresh_token. "
        "Revoke app access at https://myaccount.google.com/permissions and run again."
    )

print("\nSuccess! Add this line to your .env:\n")
print(f"GMAIL_OAUTH_REFRESH_TOKEN={refresh_token}\n")
