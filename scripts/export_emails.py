"""Export recent Gmail messages to a local JSONL pool.

PRIVACY: writes email content to data/raw/emails.jsonl on this machine only.
Prints counts, never content. Run manually by Jay. Google sees a normal
Gmail read; no LLM provider is contacted.

Rate limits: the Gmail API enforces per-user quotas and returns 403
(rateLimitExceeded / quotaExceeded) instead of 429. This script fetches
message by message with a small delay and retries with exponential backoff.
For --max 3000 expect roughly 10-30 minutes at a safe pace; run it and
let it work. Resumable: already-fetched IDs are skipped on rerun.

Usage:
    python -m scripts.export_emails --max 3000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.gmail import GmailTool  # noqa: E402

OUT_PATH = REPO_ROOT / "data" / "raw" / "emails.jsonl"

FETCH_PAUSE_SECONDS = 0.5          # pause between message fetches
RETRY_PAUSE_SECONDS = 5            # base for exponential backoff on 403/429/5xx
MAX_RETRIES = 8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=3000, help="max messages to fetch")
    parser.add_argument("--pause", type=float, default=FETCH_PAUSE_SECONDS)
    args = parser.parse_args()

    tool = GmailTool(max_results=args.max)

    access_token = tool._refresh_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    message_ids = tool._list_message_ids(headers)
    print(f"Gmail listed {len(message_ids)} message IDs")

    # Resume support: skip IDs already in the raw file.
    existing: dict[str, str] = {}
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["id"]] = line
    print(f"Resuming: {len(existing)} already on disk")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fetched = retried = failed = 0
    with OUT_PATH.open("a") as fh:
        for i, message_id in enumerate(message_ids):
            if message_id in existing:
                continue
            # Fetch one message with retries and exponential backoff.
            payload = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    payload = tool._get_message(headers, message_id)
                    break
                except Exception as exc:
                    retried += 1
                    wait = RETRY_PAUSE_SECONDS * (2 ** (attempt - 1))
                    detail = getattr(exc, "response", None)
                    reason = ""
                    if detail is not None:
                        try:
                            reason = detail.json().get("error", {}).get("errors", [{}])[0].get("reason", "")
                        except Exception:
                            pass
                    print(f"  fetch failed (attempt {attempt}, reason={reason or 'n/a'}): waiting {wait}s")
                    if attempt == MAX_RETRIES:
                        print(f"  GIVING UP on message {message_id}")
                        failed += 1
                        break
                    time.sleep(wait)
            if payload is None:
                time.sleep(args.pause)
                continue
            email = tool._message_to_email(payload)
            fh.write(
                json.dumps(
                    {
                        "id": email.id,
                        "subject": email.subject,
                        "sender": email.sender,
                        "body": email.body,
                        "timestamp": email.timestamp,
                    }
                )
                + "\n"
            )
            fetched += 1
            fh.flush()

            if (fetched + len(existing)) % 25 == 0:
                print(f"  progress: {fetched + len(existing)}/{len(message_ids)} (retried={retried}, failed={failed})")
            time.sleep(args.pause)

    print(f"Fetched {fetched} new messages ({len(existing)} were already on disk)")
    print(f"Retried {retried} times, gave up on {failed} messages")
    print(f"Wrote {OUT_PATH}")
    print("Done. Content is local-only. Do not commit or upload data/.")


if __name__ == "__main__":
    main()
