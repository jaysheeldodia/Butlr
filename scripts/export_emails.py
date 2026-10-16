"""Export recent Gmail messages to a local JSONL pool.

PRIVACY: writes email content to data/raw/emails.jsonl on this machine only.
Prints counts, never content. Run manually by Jay. Google sees a normal
Gmail read; no LLM provider is contacted.

Usage:
    python -m scripts.export_emails --max 3000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.gmail import GmailTool  # noqa: E402

OUT_PATH = REPO_ROOT / "data" / "raw" / "emails.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max", type=int, default=3000, help="max messages to fetch")
    args = parser.parse_args()

    tool = GmailTool(max_results=args.max)
    emails = tool.run()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        for email in emails:
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

    print(f"Fetched {len(emails)} messages")
    print(f"Wrote {OUT_PATH}")
    print("Done. Content is local-only. Do not commit or upload data/.")


if __name__ == "__main__":
    main()
