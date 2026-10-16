"""Prepare the raw email pool: cut quoted replies, dedupe, cluster threads.

Run manually by Jay. Reads data/raw/emails.jsonl, writes
data/corpus/pool.jsonl. Prints counts only.

Usage:
    python -m scripts.prep_corpus
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RAW_PATH = REPO_ROOT / "data" / "raw" / "emails.jsonl"
OUT_PATH = REPO_ROOT / "data" / "corpus" / "pool.jsonl"

_QUOTE_PREFIX = re.compile(r"^\s*>")
_SUBJECT_NORMALIZER = re.compile(r"^(?:\s*(?:re|fwd|fw)\s*:\s*)+", re.IGNORECASE)


def cut_quoted_replies(body: str) -> str:
    """Drop lines that are quoted reply text (start with >)."""
    kept = [line for line in body.splitlines() if not _QUOTE_PREFIX.match(line)]
    return "\n".join(kept).strip()


def thread_key(subject: str) -> str:
    """Normalize a subject so replies of one thread share a key."""
    normalized = _SUBJECT_NORMALIZER.sub("", subject).strip().lower()
    return normalized or subject.strip().lower()


def main() -> None:
    if not RAW_PATH.exists():
        print(f"Missing {RAW_PATH}. Run scripts.export_emails first.")
        sys.exit(1)

    rows = [json.loads(line) for line in RAW_PATH.read_text().splitlines() if line.strip()]
    total = len(rows)

    # Dedupe by message ID.
    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for row in rows:
        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        deduped.append(row)

    # Cut quoted text.
    for row in deduped:
        row["body"] = cut_quoted_replies(row["body"]) or row["body"]

    # Cluster by thread; keep the longest body per thread (most context).
    threads: dict[str, dict] = {}
    for row in deduped:
        key = thread_key(row["subject"])
        current = threads.get(key)
        if current is None or len(row["body"]) > len(current["body"]):
            threads[key] = row

    pool = list(threads.values())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        for row in pool:
            fh.write(json.dumps(row) + "\n")

    print(f"Raw rows:           {total}")
    print(f"After ID dedupe:    {len(deduped)}")
    print(f"After thread merge: {len(pool)}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
