"""Prepare the raw email pool: cut quoted replies, dedupe, cluster threads.

Run manually by Jay. Reads data/raw/emails.jsonl, writes
data/corpus/pool.jsonl. Prints counts only.

Default keeps ONE email per thread (the longest body). With --keep-all,
every email in a thread is kept — more rows, but reply/follow-up emails
tend to be lower quality labels. Splits stay thread-clustered either way,
so there is no train/test leakage.

Usage:
    python -m scripts.prep_corpus [--keep-all]
"""
from __future__ import annotations

import argparse
import hashlib
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


def content_key(row: dict) -> str:
    """Key for emails with identical content: same sender and same body text.

    Periodic digests (GitHub/LinkedIn round-ups) arrive again and again with
    the same text but freshly re-titled subjects, so thread_key does not merge
    them. The audit found every duplicate-content row carries 0 tasks, so
    collapsing them removes dead weight without losing a single real task.
    The sender stays in the key, so two different senders that happen to ship
    a generic body ("View in browser") are never merged together. The subject
    is deliberately NOT in the key, because re-titling is exactly how these
    digests repeat.
    """
    body = re.sub(r"\s+", " ", row.get("body") or "").strip().lower()
    sender = (row.get("sender") or "").strip().lower()
    return hashlib.sha256(f"{sender}\n{body}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="keep every email per thread instead of only the longest body",
    )
    args = parser.parse_args()

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

    # Drop rows whose content is byte-identical to one already kept (same
    # sender + subject + body). These are periodic digests; the audit showed
    # all duplicate-content rows have 0 tasks.
    seen_content: set[str] = set()
    unique: list[dict] = []
    for row in deduped:
        key = content_key(row)
        if key in seen_content:
            continue
        seen_content.add(key)
        unique.append(row)
    content_dupes = len(deduped) - len(unique)

    # Cluster by thread. Default: keep the longest body per thread (most
    # context). --keep-all: keep every email, sorted oldest first so the
    # thread reads in order; make_splits still assigns whole threads to one
    # split, so keeping all emails cannot leak across train/test.
    threads: dict[str, list[dict]] = {}
    for row in unique:
        threads.setdefault(thread_key(row["subject"]), []).append(row)

    pool: list[dict] = []
    if args.keep_all:
        for members in threads.values():
            members.sort(key=lambda r: r["timestamp"])
            pool.extend(members)
    else:
        for members in threads.values():
            pool.append(max(members, key=lambda r: len(r["body"])))
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        for row in pool:
            fh.write(json.dumps(row) + "\n")

    print(f"Raw rows:           {total}")
    print(f"After ID dedupe:    {len(deduped)}")
    print(f"Identical content dropped: {content_dupes}")
    print(f"Threads:            {len(threads)}")
    print(f"Pool rows:          {len(pool)} ({'keep-all' if args.keep_all else 'longest-per-thread'})")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
