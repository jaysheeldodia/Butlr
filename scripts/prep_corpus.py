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

    # Cluster by thread. Default: keep the longest body per thread (most
    # context). --keep-all: keep every email, sorted oldest first so the
    # thread reads in order; make_splits still assigns whole threads to one
    # split, so keeping all emails cannot leak across train/test.
    threads: dict[str, list[dict]] = {}
    for row in deduped:
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
    print(f"Threads:            {len(threads)}")
    print(f"Pool rows:          {len(pool)} ({'keep-all' if args.keep_all else 'longest-per-thread'})")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
