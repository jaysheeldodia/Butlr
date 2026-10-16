"""Weak-label the pool with the LOCAL labeler (Qwen3-8B via llama-server).

PRIVACY: this script talks ONLY to a local server on this machine
(LOCAL_LLM_URL). No cloud LLM sees any email content. Run manually by Jay;
resumable; safe to run overnight. Prints counts and timing, never content.

Prereq: llama-server serving Qwen3-8B Q4_K_M (or a smaller quant if free RAM
is under 6 GB) on 127.0.0.1:8080, thinking mode OFF.

Usage:
    python -m scripts.label_runner \
        --input data/corpus/pool.jsonl \
        --output data/labeled/weak.jsonl \
        --log data/labeled/label_log.jsonl \
        [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.extractor import EXTRACTOR_SYSTEM_PROMPT  # noqa: E402
from config import settings  # noqa: E402
from models.schemas import Email  # noqa: E402
from utils.llm_client import chat, parse_json_response  # noqa: E402

# Labeling prompt differs from the production prompt in exactly one way:
# it allows 0 tasks (an empty array) so newsletters/spam produce negatives.
LABELER_SYSTEM_PROMPT = EXTRACTOR_SYSTEM_PROMPT.replace(
    "Extract between 1 and 3 distinct tasks.",
    "Extract between 0 and 3 distinct tasks. If the email has no action item, return an empty JSON array [].",
)


def label_one(row: dict) -> tuple[list[dict], float]:
    """Call the local labeler for one email. Returns (tasks, latency_seconds)."""
    email = Email(
        id=row["id"],
        subject=row["subject"],
        sender=row["sender"],
        body=row["body"],
        timestamp=row["timestamp"],
    )
    user_prompt = (
        f"Email ID: {email.id}\n"
        f"Subject: {email.subject}\n"
        f"Sender: {email.sender}\n"
        f"Body: {email.body}\n"
        f"Timestamp: {email.timestamp}\n"
        f"Context:\nNo additional context."
    )
    start = time.perf_counter()
    response = chat(LABELER_SYSTEM_PROMPT, user_prompt)
    latency = time.perf_counter() - start
    payload = parse_json_response(response)
    return payload if isinstance(payload, list) else [], latency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/corpus/pool.jsonl")
    parser.add_argument("--output", default="data/labeled/weak.jsonl")
    parser.add_argument("--log", default="data/labeled/label_log.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    in_path = REPO_ROOT / args.input
    out_path = REPO_ROOT / args.output
    log_path = REPO_ROOT / args.log
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    done_ids: set[str] = set()
    if out_path.exists():
        done_ids = {
            json.loads(line)["email"]["id"]
            for line in out_path.read_text().splitlines()
            if line.strip()
        }
        print(f"Resuming: {len(done_ids)} rows already labeled")

    pending = [r for r in rows if r["id"] not in done_ids]
    if args.limit:
        pending = pending[: args.limit]
    print(f"Labeling {len(pending)} rows against {settings.LOCAL_LLM_URL}")

    ok = failed = 0
    total_latency = 0.0
    with out_path.open("a") as out_fh, log_path.open("a") as log_fh:
        for i, row in enumerate(pending):
            try:
                tasks, latency = label_one(row)
                out_fh.write(json.dumps({"email": row, "tasks": tasks}) + "\n")
                log_fh.write(
                    json.dumps(
                        {
                            "id": row["id"],
                            "status": "ok",
                            "latency_s": round(latency, 3),
                            "tasks": len(tasks),
                        }
                    )
                    + "\n"
                )
                ok += 1
                total_latency += latency
            except Exception as exc:  # one bad row must not stop the run
                log_fh.write(
                    json.dumps({"id": row["id"], "status": "failed", "error": str(exc)[:200]})
                    + "\n"
                )
                failed += 1
            if (i + 1) % 50 == 0:
                mean = total_latency / ok if ok else 0
                print(f"  progress {i + 1}/{len(pending)} (ok={ok}, failed={failed}, mean={mean:.1f}s)")
                out_fh.flush()
                log_fh.flush()

    mean = total_latency / ok if ok else 0
    print(f"Done. ok={ok} failed={failed} mean_latency={mean:.1f}s per email")
    print(f"Labels: {out_path}")
    print(f"Log:    {log_path}")
    print("NEXT: hand-verify the val and test rows, then run scripts.make_splits.")


if __name__ == "__main__":
    main()
