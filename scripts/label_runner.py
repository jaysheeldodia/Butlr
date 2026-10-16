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
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.extractor import EXTRACTOR_SYSTEM_PROMPT  # noqa: E402
from config import settings  # noqa: E402
from models.schemas import Email  # noqa: E402
import utils.llm_client as llm_client  # noqa: E402
from utils.llm_client import chat, parse_json_response  # noqa: E402

# This script ALWAYS talks to the LOCAL_LLM_URL endpoint, regardless of the
# .env provider. The labeler endpoint must be llama-server/Ollama on this
# machine (privacy rule), or an OpenRouter ZDR-routed endpoint explicitly
# configured by Jay. Forcing it here prevents an accidental cloud run via
# whatever LLM_PROVIDER happens to be set in .env. Both bindings are patched:
# settings (the source) and llm_client (its import-time copy).
settings.LLM_PROVIDER = "local"
llm_client.LLM_PROVIDER = "local"

# Labeling prompt differs from the production prompt in exactly one way:
# it allows 0 tasks (an empty array) so newsletters/spam produce negatives.
LABELER_SYSTEM_PROMPT = EXTRACTOR_SYSTEM_PROMPT.replace(
    "Extract between 1 and 3 distinct tasks.",
    "Extract between 0 and 3 distinct tasks. If the email has no action item, return an empty JSON array [].",
)


def label_one(row: dict, *, max_body_chars: int = 4000) -> tuple[list[dict], float, dict]:
    """Call the local labeler for one email. Returns (tasks, latency_seconds, row)."""
    body = row["body"]
    if len(body) > max_body_chars:
        # Long bodies slow prompt processing for everyone; the head of an
        # email carries the actionable content. Flagged in the log.
        body = body[:max_body_chars]
        row = dict(row, body=body, truncated=True)
    user_prompt = (
        f"Email ID: {row['id']}\n"
        f"Subject: {row['subject']}\n"
        f"Sender: {row['sender']}\n"
        f"Body: {body}\n"
        f"Timestamp: {row['timestamp']}\n"
        f"Context:\nNo additional context."
    )
    start = time.perf_counter()
    response = chat(LABELER_SYSTEM_PROMPT, user_prompt)
    latency = time.perf_counter() - start
    payload = parse_json_response(response)
    tasks = payload if isinstance(payload, list) else []
    # The stored email row is the (possibly truncated) version the labeler saw,
    # so the training row matches what was labeled.
    return tasks, latency, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/corpus/pool.jsonl")
    parser.add_argument("--output", default="data/labeled/weak.jsonl")
    parser.add_argument("--log", default="data/labeled/label_log.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="concurrent requests; requires llama-server started with --parallel >= workers",
    )
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
    lock = threading.Lock()

    def _handle(index_row: tuple[int, dict]) -> None:
        nonlocal ok, failed, total_latency
        _, row = index_row
        try:
            tasks, latency, stored_row = label_one(row)
            with lock:
                out_fh.write(json.dumps({"email": stored_row, "tasks": tasks}) + "\n")
                log_fh.write(
                    json.dumps(
                        {
                            "id": stored_row["id"],
                            "status": "ok",
                            "latency_s": round(latency, 3),
                            "tasks": len(tasks),
                            "truncated": bool(stored_row.get("truncated")),
                        }
                    )
                    + "\n"
                )
                ok += 1
                total_latency += latency
        except Exception as exc:  # one bad row must not stop the run
            with lock:
                log_fh.write(
                    json.dumps({"id": row["id"], "status": "failed", "error": str(exc)[:200]})
                    + "\n"
                )
                failed += 1
        if args.workers == 1:
            print(".", end="", flush=True)

    with out_path.open("a") as out_fh, log_path.open("a") as log_fh:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for done_count, _ in enumerate(
                pool.map(_handle, enumerate(pending)), start=1
            ):
                if done_count % 50 == 0:
                    mean = total_latency / ok if ok else 0
                    print(
                        f"\n  progress {done_count}/{len(pending)} (ok={ok}, failed={failed}, mean={mean:.1f}s)",
                        flush=True,
                    )
                    out_fh.flush()
                    log_fh.flush()

    mean = total_latency / ok if ok else 0
    print(f"Done. ok={ok} failed={failed} mean_latency={mean:.1f}s per email")
    print(f"Labels: {out_path}")
    print(f"Log:    {log_path}")
    print("NEXT: hand-verify the val and test rows, then run scripts.make_splits.")


if __name__ == "__main__":
    main()
