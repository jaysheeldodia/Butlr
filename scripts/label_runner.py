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

from config import settings  # noqa: E402
from models.schemas import Email  # noqa: E402
import utils.llm_client as llm_client  # noqa: E402
from utils.llm_client import chat, parse_json_response  # noqa: E402
from utils.text_clean import clean_body_full  # noqa: E402

# This script ALWAYS talks to the LOCAL_LLM_URL endpoint, regardless of the
# .env provider. The labeler endpoint must be llama-server/Ollama on this
# machine (privacy rule), or an OpenRouter ZDR-routed endpoint explicitly
# configured by Jay. Forcing it here prevents an accidental cloud run via
# whatever LLM_PROVIDER happens to be set in .env. Both bindings are patched:
# settings (the source) and llm_client (its import-time copy).
settings.LLM_PROVIDER = "local"
llm_client.LLM_PROVIDER = "local"

# Labeling prompt: standalone and stricter than the production extractor
# prompt, because the labeler must DECIDE whether a task exists at all,
# not just format tasks. Includes explicit noise types, cost-of-error
# framing, and relative-timeframe deadline rules (v2 of the labeler prompt,
# informed by the data audit showing 98% noise, CSS-soup bodies, 0/9
# deadlines, and a labeler that over-triggered 26x).
LABELER_SYSTEM_PROMPT = """\
You are Butlr, a personal operations assistant for a job seeker. Your job:
decide whether an email requires an ACTION from the user, and if yes,
extract the tasks. If no, return [].

MOST INBOX MAIL NEEDS NO ACTION. About 95% of the emails you see are noise.
Returning an empty JSON array [] is correct and common. Inventing a task
where none exists is the worst error you can make. When in doubt, return [].

A task exists ONLY when the email asks the USER to do something specific:
reply, apply, submit, schedule, prepare, review, confirm, decide, or follow
up. Emails that merely inform, announce, confirm an automatic event, market
a product, or contain content to read are NEVER tasks.

The body may be cleaned: HTML, CSS, quoted replies, and links were removed.
If the body is empty or unreadable (for example only CSS/HTML markup
survived), return [] — do NOT invent tasks from markup.

Typical NOISE (return []):
- Job alerts and recommendations ("new jobs near you", "you may like this
  role").
- LinkedIn updates, connection invites, group digests.
- GitHub noise: comments, reactions, review-thread replies, workflow-run and
  Dependabot notifications.
- Newsletters, digests, product updates, promotions.
- Receipts, payment confirmations, security codes, USCIS status updates
  that require nothing from the user.
- Calendar invites and event reminders by themselves.

Typical TASKS (extract them):
- A recruiter asks to schedule an interview or reply with availability.
- An application portal asks the user to submit a document or complete a form.
- Someone asks the user to review, confirm, or decide something by a date.
- A follow-up the user must make: accept an offer, reject, send documents.
- A GitHub notification that asks the USER to do something: review a pull
  request, respond to a mention, or act on an assignment (category "admin").

Rules:
- Extract between 0 and 3 distinct tasks. Most emails: 0.
- If no action item: return an empty JSON array [].
- Titles start with an action verb: Reply, Submit, Prepare, Review,
  Schedule, Complete, Send, Confirm.
- action_required=true only when the sender EXPECTS a response,
  submission, or decision from the user.
- deadline: set it when the email names a specific date OR a relative
  timeframe such as "by Friday", "within 48 hours", "before end of day",
  "by Oct 30". Otherwise null.
- Category rules:
  - "reply": the task is to answer or follow up with a person.
  - "application": a step in a job application, such as submitting documents
    or filling a form.
  - "interview": interview scheduling, preparation, or debrief.
  - "admin": paperwork, logistics, onboarding, or account setup.
  - "other": everything else.
- source must always be "email".
- Every task MUST include "id" ("<email-id>-1", "<email-id>-2", ...) and
  "related_email_id" (the email id). The output schema requires both; the
  fine-tuned model is trained on this same shape.

Example 1 (NOISE): Subject "New React jobs near you" Body "Browse 12 new
openings near you." -> []
Example 2 (TASK): Subject "Interview scheduling" Body "Please confirm
Thursday 2pm. Reply with your availability by 5pm tomorrow." ->
[{"id": "<email-id>-1", "title": "Confirm interview slot", "description":
"Reply with availability for the Thursday 2pm interview by tomorrow 5pm.",
"category": "interview", "priority": "high", "action_required": true,
"deadline": "<tomorrow 17:00 as ISO 8601>", "related_email_id": "<email-id>",
"source": "email"}]

Return only a valid JSON array matching the schema. No text outside the JSON.
"""


def label_one(row: dict, *, max_body_chars: int = 4000) -> tuple[list[dict], float, dict]:
    """Call the local labeler for one email. Returns (tasks, latency_seconds, row).

    The body is CLEANED (HTML/CSS/quoted replies stripped, via
    utils.text_clean) BEFORE the character cap is applied. This is what the
    data audit demanded: raw bodies at the 4000-char cap were often a wall
    of CSS with the real message past the cut, so both the labeler and the
    eventual training rows saw garbage. Cleaned text is also what the
    production extractor sends (agents/extractor.py uses the same helper),
    so training and serving stay consistent.
    """
    body = clean_body_full(row["body"])
    cleaned = len(body) != len(row["body"])
    if len(body) > max_body_chars:
        # Long clean bodies slow prompt processing for everyone; the head of
        # an email carries the actionable content. Flagged in the log.
        body = body[:max_body_chars]
        row = dict(row, body=body, truncated=True, cleaned=cleaned)
    else:
        row = dict(row, body=body, cleaned=cleaned)
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
    if not pending and done_ids:
        # Re-labeling trap: this script RESUMES from --output and skips any id
        # already present. Re-running against an existing file (e.g. weak.jsonl,
        # the corrected v1 labels) would silently do nothing. Use a new
        # --output path to produce a fresh labeling pass.
        print(
            f"Nothing to do: all {len(done_ids)} rows are already in {out_path}.\n"
            "To RE-label (e.g. with the v2 cleaned bodies), pick a NEW --output\n"
            "path such as data/labeled/weak_v2.jsonl. Keep weak.jsonl intact -\n"
            "it is the corrected file behind the v1 benchmark numbers."
        )
        return
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
                            "cleaned": bool(stored_row.get("cleaned")),
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
