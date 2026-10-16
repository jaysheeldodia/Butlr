"""Benchmark the API path and the local path on the locked test set.

Run manually by Jay, once per path, on the SAME machine and test file.
Reads data/splits/test.jsonl (the agent never reads this file). Writes
results/metrics_<path>.json and prints a summary table. Raw email content
stays on disk; only aggregate numbers go to results and to chat.

API run (uses the .env provider):
    LLM_PROVIDER=openai python -m scripts.benchmark --path api

Local run (llama-server must be up on LOCAL_LLM_URL):
    LLM_PROVIDER=local python -m scripts.benchmark --path local
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.schemas import Email, ExtractedTask  # noqa: E402
from utils.llm_client import chat, parse_json_response  # noqa: E402


def build_user_prompt(row: dict) -> str:
    return (
        f"Email ID: {row['email']['id']}\n"
        f"Subject: {row['email']['subject']}\n"
        f"Sender: {row['email']['sender']}\n"
        f"Body: {row['email']['body']}\n"
        f"Timestamp: {row['email']['timestamp']}\n"
        f"Context:\nNo additional context."
    )


def run_path(test_path: Path, path_name: str) -> dict:
    from agents.extractor import EXTRACTOR_SYSTEM_PROMPT

    rows = [json.loads(line) for line in test_path.read_text().splitlines() if line.strip()]
    latencies: list[float] = []
    valid = invalid = 0
    field_scores = {"priority": [0, 0], "category": [0, 0], "action_required": [0, 0]}

    for row in rows:
        truth = ExtractedTask(**row["tasks"][0]) if row["tasks"] else None
        start = time.perf_counter()
        parse_ok = True
        try:
            raw = chat(EXTRACTOR_SYSTEM_PROMPT, build_user_prompt(row))
            payload = parse_json_response(raw)
            pred_tasks = [ExtractedTask(**item) for item in payload] if isinstance(payload, list) else []
        except Exception:
            pred_tasks = []
            parse_ok = False
        latencies.append(time.perf_counter() - start)

        # Score the first predicted task against the first truth task.
        # A parse failure is ALWAYS an error — never scored as a true
        # negative, even when the truth is an empty task list. (The old
        # scoring silently counted these as correct, inflating accuracy.)
        if truth is None and not pred_tasks and parse_ok:
            valid += 1
            for field in field_scores:
                field_scores[field][0] += 1
                field_scores[field][1] += 1
        elif truth is not None and pred_tasks and parse_ok:
            pred = pred_tasks[0]
            valid += 1
            for field in field_scores:
                field_scores[field][1] += 1
                if getattr(pred, field) == getattr(truth, field):
                    field_scores[field][0] += 1
        else:
            invalid += 1

    latencies.sort()

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        idx = min(len(latencies) - 1, int(round(p * (len(latencies) - 1))))
        return round(latencies[idx], 3)

    accuracy = {
        field: round(hits / total, 4) if total else None
        for field, (hits, total) in field_scores.items()
    }
    return {
        "path": path_name,
        "n": len(rows),
        "parse_errors": invalid,
        "accuracy": accuracy,
        "latency_p50_s": pct(0.50),
        "latency_p95_s": pct(0.95),
        "mean_latency_s": round(statistics.mean(latencies), 3) if latencies else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=["api", "local"], required=True)
    parser.add_argument("--test", default="data/splits/test.jsonl")
    args = parser.parse_args()

    test_path = REPO_ROOT / args.test
    if not test_path.exists():
        print(f"Missing {test_path}. Run the data pipeline first.")
        sys.exit(1)

    metrics = run_path(test_path, args.path)
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    out = results_dir / f"metrics_{args.path}.json"
    out.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")
    print("NEXT: run the other path, then compare the two metrics files.")


if __name__ == "__main__":
    main()
