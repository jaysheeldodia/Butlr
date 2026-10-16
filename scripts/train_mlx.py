"""MLX LoRA training for the Butlr extractor (Mac, Apple Silicon only).

Run manually by Jay on the Mac. Email data never leaves the machine.
Converts data/splits/train.jsonl to the mlx-lm chat JSONL format, then
invokes mlx_lm.lora with the hyperparameters from the PRD.

Prereq (one-time):
    uv pip install mlx-lm                       # 0.32.1 or newer
    # model: mlx-community/Qwen3-1.7B-4bit (Apache-2.0)

Usage:
    python -m scripts.train_mlx \
        --train data/splits/train.jsonl --valid data/splits/val.jsonl

After training, fuse and export (also manual):
    python -m mlx_lm fuse --mlx-model ~/.cache/huggingface/hub/models--mlx-community--Qwen3-1.7B-4bit
    python -m mlx_lm convert --hf-path <fused-model-dir> -q   # optional re-quantize
    Then convert to GGUF with llama.cpp convert_hf_to_gguf.py, quantize Q4_K_M.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agents.extractor import EXTRACTOR_SYSTEM_PROMPT  # noqa: E402

BASE_MODEL = "mlx-community/Qwen3-1.7B-4bit"

# Hyperparameters per PRD (Unsloth guide, adapted for mlx-lm):
# rank 16, scale (alpha) 16, dropout 0, lr 2e-4, ~2 epochs, batch 2,
# grad accumulation 8, warmup ~5%.
HYPERPARAMS = {
    "rank": 16,
    "scale": 16,
    "dropout": 0.0,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "grad_accumulate": 8,
    "iters": 300,          # tune: ~2 epochs over your real train size
    "steps_per_eval": 50,
    "max_seq_length": 2048,
}


def to_mlx_rows(rows: list[dict], out_path: Path) -> int:
    """Write chat-format JSONL: messages + masked prompt (train on completions)."""
    count = 0
    with out_path.open("w") as fh:
        for row in rows:
            email = row["email"]
            tasks = row["tasks"]
            completion = json.dumps(
                [
                    {
                        "id": t.get("id", f"{email['id']}-1"),
                        "title": t.get("title", ""),
                        "description": t.get("description", ""),
                        "source": "email",
                        "priority": t.get("priority", "low"),
                        "category": t.get("category", "other"),
                        "action_required": bool(t.get("action_required", False)),
                        "deadline": t.get("deadline"),
                        "related_email_id": email["id"],
                    }
                    for t in tasks
                ]
            )
            user = (
                f"Email ID: {email['id']}\n"
                f"Subject: {email['subject']}\n"
                f"Sender: {email['sender']}\n"
                f"Body: {email['body']}\n"
                f"Timestamp: {email['timestamp']}\n"
                f"Context:\nNo additional context."
            )
            fh.write(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": completion},
                        ]
                    }
                )
                + "\n"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--valid", default="data/splits/val.jsonl")
    parser.add_argument("--iters", type=int, default=None, help="override PRD iters")
    args = parser.parse_args()

    train_rows = [
        json.loads(line)
        for line in (REPO_ROOT / args.train).read_text().splitlines()
        if line.strip()
    ]
    valid_rows = [
        json.loads(line)
        for line in (REPO_ROOT / args.valid).read_text().splitlines()
        if line.strip()
    ]

    work = REPO_ROOT / "data" / "mlx"
    work.mkdir(parents=True, exist_ok=True)
    n_train = to_mlx_rows(train_rows, work / "train.jsonl")
    n_valid = to_mlx_rows(valid_rows, work / "valid.jsonl")
    print(f"Converted {n_train} train rows, {n_valid} valid rows -> {work}")

    iters = args.iters or max(100, min(HYPERPARAMS["iters"], n_train))  # safe default
    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", BASE_MODEL,
        "--train",
        "--data", str(work / "train.jsonl"),
        "--val-data", str(work / "valid.jsonl") if n_valid else "",
        "--fine-tune-type", "lora",
        "--num-layers", "16",
        "--batch-size", str(HYPERPARAMS["batch_size"]),
        "--grad-accumulate", str(HYPERPARAMS["grad_accumulate"]),
        "--learning-rate", str(HYPERPARAMS["learning_rate"]),
        "--iters", str(iters),
        "--steps-per-eval", str(HYPERPARAMS["steps_per_eval"]),
        "--max-seq-length", str(HYPERPARAMS["max_seq_length"]),
        "--save-every", str(HYPERPARAMS["steps_per_eval"]),
        "--adapter-path", str(work / "adapter"),
    ]
    cmd = [c for c in cmd if c]
    print("Running:")
    print(" ".join(cmd))
    print("\nWatch for overfitting: eval loss rising while train loss falls,")
    print("or outputs that copy training examples. Stop early if you see that.")

    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
