"""Split the labeled pool into train/val/test by thread cluster.

Run manually by Jay AFTER hand-verifying the rows you care about.
Allocation: 150 val, 150 test, rest train. Whole threads go to one side,
so no thread leaks across splits. Prints the test-set SHA256; lock the
test file afterwards. The AI agent never reads these files.

Usage:
    python -m scripts.make_splits --labeled data/labeled/weak.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SUBJECT_NORMALIZER = re.compile(r"^(?:\s*(?:re|fwd|fw)\s*:\s*)+", re.IGNORECASE)


def thread_key(subject: str) -> str:
    return SUBJECT_NORMALIZER.sub("", subject).strip().lower() or subject.strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labeled", default="data/labeled/weak.jsonl")
    parser.add_argument("--val-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    labeled_path = REPO_ROOT / args.labeled
    rows = [json.loads(line) for line in labeled_path.read_text().splitlines() if line.strip()]

    # Cluster by thread.
    threads: dict[str, list[dict]] = {}
    for row in rows:
        threads.setdefault(thread_key(row["email"]["subject"]), []).append(row)

    keys = sorted(threads)  # stable order before shuffle
    random.Random(args.seed).shuffle(keys)

    def take(bucket: list[str], n: int) -> list[str]:
        chosen: list[str] = []
        total = 0
        while bucket and total < n:
            key = bucket.pop()
            chosen.append(key)
            total += len(threads[key])
        return chosen

    test_keys = take(keys, args.test_size)
    val_keys = take(keys, args.val_size)
    train_keys = keys

    splits = {"train": train_keys, "val": val_keys, "test": test_keys}
    out_dir = REPO_ROOT / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, keys_ in splits.items():
        path = out_dir / f"{name}.jsonl"
        count = 0
        with path.open("w") as fh:
            for key in keys_:
                for row in threads[key]:
                    fh.write(json.dumps(row) + "\n")
                    count += 1
        print(f"{name}: {count} rows -> {path}")

    test_path = out_dir / "test.jsonl"
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    print(f"TEST SET SHA256: {digest}")
    print("Lock this file. Record the hash. The agent must never read it.")


if __name__ == "__main__":
    main()
