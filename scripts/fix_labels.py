"""Fix labels by email ID: set tasks to [] (noise) or downgrade priority.

Run manually by Jay. Operates on the labeled file IN PLACE (default
data/labeled/weak.jsonl) and writes a .bak backup ONCE.

Usage:
    # mark rows as noise (empty tasks):
    python -m scripts.fix_labels --noise ID1 ID2 ID3

    # downgrade priority to low:
    python -m scripts.fix_labels --low-priority ID4 ID5

    # interactive: shows each row WITH tasks, one at a time; you decide n(oise)
    # or k(eep) - this is the fast pass for the GitHub-notification rows:
    python -m scripts.fix_labels --interactive

    # correcting a v2 run instead of the v1 file:
    python -m scripts.fix_labels --labeled data/labeled/weak_v2.jsonl --interactive

Backup rule: the .bak is written only if it does not already exist. It must
stay the RAW labeler output, because audit_empties.py uses it as "what the
labeler actually said" when measuring recall. Overwriting it with corrected
labels would silently make the audit report ~0 labeler-empties.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATH = REPO_ROOT / "data" / "labeled" / "weak.jsonl"
BACKUP = PATH.with_suffix(".jsonl.bak")


def load_rows() -> list[dict]:
    return [json.loads(l) for l in PATH.read_text().splitlines() if l.strip()]


def save_rows(rows: list[dict]) -> None:
    PATH.write_text("".join(json.dumps(r) + "\n" for r in rows))


def index(rows: list[dict]) -> dict[str, dict]:
    return {r["email"]["id"]: r for r in rows}


def main() -> None:
    global PATH, BACKUP
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeled",
        default="data/labeled/weak.jsonl",
        help="labeled file to edit in place (default: data/labeled/weak.jsonl). "
        "Point this at weak_v2.jsonl when correcting a v2 run.",
    )
    parser.add_argument("--noise", nargs="*", default=[], help="IDs to set tasks=[]")
    parser.add_argument("--low-priority", nargs="*", default=[], dest="low_priority", help="IDs to downgrade priority to low")
    parser.add_argument("--interactive", action="store_true", help="walk task rows one by one")
    args = parser.parse_args()

    PATH = REPO_ROOT / args.labeled
    BACKUP = PATH.with_suffix(".jsonl.bak")

    if not PATH.exists():
        print(f"Missing {PATH}")
        available = sorted(p.name for p in PATH.parent.glob("*.jsonl")) if PATH.parent.exists() else []
        if available:
            print("Files present: " + ", ".join(available))
        sys.exit(1)

    # Back up ONCE. The .bak has to stay the RAW labeler output: audit_empties
    # uses it as "what the labeler actually said" when measuring recall.
    # Copying over it on every run would replace that reference with corrected
    # labels and silently report ~0 labeler-empties.
    if BACKUP.exists():
        print(f"Keeping existing backup: {BACKUP} (not overwritten)")
    else:
        shutil.copy(PATH, BACKUP)
        print(f"Backup written: {BACKUP}")
    print(f"Editing: {PATH}")

    rows = load_rows()
    by_id = index(rows)

    changed = 0
    for email_id in args.noise:
        row = by_id.get(email_id)
        if row is None:
            print(f"not found: {email_id}")
            continue
        n = len(row["tasks"])
        row["tasks"] = []
        changed += 1
        print(f"noise: {email_id} (cleared {n} task(s))")

    for email_id in args.low_priority:
        row = by_id.get(email_id)
        if row is None:
            print(f"not found: {email_id}")
            continue
        for t in row["tasks"]:
            t["priority"] = "low"
        changed += 1
        print(f"low-priority: {email_id}")

    if args.interactive:
        print("n = noise (empty tasks) | k = keep | q = quit")
        for row in rows:
            if not row["tasks"]:
                continue
            email = row["email"]
            t0 = row["tasks"][0]
            print(f"\nID {email['id']}")
            print(f"  subject : {email['subject'][:80]}")
            print(f"  sender  : {email['sender'][:60]}")
            for t in row["tasks"]:
                print(f"  task    : [{t['priority']}/{t['category']}] {t['title'][:70]}")
                if t.get("deadline"):
                    print(f"  deadline: {t['deadline']}")
            answer = input("  [n/k/q] ").strip().lower()
            if answer == "q":
                break
            if answer == "n":
                row["tasks"] = []
                changed += 1

    if changed:
        save_rows(rows)
        print(f"\nSaved {changed} change(s). Backup at {BACKUP}")
        print("NEXT: python -m scripts.make_splits  (new test hash!)")
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
