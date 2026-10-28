"""Check whether data/labeled/weak.jsonl is the corrected pool used for the
final v1 run. Prints counts, distributions, and IDs only - never email content.

Run from the repo root on the Mac:
    python3 scripts/check_weak_jsonl.py data [data2 data3 ...]
Every argument is a data directory to compare (e.g. a backup copy).
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def inspect(data_dir: Path) -> None:
    labeled = data_dir / "labeled" / "weak.jsonl"
    print(f"\n=== {data_dir} ===")
    if not labeled.exists():
        print("missing:", labeled)
        return

    rows = load(labeled)
    n = len(rows)
    print("file:", labeled.resolve())
    print("rows:", n)

    email_ids = [r.get("email", {}).get("id", "MISSING") for r in rows]
    print("unique email ids:", len(set(email_ids)))
    print("rows without email.id:", sum(i == "MISSING" for i in email_ids))
    print("rows without 'tasks' key:", sum("tasks" not in r for r in rows))
    print("rows where tasks is not a list:", sum(not isinstance(r.get("tasks"), list) for r in rows))

    empty = sum(1 for r in rows if not r.get("tasks"))
    print(f"empty-task rows: {empty} ({empty / n:.1%})")

    cats: Counter = Counter()
    prios: Counter = Counter()
    task_items = 0
    for r in rows:
        for t in r.get("tasks") or []:
            task_items += 1
            cats[t.get("category", "MISSING")] += 1
            prios[t.get("priority", "MISSING")] += 1
    print("task items total:", task_items)
    print("category counts:", dict(cats))
    print("priority counts:", dict(prios))

    # Schema validity - only when run from the repo root (pydantic available).
    try:
        sys.path.insert(0, str(Path.cwd()))
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from models.schemas import ExtractedTask

        bad = 0
        for r in rows:
            for t in r.get("tasks") or []:
                try:
                    ExtractedTask(**t)
                except Exception:
                    bad += 1
        print("task items failing ExtractedTask schema:", bad)
    except ImportError:
        print("(schema check skipped: run from repo root to enable)")

    # Splits overlap: is this file the exact source of train/val/test?
    id_set = set(email_ids)
    for name in ("train", "val", "test"):
        p = data_dir / "splits" / f"{name}.jsonl"
        if not p.exists():
            print(f"{name}: no split file")
            continue
        s = [r.get("email", {}).get("id") for r in load(p)]
        sids = set(s)
        print(
            f"{name}: {len(s)} rows | in weak.jsonl: {len(sids & id_set)} "
            f"| NOT in weak.jsonl: {len(sids - id_set)} "
            f"| weak rows missing from split: {len(id_set - sids)}"
        )

    ts = [r.get("email", {}).get("timestamp", "") for r in rows if r.get("email", {}).get("timestamp")]
    if ts:
        print("timestamp range:", min(ts), "->", max(ts))

    print("sha256:", hashlib.sha256(labeled.read_bytes()).hexdigest())
    bak = labeled.with_suffix(".jsonl.bak")
    if bak.exists():
        b = load(bak)
        print(f"backup .jsonl.bak: EXISTS ({len(b)} rows) - fix_labels ran")
        print("backup empty ratio: {:.1%}".format(sum(1 for r in b if not r.get("tasks")) / len(b)))
    else:
        print("backup .jsonl.bak: missing (never touched by fix_labels, or backup removed)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for arg in sys.argv[1:]:
        inspect(Path(arg))


if __name__ == "__main__":
    main()
