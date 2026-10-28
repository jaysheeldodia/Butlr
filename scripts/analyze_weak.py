"""Data analysis over the Butlr labeled pool. Prints numbers only -
no email bodies, subjects, or sender addresses (domains only).

Run from the repo root on the Mac:
    python3 scripts/analyze_weak.py data
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

KEYWORDS = [
    "application", "interview", "reply", "submit", "schedule", "confirm",
    "deadline", "due", "urgent", "please", "invoice", "invite", "offer",
    "update", "verification", "receipt", "newsletter", "digest",
    "notification", "welcome", "weekly", "monthly", "verify", "session",
    "resume", "salary", "visa", "relocate",
]

SUBJECT_NORM = re.compile(r"^\s*(?:(?:re|fwd|fw|sv|vs):\s*)+", re.IGNORECASE)
EMAIL_RE = re.compile(r"@([\w.-]+)")
HTML_TAG_RE = re.compile(r"<[a-zA-Z/][^>]*>")
CSS_RE = re.compile(r"(?:style\s*=|class\s*=|#[0-9a-fA-F]{3,6}\b)")
URL_RE = re.compile(r"https?://\S+")
QUOTE_RE = re.compile(r"^>\s?", re.MULTILINE)
BASE64_RE = re.compile(r"[A-Za-z0-9+/]{200,}={0,2}")


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def domain_of(sender: str) -> str:
    m = EMAIL_RE.search(sender or "")
    return m.group(1).lower() if m else "(no-address)"


def norm_subject(subject: str) -> str:
    s = re.sub(r"\s+", " ", (subject or "").strip().lower())
    return SUBJECT_NORM.sub("", s)


def stats(arr) -> str:
    if not arr:
        return "n=0"
    s = sorted(arr)
    return (f"n={len(s)} mean={sum(s) / len(s):.0f} "
            f"p50={s[len(s) // 2]} p90={s[int(len(s) * 0.9)]} max={s[-1]}")


def main() -> None:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    labeled = data / "labeled" / "weak.jsonl"
    rows = load(labeled)
    n = len(rows)

    def has_tasks(r) -> bool:
        return bool(r.get("tasks"))

    n_task = sum(1 for r in rows if has_tasks(r))
    n_empty = n - n_task
    print(f"== POOL: {n} rows | {n_task} with tasks | {n_empty} empty "
          f"({n_empty / n:.1%}) | ratio {n_empty}:{max(n_task, 1)} ==")

    print("\n== Q2. WHO MAILS: top domains by volume and by task density ==")
    dom_all = Counter(domain_of(r["email"].get("sender", "")) for r in rows)
    dom_task = Counter(
        domain_of(r["email"].get("sender", "")) for r in rows if has_tasks(r))
    top3_share = sum(c for _, c in dom_all.most_common(3)) / n
    print(f"unique domains: {len(dom_all)}, top-3 share of volume: {top3_share:.0%}")
    for d, c in dom_all.most_common(12):
        t = dom_task.get(d, 0)
        print(f"  {d:30s} vol={c:4d} tasks={t}  task-rate={t / c:.0%}")
    print("  -- task-density leaders (>=2 emails, task-rate >= 50%) --")
    for d, c in dom_all.most_common():
        t = dom_task.get(d, 0)
        if c >= 2 and t / c >= 0.5:
            print(f"  {d:30s} vol={c:4d} tasks={t}  task-rate={t / c:.0%}")

    print("\n== Q3. TASK vs NO-TASK ==")
    cats = Counter()
    prios = Counter()
    deadlines = 0
    action_req = 0
    for r in rows:
        for t in r.get("tasks") or []:
            cats[t.get("category", "?")] += 1
            prios[t.get("priority", "?")] += 1
            if t.get("deadline"):
                deadlines += 1
            if t.get("action_required"):
                action_req += 1
    print(f"without tasks: {n_empty} | with tasks: {n_task} "
          f"| imbalance {n_empty / max(n_task, 1):.0f}:1")
    print("categories:", dict(cats), "| priorities:", dict(prios),
          f"| tasks with deadline: {deadlines}/{n_task} "
          f"| action_required: {action_req}/{n_task}")
    id_task = {r["email"]["id"]: has_tasks(r) for r in rows}
    for name in ("train", "val", "test"):
        p = data / "splits" / f"{name}.jsonl"
        if not p.exists():
            continue
        s = load(p)
        t = sum(1 for r in s if id_task.get(r["email"]["id"]))
        print(f"  split {name}: {len(s)} rows, {t} with tasks")

    print("\n== Q4. BODY DIRT (classification counts, no content) ==")
    html = sum(1 for r in rows if HTML_TAG_RE.search(r["email"].get("body", "")))
    css = sum(1 for r in rows if CSS_RE.search(r["email"].get("body", "")))
    urlz = sum(1 for r in rows if URL_RE.search(r["email"].get("body", "")))
    quoted = sum(1 for r in rows if QUOTE_RE.search(r["email"].get("body", "")))
    b64 = sum(1 for r in rows if BASE64_RE.search(r["email"].get("body", "")))
    print(f"rows containing HTML tags: {html} ({html / n:.0%})")
    print(f"rows containing CSS/style classes: {css} ({css / n:.0%})")
    print(f"rows containing URLs: {urlz} ({urlz / n:.0%})")
    print(f"rows containing quoted reply lines ('>'): {quoted} ({quoted / n:.0%})")
    print(f"rows containing long base64 blobs: {b64} ({b64 / n:.0%})")

    print("\n== Q9. NEAR-DUPLICATES (same normalized body) ==")
    by_hash = defaultdict(list)
    for r in rows:
        norm = re.sub(r"\s+", " ", (r["email"].get("body", "") or ""))
        by_hash[hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]].append(r)
    dup_bodies = sum(1 for v in by_hash.values() if len(v) > 1)
    dup_rows = sum(len(v) for v in by_hash.values() if len(v) > 1)
    print(f"unique bodies: {len(by_hash)} | duplicated-content bodies: {dup_bodies} "
          f"affecting {dup_rows} rows ({dup_rows / n:.0%})")
    for v in sorted(by_hash.values(), key=len, reverse=True)[:5]:
        if len(v) > 1:
            t = sum(1 for r in v if has_tasks(r))
            print(f"  {len(v)} copies of one content, tasks among them: {t}")

    print("\n== Q10. LABELER AGREEMENT (backup = 4B weak labels) ==")
    bak = labeled.with_suffix(".jsonl.bak")
    if bak.exists():
        b = {r["email"]["id"]: bool(r.get("tasks")) for r in load(bak)}
        cur = {r["email"]["id"]: has_tasks(r) for r in rows}
        common = set(b) & set(cur)
        kept = sum(1 for i in common if b[i] == cur[i])
        flip_empty = sum(1 for i in common if b[i] and not cur[i])
        flip_task = sum(1 for i in common if not b[i] and cur[i])
        b_total = sum(1 for v in b.values() if v)
        print(f"agreement: {kept / len(common):.0%} | kept {kept} | "
              f"flipped to empty {flip_empty} | flipped to task {flip_task}")
        print(f"4B labeler said 'task': {b_total} ({b_total / len(common):.0%}) "
              f"-> after corrections: {n_task} ({n_task / len(common):.0%})")

    print("\n== Q5-Q7. LENGTHS (chars), by group ==")
    raw_all = [len(r["email"].get("body", "")) for r in rows]
    raw_task = [len(r["email"].get("body", "")) for r in rows if has_tasks(r)]
    raw_empty = [len(r["email"].get("body", "")) for r in rows if not has_tasks(r)]
    print(f"all    : {stats(raw_all)}")
    print(f"task   : {stats(raw_task)}")
    print(f"no-task: {stats(raw_empty)}")

    print("\n== Q1/Q13. TIME SKEW ==")
    dates = Counter()
    task_dates = Counter()
    for r in rows:
        m = re.match(r"(\d{4}-\d{2}-\d{2})T", r["email"].get("timestamp", ""))
        if m:
            d = m.group(1)
            dates[d] += 1
            if has_tasks(r):
                task_dates[d] += 1
    if dates:
        print(f"range {min(dates)} -> {max(dates)} ({len(dates)} days)")
        print("busiest days (vol | tasks): "
              + ", ".join(f"{d}:{dates[d]}({task_dates.get(d, 0)})"
                          for d, _ in dates.most_common(5)))
    else:
        print("no timestamps parsed")

    print("\n== THREADS (subject-clustered, subjects hashed) ==")
    threads = defaultdict(list)
    for r in rows:
        threads[norm_subject(r["email"].get("subject", ""))].append(r)
    multi = {s: v for s, v in threads.items() if len(v) > 1}
    print(f"clusters: {len(threads)} | multi-message: {len(multi)}")
    for s, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:6]:
        t = sum(1 for r in v if has_tasks(r))
        print(f"  subject#{hashlib.sha1(s.encode()).hexdigest()[:8]} "
              f"size={len(v)} tasks={t}")

    print("\n== Q11. KEYWORD TASK-DENSITY (v2 query design) ==")
    for kw in KEYWORDS:
        hits = [r for r in rows
                if kw in (r["email"].get("body", "") or "").lower()
                or kw in (r["email"].get("subject", "") or "").lower()]
        if len(hits) >= 2:
            t = sum(1 for r in hits if has_tasks(r))
            print(f"  {kw:12s} vol={len(hits):4d} tasks={t} rate={t / len(hits):.0%}")

    print("\n== Q12. FILTER SIMULATION: exclude top-3 volume domains ==")
    top3 = [d for d, _ in dom_all.most_common(3)]
    lost = sum(1 for r in rows
               if has_tasks(r) and domain_of(r["email"].get("sender", "")) in top3)
    shrunk = n - sum(dom_all[d] for d in top3)
    print(f"excluding {top3}: pool {n} -> {shrunk} (-{n - shrunk}, "
          f"{(n - shrunk) / n:.0%}), tasks lost: {lost}/{n_task}")


if __name__ == "__main__":
    main()
