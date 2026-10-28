"""Audit the labeler-'empty' rows: the rows the 4B labeler said had NO tasks
and that never appeared in your correction pass (fix_labels only showed rows
WITH tasks). We don't know if the labeler missed real tasks in these rows.

Part A: GitHub sub-sender breakdown (counts only - safe to paste).
Part B: interactive audit - shows sampled rows on YOUR screen only.
Keys: y (looks like a task) / n (noise) / r (show raw body) / s (skip).
For y, then answer d (has deadline) / n (no deadline).
Each verdict is saved immediately to data/labeled/audit_empties.jsonl
(IDs + verdicts only, no content), so Ctrl-C loses nothing.

Usage:
    python3 scripts/audit_empties.py data [--limit 50] [--seed 42]
    python3 scripts/audit_empties.py data --labeled data/labeled/weak_v2.jsonl

Paste back only the SUMMARY block, never row content.
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

EMAIL_IN_ANGLE = re.compile(r"<([^>]+)>")
QUOTE_RE = re.compile(r"^[ \t]*>.*(?:\n|\Z)", re.MULTILINE)
SKIP_BLOCK_RE = re.compile(
    r"(?is)<(style|script|head|title|noscript)\b[^>]*>.*?(?:</\1\s*>|\Z)")
# Structural CSS lines that can sit under the density threshold when the CSS
# is pretty-printed: at-rules, selector/brace lines, `prop: value;`.
CSS_LINE_RE = re.compile(
    r"^@(?:media|font-face|import|keyframes|charset|supports|page|-[\w-]+)\b"
    r"|[{}]\s*$"
    r"|^-?[a-z][a-z-]*\s*:\s*[^;]+;\s*$")
# `selector { ... }` on one rule; an unclosed block runs to end of input.
CSS_RULE_RE = re.compile(r"[^{}\n]*\{[^{}]*(?:\}|\Z)")
CSS_DENSITY_MAX = 0.05
DENSITY_MIN_LEN = 40  # shorter lines ("Thanks!") are judged by CSS_LINE_RE only
BLOCK_TAGS = {"p", "div", "br", "tr", "li", "td", "h1", "h2", "h3", "table"}


def load(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def sender_address(sender: str) -> str:
    m = EMAIL_IN_ANGLE.search(sender or "")
    return (m.group(1) if m else (sender or "")).strip().lower()


class _TextExtractor(HTMLParser):
    """Extract visible text; skip style/script/head/title/noscript content.

    Guards against unclosed <style>/<script> blocks (common when a body was
    truncated by the 4000-char labeler cap): after 3000 skipped chars we
    stop skipping, so content after a cut-off tag is not silently lost.
    """

    SKIP = {"style", "script", "head", "title", "template", "noscript"}
    SKIP_ESCAPE = 3000

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.skip_depth = 0
        self.skip_chars = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip_depth += 1
            self.skip_chars = 0
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip_depth:
            self.skip_depth -= 1
        elif tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth:
            self.skip_chars += len(data)
            if self.skip_chars > self.SKIP_ESCAPE:
                self.skip_depth = 0  # unbalanced block; un-stick the parser
                self.parts.append(data)
            return
        self.parts.append(data)


def _css_density(s: str) -> float:
    """Share of CSS signature chars ({, }, ;, !, @) in a string."""
    if not s:
        return 0.0
    sig = sum(1 for ch in s if ch in "{};!@")
    return sig / len(s)


def _is_css_like(s: str) -> bool:
    if CSS_LINE_RE.search(s):
        return True
    return len(s) >= DENSITY_MIN_LEN and _css_density(s) > CSS_DENSITY_MAX


def clean_body(body: str, max_chars: int = 900) -> str:
    """Local-only preview: quoted lines cut, HTML/CSS stripped, entities decoded.

    CSS (even bare text without <style> tags, common after truncation) is
    removed by signature density, so a body that lost its message to the
    4000-char cap returns "" instead of a CSS wall.
    """
    text = QUOTE_RE.sub("", body or "")
    # Closing tag may be cut off by truncation: \Z matches to end of input.
    text = SKIP_BLOCK_RE.sub("\n", text)
    try:
        parser = _TextExtractor()
        parser.feed(text)
        parser.close()
        text = "".join(parser.parts)  # convert_charrefs=True decodes entities
    except Exception:
        text = unescape(re.sub(r"<[^>]*>", "\n", text))
    # Innermost rules go first, so loop until nested @media blocks are gone.
    prev = None
    while prev != text:
        prev, text = text, CSS_RULE_RE.sub("\n", text)
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if _is_css_like(ln):
            continue
        # long no-space blobs (base64, URLs)
        if len(ln) > 200 and " " not in ln:
            continue
        lines.append(ln)
    text = re.sub(r"\s+", " ", " ".join(lines)).strip()
    if not text or _is_css_like(text):
        return ""
    return text[:max_chars]


def is_html_heavy(body: str) -> bool:
    low = (body or "").lower()
    return bool(re.search(r"<style|@media|class=", low)) or "@media only screen" in low


def select_labeler_empties(rows: list[dict], backups: list[dict]) -> list[dict]:
    """Rows to audit: those the LABELER said had no tasks.

    Two cases, and getting this wrong audits the wrong rows:

    - A CORRECTED file (v1 weak.jsonl) holds Jay's hand edits, so its own
      "empty" rows are partly human decisions. The pre-correction .bak is what
      the labeler actually said, so that is the truth to audit against.
    - A FRESH v2 file has no .bak at all. There the file's own empty rows ARE
      the labeler's empties. Falling back to the .bak logic here would mark
      EVERY row as labeler-empty (missing id -> default True), so the sample
      would include rows the v2 labeler positively labeled as tasks.
    """
    if backups:
        backup_empty = {r["email"]["id"]: not r.get("tasks") for r in backups}
        return [r for r in rows if backup_empty.get(r["email"]["id"], True)]
    return [r for r in rows if not r.get("tasks")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="?", default="data")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--labeled",
        default=None,
        help="labeled JSONL to audit (default: <data>/labeled/weak.jsonl). "
        "Point this at weak_v2.jsonl after a v2 re-label.",
    )
    ap.add_argument(
        "--verdicts",
        default=None,
        help="where verdicts are saved. Default audit_empties.jsonl, or "
        "audit_<labeled-stem>.jsonl when --labeled is given.",
    )
    args = ap.parse_args()

    data = Path(args.data)
    labeled = Path(args.labeled) if args.labeled else data / "labeled" / "weak.jsonl"
    bak = labeled.with_suffix(".jsonl.bak")
    if args.verdicts:
        audit_path = Path(args.verdicts)
    elif args.labeled:
        # Separate verdict file per labeled file: verdicts resume by email_id,
        # so sharing one file across v1 and v2 would skip every already-judged
        # id and silently reuse the old verdicts.
        audit_path = labeled.with_name(f"audit_{labeled.stem}.jsonl")
    else:
        audit_path = data / "labeled" / "audit_empties.jsonl"

    print(f"auditing: {labeled}")
    print(f"verdicts: {audit_path}")
    if not labeled.exists():
        print(f"\nERROR: no such file: {labeled}")
        available = sorted(p.name for p in labeled.parent.glob("*.jsonl")) if labeled.parent.exists() else []
        if available:
            print(f"Files present in {labeled.parent}:")
            for name in available:
                print(f"  {name}")
        else:
            print(f"(no .jsonl files in {labeled.parent})")
        print("\nCheck --labeled, or whether scripts.label_runner has been run yet.")
        sys.exit(1)
    rows = load(labeled)
    backups = load(bak) if bak.exists() else []
    by_id = {r["email"]["id"]: r for r in rows}

    labeler_empty = select_labeler_empties(rows, backups)
    print(f"rows the labeler called 'empty' (never shown to you): {len(labeler_empty)}")

    # ---- Part A: GitHub sub-sender breakdown (counts only) ----
    print("\n== PART A. github.com sub-senders ==")
    gh = [r for r in rows
          if "@github.com" in sender_address(r["email"].get("sender", ""))]
    gh_senders = Counter(sender_address(r["email"].get("sender", "")) for r in gh)
    gh_tasks = Counter(
        sender_address(r["email"].get("sender", ""))
        for r in gh if r.get("tasks"))
    print(f"github emails: {len(gh)}, with tasks: {sum(1 for r in gh if r.get('tasks'))}")
    for s, c in gh_senders.most_common(10):
        print(f"  {s:35s} vol={c:4d} tasks={gh_tasks.get(s, 0)}")
    subj = Counter()
    for r in gh:
        sub = (r["email"].get("subject", "") or "").lower()
        if "review requested" in sub:
            subj["review requested"] += 1
        elif "mentioned" in sub:
            subj["mentioned"] += 1
        elif "requested changes" in sub:
            subj["requested changes"] += 1
        elif "comment on" in sub or "commented on" in sub:
            subj["comment on"] += 1
        elif "dependabot" in sub:
            subj["dependabot"] += 1
        elif "workflow" in sub or "action run" in sub:
            subj["workflow/actions"] += 1
        elif "security" in sub:
            subj["security"] += 1
        else:
            subj["other"] += 1
    print("subject buckets:", dict(subj.most_common(8)))

    # ---- Part B: interactive audit ----
    already = set()
    if audit_path.exists():
        already = {json.loads(l).get("email_id") for l in
                   audit_path.read_text().splitlines() if l.strip()}
        print(f"\naudit file exists: {len(already)} ids already judged - resuming")

    avail = [r for r in labeler_empty if r["email"]["id"] not in already]
    rng = random.Random(args.seed)
    rng.shuffle(avail)
    sample = avail[: args.limit]
    print(f"sampling {len(sample)} of {len(avail)} unjudged rows (seed {args.seed})\n")

    verdicts = []
    n_no_text = 0
    with audit_path.open("a") as fh:
        for i, r in enumerate(sample, 1):
            em = r["email"]
            raw_body = em.get("body", "") or ""
            heavy = is_html_heavy(raw_body)
            preview = clean_body(raw_body)
            print(f"\n--- [{i}/{len(sample)}] id={em['id']} "
                  f"from={em.get('sender', '')} {em.get('timestamp', '')}"
                  + (f" [HTML-heavy body, {len(raw_body)} chars]" if heavy
                     else f" [body {len(raw_body)} chars]"))
            print(f"subj: {em.get('subject', '')}")
            if preview:
                print(f"body: {preview}")
            else:
                n_no_text += 1
                print("body: [NO VISIBLE TEXT - press r to inspect raw]")
            while True:
                try:
                    verdict = input(
                        "[y] task  [n] noise  [r] raw  [s] skip  -> ").strip().lower()
                except (KeyboardInterrupt, EOFError):
                    print("\ninterrupted - judging so far is already saved")
                    sys.exit(0)
                if verdict == "r":
                    print("-" * 60)
                    print("RAW BODY (local screen only):")
                    print((em.get("body", "") or "")[:1600])
                    print("-" * 60)
                    continue
                if verdict in ("y", "n", "s"):
                    break
            deadline = None
            if verdict == "y":
                d = input("  deadline? [d] yes [n] no -> ").strip().lower()
                deadline = d == "d"
            rec = {"email_id": em["id"], "verdict": verdict, "deadline": deadline}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            verdicts.append(rec)

    judged = [v for v in verdicts if v["verdict"] in ("y", "n")]
    tasks_found = sum(1 for v in judged if v["verdict"] == "y")
    with_deadline = sum(1 for v in judged
                        if v["verdict"] == "y" and v.get("deadline"))
    if judged:
        rate = tasks_found / len(judged)
        est = rate * len(labeler_empty)
        print("\n========== SUMMARY (paste this back) ==========")
        print(f"labeler-empty rows in pool: {len(labeler_empty)}")
        print(f"audit judged this run: {len(judged)} "
              f"({sum(1 for v in verdicts if v['verdict'] == 's')} skipped)")
        print(f"rows with NO VISIBLE TEXT in stored body: {n_no_text} "
              f"of {len(sample)} sampled -> extrapolates to ~"
              f"{n_no_text * len(labeler_empty) // max(len(sample), 1)} "
              f"of the {len(labeler_empty)} empties")
        print(f"looks-like-task: {tasks_found} "
              f"({rate:.0%} of judged) -> extrapolates to ~{est:.0f} "
              f"missed tasks in the {len(labeler_empty)} empties")
        print(f"of task-looking rows, with deadline: {with_deadline}")
        doms = Counter()
        for v in judged:
            if v["verdict"] == "y":
                r = by_id.get(v["email_id"])
                if r:
                    addr = sender_address(r["email"].get("sender", ""))
                    doms[addr.split("@")[-1]] += 1
        print("task-looking rows by domain:", dict(doms.most_common(5)))
        print(f"verdicts appended to: {audit_path}")
    else:
        print("\nnothing judged this run")


if __name__ == "__main__":
    main()
