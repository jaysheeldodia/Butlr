"""Anonymize a corpus file with deterministic placeholders.

Run manually by Jay, BEFORE any upload (Colab path only). Replaces email
addresses, phone numbers, URLs, and sender display names with stable
placeholders. Dates stay, because deadline extraction needs them.
Reads JSONL in, writes JSONL out. Prints counts only.

Usage:
    python -m scripts.anonymize data/corpus/pool.jsonl data/corpus/pool_anon.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{8,}\d)")
_URL_RE = re.compile(r"https?://\S+")
_DISPLAY_NAME_RE = re.compile(r"^(.*?)\s*<[^>]+>\s*$")


def anonymize_text(text: str, mapping: dict[str, str]) -> str:
    def _map(match: re.Match) -> str:
        value = match.group(0)
        if value not in mapping:
            mapping[value] = f"[[REDACTED_{len(mapping) + 1}]]"
        return mapping[value]

    text = _URL_RE.sub(_map, text)
    text = _EMAIL_RE.sub(_map, text)
    text = _PHONE_RE.sub(_map, text)
    return text


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python -m scripts.anonymize IN.jsonl OUT.jsonl")
        sys.exit(1)

    in_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]

    mapping: dict[str, str] = {}
    for row in rows:
        # Display name part of "Name <email@x>" becomes a placeholder too.
        name_match = _DISPLAY_NAME_RE.match(row["sender"])
        if name_match and name_match.group(1).strip():
            name = name_match.group(1).strip()
            if name not in mapping:
                mapping[name] = f"[[PERSON_{len(mapping) + 1}]]"
            row["sender"] = mapping[name] + re.sub(r"^.*?(<.*>)?$", r" \1", row["sender"]).strip()
        for field in ("subject", "body"):
            row[field] = anonymize_text(row[field], mapping)

    out_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"Anonymized {len(rows)} rows")
    print(f"Replaced {len(mapping)} distinct strings")
    print(f"Wrote {out_path}")
    print("Spot-check the output by hand before any upload.")


if __name__ == "__main__":
    main()
