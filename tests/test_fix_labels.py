"""Tests for fix_labels: --labeled targeting and the write-once backup rule.

Two traps this guards:
1. The script was hardcoded to weak.jsonl, so correcting a v2 run would have
   silently edited the v1 file instead.
2. It overwrote its .bak on EVERY run. audit_empties uses that .bak as "what
   the labeler actually said", so overwriting it with corrected labels would
   have made the audit report ~0 labeler-empties - i.e. a clean bill of health
   invented from a corrupted reference.
"""
from __future__ import annotations

import json
import sys

import scripts.fix_labels as fl


def _row(email_id: str, tasks: list) -> dict:
    return {
        "email": {
            "id": email_id,
            "subject": "s",
            "sender": "a@b.com",
            "body": "b",
            "timestamp": "2026-09-22T00:00:00Z",
        },
        "tasks": tasks,
    }


def _write(path, rows) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_edits_the_file_given_by_labeled(tmp_path, monkeypatch):
    labeled = tmp_path / "weak_v2.jsonl"
    _write(labeled, [_row("a", [{"title": "x", "priority": "high"}])])
    monkeypatch.setattr(fl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fix_labels", "--labeled", "weak_v2.jsonl", "--noise", "a"])

    fl.main()

    rows = [json.loads(line) for line in labeled.read_text().splitlines() if line.strip()]
    assert rows[0]["tasks"] == []


def test_existing_backup_is_never_overwritten(tmp_path, monkeypatch):
    labeled = tmp_path / "weak_v2.jsonl"
    bak = tmp_path / "weak_v2.jsonl.bak"
    _write(labeled, [_row("a", [{"title": "x", "priority": "high"}])])
    bak.write_text("RAW LABELER OUTPUT\n")
    monkeypatch.setattr(fl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fix_labels", "--labeled", "weak_v2.jsonl", "--noise", "a"])

    fl.main()

    assert bak.read_text() == "RAW LABELER OUTPUT\n"  # reference preserved
    rows = [json.loads(line) for line in labeled.read_text().splitlines() if line.strip()]
    assert rows[0]["tasks"] == []  # the edit still happened


def test_backup_created_on_first_run(tmp_path, monkeypatch):
    labeled = tmp_path / "weak_v2.jsonl"
    bak = tmp_path / "weak_v2.jsonl.bak"
    _write(labeled, [_row("a", [{"title": "x", "priority": "high"}])])
    monkeypatch.setattr(fl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fix_labels", "--labeled", "weak_v2.jsonl", "--noise", "a"])

    fl.main()

    assert bak.exists()
    assert "RAW" not in bak.read_text()


def test_missing_file_lists_available(tmp_path, monkeypatch, capsys):
    (tmp_path / "weak.jsonl").write_text("")
    monkeypatch.setattr(fl, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["fix_labels", "--labeled", "nope.jsonl", "--noise", "a"])

    try:
        fl.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:  # pragma: no cover
        raise AssertionError("expected SystemExit")

    assert "weak.jsonl" in capsys.readouterr().out
