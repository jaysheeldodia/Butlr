"""Tests for audit_empties row selection.

The bug this guards: a fresh v2 labeled file has no .bak, and the old logic
defaulted every missing id to "labeler said empty" - so it would sample rows
the v2 labeler had positively labeled as tasks.
"""
from __future__ import annotations

from scripts.audit_empties import select_labeler_empties


def _row(id: str, tasks: list) -> dict:
    return {"email": {"id": id}, "tasks": tasks}


def test_fresh_file_uses_its_own_empty_rows():
    rows = [_row("a", []), _row("b", [{"title": "Reply"}])]
    selected = select_labeler_empties(rows, backups=[])
    assert [r["email"]["id"] for r in selected] == ["a"]


def test_corrected_file_uses_the_backup_not_the_hand_edits():
    # b: labeler said empty, Jay later added a task by hand -> still auditable.
    # a: labeler found a task, Jay deleted it -> NOT a labeler miss.
    rows = [_row("a", []), _row("b", [{"title": "Reply"}])]
    backups = [_row("a", [{"title": "Reply"}]), _row("b", [])]
    selected = select_labeler_empties(rows, backups)
    assert [r["email"]["id"] for r in selected] == ["b"]


def test_unknown_id_with_backup_is_treated_as_empty():
    rows = [_row("a", [{"title": "Reply"}]), _row("zz", [])]
    backups = [_row("a", [{"title": "Reply"}])]
    selected = select_labeler_empties(rows, backups)
    assert [r["email"]["id"] for r in selected] == ["zz"]
