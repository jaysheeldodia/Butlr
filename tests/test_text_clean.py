"""Tests for the v2 labeler/data fixes driven by analysis.md:

- clean_body_full() surfaces the message hidden behind a CSS wall (the exact
  failure found in Pass 3: a 4000-char body that was 100% CSS).
- label_runner cleans BEFORE the cap, so the labeler sees readable text.
- the production extractor sends the same cleaned text (train/serve parity).
- the labeler prompt keeps actionable GitHub mail while dropping GitHub noise.
- prep_corpus collapses identical-content periodic digests.
"""
from __future__ import annotations

import json
import sys

from models.schemas import Email
from utils.text_clean import clean_body_full

# ~5000 chars of CSS, the way a truncated MINIFIED email body looked: the wall
# fills the first 4000 chars and the real message sits past the cap.
CSS_WALL = "@media only screen and (max-width: 600px) { .col { width: 100%; } }\n" * 90


def _css_walled_body(message: str) -> str:
    return (
        "<html><head><style>"
        + CSS_WALL
        + "</style></head><body><div class='wrap'><p>"
        + message
        + "</p></div></body></html>"
    )


# ── utils.text_clean ─────────────────────────────────────────────────────────

def test_clean_surfaces_text_behind_css_wall():
    raw = _css_walled_body("Please submit your transcript by Friday.")
    assert "submit" not in raw[:4000]  # the audit's failure mode
    cleaned = clean_body_full(raw)
    assert "submit your transcript by Friday" in cleaned
    assert "<style" not in cleaned and "@media" not in cleaned


def test_clean_returns_empty_when_only_markup_survived():
    assert clean_body_full("<div>@media only screen { .a { color: #fff; } }</div>") == ""
    assert clean_body_full("") == ""
    assert clean_body_full("<style>.a{color:red}</style>") == ""


def test_clean_passes_plain_text_through():
    plain = "Please confirm Thursday 2pm. Reply with your availability by 5pm."
    assert clean_body_full(plain) == plain


def test_clean_cuts_quoted_reply_lines():
    cleaned = clean_body_full("Please review this.\n> old quoted reply text here")
    assert "review this" in cleaned
    assert "quoted reply" not in cleaned


# ── label_runner: clean before the cap ───────────────────────────────────────

def test_label_one_cleans_before_capping(monkeypatch):
    import scripts.label_runner as lr

    captured: dict = {}

    def fake_chat(system_prompt, user_prompt):
        captured["user"] = user_prompt
        return "[]"

    monkeypatch.setattr(lr, "chat", fake_chat)
    row = {
        "id": "e1",
        "subject": "Application",
        "sender": "jobs@example.com",
        "body": _css_walled_body("Please submit your transcript by Friday."),
        "timestamp": "2026-09-22T00:00:00Z",
    }
    tasks, _latency, stored = lr.label_one(row)

    assert tasks == []
    assert "submit your transcript by Friday" in captured["user"]  # labeler SAW it
    assert "<style" not in captured["user"] and "@media" not in captured["user"]
    assert stored["cleaned"] is True
    assert stored["body"] != row["body"]


def test_label_one_flags_truncation_on_long_clean_body(monkeypatch):
    import scripts.label_runner as lr

    monkeypatch.setattr(lr, "chat", lambda s, u: "[]")
    long_plain = "Please review the onboarding checklist before your first day. " * 200
    row = {
        "id": "e2",
        "subject": "Onboarding",
        "sender": "hr@example.com",
        "body": long_plain,
        "timestamp": "2026-09-22T00:00:00Z",
    }
    _tasks, _latency, stored = lr.label_one(row)
    assert stored["truncated"] is True
    assert len(stored["body"]) == 4000


def test_labeler_prompt_keeps_actionable_github_and_allows_empty():
    from scripts.label_runner import LABELER_SYSTEM_PROMPT as prompt

    assert "Returning an empty JSON array [] is correct" in prompt
    assert "review a pull" in prompt            # actionable GitHub stays a task
    assert "workflow-run and" in prompt         # GitHub noise still drops out
    assert "Extract between 0 and 3 distinct tasks" in prompt


def test_labeler_prompt_covers_every_grammar_required_field():
    """Prompt and llama-server grammar must agree, or GBNF forces the model to
    invent fields the prompt never described (id, related_email_id)."""
    import json
    from pathlib import Path

    import scripts.label_runner as lr

    schema = json.loads((Path(lr.REPO_ROOT) / "scripts" / "task_schema.json").read_text())
    for field in schema["items"]["required"]:
        assert f'"{field}"' in lr.LABELER_SYSTEM_PROMPT, field


# ── extractor: serving sees the same cleaned text as training ────────────────

def test_extractor_sends_cleaned_body(monkeypatch):
    import agents.extractor as ex

    captured: dict = {}

    def fake_chat(system_prompt, user_prompt):
        captured["user"] = user_prompt
        return "[]"

    monkeypatch.setattr(ex, "chat", fake_chat)
    email = Email(
        id="e3",
        subject="Application",
        sender="jobs@example.com",
        body="<html><body><p>Please reply by Friday.</p></body></html>",
        timestamp="2026-09-22T00:00:00Z",
    )
    ex.extract_tasks(email)

    assert "Please reply by Friday." in captured["user"]
    assert "<p>" not in captured["user"]  # raw markup no longer reaches the model


# ── prep_corpus: identical-content dedupe ────────────────────────────────────

def test_content_key_keeps_different_senders_apart():
    from scripts.prep_corpus import content_key

    a = {"sender": "a@x.com", "subject": "Weekly digest", "body": "Same body."}
    b = {"sender": "b@x.com", "subject": "Weekly digest", "body": "Same body."}
    assert content_key(a) != content_key(b)


def test_content_key_ignores_retitled_subject():
    from scripts.prep_corpus import content_key

    a = {"sender": "notifications@github.com", "subject": "[repo] Weekly digest #12",
         "body": "Your round-up."}
    b = {"sender": "notifications@github.com", "subject": "[repo] Weekly digest #13",
         "body": "Your round-up."}
    assert content_key(a) == content_key(b)


def test_prep_corpus_drops_identical_content_rows(tmp_path, monkeypatch):
    import scripts.prep_corpus as prep

    raw = tmp_path / "emails.jsonl"
    out = tmp_path / "pool.jsonl"
    rows = [
        {"id": "a1", "subject": "Weekly digest", "sender": "notifications@github.com",
         "body": "Your weekly round-up of activity.", "timestamp": "2026-09-22T00:00:00Z"},
        {"id": "a2", "subject": "Weekly round-up", "sender": "notifications@github.com",
         "body": "Your weekly round-up of activity.", "timestamp": "2026-09-23T00:00:00Z"},
        {"id": "a3", "subject": "Interview scheduling", "sender": "recruiter@x.com",
         "body": "Please confirm Thursday.", "timestamp": "2026-09-24T00:00:00Z"},
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(prep, "RAW_PATH", raw)
    monkeypatch.setattr(prep, "OUT_PATH", out)
    monkeypatch.setattr(sys, "argv", ["prep_corpus"])

    prep.main()

    pool = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    ids = {r["id"] for r in pool}
    assert len(pool) == 2
    assert "a1" in ids and "a3" in ids  # first copy kept, duplicate dropped
    assert "a2" not in ids


# ── newsletter/mail footers: survive cleaning, so the prompt must handle them ─

def test_footer_text_survives_cleaning():
    """Documents WHY the labeler prompt needs an explicit footer rule.

    clean_body strips quoted lines, HTML/CSS and long blobs, but a plain-text
    footer is ordinary prose, so it survives. Every GitHub notification
    therefore reaches the labeler containing "Reply to this email directly".
    """
    body = (
        "<div><p>Your workflow run failed.</p>"
        "<p>Reply to this email directly, view it on GitHub, or unsubscribe.</p>"
        "</div>"
    )
    cleaned = clean_body_full(body)
    assert "workflow run failed" in cleaned
    assert "Reply to this email directly" in cleaned


def test_labeler_prompt_treats_footers_as_noise():
    from scripts.label_runner import LABELER_SYSTEM_PROMPT as prompt

    assert "Reply to this email directly" in prompt
    assert "NOT a request to reply" in prompt
    assert "unsubscribe" in prompt.lower()
    # the old prompt claimed links were removed; cleaning keeps short URLs,
    # so telling the model otherwise is false.
    assert "links were removed" not in prompt
