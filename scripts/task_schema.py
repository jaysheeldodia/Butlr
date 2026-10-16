"""Task-extraction JSON schema, derived from models.ExtractedTask.

Used by llama-server (--json-schema-file) to constrain local model output to
valid JSON. Jay reviews and edits this file by hand if the schema changes.

Note: llama-server applies this to the WHOLE completion. The model is trained
to emit a bare JSON array. If your GGUF wraps output in chat fences, use the
GBNF grammar route instead (see llama.cpp grammars/README.md).
"""
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "title": {"type": "string"},
            "description": {"type": "string"},
            "source": {"type": "string", "enum": ["email", "calendar"]},
            "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            "category": {
                "type": "string",
                "enum": ["reply", "application", "interview", "admin", "other"],
            },
            "action_required": {"type": "boolean"},
            "deadline": {"type": ["string", "null"]},
            "related_email_id": {"type": ["string", "null"]},
        },
        "required": [
            "id",
            "title",
            "description",
            "source",
            "priority",
            "category",
            "action_required",
            "deadline",
            "related_email_id",
        ],
    },
}

if __name__ == "__main__":
    import json
    from pathlib import Path

    out = Path(__file__).resolve().parent / "task_schema.json"
    out.write_text(json.dumps(schema, indent=2))
    print(f"Wrote {out}")
