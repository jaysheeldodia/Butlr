from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class Email(BaseModel):
    id: str
    subject: str
    sender: str
    body: str
    timestamp: str


class CalendarEvent(BaseModel):
    id: str
    title: str
    description: str
    datetime: str
    location: Optional[str] = None


class ClassificationResult(BaseModel):
    category: Literal["requires_reply", "informational", "newsletter", "spam"]
    priority: Literal["high", "medium", "low"]
    is_noise: bool


class ExtractedTask(BaseModel):
    id: str
    title: str
    description: str
    source: Literal["email", "calendar"]
    priority: Literal["high", "medium", "low"]
    category: Literal["reply", "application", "interview", "admin", "other"] = "other"
    action_required: bool
    deadline: Optional[str] = None
    related_email_id: Optional[str] = None


class DraftResponse(BaseModel):
    subject: str
    body: str


class PipelineResult(BaseModel):
    tasks: List[ExtractedTask]
    drafts: Dict[str, DraftResponse]
