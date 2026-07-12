from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Event(BaseModel):
    """Event data class representing a single event in agent execution.

    Attributes:
        type: Type of event (thought, action, observation, text_token, final_answer, error).
        content: Text content of the event.
        metadata: Additional event metadata.
        timestamp: Event timestamp, defaults to current UTC time.
    """
    type: str
    content: str = ""
    metadata: dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)