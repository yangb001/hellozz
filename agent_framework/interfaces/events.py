from pydantic import BaseModel, Field
from typing import Any, Optional, Dict
from datetime import datetime, timezone
from dataclasses import dataclass

# EventType is defined in enums.py - re-export here for backward compatibility
from .enums import EventType


@dataclass
class StreamingEventData:
    """Data for streaming start/end events.

    Attributes:
        stream_id: Unique identifier for the stream.
        is_complete: Whether the stream is complete.
    """
    stream_id: str = ""
    is_complete: bool = False


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class ToolCallEventData:
    """Data for tool call events.

    Attributes:
        tool_call_id: Unique identifier for the tool call.
        tool_name: Name of the tool being called.
        arguments: Tool arguments (may be partial during streaming).
        is_complete: Whether the tool call is complete.
    """
    tool_call_id: str
    tool_name: str
    arguments: str = ""
    is_complete: bool = False


@dataclass
class ReasoningEventData:
    """Data for reasoning content events.

    Attributes:
        content: Reasoning content (may be partial during streaming).
        is_complete: Whether the reasoning is complete.
    """
    content: str = ""
    is_complete: bool = False


class ThinkingData(BaseModel):
    """Thinking process data.

    Attributes:
        step: Thinking step number.
        label: Step label (e.g., "analyze problem", "select tool").
        content: Thinking content.
        duration_ms: Duration in milliseconds.
        token_count: Token count.
    """
    step: int = 0
    label: str = ""
    content: str = ""
    duration_ms: Optional[int] = None
    token_count: Optional[int] = None


class Event(BaseModel):
    """Event data class representing a single event in agent execution.

    Attributes:
        type: Type of event (thought, action, observation, text_token, final_answer, error).
        content: Text content of the event.
        metadata: Additional event metadata.
        timestamp: Event timestamp, defaults to current UTC time.
        thinking: Thinking process data for thinking events.
        sequence: Event sequence number.
    """
    type: str
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)
    thinking: Optional[ThinkingData] = None
    sequence: int = 0