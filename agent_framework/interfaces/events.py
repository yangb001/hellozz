from pydantic import BaseModel, Field
from typing import Any, Optional
from datetime import datetime, timezone
from dataclasses import dataclass


class EventType:
    """Event type constants for agent execution events."""

    # Core event types
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    TEXT_TOKEN = "text_token"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"

    # Streaming event types
    REASONING_CONTENT = "reasoning_content"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_ARGUMENT = "tool_call_argument"
    TOOL_CALL_END = "tool_call_end"
    STREAMING_START = "streaming_start"
    STREAMING_END = "streaming_end"


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