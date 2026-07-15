"""Enumeration types for the agent framework."""
from enum import Enum


class SessionStatus(str, Enum):
    """Session lifecycle status."""
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class EventType(str, Enum):
    """Event types emitted during agent execution."""
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


class MessageRole(str, Enum):
    """Message sender role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"