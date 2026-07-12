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


class MessageRole(str, Enum):
    """Message sender role."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"