from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Message(BaseModel):
    """Message data class representing a single message in a conversation."""
    role: str
    content: Optional[str] = None
    sender_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class SessionContext(BaseModel):
    """Session context containing all session state and history."""
    session_id: str
    session_type: str = "private"
    participants: List[str] = []
    status: str = "active"
    messages: List[Message] = []
    agent_state: Dict[str, Any] = Field(default_factory=dict)
    tool_instances: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    last_active: datetime = Field(default_factory=utc_now)