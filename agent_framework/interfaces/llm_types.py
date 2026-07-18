"""LLM Types - Common type definitions for LLM interactions.

This module provides unified type definitions for LLM responses,
including tool calls, chat messages, and chat responses.

These types are used across the LLM gateway, planners, and runtime
to ensure consistent data structures.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FunctionCall:
    """Represents a function call requested by the LLM.

    Attributes:
        name: The name of the function to call.
        arguments: The JSON arguments string for the function.
    """
    name: str
    arguments: str = ""


@dataclass
class ToolCall:
    """Represents a tool call requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call.
        type: The type of tool call (usually "function").
        function: The FunctionCall details (name and arguments).
    """
    id: str
    type: str = "function"
    function: FunctionCall = field(default_factory=FunctionCall)


@dataclass
class ChatResponse:
    """Represents a chat response from the LLM.

    Attributes:
        content: The text content of the response (if no tool calls).
        tool_calls: List of tool calls requested by the LLM (if any).

    Properties:
        has_tool_calls: Whether the response contains tool calls.
    """
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return self.tool_calls is not None and len(self.tool_calls) > 0


@dataclass
class ChatMessage:
    """Represents a message in the chat completions format.

    Attributes:
        role: Role of the message sender (system, user, assistant, tool).
        content: Text content of the message.
        name: Optional name for the sender (used for tool messages).
        tool_call_id: Optional ID linking a tool result to its call.
        tool_calls: Optional list of tool calls (for assistant messages with tool calls).
    """
    role: str
    content: str = ""
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None