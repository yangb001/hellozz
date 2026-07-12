"""Shared type aliases for the agent framework."""
from typing import AsyncIterator, Union

# Identifier types
SessionId = str
UserId = str
ToolName = str

# LLM interaction types
Prompt = str
Response = Union[str, AsyncIterator[str]]

# Stream types
EventStream = AsyncIterator["Event"]

# Result types
MemoryResult = str
ToolResult = str


# Avoid circular import for type hints
from .events import Event