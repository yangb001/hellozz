"""BufferMemory - Short-term memory using in-message buffer with token-based truncation."""
from collections import defaultdict
from typing import List, Optional

from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.session import Message


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Uses a simple heuristic: ~4 characters per token on average.
    This avoids external dependencies while providing reasonable truncation.
    """
    return len(text) // 4 or 1  # at least 1 token for non-empty text


class BufferMemory(BaseMemory):
    """Short-term memory implementation using an in-memory buffer.

    Stores messages per session in a defaultdict. When the buffer exceeds
    max_tokens, oldest messages are removed to stay within the limit.

    Args:
        max_tokens: Maximum approximate token count to retain per session.
                    Defaults to 2000.
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.buffers: dict[str, List[Message]] = defaultdict(list)

    async def add(self, session_id: str, msg: Message) -> None:
        """Add a message to the session buffer.

        After adding, truncates the buffer from the oldest messages
        if the total estimated token count exceeds max_tokens.

        Args:
            session_id: Unique identifier for the session.
            msg: Message to add.
        """
        self.buffers[session_id].append(msg)
        self._truncate(session_id)

    async def get_recent(self, session_id: str, n: int = 10) -> str:
        """Get the content of the most recent n messages, joined by newlines.

        Args:
            session_id: Unique identifier for the session.
            n: Number of recent messages to retrieve. Defaults to 10.

        Returns:
            Newline-joined content of recent messages, or empty string
            if no messages exist.
        """
        messages = self.buffers.get(session_id, [])
        if not messages:
            return ""
        recent = messages[-n:]
        return "\n".join(m.content for m in recent)

    def get_recent_messages(self, session_id: str, n: int = 20) -> List[Message]:
        """Get the most recent n Message objects from the buffer.

        Unlike get_recent(), this returns the original Message objects
        preserving all fields (role, content, sender_id, timestamp),
        which is needed for operations like memory extraction that
        require access to message metadata.

        Args:
            session_id: Unique identifier for the session.
            n: Number of recent messages to retrieve. Defaults to 20.

        Returns:
            List of Message objects, or empty list if no messages exist.
        """
        messages = self.buffers.get(session_id, [])
        if not messages:
            return []
        return list(messages[-n:])

    async def save(self, session_id: str, message: Message) -> None:
        """Save a message (delegates to add).

        Implements BaseMemory.save interface.

        Args:
            session_id: Unique identifier for the session.
            message: Message to save.
        """
        await self.add(session_id, message)

    async def retrieve(
        self,
        session_id: str,
        query: str,
        user_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """Retrieve recent messages as text.

        For BufferMemory, this returns the most recent messages
        regardless of the query content (no semantic search).

        Args:
            session_id: Unique identifier for the session.
            query: Ignored for buffer memory.
            user_ids: Ignored for buffer memory.
            top_k: Number of recent messages to return.

        Returns:
            Newline-joined content of recent messages.
        """
        return await self.get_recent(session_id, n=top_k)

    async def clear(self, session_id: str) -> None:
        """Clear all buffered messages for a session.

        Args:
            session_id: Unique identifier for the session.
        """
        self.buffers[session_id] = []

    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        """No-op for BufferMemory.

        Long-term memory extraction is handled by MemoryManager/Extractor,
        not by the buffer itself.

        Args:
            session_id: Ignored.
            force: Ignored.
        """
        pass

    def _truncate(self, session_id: str) -> None:
        """Remove oldest messages until the buffer is within max_tokens.

        Uses estimated token count (len/4 heuristic).

        Args:
            session_id: Unique identifier for the session.
        """
        messages = self.buffers[session_id]
        while messages:
            total_tokens = sum(_estimate_tokens(m.content) for m in messages)
            if total_tokens <= self.max_tokens:
                break
            messages.pop(0)  # Remove oldest message
