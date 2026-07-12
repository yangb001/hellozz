"""Base memory interface - Defines the contract for memory implementations."""
from abc import ABC, abstractmethod
from typing import List, Optional


class BaseMemory(ABC):
    """Abstract base class for memory implementations.

    A memory implementation handles storage and retrieval of conversation
    history and extracted facts for agent sessions.

    All memory operations are session-scoped, allowing isolation between
    different conversation contexts.
    """

    @abstractmethod
    async def save(self, session_id: str, message: "Message") -> None:
        """Save a message/event to memory.

        Args:
            session_id: Unique identifier for the session.
            message: Message object to save.
        """
        ...

    @abstractmethod
    async def retrieve(
        self,
        session_id: str,
        query: str,
        user_ids: Optional[List[str]] = None,
        top_k: int = 5
    ) -> str:
        """Retrieve relevant memories as concatenated text.

        Args:
            session_id: Unique identifier for the session.
            query: Query string to search for relevant memories.
            user_ids: Optional list of user IDs for cross-session memory.
            top_k: Number of top results to retrieve.

        Returns:
            Concatenated relevant memory text.
        """
        ...

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear all memories for a session.

        Args:
            session_id: Unique identifier for the session.
        """
        ...

    @abstractmethod
    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        """Trigger extraction of long-term memories.

        Args:
            session_id: Unique identifier for the session.
            force: If True, force extraction even if conditions not met.
        """
        ...


# Avoid circular import at module level
from .session import Message