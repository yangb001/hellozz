"""VectorMemory - Long-term memory implementation using vector storage.

This module provides VectorMemory class for storing and retrieving
long-term memories using vector similarity search.

Collections are named with prefixes:
- session_{session_id} for session-scoped memories
- user_{user_id} for user-scoped memories (cross-session)
"""
from typing import Optional

from ..infrastructure.storage.vector_store import VectorStore


class VectorMemory:
    """Long-term memory implementation using vector storage.

    This class stores memories as vectors and retrieves them using
    similarity search. It supports both session-scoped and user-scoped
    memory storage.

    Collection naming convention:
    - Session memories: session_{session_id}
    - User memories: user_{user_id}
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        """Initialize VectorMemory with a vector store.

        Args:
            vector_store: VectorStore implementation for storage backend.
                         If None, vector memory operations will be no-ops.
        """
        self.store = vector_store

    async def add(
        self,
        session_id: str,
        text: str,
        metadata: Optional[dict] = None
    ) -> None:
        """Add a memory to session-scoped storage.

        Args:
            session_id: Unique identifier for the session.
            text: Text content to store as memory.
            metadata: Optional metadata to associate with the memory.
        """
        if self.store is None:
            return
        collection = f"session_{session_id}"
        await self.store.add(
            collection=collection,
            text=text,
            metadata=metadata
        )

    async def query(
        self,
        session_id: str,
        query: str,
        top_k: int = 5
    ) -> str:
        """Query session-scoped memories.

        Args:
            session_id: Unique identifier for the session.
            query: Query text to search for relevant memories.
            top_k: Maximum number of results to return.

        Returns:
            Concatenated text of relevant memories, separated by newlines.
        """
        if self.store is None:
            return ""
        collection = f"session_{session_id}"
        results = await self.store.query(
            collection=collection,
            query=query,
            top_k=top_k
        )
        return "\n".join([r.text for r in results])

    async def add_user(
        self,
        user_id: str,
        text: str
    ) -> None:
        """Add a memory to user-scoped storage (cross-session).

        Args:
            user_id: Unique identifier for the user.
            text: Text content to store as user memory.
        """
        if self.store is None:
            return
        collection = f"user_{user_id}"
        await self.store.add(
            collection=collection,
            text=text
        )

    async def query_user(
        self,
        user_id: str,
        query: str,
        top_k: int = 5
    ) -> str:
        """Query user-scoped memories (cross-session).

        Args:
            user_id: Unique identifier for the user.
            query: Query text to search for relevant memories.
            top_k: Maximum number of results to return.

        Returns:
            Concatenated text of relevant user memories, separated by newlines.
        """
        if self.store is None:
            return ""
        collection = f"user_{user_id}"
        results = await self.store.query(
            collection=collection,
            query=query,
            top_k=top_k
        )
        return "\n".join([r.text for r in results])
