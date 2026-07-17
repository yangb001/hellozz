"""MemoryManager - Unified entry point coordinating short-term and long-term memory.

This module implements the MemoryManager class which orchestrates BufferMemory
(short-term), VectorMemory (long-term), and MemoryExtractor to provide a
coherent memory system for agent sessions.

参考：详细设计.md 第6.1节
"""
import asyncio
from typing import List, Optional

from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.session import Message
from agent_framework.memory.buffer_memory import BufferMemory
from agent_framework.memory.vector_memory import VectorMemory
from agent_framework.memory.extractor import MemoryExtractor
from agent_framework.core.config import MemoryConfig


class MemoryManager(BaseMemory):
    """Unified memory entry point implementing the BaseMemory interface.

    Coordinates between BufferMemory (short-term), VectorMemory (long-term),
    and MemoryExtractor to provide a complete memory system.

    Args:
        short_term: BufferMemory instance for short-term conversation history.
        long_term: VectorMemory instance for long-term vector-based storage.
        extractor: MemoryExtractor instance for LLM-based fact extraction.
        config: MemoryConfig controlling extraction trigger behavior.
    """

    def __init__(
        self,
        short_term: BufferMemory,
        long_term: VectorMemory,
        extractor: MemoryExtractor,
        config: MemoryConfig,
    ):
        self.short_term = short_term
        self.long_term = long_term
        self.extractor = extractor
        self.config = config
        self._turn_counter: dict[str, int] = {}

    async def save(self, session_id: str, message: Message) -> None:
        """Save a message to short-term memory, optionally triggering extraction.

        Always adds the message to the short-term buffer. Then, depending on
        the configured trigger strategy:
        - "smart": Calls extractor.is_important() and extracts if important.
        - "every_n_turns": Tracks message count and extracts every N turns.

        Args:
            session_id: Unique identifier for the session.
            message: Message to save.
        """
        await self.short_term.add(session_id, message)

        trigger = self.config.trigger.lower() if self.config.trigger else ""
        if trigger == "smart":
            # Run importance check and extraction asynchronously to avoid blocking the agent
            asyncio.create_task(self._smart_extract(session_id, message))
        elif trigger == "every_n_turns":
            self._turn_counter[session_id] = self._turn_counter.get(session_id, 0) + 1
            if self._turn_counter[session_id] >= self.config.every_n:
                self._turn_counter[session_id] = 0
                await self.extract_long_term(session_id)
        # else: trigger is "off" or unknown - do nothing

    async def retrieve(
        self,
        session_id: str,
        query: str,
        user_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> str:
        """Retrieve combined short-term, long-term, and user memories.

        Concatenates memory from three sources:
        1. Short-term buffer (recent N messages)
        2. Long-term vector store (semantic search on session memories)
        3. User memories (cross-session, per-user memories)

        Args:
            session_id: Unique identifier for the session.
            query: Query string for semantic search.
            user_ids: Optional list of user IDs for cross-session memory.
            top_k: Number of top results from long-term memory.

        Returns:
            Concatenated memory text from all sources.
        """
        short_ctx = await self.short_term.get_recent(session_id, n=10)
        long_ctx = await self.long_term.query(session_id, query, top_k=top_k)

        user_ctx = ""
        if user_ids:
            for uid in user_ids:
                mem = await self.long_term.query_user(uid, query, top_k=2)
                user_ctx += mem

        return f"{user_ctx}\n{long_ctx}\nRecent: {short_ctx}"

    async def _smart_extract(self, session_id: str, message: Message) -> None:
        """Asynchronously check message importance and extract if important.

        Args:
            session_id: Unique identifier for the session.
            message: Message to check for importance.
        """
        try:
            if await self.extractor.is_important(message):
                await self.extract_long_term(session_id, force=True)
        except Exception as e:
            logger = __import__('logging').getLogger(__name__)
            logger.warning(f"Failed to extract long-term memory: {e}")

    async def clear(self, session_id: str) -> None:
        """Clear all memories for a session.

        Clears the short-term buffer and resets the turn counter.

        Args:
            session_id: Unique identifier for the session.
        """
        await self.short_term.clear(session_id)
        self._turn_counter[session_id] = 0

    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        """Extract facts from recent messages and store in long-term memory.

        Gets recent messages from the short-term buffer, uses the extractor
        to identify important facts, and stores them in the vector memory.
        If a fact has a user_id, it's also stored in user-scoped memory.

        Args:
            session_id: Unique identifier for the session.
            force: If True, force extraction (currently unused, reserved for
                   future conditional logic).
        """
        recent = self.short_term.get_recent_messages(session_id, n=20)
        facts = await self.extractor.extract(recent)

        for fact in facts:
            await self.long_term.add(session_id, fact.content, fact.metadata)
            if fact.user_id:
                await self.long_term.add_user(fact.user_id, fact.content)
