"""Tests for VectorMemory long-term memory implementation.

This module tests:
- VectorMemory class creation and initialization
- add, query, add_user, query_user methods
- Collection naming conventions (session_{id}, user_{id})
- Integration with VectorStore interface
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVectorMemoryInit:
    """Test VectorMemory initialization."""

    def test_import_vector_memory(self):
        """Test that VectorMemory can be imported."""
        from agent_framework.memory.vector_memory import VectorMemory
        assert VectorMemory is not None

    def test_vector_memory_init_with_store(self):
        """Test VectorMemory initialization with VectorStore."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        # Create mock store
        mock_store = MagicMock(spec=VectorStore)

        memory = VectorMemory(vector_store=mock_store)
        assert memory is not None
        assert memory.store == mock_store


class TestVectorMemoryAdd:
    """Test VectorMemory add method."""

    @pytest.mark.asyncio
    async def test_add_calls_store_add(self):
        """Test that add method calls store.add with correct parameters."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.add = AsyncMock(return_value="doc-id-123")

        memory = VectorMemory(vector_store=mock_store)

        await memory.add(
            session_id="session-123",
            text="This is a test memory.",
            metadata={"source": "test"}
        )

        # Verify store.add was called with session_ prefix
        mock_store.add.assert_called_once_with(
            collection="session_session-123",
            text="This is a test memory.",
            metadata={"source": "test"}
        )

    @pytest.mark.asyncio
    async def test_add_without_metadata(self):
        """Test add method without metadata."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.add = AsyncMock(return_value="doc-id-456")

        memory = VectorMemory(vector_store=mock_store)

        await memory.add(
            session_id="session-456",
            text="Another memory."
        )

        mock_store.add.assert_called_once_with(
            collection="session_session-456",
            text="Another memory.",
            metadata=None
        )


class TestVectorMemoryQuery:
    """Test VectorMemory query method."""

    @pytest.mark.asyncio
    async def test_query_returns_joined_text(self):
        """Test that query returns concatenated text from results."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        mock_store = MagicMock(spec=VectorStore)
        mock_store.query = AsyncMock(return_value=[
            SearchResult(id="1", text="First result.", score=0.9),
            SearchResult(id="2", text="Second result.", score=0.8),
        ])

        memory = VectorMemory(vector_store=mock_store)

        result = await memory.query(
            session_id="session-789",
            query="What is AI?",
            top_k=5
        )

        # Verify store.query was called with session_ prefix
        mock_store.query.assert_called_once_with(
            collection="session_session-789",
            query="What is AI?",
            top_k=5
        )

        # Verify result is joined text
        assert result == "First result.\nSecond result."

    @pytest.mark.asyncio
    async def test_query_empty_results(self):
        """Test query with no results."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.query = AsyncMock(return_value=[])

        memory = VectorMemory(vector_store=mock_store)

        result = await memory.query(
            session_id="session-empty",
            query="nonexistent",
            top_k=5
        )

        assert result == ""

    @pytest.mark.asyncio
    async def test_query_with_custom_top_k(self):
        """Test query with custom top_k parameter."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        mock_store = MagicMock(spec=VectorStore)
        mock_store.query = AsyncMock(return_value=[
            SearchResult(id="1", text="Result.", score=0.9),
        ])

        memory = VectorMemory(vector_store=mock_store)

        await memory.query(
            session_id="session-custom",
            query="test",
            top_k=3
        )

        mock_store.query.assert_called_once_with(
            collection="session_session-custom",
            query="test",
            top_k=3
        )


class TestVectorMemoryAddUser:
    """Test VectorMemory add_user method."""

    @pytest.mark.asyncio
    async def test_add_user_calls_store_add(self):
        """Test that add_user calls store.add with user_ prefix."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.add = AsyncMock(return_value="user-doc-123")

        memory = VectorMemory(vector_store=mock_store)

        await memory.add_user(
            user_id="user-001",
            text="User prefers dark mode."
        )

        # Verify store.add was called with user_ prefix
        mock_store.add.assert_called_once_with(
            collection="user_user-001",
            text="User prefers dark mode."
        )


class TestVectorMemoryQueryUser:
    """Test VectorMemory query_user method."""

    @pytest.mark.asyncio
    async def test_query_user_returns_joined_text(self):
        """Test that query_user returns concatenated text."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        mock_store = MagicMock(spec=VectorStore)
        mock_store.query = AsyncMock(return_value=[
            SearchResult(id="1", text="User likes Python.", score=0.95),
            SearchResult(id="2", text="User works at tech company.", score=0.85),
        ])

        memory = VectorMemory(vector_store=mock_store)

        result = await memory.query_user(
            user_id="user-001",
            query="What do we know about the user?",
            top_k=5
        )

        # Verify store.query was called with user_ prefix
        mock_store.query.assert_called_once_with(
            collection="user_user-001",
            query="What do we know about the user?",
            top_k=5
        )

        # Verify result is joined text
        assert result == "User likes Python.\nUser works at tech company."

    @pytest.mark.asyncio
    async def test_query_user_empty_results(self):
        """Test query_user with no results."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.query = AsyncMock(return_value=[])

        memory = VectorMemory(vector_store=mock_store)

        result = await memory.query_user(
            user_id="user-empty",
            query="preferences",
            top_k=5
        )

        assert result == ""


class TestVectorMemoryCollectionNaming:
    """Test collection naming conventions."""

    @pytest.mark.asyncio
    async def test_session_collection_naming(self):
        """Test that session collections use session_ prefix."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.add = AsyncMock(return_value="id")
        mock_store.query = AsyncMock(return_value=[])

        memory = VectorMemory(vector_store=mock_store)

        # Test various session IDs
        test_cases = [
            ("abc123", "session_abc123"),
            ("session-with-dashes", "session_session-with-dashes"),
            ("12345", "session_12345"),
        ]

        for session_id, expected_collection in test_cases:
            await memory.add(session_id=session_id, text="test")
            mock_store.add.assert_called_with(
                collection=expected_collection,
                text="test",
                metadata=None
            )

    @pytest.mark.asyncio
    async def test_user_collection_naming(self):
        """Test that user collections use user_ prefix."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        mock_store = MagicMock(spec=VectorStore)
        mock_store.add = AsyncMock(return_value="id")
        mock_store.query = AsyncMock(return_value=[])

        memory = VectorMemory(vector_store=mock_store)

        # Test various user IDs
        test_cases = [
            ("user001", "user_user001"),
            ("user-with-dashes", "user_user-with-dashes"),
            ("99999", "user_99999"),
        ]

        for user_id, expected_collection in test_cases:
            await memory.add_user(user_id=user_id, text="test")
            mock_store.add.assert_called_with(
                collection=expected_collection,
                text="test"
            )


class TestVectorMemoryIntegration:
    """Integration tests for VectorMemory with real VectorStore."""

    @pytest.mark.asyncio
    async def test_full_session_workflow(self):
        """Test complete session memory workflow."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        # Create a more realistic mock
        class MockVectorStore:
            def __init__(self):
                self.data = {}

            async def add(self, collection, text, metadata=None, id=None):
                if collection not in self.data:
                    self.data[collection] = []
                doc_id = id or f"doc-{len(self.data[collection])}"
                self.data[collection].append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata
                })
                return doc_id

            async def query(self, collection, query, top_k=5):
                if collection not in self.data:
                    return []
                # Simple mock: return all docs up to top_k
                docs = self.data[collection][:top_k]
                return [
                    SearchResult(
                        id=doc["id"],
                        text=doc["text"],
                        score=0.9,
                        metadata=doc["metadata"]
                    )
                    for doc in docs
                ]

        store = MockVectorStore()
        memory = VectorMemory(vector_store=store)

        # Add session memories
        await memory.add("session-1", "User asked about Python.")
        await memory.add("session-1", "User prefers async code.")

        # Query session memories
        result = await memory.query("session-1", "What does user prefer?")
        assert "User asked about Python." in result
        assert "User prefers async code." in result

    @pytest.mark.asyncio
    async def test_full_user_workflow(self):
        """Test complete user memory workflow."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        class MockVectorStore:
            def __init__(self):
                self.data = {}

            async def add(self, collection, text, metadata=None, id=None):
                if collection not in self.data:
                    self.data[collection] = []
                doc_id = id or f"doc-{len(self.data[collection])}"
                self.data[collection].append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata
                })
                return doc_id

            async def query(self, collection, query, top_k=5):
                if collection not in self.data:
                    return []
                docs = self.data[collection][:top_k]
                return [
                    SearchResult(
                        id=doc["id"],
                        text=doc["text"],
                        score=0.9,
                        metadata=doc["metadata"]
                    )
                    for doc in docs
                ]

        store = MockVectorStore()
        memory = VectorMemory(vector_store=store)

        # Add user memories
        await memory.add_user("user-100", "User is a software engineer.")
        await memory.add_user("user-100", "User likes Rust and Python.")

        # Query user memories
        result = await memory.query_user("user-100", "What is user's job?")
        assert "User is a software engineer." in result
        assert "User likes Rust and Python." in result

    @pytest.mark.asyncio
    async def test_session_and_user_isolation(self):
        """Test that session and user memories are isolated."""
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        class MockVectorStore:
            def __init__(self):
                self.data = {}

            async def add(self, collection, text, metadata=None, id=None):
                if collection not in self.data:
                    self.data[collection] = []
                doc_id = id or f"doc-{len(self.data[collection])}"
                self.data[collection].append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata
                })
                return doc_id

            async def query(self, collection, query, top_k=5):
                if collection not in self.data:
                    return []
                docs = self.data[collection][:top_k]
                return [
                    SearchResult(
                        id=doc["id"],
                        text=doc["text"],
                        score=0.9,
                        metadata=doc["metadata"]
                    )
                    for doc in docs
                ]

        store = MockVectorStore()
        memory = VectorMemory(vector_store=store)

        # Add session memory
        await memory.add("session-1", "Session specific memory.")

        # Add user memory
        await memory.add_user("user-1", "User specific memory.")

        # Query session should not return user memory
        session_result = await memory.query("session-1", "memory")
        assert "Session specific memory." in session_result
        assert "User specific memory." not in session_result

        # Query user should not return session memory
        user_result = await memory.query_user("user-1", "memory")
        assert "User specific memory." in user_result
        assert "Session specific memory." not in user_result
