"""Independent tests for MemoryExtractor fix verification.

This test file verifies the fix for the AttributeError that occurred when
MemoryManager.extract_long_term() tried to call get_recent_messages() on
BufferMemory.

Design Reference:
- BufferMemory.get_recent_messages() returns List[Message] (not strings)
- MemoryManager.extract_long_term() uses get_recent_messages() to get Message objects
- End-to-end flow should work without AttributeError

Test Coverage:
1. BufferMemory.get_recent_messages() returns correct Message objects
2. MemoryManager.extract_long_term() works with real components
3. End-to-end flow from save to extraction works
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework.interfaces.session import Message
from agent_framework.memory.buffer_memory import BufferMemory
from agent_framework.memory.memory_manager import MemoryManager, MemoryConfig
from agent_framework.memory.extractor import MemoryExtractor, MemoryFact
from agent_framework.memory.vector_memory import VectorMemory


class TestBufferMemoryGetRecentMessages:
    """Test BufferMemory.get_recent_messages() returns correct Message objects."""

    def test_get_recent_messages_returns_list(self):
        """get_recent_messages should return a list, not a string."""
        buffer = BufferMemory(max_tokens=1000)
        result = buffer.get_recent_messages("session-1", n=5)
        assert isinstance(result, list), f"Expected list, got {type(result)}"

    def test_get_recent_messages_returns_empty_list_for_empty_buffer(self):
        """get_recent_messages should return empty list when no messages exist."""
        buffer = BufferMemory(max_tokens=1000)
        result = buffer.get_recent_messages("session-1", n=5)
        assert result == [], "Should return empty list for empty buffer"

    def test_get_recent_messages_returns_message_objects(self):
        """get_recent_messages should return Message objects, not strings."""
        buffer = BufferMemory(max_tokens=1000)
        msg = Message(role="user", content="Hello", sender_id="user-1")
        buffer.buffers["session-1"] = [msg]

        result = buffer.get_recent_messages("session-1", n=5)
        assert len(result) == 1, f"Expected 1 message, got {len(result)}"
        assert isinstance(result[0], Message), f"Expected Message object, got {type(result[0])}"

    def test_get_recent_messages_preserves_message_fields(self):
        """get_recent_messages should preserve all Message fields (role, content, sender_id, timestamp)."""
        buffer = BufferMemory(max_tokens=1000)
        msg = Message(role="user", content="Hello", sender_id="user-1")
        buffer.buffers["session-1"] = [msg]

        result = buffer.get_recent_messages("session-1", n=5)
        retrieved_msg = result[0]

        assert retrieved_msg.role == "user", f"Expected role='user', got '{retrieved_msg.role}'"
        assert retrieved_msg.content == "Hello", f"Expected content='Hello', got '{retrieved_msg.content}'"
        assert retrieved_msg.sender_id == "user-1", f"Expected sender_id='user-1', got '{retrieved_msg.sender_id}'"
        assert retrieved_msg.timestamp is not None, "Timestamp should not be None"

    def test_get_recent_messages_returns_most_recent_n(self):
        """get_recent_messages should return the most recent n messages."""
        buffer = BufferMemory(max_tokens=10000)
        messages = [
            Message(role="user", content=f"Message {i}", sender_id="user-1")
            for i in range(10)
        ]
        buffer.buffers["session-1"] = messages

        result = buffer.get_recent_messages("session-1", n=3)
        assert len(result) == 3, f"Expected 3 messages, got {len(result)}"

        # Should be the last 3 messages
        assert result[0].content == "Message 7", f"Expected 'Message 7', got '{result[0].content}'"
        assert result[1].content == "Message 8", f"Expected 'Message 8', got '{result[1].content}'"
        assert result[2].content == "Message 9", f"Expected 'Message 9', got '{result[2].content}'"

    def test_get_recent_messages_default_n_is_20(self):
        """get_recent_messages should default to n=20 per implementation."""
        import inspect
        sig = inspect.signature(BufferMemory.get_recent_messages)
        params = sig.parameters

        assert 'n' in params, "get_recent_messages should have 'n' parameter"
        assert params['n'].default == 20, f"Default for 'n' should be 20, got {params['n'].default}"

    def test_get_recent_messages_method_exists(self):
        """BufferMemory must have get_recent_messages method."""
        assert hasattr(BufferMemory, 'get_recent_messages'), \
            "BufferMemory should have get_recent_messages method"

    def test_get_recent_messages_is_not_async(self):
        """get_recent_messages is a synchronous method (not async)."""
        import inspect
        assert not inspect.iscoroutinefunction(BufferMemory.get_recent_messages), \
            "get_recent_messages should be synchronous (not async)"


class TestMemoryManagerExtractLongTerm:
    """Test MemoryManager.extract_long_term() works correctly."""

    @pytest.fixture
    def mock_extractor(self):
        """Create a mock MemoryExtractor."""
        extractor = AsyncMock(spec=MemoryExtractor)
        extractor.extract.return_value = [
            MemoryFact(
                content="User likes Python",
                metadata={"type": "preference"},
                user_id="user-1"
            )
        ]
        return extractor

    @pytest.fixture
    def mock_vector_memory(self):
        """Create a mock VectorMemory."""
        vector_mem = AsyncMock(spec=VectorMemory)
        return vector_mem

    @pytest.fixture
    def memory_manager(self, mock_extractor, mock_vector_memory):
        """Create a MemoryManager with real BufferMemory and mocked dependencies."""
        buffer = BufferMemory(max_tokens=2000)
        config = MemoryConfig(trigger="every_n_turns", every_n=5)
        manager = MemoryManager(
            short_term=buffer,
            long_term=mock_vector_memory,
            extractor=mock_extractor,
            config=config
        )
        return manager

    @pytest.mark.asyncio
    async def test_extract_long_term_calls_get_recent_messages(self, memory_manager):
        """extract_long_term should call get_recent_messages on BufferMemory."""
        # Add messages to buffer
        for i in range(5):
            msg = Message(role="user", content=f"Message {i}", sender_id="user-1")
            await memory_manager.short_term.add("session-1", msg)

        # Call extract_long_term
        await memory_manager.extract_long_term("session-1")

        # Verify extractor was called with Message objects (not strings)
        memory_manager.extractor.extract.assert_called_once()
        call_args = memory_manager.extractor.extract.call_args
        messages_arg = call_args[0][0]  # First positional argument

        assert isinstance(messages_arg, list), "extractor.extract should receive a list"
        assert len(messages_arg) > 0, "Should pass messages to extractor"
        assert isinstance(messages_arg[0], Message), \
            f"Expected Message objects, got {type(messages_arg[0])}"

    @pytest.mark.asyncio
    async def test_extract_long_term_with_empty_buffer(self, memory_manager):
        """extract_long_term should handle empty buffer gracefully."""
        # Call extract_long_term with no messages
        await memory_manager.extract_long_term("session-1")

        # Should still call extractor with empty list
        memory_manager.extractor.extract.assert_called_once()
        call_args = memory_manager.extractor.extract.call_args
        messages_arg = call_args[0][0]

        assert isinstance(messages_arg, list), "Should pass a list"
        assert len(messages_arg) == 0, "Should pass empty list for empty buffer"

    @pytest.mark.asyncio
    async def test_extract_long_term_stores_facts_in_vector_memory(
        self, memory_manager, mock_vector_memory
    ):
        """extract_long_term should store extracted facts in vector memory."""
        # Add messages to buffer
        msg = Message(role="user", content="I love Python", sender_id="user-1")
        await memory_manager.short_term.add("session-1", msg)

        # Call extract_long_term
        await memory_manager.extract_long_term("session-1")

        # Verify facts were stored in vector memory
        mock_vector_memory.add.assert_called_once()
        call_args = mock_vector_memory.add.call_args

        assert call_args[0][0] == "session-1", "Should store with correct session_id"
        assert call_args[0][1] == "User likes Python", "Should store fact content"

    @pytest.mark.asyncio
    async def test_extract_long_term_stores_user_facts(self, memory_manager, mock_vector_memory):
        """extract_long_term should store user-specific facts in user memory."""
        # Add messages to buffer
        msg = Message(role="user", content="I love Python", sender_id="user-1")
        await memory_manager.short_term.add("session-1", msg)

        # Call extract_long_term
        await memory_manager.extract_long_term("session-1")

        # Verify user fact was stored
        mock_vector_memory.add_user.assert_called_once()
        call_args = mock_vector_memory.add_user.call_args

        assert call_args[0][0] == "user-1", "Should store with correct user_id"
        assert call_args[0][1] == "User likes Python", "Should store fact content"

    @pytest.mark.asyncio
    async def test_extract_long_term_no_attribute_error(self, memory_manager):
        """extract_long_term should not raise AttributeError (the original bug)."""
        # Add messages to buffer
        for i in range(3):
            msg = Message(role="user", content=f"Message {i}", sender_id="user-1")
            await memory_manager.short_term.add("session-1", msg)

        # This should NOT raise AttributeError
        try:
            await memory_manager.extract_long_term("session-1")
        except AttributeError as e:
            pytest.fail(f"AttributeError raised: {e}")


class TestEndToEndFlow:
    """Test end-to-end flow from save to extraction."""

    @pytest.fixture
    def mock_extractor(self):
        """Create a mock MemoryExtractor that returns facts."""
        extractor = AsyncMock(spec=MemoryExtractor)
        extractor.extract.return_value = [
            MemoryFact(
                content="User prefers dark mode",
                metadata={"type": "preference"},
                user_id="user-1"
            )
        ]
        # is_important returns True to trigger extraction
        extractor.is_important.return_value = True
        return extractor

    @pytest.fixture
    def mock_vector_memory(self):
        """Create a mock VectorMemory."""
        return AsyncMock(spec=VectorMemory)

    @pytest.mark.asyncio
    async def test_save_triggers_extraction_with_smart_trigger(self, mock_extractor, mock_vector_memory):
        """With smart trigger, important messages should trigger extraction."""
        buffer = BufferMemory(max_tokens=2000)
        config = MemoryConfig(trigger="smart")
        manager = MemoryManager(
            short_term=buffer,
            long_term=mock_vector_memory,
            extractor=mock_extractor,
            config=config
        )

        # Save an important message
        msg = Message(role="user", content="I prefer dark mode", sender_id="user-1")
        await manager.save("session-1", msg)

        # Verify extraction was triggered
        mock_extractor.is_important.assert_called_once_with(msg)
        mock_extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_triggers_extraction_with_every_n_turns(self, mock_extractor, mock_vector_memory):
        """With every_n_turns trigger, extraction should happen every N messages."""
        buffer = BufferMemory(max_tokens=2000)
        config = MemoryConfig(trigger="every_n_turns", every_n=3)
        manager = MemoryManager(
            short_term=buffer,
            long_term=mock_vector_memory,
            extractor=mock_extractor,
            config=config
        )

        # Save 3 messages (should trigger extraction on 3rd)
        for i in range(3):
            msg = Message(role="user", content=f"Message {i}", sender_id="user-1")
            await manager.save("session-1", msg)

        # Extraction should have been triggered once
        mock_extractor.extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_end_to_end_no_attribute_error(self, mock_extractor, mock_vector_memory):
        """Complete flow should work without AttributeError."""
        buffer = BufferMemory(max_tokens=2000)
        config = MemoryConfig(trigger="every_n_turns", every_n=2)
        manager = MemoryManager(
            short_term=buffer,
            long_term=mock_vector_memory,
            extractor=mock_extractor,
            config=config
        )

        # Simulate a conversation
        messages = [
            Message(role="user", content="Hello, I'm John", sender_id="user-1"),
            Message(role="assistant", content="Hi John! How can I help?"),
            Message(role="user", content="I need help with Python", sender_id="user-1"),
        ]

        try:
            for msg in messages:
                await manager.save("session-1", msg)
        except AttributeError as e:
            pytest.fail(f"AttributeError in end-to-end flow: {e}")

        # Verify messages were stored in buffer
        recent = buffer.get_recent_messages("session-1", n=10)
        assert len(recent) == 3, f"Expected 3 messages in buffer, got {len(recent)}"

    @pytest.mark.asyncio
    async def test_extract_long_term_with_real_buffer(self, mock_extractor, mock_vector_memory):
        """extract_long_term should work with real BufferMemory."""
        buffer = BufferMemory(max_tokens=2000)
        config = MemoryConfig(trigger="every_n_turns", every_n=5)
        manager = MemoryManager(
            short_term=buffer,
            long_term=mock_vector_memory,
            extractor=mock_extractor,
            config=config
        )

        # Add messages
        for i in range(5):
            msg = Message(role="user", content=f"Message {i}", sender_id="user-1")
            await buffer.add("session-1", msg)

        # Call extract_long_term
        await manager.extract_long_term("session-1")

        # Verify extractor received Message objects
        call_args = mock_extractor.extract.call_args[0][0]
        assert len(call_args) == 5, f"Expected 5 messages, got {len(call_args)}"
        assert all(isinstance(m, Message) for m in call_args), \
            "All items should be Message objects"


class TestCodeQualityReview:
    """Review code quality of the fix."""

    def test_get_recent_messages_returns_copy(self):
        """get_recent_messages should return a copy, not a reference to internal list."""
        buffer = BufferMemory(max_tokens=1000)
        msg = Message(role="user", content="Hello", sender_id="user-1")
        buffer.buffers["session-1"] = [msg]

        result = buffer.get_recent_messages("session-1", n=5)

        # Modifying the result should not affect internal buffer
        result.append(Message(role="user", content="Injected", sender_id="attacker"))

        internal = buffer.buffers["session-1"]
        assert len(internal) == 1, "Internal buffer should not be modified"
        assert internal[0].content == "Hello", "Original message should be unchanged"

    def test_get_recent_messages_handles_none_session(self):
        """get_recent_messages should handle non-existent session gracefully."""
        buffer = BufferMemory(max_tokens=1000)
        result = buffer.get_recent_messages("non-existent-session", n=5)
        assert result == [], "Should return empty list for non-existent session"

    def test_buffer_memory_inherits_base_memory(self):
        """BufferMemory should properly inherit from BaseMemory."""
        from agent_framework.interfaces.base_memory import BaseMemory
        assert issubclass(BufferMemory, BaseMemory), \
            "BufferMemory should inherit from BaseMemory"

    def test_memory_manager_inherits_base_memory(self):
        """MemoryManager should properly inherit from BaseMemory."""
        from agent_framework.interfaces.base_memory import BaseMemory
        assert issubclass(MemoryManager, BaseMemory), \
            "MemoryManager should inherit from BaseMemory"

    def test_extractor_accepts_message_list(self):
        """MemoryExtractor.extract should accept List[Message] per design."""
        import inspect
        sig = inspect.signature(MemoryExtractor.extract)
        params = sig.parameters

        assert 'messages' in params, "extract should have 'messages' parameter"
        # The type hint should be List[Message]
        annotation = params['messages'].annotation
        # Check it's a list type (exact annotation may vary)
        assert annotation is not None, "messages parameter should have type annotation"

    def test_memory_fact_dataclass(self):
        """MemoryFact should be a proper dataclass with required fields."""
        from dataclasses import fields
        fact_fields = {f.name for f in fields(MemoryFact)}
        assert 'content' in fact_fields, "MemoryFact should have 'content' field"
        assert 'metadata' in fact_fields, "MemoryFact should have 'metadata' field"
        assert 'user_id' in fact_fields, "MemoryFact should have 'user_id' field"

    def test_memory_fact_default_values(self):
        """MemoryFact should have correct default values."""
        fact = MemoryFact(content="test")
        assert fact.metadata == {}, f"Default metadata should be empty dict, got {fact.metadata}"
        assert fact.user_id is None, f"Default user_id should be None, got {fact.user_id}"
