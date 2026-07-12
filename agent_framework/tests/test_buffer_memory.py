"""Tests for BufferMemory short-term memory implementation."""
import pytest
import asyncio
from agent_framework.interfaces.session import Message
from agent_framework.memory.buffer_memory import BufferMemory


@pytest.fixture
def buffer():
    """Create a fresh BufferMemory instance."""
    return BufferMemory(max_tokens=500)


@pytest.fixture
def sample_messages():
    """Create sample messages for testing."""
    return [
        Message(role="user", content="Hello, how are you?"),
        Message(role="assistant", content="I'm doing well, thank you!"),
        Message(role="user", content="What's the weather today?"),
        Message(role="assistant", content="It's sunny and warm."),
        Message(role="user", content="Tell me a joke."),
    ]


class TestBufferMemoryInit:
    """Test BufferMemory initialization."""

    def test_default_max_tokens(self):
        buf = BufferMemory()
        assert buf.max_tokens == 2000

    def test_custom_max_tokens(self):
        buf = BufferMemory(max_tokens=500)
        assert buf.max_tokens == 500

    def test_buffers_is_empty_dict(self):
        buf = BufferMemory()
        assert len(buf.buffers) == 0


class TestBufferMemoryAdd:
    """Test BufferMemory.add method."""

    @pytest.mark.asyncio
    async def test_add_single_message(self, buffer):
        msg = Message(role="user", content="Hello")
        await buffer.add("session1", msg)
        assert len(buffer.buffers["session1"]) == 1
        assert buffer.buffers["session1"][0].content == "Hello"

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self, buffer, sample_messages):
        for msg in sample_messages:
            await buffer.add("session1", msg)
        assert len(buffer.buffers["session1"]) == len(sample_messages)

    @pytest.mark.asyncio
    async def test_add_to_different_sessions(self, buffer):
        msg1 = Message(role="user", content="Session 1 msg")
        msg2 = Message(role="user", content="Session 2 msg")
        await buffer.add("s1", msg1)
        await buffer.add("s2", msg2)
        assert len(buffer.buffers["s1"]) == 1
        assert len(buffer.buffers["s2"]) == 1
        assert buffer.buffers["s1"][0].content == "Session 1 msg"
        assert buffer.buffers["s2"][0].content == "Session 2 msg"

    @pytest.mark.asyncio
    async def test_truncation_by_max_tokens(self):
        """When buffer exceeds max_tokens, oldest messages are removed."""
        buf = BufferMemory(max_tokens=50)
        for i in range(20):
            msg = Message(role="user", content=f"Message number {i} here")
            await buf.add("s1", msg)
        # Buffer should have been truncated to fewer than 20 messages
        remaining = buf.buffers["s1"]
        assert len(remaining) < 20
        # Verify estimated tokens are within limit
        from agent_framework.memory.buffer_memory import _estimate_tokens
        total_tokens = sum(_estimate_tokens(m.content) for m in remaining)
        assert total_tokens <= 50

    @pytest.mark.asyncio
    async def test_truncation_preserves_recent_messages(self):
        """After truncation, the most recent messages should remain."""
        buf = BufferMemory(max_tokens=50)
        for i in range(20):
            msg = Message(role="user", content=f"Message number {i} here")
            await buf.add("s1", msg)
        # The last message should still be present
        assert buf.buffers["s1"][-1].content == "Message number 19 here"


class TestBufferMemoryGetRecent:
    """Test BufferMemory.get_recent method."""

    @pytest.mark.asyncio
    async def test_get_recent_default(self, buffer, sample_messages):
        for msg in sample_messages:
            await buffer.add("s1", msg)
        result = await buffer.get_recent("s1")
        # Default n=10, we have 5 messages, all should be returned
        assert "Hello, how are you?" in result
        assert "Tell me a joke." in result

    @pytest.mark.asyncio
    async def test_get_recent_n(self, buffer, sample_messages):
        for msg in sample_messages:
            await buffer.add("s1", msg)
        result = await buffer.get_recent("s1", n=2)
        # Only last 2 messages
        assert "Tell me a joke." in result
        assert "It's sunny and warm." in result
        assert "Hello, how are you?" not in result

    @pytest.mark.asyncio
    async def test_get_recent_empty_session(self, buffer):
        result = await buffer.get_recent("nonexistent")
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_recent_n_greater_than_available(self, buffer):
        msg = Message(role="user", content="Only message")
        await buffer.add("s1", msg)
        result = await buffer.get_recent("s1", n=100)
        assert "Only message" in result

    @pytest.mark.asyncio
    async def test_get_recent_returns_joined_content(self, buffer):
        msgs = [
            Message(role="user", content="A"),
            Message(role="assistant", content="B"),
            Message(role="user", content="C"),
        ]
        for m in msgs:
            await buffer.add("s1", m)
        result = await buffer.get_recent("s1", n=3)
        # Should be newline-joined content
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "A"
        assert lines[1] == "B"
        assert lines[2] == "C"


class TestBufferMemoryBaseMemoryInterface:
    """Test that BufferMemory correctly implements BaseMemory interface."""

    @pytest.mark.asyncio
    async def test_save_delegates_to_add(self, buffer):
        msg = Message(role="user", content="test save")
        await buffer.save("s1", msg)
        assert len(buffer.buffers["s1"]) == 1
        assert buffer.buffers["s1"][0].content == "test save"

    @pytest.mark.asyncio
    async def test_retrieve_returns_recent(self, buffer):
        msgs = [
            Message(role="user", content="First"),
            Message(role="assistant", content="Second"),
        ]
        for m in msgs:
            await buffer.save("s1", m)
        result = await buffer.retrieve("s1", query="test")
        assert "First" in result
        assert "Second" in result

    @pytest.mark.asyncio
    async def test_clear_removes_session_data(self, buffer):
        msg = Message(role="user", content="to be cleared")
        await buffer.add("s1", msg)
        assert len(buffer.buffers["s1"]) == 1
        await buffer.clear("s1")
        assert len(buffer.buffers["s1"]) == 0

    @pytest.mark.asyncio
    async def test_clear_nonexistent_session(self, buffer):
        # Should not raise
        await buffer.clear("nonexistent")

    @pytest.mark.asyncio
    async def test_extract_long_term_is_noop(self, buffer):
        # BufferMemory doesn't extract long-term, should be a no-op
        await buffer.extract_long_term("s1")
        await buffer.extract_long_term("s1", force=True)


class TestBufferMemoryIsolation:
    """Test session isolation in BufferMemory."""

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self, buffer):
        await buffer.add("s1", Message(role="user", content="S1 message"))
        await buffer.add("s2", Message(role="user", content="S2 message"))
        result1 = await buffer.get_recent("s1")
        result2 = await buffer.get_recent("s2")
        assert "S1 message" in result1
        assert "S2 message" not in result1
        assert "S2 message" in result2
        assert "S1 message" not in result2

    @pytest.mark.asyncio
    async def test_clear_one_session_keeps_others(self, buffer):
        await buffer.add("s1", Message(role="user", content="S1"))
        await buffer.add("s2", Message(role="user", content="S2"))
        await buffer.clear("s1")
        assert len(buffer.buffers["s1"]) == 0
        assert len(buffer.buffers["s2"]) == 1
