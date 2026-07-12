"""Independent test cases for BufferMemory short-term memory.

This module contains independent verification tests for the BufferMemory
implementation, following the detailed design specification in section 6.2.

Test categories:
1. BaseMemory interface compliance
2. Constructor and initialization
3. add / save message operations
4. get_recent retrieval
5. retrieve (BaseMemory interface)
6. clear operation
7. extract_long_term (no-op verification)
8. Token-based truncation logic
9. Multi-session isolation
10. Boundary conditions
"""
import pytest
from collections import defaultdict
from typing import get_type_hints

from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.session import Message
from agent_framework.memory.buffer_memory import BufferMemory, _estimate_tokens


# ============================================================================
# Helper
# ============================================================================


def _make_msg(content: str, role: str = "user") -> Message:
    """Create a test Message with given content."""
    return Message(role=role, content=content)


# ============================================================================
# 1. BaseMemory Interface Compliance
# ============================================================================


class TestBaseMemoryCompliance:
    """Verify BufferMemory correctly implements BaseMemory."""

    def test_buffer_memory_inherits_base_memory(self):
        """BufferMemory must be a subclass of BaseMemory."""
        assert issubclass(BufferMemory, BaseMemory), \
            "BufferMemory 应继承自 BaseMemory"

    def test_buffer_memory_is_not_abstract(self):
        """BufferMemory must be concrete (instantiable)."""
        mem = BufferMemory()
        assert isinstance(mem, BaseMemory)

    def test_has_save_method(self):
        """BufferMemory must implement save method from BaseMemory."""
        assert hasattr(BufferMemory, "save"), "BufferMemory 缺少 save 方法"

    def test_has_retrieve_method(self):
        """BufferMemory must implement retrieve method from BaseMemory."""
        assert hasattr(BufferMemory, "retrieve"), "BufferMemory 缺少 retrieve 方法"

    def test_has_clear_method(self):
        """BufferMemory must implement clear method from BaseMemory."""
        assert hasattr(BufferMemory, "clear"), "BufferMemory 缺少 clear 方法"

    def test_has_extract_long_term_method(self):
        """BufferMemory must implement extract_long_term method from BaseMemory."""
        assert hasattr(BufferMemory, "extract_long_term"), \
            "BufferMemory 缺少 extract_long_term 方法"

    def test_save_is_async(self):
        """save method must be async."""
        import inspect
        assert inspect.iscoroutinefunction(BufferMemory.save), \
            "save 方法应为 async"

    def test_retrieve_is_async(self):
        """retrieve method must be async."""
        import inspect
        assert inspect.iscoroutinefunction(BufferMemory.retrieve), \
            "retrieve 方法应为 async"

    def test_clear_is_async(self):
        """clear method must be async."""
        import inspect
        assert inspect.iscoroutinefunction(BufferMemory.clear), \
            "clear 方法应为 async"

    def test_extract_long_term_is_async(self):
        """extract_long_term method must be async."""
        import inspect
        assert inspect.iscoroutinefunction(BufferMemory.extract_long_term), \
            "extract_long_term 方法应为 async"


# ============================================================================
# 2. Constructor and Initialization
# ============================================================================


class TestInitialization:
    """Test BufferMemory constructor and default values."""

    def test_default_max_tokens(self):
        """Default max_tokens should be 2000."""
        mem = BufferMemory()
        assert mem.max_tokens == 2000

    def test_custom_max_tokens(self):
        """max_tokens can be customized."""
        mem = BufferMemory(max_tokens=500)
        assert mem.max_tokens == 500

    def test_buffers_is_defaultdict(self):
        """buffers must be a defaultdict(list)."""
        mem = BufferMemory()
        assert isinstance(mem.buffers, defaultdict)
        # Accessing a missing key should return empty list
        assert mem.buffers["nonexistent"] == []

    def test_buffers_initially_empty(self):
        """buffers should start empty."""
        mem = BufferMemory()
        assert len(mem.buffers) == 0

    def test_max_tokens_zero(self):
        """BufferMemory accepts max_tokens=0."""
        mem = BufferMemory(max_tokens=0)
        assert mem.max_tokens == 0

    def test_max_tokens_large_value(self):
        """BufferMemory accepts large max_tokens."""
        mem = BufferMemory(max_tokens=1000000)
        assert mem.max_tokens == 1000000


# ============================================================================
# 3. add / save Message Operations
# ============================================================================


class TestAddSaveOperations:
    """Test add and save message operations."""

    @pytest.mark.asyncio
    async def test_add_single_message(self):
        """add appends a message to the session buffer."""
        mem = BufferMemory()
        msg = _make_msg("hello")
        await mem.add("s1", msg)
        assert len(mem.buffers["s1"]) == 1
        assert mem.buffers["s1"][0].content == "hello"

    @pytest.mark.asyncio
    async def test_add_multiple_messages(self):
        """add appends multiple messages in order."""
        mem = BufferMemory()
        for i in range(5):
            await mem.add("s1", _make_msg(f"msg-{i}"))
        assert len(mem.buffers["s1"]) == 5
        assert mem.buffers["s1"][0].content == "msg-0"
        assert mem.buffers["s1"][4].content == "msg-4"

    @pytest.mark.asyncio
    async def test_save_delegates_to_add(self):
        """save should produce the same result as add."""
        mem = BufferMemory()
        msg = _make_msg("via save")
        await mem.save("s1", msg)
        assert len(mem.buffers["s1"]) == 1
        assert mem.buffers["s1"][0].content == "via save"

    @pytest.mark.asyncio
    async def test_add_preserves_message_fields(self):
        """add preserves all Message fields."""
        mem = BufferMemory()
        msg = Message(role="assistant", content="response", sender_id="bot-1")
        await mem.add("s1", msg)
        stored = mem.buffers["s1"][0]
        assert stored.role == "assistant"
        assert stored.content == "response"
        assert stored.sender_id == "bot-1"

    @pytest.mark.asyncio
    async def test_add_different_roles(self):
        """Messages with different roles are stored correctly."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("user msg", role="user"))
        await mem.add("s1", _make_msg("assistant msg", role="assistant"))
        await mem.add("s1", _make_msg("system msg", role="system"))
        assert len(mem.buffers["s1"]) == 3
        assert mem.buffers["s1"][0].role == "user"
        assert mem.buffers["s1"][1].role == "assistant"
        assert mem.buffers["s1"][2].role == "system"


# ============================================================================
# 4. get_recent Retrieval
# ============================================================================


class TestGetRecent:
    """Test get_recent method."""

    @pytest.mark.asyncio
    async def test_get_recent_empty_buffer(self):
        """get_recent returns empty string for empty buffer."""
        mem = BufferMemory()
        result = await mem.get_recent("s1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_get_recent_default_n(self):
        """get_recent defaults to n=10."""
        mem = BufferMemory()
        for i in range(15):
            await mem.add("s1", _make_msg(f"msg-{i}"))
        result = await mem.get_recent("s1")
        lines = result.split("\n")
        assert len(lines) == 10
        assert lines[0] == "msg-5"  # oldest in the 10-message window
        assert lines[9] == "msg-14"

    @pytest.mark.asyncio
    async def test_get_recent_n_less_than_buffer(self):
        """get_recent returns last n messages when n < buffer size."""
        mem = BufferMemory()
        for i in range(10):
            await mem.add("s1", _make_msg(f"msg-{i}"))
        result = await mem.get_recent("s1", n=3)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[0] == "msg-7"
        assert lines[2] == "msg-9"

    @pytest.mark.asyncio
    async def test_get_recent_n_greater_than_buffer(self):
        """get_recent returns all messages when n > buffer size."""
        mem = BufferMemory()
        for i in range(3):
            await mem.add("s1", _make_msg(f"msg-{i}"))
        result = await mem.get_recent("s1", n=100)
        lines = result.split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_get_recent_n_equals_one(self):
        """get_recent with n=1 returns only the latest message."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("first"))
        await mem.add("s1", _make_msg("second"))
        result = await mem.get_recent("s1", n=1)
        assert result == "second"

    @pytest.mark.asyncio
    async def test_get_recent_joins_with_newline(self):
        """get_recent joins messages with newline character."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("a"))
        await mem.add("s1", _make_msg("b"))
        await mem.add("s1", _make_msg("c"))
        result = await mem.get_recent("s1", n=3)
        assert result == "a\nb\nc"

    @pytest.mark.asyncio
    async def test_get_recent_unknown_session(self):
        """get_recent returns empty string for unknown session."""
        mem = BufferMemory()
        result = await mem.get_recent("unknown-session")
        assert result == ""


# ============================================================================
# 5. retrieve (BaseMemory Interface)
# ============================================================================


class TestRetrieve:
    """Test retrieve method (BaseMemory interface)."""

    @pytest.mark.asyncio
    async def testretrieve_returns_string(self):
        """retrieve must return a string."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("hello"))
        result = await mem.retrieve("s1", "query")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def testretrieve_uses_top_k(self):
        """retrieve delegates to get_recent with top_k."""
        mem = BufferMemory()
        for i in range(10):
            await mem.add("s1", _make_msg(f"msg-{i}"))
        result = await mem.retrieve("s1", "any query", top_k=3)
        lines = result.split("\n")
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def testretrieve_default_top_k(self):
        """retrieve defaults to top_k=5."""
        mem = BufferMemory()
        for i in range(10):
            await mem.add("s1", _make_msg(f"msg-{i}"))
        result = await mem.retrieve("s1", "query")
        lines = result.split("\n")
        assert len(lines) == 5

    @pytest.mark.asyncio
    async def testretrieve_ignores_query(self):
        """BufferMemory retrieve ignores query content (no semantic search)."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("cat"))
        await mem.add("s1", _make_msg("dog"))
        # Same result regardless of query
        result1 = await mem.retrieve("s1", "cat")
        result2 = await mem.retrieve("s1", "elephant")
        assert result1 == result2

    @pytest.mark.asyncio
    async def testretrieve_ignores_user_ids(self):
        """BufferMemory retrieve ignores user_ids parameter."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("hello"))
        result = await mem.retrieve("s1", "q", user_ids=["u1", "u2"])
        assert result == "hello"

    @pytest.mark.asyncio
    async def testretrieve_empty_buffer(self):
        """retrieve returns empty string for empty buffer."""
        mem = BufferMemory()
        result = await mem.retrieve("s1", "query")
        assert result == ""


# ============================================================================
# 6. clear Operation
# ============================================================================


class TestClear:
    """Test clear operation."""

    @pytest.mark.asyncio
    async def testclear_removes_messages(self):
        """clear removes all messages for a session."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg1"))
        await mem.add("s1", _make_msg("msg2"))
        await mem.clear("s1")
        assert mem.buffers["s1"] == []

    @pytest.mark.asyncio
    async def testclear_idempotent(self):
        """clear can be called multiple times without error."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg"))
        await mem.clear("s1")
        await mem.clear("s1")  # second clear
        assert mem.buffers["s1"] == []

    @pytest.mark.asyncio
    async def testclear_unknown_session(self):
        """clear on unknown session does not raise."""
        mem = BufferMemory()
        await mem.clear("unknown")  # should not raise

    @pytest.mark.asyncio
    async def testclear_does_not_affect_other_sessions(self):
        """clear only affects the specified session."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg1"))
        await mem.add("s2", _make_msg("msg2"))
        await mem.clear("s1")
        assert len(mem.buffers["s1"]) == 0
        assert len(mem.buffers["s2"]) == 1

    @pytest.mark.asyncio
    async def testclear_then_add_works(self):
        """After clear, new messages can be added."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("old"))
        await mem.clear("s1")
        await mem.add("s1", _make_msg("new"))
        assert len(mem.buffers["s1"]) == 1
        assert mem.buffers["s1"][0].content == "new"


# ============================================================================
# 7. extract_long_term (No-op Verification)
# ============================================================================


class TestExtractLongTerm:
    """Test that extract_long_term is a no-op for BufferMemory."""

    @pytest.mark.asyncio
    async def test_extract_long_term_no_op(self):
        """extract_long_term does not modify buffer."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg"))
        await mem.extract_long_term("s1")
        assert len(mem.buffers["s1"]) == 1

    @pytest.mark.asyncio
    async def test_extract_long_term_force_no_op(self):
        """extract_long_term with force=True is still no-op."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg"))
        await mem.extract_long_term("s1", force=True)
        assert len(mem.buffers["s1"]) == 1

    @pytest.mark.asyncio
    async def test_extract_long_term_unknown_session(self):
        """extract_long_term on unknown session does not raise."""
        mem = BufferMemory()
        await mem.extract_long_term("unknown")

    @pytest.mark.asyncio
    async def test_extract_long_term_returns_none(self):
        """extract_long_term returns None."""
        mem = BufferMemory()
        result = await mem.extract_long_term("s1")
        assert result is None


# ============================================================================
# 8. Token-based Truncation Logic
# ============================================================================


class TestTokenTruncation:
    """Test token-based buffer truncation."""

    @pytest.mark.asyncio
    async def test_truncation_removes_oldest(self):
        """When exceeding max_tokens, oldest messages are removed."""
        mem = BufferMemory(max_tokens=10)
        # Each message ~5 chars = ~2 tokens (len//4 or 1)
        # Add enough to exceed 10 tokens
        for i in range(20):
            await mem.add("s1", _make_msg(f"message-{i}"))
        # Buffer should be truncated
        total = sum(_estimate_tokens(m.content) for m in mem.buffers["s1"])
        assert total <= 10

    @pytest.mark.asyncio
    async def test_truncation_preserves_newest(self):
        """After truncation, newest messages remain."""
        mem = BufferMemory(max_tokens=20)
        await mem.add("s1", _make_msg("old message"))
        await mem.add("s1", _make_msg("new message"))
        # Even if truncated, the newest should survive
        recent = await mem.get_recent("s1", n=1)
        assert recent == "new message"

    @pytest.mark.asyncio
    async def test_no_truncation_under_limit(self):
        """No truncation when under max_tokens."""
        mem = BufferMemory(max_tokens=10000)
        for i in range(5):
            await mem.add("s1", _make_msg(f"msg-{i}"))
        assert len(mem.buffers["s1"]) == 5

    @pytest.mark.asyncio
    async def test_truncation_with_single_large_message(self):
        """Single message exceeding max_tokens is removed by truncation."""
        mem = BufferMemory(max_tokens=5)
        large_msg = _make_msg("a" * 100)  # ~25 tokens, exceeds max_tokens
        await mem.add("s1", large_msg)
        # Implementation removes even the last message if it exceeds max_tokens
        assert len(mem.buffers["s1"]) == 0

    @pytest.mark.asyncio
    async def test_truncation_multiple_sessions_independent(self):
        """Truncation is per-session, not global."""
        mem = BufferMemory(max_tokens=10)
        for i in range(10):
            await mem.add("s1", _make_msg(f"s1-msg-{i}"))
            await mem.add("s2", _make_msg(f"s2-msg-{i}"))
        # Both sessions should be independently truncated
        s1_tokens = sum(_estimate_tokens(m.content) for m in mem.buffers["s1"])
        s2_tokens = sum(_estimate_tokens(m.content) for m in mem.buffers["s2"])
        assert s1_tokens <= 10
        assert s2_tokens <= 10


# ============================================================================
# 9. Multi-session Isolation
# ============================================================================


class TestMultiSessionIsolation:
    """Test that sessions are properly isolated."""

    @pytest.mark.asyncio
    async def test_different_sessions_independent(self):
        """Messages in different sessions do not mix."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("session 1 msg"))
        await mem.add("s2", _make_msg("session 2 msg"))
        r1 = await mem.get_recent("s1", n=1)
        r2 = await mem.get_recent("s2", n=1)
        assert r1 == "session 1 msg"
        assert r2 == "session 2 msg"

    @pytest.mark.asyncio
    async def testclear_one_session_preserves_other(self):
        """Clearing one session does not affect another."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg1"))
        await mem.add("s2", _make_msg("msg2"))
        await mem.clear("s1")
        assert await mem.get_recent("s1") == ""
        assert await mem.get_recent("s2", n=1) == "msg2"

    @pytest.mark.asyncio
    async def test_session_created_on_first_add(self):
        """Session buffer is created on first add via defaultdict."""
        mem = BufferMemory()
        assert "new-session" not in mem.buffers
        await mem.add("new-session", _make_msg("first"))
        assert "new-session" in mem.buffers

    @pytest.mark.asyncio
    async def test_many_sessions(self):
        """BufferMemory handles many sessions."""
        mem = BufferMemory()
        for i in range(100):
            await mem.add(f"session-{i}", _make_msg(f"msg-{i}"))
        assert len(mem.buffers) == 100
        for i in range(100):
            result = await mem.get_recent(f"session-{i}", n=1)
            assert result == f"msg-{i}"


# ============================================================================
# 10. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_content_message(self):
        """BufferMemory handles messages with empty content."""
        mem = BufferMemory()
        msg = _make_msg("")
        await mem.add("s1", msg)
        result = await mem.get_recent("s1", n=1)
        assert result == ""

    @pytest.mark.asyncio
    async def test_very_long_content(self):
        """BufferMemory truncates very long message that exceeds max_tokens."""
        mem = BufferMemory(max_tokens=100)
        long_content = "x" * 10000  # ~2500 tokens, exceeds 100
        await mem.add("s1", _make_msg(long_content))
        # Single large message exceeding max_tokens is removed by truncation
        result = await mem.get_recent("s1", n=1)
        assert result == ""

    @pytest.mark.asyncio
    async def test_very_long_content_within_limit(self):
        """BufferMemory stores long content when within max_tokens."""
        mem = BufferMemory(max_tokens=10000)
        long_content = "x" * 1000  # ~250 tokens, within 10000
        await mem.add("s1", _make_msg(long_content))
        result = await mem.get_recent("s1", n=1)
        assert result == long_content

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """BufferMemory handles Unicode content."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("你好世界"))
        await mem.add("s1", _make_msg("emoji test"))
        result = await mem.get_recent("s1", n=2)
        assert "你好世界" in result
        assert "emoji test" in result

    @pytest.mark.asyncio
    async def test_whitespace_content(self):
        """BufferMemory handles whitespace-only content."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("   "))
        result = await mem.get_recent("s1", n=1)
        assert result == "   "

    @pytest.mark.asyncio
    async def test_newline_in_content(self):
        """BufferMemory handles newlines within content."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("line1\nline2\nline3"))
        result = await mem.get_recent("s1", n=1)
        assert result == "line1\nline2\nline3"

    @pytest.mark.asyncio
    async def test_get_recent_n_zero(self):
        """get_recent with n=0 returns all messages (Python slice behavior)."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg"))
        # Python: list[-0:] is equivalent to list[0:], returns full list
        result = await mem.get_recent("s1", n=0)
        assert result == "msg"

    @pytest.mark.asyncio
    async def test_get_recent_n_negative(self):
        """get_recent with negative n returns empty string (slice behavior)."""
        mem = BufferMemory()
        await mem.add("s1", _make_msg("msg"))
        result = await mem.get_recent("s1", n=-1)
        # Python slice with negative returns empty
        assert result == ""


# ============================================================================
# 11. _estimate_tokens Helper
# ============================================================================


class TestEstimateTokens:
    """Test the _estimate_tokens helper function."""

    def test_empty_string(self):
        """Empty string returns 1 token (min 1 for non-empty text logic)."""
        # len("")//4 = 0, but 0 or 1 = 1
        assert _estimate_tokens("") == 1

    def test_short_text(self):
        """Short text returns 1 token."""
        assert _estimate_tokens("hi") == 1  # 2//4 = 0, 0 or 1 = 1

    def test_four_chars(self):
        """4 characters = 1 token."""
        assert _estimate_tokens("abcd") == 1  # 4//4 = 1

    def test_eight_chars(self):
        """8 characters = 2 tokens."""
        assert _estimate_tokens("abcdefgh") == 2  # 8//4 = 2

    def test_long_text(self):
        """Long text scales linearly."""
        text = "a" * 400
        assert _estimate_tokens(text) == 100  # 400//4 = 100

    def test_returns_int(self):
        """Result is always an integer."""
        assert isinstance(_estimate_tokens("test"), int)
        assert isinstance(_estimate_tokens(""), int)
