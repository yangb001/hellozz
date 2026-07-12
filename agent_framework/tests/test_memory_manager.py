"""Tests for MemoryManager unified memory entry point."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_framework.interfaces.session import Message
from agent_framework.memory.memory_manager import MemoryManager, MemoryConfig
from agent_framework.memory.extractor import MemoryFact


@pytest.fixture
def mock_buffer():
    """Mock BufferMemory."""
    buf = AsyncMock()
    buf.get_recent = AsyncMock(return_value="recent messages")
    buf.add = AsyncMock()
    buf.clear = AsyncMock()
    return buf


@pytest.fixture
def mock_vector():
    """Mock VectorMemory."""
    vec = AsyncMock()
    vec.query = AsyncMock(return_value="long term context")
    vec.query_user = AsyncMock(return_value="user context")
    vec.add = AsyncMock()
    vec.add_user = AsyncMock()
    return vec


@pytest.fixture
def mock_extractor():
    """Mock MemoryExtractor."""
    ext = AsyncMock()
    ext.extract = AsyncMock(return_value=[])
    ext.is_important = AsyncMock(return_value=False)
    return ext


@pytest.fixture
def manager(mock_buffer, mock_vector, mock_extractor):
    """Create MemoryManager with mocked dependencies."""
    config = MemoryConfig(trigger="smart")
    return MemoryManager(
        short_term=mock_buffer,
        long_term=mock_vector,
        extractor=mock_extractor,
        config=config,
    )


@pytest.fixture
def sample_message():
    return Message(role="user", content="I love pizza")


class TestMemoryConfig:
    """Test MemoryConfig dataclass."""

    def test_default_config(self):
        config = MemoryConfig()
        assert config.trigger == "smart"
        assert config.every_n == 5

    def test_custom_config(self):
        config = MemoryConfig(trigger="every_n_turns", every_n=10)
        assert config.trigger == "every_n_turns"
        assert config.every_n == 10

    def test_smart_trigger(self):
        config = MemoryConfig(trigger="smart")
        assert config.trigger == "smart"


class TestMemoryManagerInit:
    """Test MemoryManager initialization."""

    def test_stores_dependencies(self, mock_buffer, mock_vector, mock_extractor):
        config = MemoryConfig(trigger="smart")
        mgr = MemoryManager(mock_buffer, mock_vector, mock_extractor, config)
        assert mgr.short_term is mock_buffer
        assert mgr.long_term is mock_vector
        assert mgr.extractor is mock_extractor
        assert mgr.config.trigger == "smart"

    def test_turn_counter_initialized(self, mock_buffer, mock_vector, mock_extractor):
        config = MemoryConfig()
        mgr = MemoryManager(mock_buffer, mock_vector, mock_extractor, config)
        assert mgr._turn_counter == {}


class TestMemoryManagerSave:
    """Test MemoryManager.save method."""

    @pytest.mark.asyncio
    async def test_save_adds_to_short_term(self, manager, mock_buffer, sample_message):
        await manager.save("s1", sample_message)
        mock_buffer.add.assert_awaited_once_with("s1", sample_message)

    @pytest.mark.asyncio
    async def test_save_smart_trigger_checks_importance(
        self, manager, mock_extractor, sample_message
    ):
        mock_extractor.is_important.return_value = False
        await manager.save("s1", sample_message)
        mock_extractor.is_important.assert_awaited_once_with(sample_message)

    @pytest.mark.asyncio
    async def test_save_smart_trigger_extracts_when_important(
        self, manager, mock_extractor, mock_buffer, mock_vector
    ):
        mock_extractor.is_important.return_value = True
        fact = MemoryFact(content="User loves pizza", metadata={}, user_id="u1")
        mock_extractor.extract.return_value = [fact]
        mock_buffer.get_recent.return_value = "recent msgs"

        await manager.save("s1", Message(role="user", content="I love pizza"))

        mock_extractor.extract.assert_awaited_once()
        mock_vector.add.assert_awaited_once_with("s1", "User loves pizza", {})

    @pytest.mark.asyncio
    async def test_save_smart_trigger_extracts_user_fact(
        self, manager, mock_extractor, mock_buffer, mock_vector
    ):
        mock_extractor.is_important.return_value = True
        fact = MemoryFact(content="User is a doctor", metadata={"type": "profession"}, user_id="u1")
        mock_extractor.extract.return_value = [fact]
        mock_buffer.get_recent.return_value = "recent msgs"

        await manager.save("s1", Message(role="user", content="I am a doctor"))

        mock_vector.add.assert_awaited_once_with("s1", "User is a doctor", {"type": "profession"})
        mock_vector.add_user.assert_awaited_once_with("u1", "User is a doctor")

    @pytest.mark.asyncio
    async def test_save_every_n_turns_trigger(self, mock_buffer, mock_vector, mock_extractor):
        config = MemoryConfig(trigger="every_n_turns", every_n=3)
        mgr = MemoryManager(mock_buffer, mock_vector, mock_extractor, config)

        fact = MemoryFact(content="fact", metadata={}, user_id=None)
        mock_extractor.extract.return_value = [fact]
        mock_buffer.get_recent.return_value = "msgs"

        msg = Message(role="user", content="test")
        # First two saves should not trigger extraction
        await mgr.save("s1", msg)
        await mgr.save("s1", msg)
        mock_extractor.extract.assert_not_awaited()

        # Third save triggers extraction
        await mgr.save("s1", msg)
        mock_extractor.extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_every_n_turns_resets_counter(
        self, mock_buffer, mock_vector, mock_extractor
    ):
        config = MemoryConfig(trigger="every_n_turns", every_n=2)
        mgr = MemoryManager(mock_buffer, mock_vector, mock_extractor, config)
        mock_extractor.extract.return_value = []
        mock_buffer.get_recent.return_value = ""

        msg = Message(role="user", content="test")
        await mgr.save("s1", msg)
        await mgr.save("s1", msg)  # triggers (n=2)
        mock_extractor.extract.assert_awaited_once()

        # Counter resets, needs 2 more
        await mgr.save("s1", msg)
        assert mock_extractor.extract.await_count == 1  # still only 1
        await mgr.save("s1", msg)
        assert mock_extractor.extract.await_count == 2  # now 2

    @pytest.mark.asyncio
    async def test_save_tracks_separate_session_counters(
        self, mock_buffer, mock_vector, mock_extractor
    ):
        config = MemoryConfig(trigger="every_n_turns", every_n=2)
        mgr = MemoryManager(mock_buffer, mock_vector, mock_extractor, config)
        mock_extractor.extract.return_value = []
        mock_buffer.get_recent.return_value = ""

        msg = Message(role="user", content="test")
        await mgr.save("s1", msg)
        await mgr.save("s2", msg)
        # Neither session has hit n=2 yet
        mock_extractor.extract.assert_not_awaited()


class TestMemoryManagerRetrieve:
    """Test MemoryManager.retrieve method."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_combined_context(self, manager):
        result = await manager.retrieve("s1", query="test")
        assert "recent messages" in result
        assert "long term context" in result

    @pytest.mark.asyncio
    async def test_retrieve_queries_short_term(self, manager, mock_buffer):
        await manager.retrieve("s1", query="test", top_k=5)
        mock_buffer.get_recent.assert_awaited_once_with("s1", n=10)

    @pytest.mark.asyncio
    async def test_retrieve_queries_long_term(self, manager, mock_vector):
        await manager.retrieve("s1", query="what is pizza", top_k=3)
        mock_vector.query.assert_awaited_once_with("s1", "what is pizza", top_k=3)

    @pytest.mark.asyncio
    async def test_retrieve_with_user_ids(self, manager, mock_vector):
        result = await manager.retrieve(
            "s1", query="test", user_ids=["u1", "u2"]
        )
        assert mock_vector.query_user.await_count == 2
        mock_vector.query_user.assert_any_await("u1", "test", top_k=2)
        mock_vector.query_user.assert_any_await("u2", "test", top_k=2)
        assert "user context" in result

    @pytest.mark.asyncio
    async def test_retrieve_without_user_ids(self, manager, mock_vector):
        result = await manager.retrieve("s1", query="test")
        mock_vector.query_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retrieve_format(self, manager):
        result = await manager.retrieve("s1", query="test")
        # Should contain all three parts
        assert "long term context" in result
        assert "recent messages" in result


class TestMemoryManagerClear:
    """Test MemoryManager.clear method."""

    @pytest.mark.asyncio
    async def test_clear_clears_short_term(self, manager, mock_buffer):
        await manager.clear("s1")
        mock_buffer.clear.assert_awaited_once_with("s1")

    @pytest.mark.asyncio
    async def test_clear_resets_turn_counter(self, manager):
        manager._turn_counter["s1"] = 5
        await manager.clear("s1")
        assert manager._turn_counter.get("s1", 0) == 0


class TestMemoryManagerExtractLongTerm:
    """Test MemoryManager.extract_long_term method."""

    @pytest.mark.asyncio
    async def test_extract_gets_recent_messages(self, manager, mock_buffer):
        mock_extractor = manager.extractor
        mock_extractor.extract.return_value = []
        mock_buffer.get_recent.return_value = "recent"

        await manager.extract_long_term("s1")
        mock_buffer.get_recent.assert_awaited_once_with("s1", n=20)

    @pytest.mark.asyncio
    async def test_extract_calls_extractor(self, manager, mock_buffer, mock_extractor):
        mock_buffer.get_recent.return_value = "recent msgs"
        mock_extractor.extract.return_value = []
        await manager.extract_long_term("s1")
        mock_extractor.extract.assert_awaited_once_with("recent msgs")

    @pytest.mark.asyncio
    async def test_extract_stores_facts_in_vector(
        self, manager, mock_buffer, mock_extractor, mock_vector
    ):
        mock_buffer.get_recent.return_value = "recent"
        facts = [
            MemoryFact(content="fact1", metadata={"k": "v"}, user_id=None),
            MemoryFact(content="fact2", metadata={}, user_id="u1"),
        ]
        mock_extractor.extract.return_value = facts

        await manager.extract_long_term("s1")

        mock_vector.add.assert_any_await("s1", "fact1", {"k": "v"})
        mock_vector.add.assert_any_await("s1", "fact2", {})
        mock_vector.add_user.assert_awaited_once_with("u1", "fact2")

    @pytest.mark.asyncio
    async def test_extract_no_user_id_skips_user_memory(
        self, manager, mock_buffer, mock_extractor, mock_vector
    ):
        mock_buffer.get_recent.return_value = "recent"
        facts = [MemoryFact(content="fact", metadata={}, user_id=None)]
        mock_extractor.extract.return_value = facts

        await manager.extract_long_term("s1")
        mock_vector.add.assert_awaited_once()
        mock_vector.add_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_extract_with_force(self, manager, mock_buffer, mock_extractor):
        mock_buffer.get_recent.return_value = "recent"
        mock_extractor.extract.return_value = []
        await manager.extract_long_term("s1", force=True)
        # Force flag is passed through; extraction still runs
        mock_extractor.extract.assert_awaited_once()


class TestMemoryManagerBaseMemoryInterface:
    """Test that MemoryManager correctly implements BaseMemory interface."""

    def test_is_subclass_of_base_memory(self):
        from agent_framework.interfaces.base_memory import BaseMemory
        assert issubclass(MemoryManager, BaseMemory)

    @pytest.mark.asyncio
    async def test_has_all_required_methods(self):
        assert hasattr(MemoryManager, "save")
        assert hasattr(MemoryManager, "retrieve")
        assert hasattr(MemoryManager, "clear")
        assert hasattr(MemoryManager, "extract_long_term")
