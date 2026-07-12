"""Independent tests for MemoryManager implementation.

验证内容（基于详细设计.md 第6.1节）：
- BaseMemory 接口实现
- 组合 BufferMemory、VectorMemory、MemoryExtractor
- save/retrieve/clear/extract_long_term 方法
- 记忆拼接逻辑
- 触发策略配置（smart, every_n_turns）

本测试文件完全独立编写，不使用开发者编写的测试用例。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from agent_framework.memory.memory_manager import MemoryManager, MemoryConfig
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.session import Message
from agent_framework.memory.extractor import MemoryFact


# ─────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────

def _make_message(content: str, role: str = "user", sender_id: str = None) -> Message:
    """创建测试用 Message。"""
    return Message(role=role, content=content, sender_id=sender_id)


def _make_memory_manager(
    trigger: str = "smart",
    every_n: int = 5,
) -> tuple[MemoryManager, AsyncMock, AsyncMock, AsyncMock]:
    """创建带 mock 依赖的 MemoryManager。"""
    mock_short_term = AsyncMock()
    mock_long_term = AsyncMock()
    mock_extractor = AsyncMock()
    config = MemoryConfig(trigger=trigger, every_n=every_n)
    mm = MemoryManager(mock_short_term, mock_long_term, mock_extractor, config)
    return mm, mock_short_term, mock_long_term, mock_extractor


# ─────────────────────────────────────────────────────────
# 1. MemoryConfig 配置
# ─────────────────────────────────────────────────────────

class TestMemoryConfig:
    """验证 MemoryConfig 数据类。"""

    def test_default_trigger_is_smart(self):
        """默认触发策略应为 smart。"""
        config = MemoryConfig()
        assert config.trigger == "smart"

    def test_default_every_n_is_5(self):
        """默认 every_n 应为 5。"""
        config = MemoryConfig()
        assert config.every_n == 5

    def test_custom_config(self):
        """应支持自定义配置。"""
        config = MemoryConfig(trigger="every_n_turns", every_n=10)
        assert config.trigger == "every_n_turns"
        assert config.every_n == 10

    def test_is_dataclass(self):
        """MemoryConfig 应是 dataclass。"""
        import dataclasses
        assert dataclasses.is_dataclass(MemoryConfig)


# ─────────────────────────────────────────────────────────
# 2. 初始化与继承
# ─────────────────────────────────────────────────────────

class TestMemoryManagerInit:
    """验证 MemoryManager 初始化和继承关系。"""

    def test_is_subclass_of_base_memory(self):
        """MemoryManager 必须是 BaseMemory 的子类。"""
        assert issubclass(MemoryManager, BaseMemory)

    def test_stores_components(self):
        """初始化后应保存所有组件引用。"""
        mm, short, long, extractor = _make_memory_manager()
        assert mm.short_term is short
        assert mm.long_term is long
        assert mm.extractor is extractor

    def test_stores_config(self):
        """初始化后应保存配置。"""
        mm, _, _, _ = _make_memory_manager(trigger="every_n_turns", every_n=10)
        assert mm.config.trigger == "every_n_turns"
        assert mm.config.every_n == 10

    def test_initializes_turn_counter(self):
        """初始化后 turn_counter 应为空字典。"""
        mm, _, _, _ = _make_memory_manager()
        assert mm._turn_counter == {}

    def test_implements_save(self):
        """必须实现 save 方法。"""
        assert hasattr(MemoryManager, "save")

    def test_implements_retrieve(self):
        """必须实现 retrieve 方法。"""
        assert hasattr(MemoryManager, "retrieve")

    def test_implements_clear(self):
        """必须实现 clear 方法。"""
        assert hasattr(MemoryManager, "clear")

    def test_implements_extract_long_term(self):
        """必须实现 extract_long_term 方法。"""
        assert hasattr(MemoryManager, "extract_long_term")


# ─────────────────────────────────────────────────────────
# 3. save 方法
# ─────────────────────────────────────────────────────────

class TestSaveMethod:
    """验证 save 方法行为。"""

    @pytest.mark.asyncio
    async def test_save_adds_to_short_term(self):
        """save 应将消息添加到短期记忆。"""
        mm, short, _, _ = _make_memory_manager()
        msg = _make_message("hello")

        await mm.save("sess1", msg)

        short.add.assert_awaited_once_with("sess1", msg)

    @pytest.mark.asyncio
    async def test_save_smart_trigger_important(self):
        """smart 触发策略下，重要消息应触发提取。"""
        mm, short, long, extractor = _make_memory_manager(trigger="smart")
        extractor.is_important = AsyncMock(return_value=True)
        extractor.extract = AsyncMock(return_value=[])
        short.get_recent_messages = MagicMock(return_value=[])
        msg = _make_message("important info")

        await mm.save("sess1", msg)

        extractor.is_important.assert_awaited_once_with(msg)
        extractor.extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_smart_trigger_not_important(self):
        """smart 触发策略下，不重要消息不应触发提取。"""
        mm, short, long, extractor = _make_memory_manager(trigger="smart")
        extractor.is_important = AsyncMock(return_value=False)
        msg = _make_message("trivial")

        await mm.save("sess1", msg)

        extractor.is_important.assert_awaited_once_with(msg)
        extractor.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_every_n_turns_triggers_at_threshold(self):
        """every_n_turns 策略下，达到阈值时应触发提取。"""
        mm, short, long, extractor = _make_memory_manager(trigger="every_n_turns", every_n=3)
        extractor.extract = AsyncMock(return_value=[])
        short.get_recent_messages = MagicMock(return_value=[])

        for i in range(3):
            await mm.save("sess1", _make_message(f"msg {i}"))

        # 第3条消息时触发
        extractor.extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_every_n_turns_resets_counter(self):
        """every_n_turns 策略下，触发后计数器应重置。"""
        mm, short, long, extractor = _make_memory_manager(trigger="every_n_turns", every_n=2)
        extractor.extract = AsyncMock(return_value=[])
        short.get_recent_messages = MagicMock(return_value=[])

        await mm.save("sess1", _make_message("msg1"))
        assert mm._turn_counter["sess1"] == 1

        await mm.save("sess1", _make_message("msg2"))
        assert mm._turn_counter["sess1"] == 0  # 重置

    @pytest.mark.asyncio
    async def test_save_every_n_turns_no_trigger_before_threshold(self):
        """every_n_turns 策略下，未达阈值时不触发提取。"""
        mm, short, long, extractor = _make_memory_manager(trigger="every_n_turns", every_n=5)
        extractor.extract = AsyncMock(return_value=[])
        short.get_recent_messages = MagicMock(return_value=[])

        for i in range(4):
            await mm.save("sess1", _make_message(f"msg {i}"))

        extractor.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_save_tracks_turn_counter_per_session(self):
        """every_n_turns 策略下，计数器应按会话独立追踪。"""
        mm, short, long, extractor = _make_memory_manager(trigger="every_n_turns", every_n=2)
        extractor.extract = AsyncMock(return_value=[])
        short.get_recent_messages = MagicMock(return_value=[])

        await mm.save("sess1", _make_message("msg1"))
        await mm.save("sess2", _make_message("msg2"))

        assert mm._turn_counter["sess1"] == 1
        assert mm._turn_counter["sess2"] == 1


# ─────────────────────────────────────────────────────────
# 4. retrieve 方法
# ─────────────────────────────────────────────────────────

class TestRetrieveMethod:
    """验证 retrieve 方法行为。"""

    @pytest.mark.asyncio
    async def test_retrieve_gets_short_term_recent(self):
        """retrieve 应从短期记忆获取最近消息。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="recent messages")
        long.query = AsyncMock(return_value="long term")

        await mm.retrieve("sess1", "query")

        short.get_recent.assert_awaited_once_with("sess1", n=10)

    @pytest.mark.asyncio
    async def test_retrieve_queries_long_term(self):
        """retrieve 应从长期记忆查询。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="")
        long.query = AsyncMock(return_value="")

        await mm.retrieve("sess1", "search query", top_k=3)

        long.query.assert_awaited_once_with("sess1", "search query", top_k=3)

    @pytest.mark.asyncio
    async def test_retrieve_default_top_k_is_5(self):
        """retrieve 的 top_k 默认值应为 5。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="")
        long.query = AsyncMock(return_value="")

        await mm.retrieve("sess1", "query")

        long.query.assert_awaited_once_with("sess1", "query", top_k=5)

    @pytest.mark.asyncio
    async def test_retrieve_concatenates_memories(self):
        """retrieve 应拼接短期、长期和用户记忆。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="short term text")
        long.query = AsyncMock(return_value="long term text")

        result = await mm.retrieve("sess1", "query")

        assert "short term text" in result
        assert "long term text" in result
        assert "Recent:" in result

    @pytest.mark.asyncio
    async def test_retrieve_with_user_ids(self):
        """传入 user_ids 时应查询用户记忆。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="")
        long.query = AsyncMock(return_value="")
        long.query_user = AsyncMock(return_value="user memory")

        result = await mm.retrieve("sess1", "query", user_ids=["user1", "user2"])

        assert long.query_user.await_count == 2
        long.query_user.assert_any_await("user1", "query", top_k=2)
        long.query_user.assert_any_await("user2", "query", top_k=2)

    @pytest.mark.asyncio
    async def test_retrieve_without_user_ids(self):
        """不传 user_ids 时不应查询用户记忆。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="")
        long.query = AsyncMock(return_value="")

        await mm.retrieve("sess1", "query")

        long.query_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retrieve_empty_user_ids(self):
        """空 user_ids 列表不应查询用户记忆。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="")
        long.query = AsyncMock(return_value="")

        await mm.retrieve("sess1", "query", user_ids=[])

        long.query_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_retrieve_format(self):
        """retrieve 返回格式应为 '{user_ctx}\\n{long_ctx}\\nRecent: {short_ctx}'。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="S")
        long.query = AsyncMock(return_value="L")

        result = await mm.retrieve("sess1", "query")

        assert result == "\nL\nRecent: S"

    @pytest.mark.asyncio
    async def test_retrieve_with_user_format(self):
        """带用户记忆时格式应包含用户记忆。"""
        mm, short, long, _ = _make_memory_manager()
        short.get_recent = AsyncMock(return_value="S")
        long.query = AsyncMock(return_value="L")
        long.query_user = AsyncMock(return_value="U")

        result = await mm.retrieve("sess1", "query", user_ids=["user1"])

        assert result == "U\nL\nRecent: S"


# ─────────────────────────────────────────────────────────
# 5. clear 方法
# ─────────────────────────────────────────────────────────

class TestClearMethod:
    """验证 clear 方法行为。"""

    @pytest.mark.asyncio
    async def test_clear_calls_short_term_clear(self):
        """clear 应调用短期记忆的 clear 方法。"""
        mm, short, _, _ = _make_memory_manager()

        await mm.clear("sess1")

        short.clear.assert_awaited_once_with("sess1")

    @pytest.mark.asyncio
    async def test_clear_resets_turn_counter(self):
        """clear 应重置会话的 turn_counter。"""
        mm, short, _, _ = _make_memory_manager()
        mm._turn_counter["sess1"] = 3

        await mm.clear("sess1")

        assert mm._turn_counter["sess1"] == 0

    @pytest.mark.asyncio
    async def test_clear_initializes_counter_for_new_session(self):
        """clear 应为新会话初始化 turn_counter。"""
        mm, short, _, _ = _make_memory_manager()

        await mm.clear("new_session")

        assert mm._turn_counter["new_session"] == 0


# ─────────────────────────────────────────────────────────
# 6. extract_long_term 方法
# ─────────────────────────────────────────────────────────

class TestExtractLongTerm:
    """验证 extract_long_term 方法行为。"""

    @pytest.mark.asyncio
    async def test_extract_gets_recent_messages(self):
        """extract_long_term 应获取最近 20 条消息。"""
        mm, short, long, extractor = _make_memory_manager()
        short.get_recent_messages = MagicMock(return_value=[])
        extractor.extract = AsyncMock(return_value=[])

        await mm.extract_long_term("sess1")

        short.get_recent_messages.assert_called_once_with("sess1", n=20)

    @pytest.mark.asyncio
    async def test_extract_calls_extractor(self):
        """extract_long_term 应调用 extractor.extract。"""
        mm, short, long, extractor = _make_memory_manager()
        msgs = [_make_message("recent message 1"), _make_message("recent message 2")]
        short.get_recent_messages = MagicMock(return_value=msgs)
        extractor.extract = AsyncMock(return_value=[])

        await mm.extract_long_term("sess1")

        extractor.extract.assert_awaited_once_with(msgs)

    @pytest.mark.asyncio
    async def test_extract_adds_facts_to_long_term(self):
        """extract_long_term 应将提取的事实添加到长期记忆。"""
        mm, short, long, extractor = _make_memory_manager()
        short.get_recent_messages = MagicMock(return_value=[])
        facts = [
            MemoryFact(content="fact1", metadata={"type": "preference"}),
            MemoryFact(content="fact2", metadata={"type": "info"}),
        ]
        extractor.extract = AsyncMock(return_value=facts)

        await mm.extract_long_term("sess1")

        assert long.add.await_count == 2
        long.add.assert_any_await("sess1", "fact1", {"type": "preference"})
        long.add.assert_any_await("sess1", "fact2", {"type": "info"})

    @pytest.mark.asyncio
    async def test_extract_adds_user_facts(self):
        """有 user_id 的事实应同时写入用户记忆。"""
        mm, short, long, extractor = _make_memory_manager()
        short.get_recent_messages = MagicMock(return_value=[])
        facts = [
            MemoryFact(content="user fact", metadata={}, user_id="user1"),
        ]
        extractor.extract = AsyncMock(return_value=facts)

        await mm.extract_long_term("sess1")

        long.add.assert_awaited_once_with("sess1", "user fact", {})
        long.add_user.assert_awaited_once_with("user1", "user fact")

    @pytest.mark.asyncio
    async def test_extract_no_user_facts_skips_add_user(self):
        """没有 user_id 的事实不应调用 add_user。"""
        mm, short, long, extractor = _make_memory_manager()
        short.get_recent_messages = MagicMock(return_value=[])
        facts = [
            MemoryFact(content="general fact", metadata={}),
        ]
        extractor.extract = AsyncMock(return_value=facts)

        await mm.extract_long_term("sess1")

        long.add.assert_awaited_once()
        long.add_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_extract_empty_facts(self):
        """无提取事实时不应调用 long_term.add。"""
        mm, short, long, extractor = _make_memory_manager()
        short.get_recent_messages = MagicMock(return_value=[])
        extractor.extract = AsyncMock(return_value=[])

        await mm.extract_long_term("sess1")

        long.add.assert_not_awaited()
        long.add_user.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_extract_multiple_facts_mixed(self):
        """混合有无 user_id 的事实应正确处理。"""
        mm, short, long, extractor = _make_memory_manager()
        short.get_recent_messages = MagicMock(return_value=[])
        facts = [
            MemoryFact(content="general", metadata={}),
            MemoryFact(content="user specific", metadata={}, user_id="user1"),
            MemoryFact(content="another general", metadata={}),
        ]
        extractor.extract = AsyncMock(return_value=facts)

        await mm.extract_long_term("sess1")

        assert long.add.await_count == 3
        long.add_user.assert_awaited_once_with("user1", "user specific")


# ─────────────────────────────────────────────────────────
# 7. 触发策略配置
# ─────────────────────────────────────────────────────────

class TestTriggerStrategy:
    """验证触发策略配置。"""

    @pytest.mark.asyncio
    async def test_smart_trigger_calls_is_important(self):
        """smart 策略应调用 extractor.is_important。"""
        mm, _, _, extractor = _make_memory_manager(trigger="smart")
        extractor.is_important = AsyncMock(return_value=False)

        await mm.save("sess1", _make_message("test"))

        extractor.is_important.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_every_n_turns_does_not_call_is_important(self):
        """every_n_turns 策略不应调用 extractor.is_important。"""
        mm, _, _, extractor = _make_memory_manager(trigger="every_n_turns")
        extractor.is_important = AsyncMock(return_value=False)

        await mm.save("sess1", _make_message("test"))

        extractor.is_important.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_unknown_trigger_does_not_extract(self):
        """未知触发策略不应触发提取。"""
        mm, _, _, extractor = _make_memory_manager(trigger="unknown")
        extractor.is_important = AsyncMock(return_value=False)
        extractor.extract = AsyncMock(return_value=[])

        await mm.save("sess1", _make_message("test"))

        extractor.is_important.assert_not_awaited()
        extractor.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_smart_trigger_extract_on_important(self):
        """smart 策略下重要消息应触发完整提取流程。"""
        mm, short, long, extractor = _make_memory_manager(trigger="smart")
        extractor.is_important = AsyncMock(return_value=True)
        msgs = [_make_message("recent message")]
        short.get_recent_messages = MagicMock(return_value=msgs)
        facts = [MemoryFact(content="extracted", metadata={}, user_id="user1")]
        extractor.extract = AsyncMock(return_value=facts)

        await mm.save("sess1", _make_message("important"))

        extractor.extract.assert_awaited_once_with(msgs)
        long.add.assert_awaited_once_with("sess1", "extracted", {})
        long.add_user.assert_awaited_once_with("user1", "extracted")
