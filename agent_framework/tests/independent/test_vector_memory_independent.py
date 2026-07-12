"""Independent tests for VectorMemory implementation.

验证内容（基于详细设计.md 第6.3节）：
- VectorMemory 使用 VectorStore 的正确性
- add/query/add_user/query_user 方法
- 集合命名约定（session_{session_id}, user_{user_id}）
- 边界条件

本测试文件完全独立编写，不使用开发者编写的测试用例。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, call

from agent_framework.memory.vector_memory import VectorMemory
from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult


# ─────────────────────────────────────────────────────────
# 1. 初始化与依赖注入
# ─────────────────────────────────────────────────────────

class TestVectorMemoryInit:
    """验证 VectorMemory 初始化逻辑。"""

    def test_stores_vector_store_reference(self):
        """初始化后应保存 vector_store 引用。"""
        mock_store = MagicMock(spec=VectorStore)
        vm = VectorMemory(mock_store)
        assert vm.store is mock_store

    def test_accepts_vector_store_implementation(self):
        """应接受任何 VectorStore 子类实现。"""
        class CustomStore(VectorStore):
            async def add(self, collection, text, metadata=None, id=None):
                return "id"
            async def query(self, collection, query, top_k=5):
                return []
            async def delete(self, collection, ids=None):
                pass
            async def create_collection(self, collection, dimension):
                pass

        store = CustomStore()
        vm = VectorMemory(store)
        assert vm.store is store


# ─────────────────────────────────────────────────────────
# 2. add 方法
# ─────────────────────────────────────────────────────────

class TestAddMethod:
    """验证 add 方法行为。"""

    @pytest.mark.asyncio
    async def test_add_calls_store_with_session_collection(self):
        """add 应使用 session_{session_id} 作为集合名调用 store.add。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add("sess123", "test memory")

        mock_store.add.assert_awaited_once_with(
            collection="session_sess123",
            text="test memory",
            metadata=None,
        )

    @pytest.mark.asyncio
    async def test_add_passes_metadata(self):
        """add 应将 metadata 传递给 store.add。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        metadata = {"source": "chat", "importance": "high"}
        await vm.add("sess123", "test memory", metadata=metadata)

        mock_store.add.assert_awaited_once_with(
            collection="session_sess123",
            text="test memory",
            metadata=metadata,
        )

    @pytest.mark.asyncio
    async def test_add_with_none_metadata(self):
        """metadata 默认为 None 时应正确传递。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add("sess123", "test memory")

        _, kwargs = mock_store.add.call_args
        assert kwargs["metadata"] is None

    @pytest.mark.asyncio
    async def test_add_returns_none(self):
        """add 方法应返回 None。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        result = await vm.add("sess123", "test memory")
        assert result is None


# ─────────────────────────────────────────────────────────
# 3. query 方法
# ─────────────────────────────────────────────────────────

class TestQueryMethod:
    """验证 query 方法行为。"""

    @pytest.mark.asyncio
    async def test_query_calls_store_with_session_collection(self):
        """query 应使用 session_{session_id} 作为集合名调用 store.query。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query("sess123", "search term")

        mock_store.query.assert_awaited_once_with(
            collection="session_sess123",
            query="search term",
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_query_passes_top_k(self):
        """query 应将 top_k 参数传递给 store.query。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query("sess123", "search term", top_k=10)

        mock_store.query.assert_awaited_once_with(
            collection="session_sess123",
            query="search term",
            top_k=10,
        )

    @pytest.mark.asyncio
    async def test_query_returns_joined_text(self):
        """query 应将结果文本用换行符拼接返回。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = [
            SearchResult(id="1", text="memory one", score=0.9),
            SearchResult(id="2", text="memory two", score=0.8),
            SearchResult(id="3", text="memory three", score=0.7),
        ]
        vm = VectorMemory(mock_store)

        result = await vm.query("sess123", "search term")

        assert result == "memory one\nmemory two\nmemory three"

    @pytest.mark.asyncio
    async def test_query_returns_empty_string_when_no_results(self):
        """无结果时 query 应返回空字符串。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        result = await vm.query("sess123", "search term")

        assert result == ""

    @pytest.mark.asyncio
    async def test_query_single_result(self):
        """单个结果时应返回该结果的文本（无换行符）。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = [
            SearchResult(id="1", text="only memory", score=0.95),
        ]
        vm = VectorMemory(mock_store)

        result = await vm.query("sess123", "search term")

        assert result == "only memory"

    @pytest.mark.asyncio
    async def test_query_default_top_k_is_5(self):
        """query 的 top_k 默认值应为 5。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query("sess123", "search term")

        _, kwargs = mock_store.query.call_args
        assert kwargs["top_k"] == 5


# ─────────────────────────────────────────────────────────
# 4. add_user 方法
# ─────────────────────────────────────────────────────────

class TestAddUserMethod:
    """验证 add_user 方法行为。"""

    @pytest.mark.asyncio
    async def test_add_user_calls_store_with_user_collection(self):
        """add_user 应使用 user_{user_id} 作为集合名调用 store.add。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add_user("user456", "user preference")

        mock_store.add.assert_awaited_once_with(
            collection="user_user456",
            text="user preference",
        )

    @pytest.mark.asyncio
    async def test_add_user_does_not_pass_metadata(self):
        """add_user 不应传递 metadata 参数。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add_user("user456", "user preference")

        _, kwargs = mock_store.add.call_args
        assert "metadata" not in kwargs

    @pytest.mark.asyncio
    async def test_add_user_returns_none(self):
        """add_user 方法应返回 None。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        result = await vm.add_user("user456", "user preference")
        assert result is None


# ─────────────────────────────────────────────────────────
# 5. query_user 方法
# ─────────────────────────────────────────────────────────

class TestQueryUserMethod:
    """验证 query_user 方法行为。"""

    @pytest.mark.asyncio
    async def test_query_user_calls_store_with_user_collection(self):
        """query_user 应使用 user_{user_id} 作为集合名调用 store.query。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query_user("user456", "search term")

        mock_store.query.assert_awaited_once_with(
            collection="user_user456",
            query="search term",
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_query_user_passes_top_k(self):
        """query_user 应将 top_k 参数传递给 store.query。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query_user("user456", "search term", top_k=3)

        mock_store.query.assert_awaited_once_with(
            collection="user_user456",
            query="search term",
            top_k=3,
        )

    @pytest.mark.asyncio
    async def test_query_user_returns_joined_text(self):
        """query_user 应将结果文本用换行符拼接返回。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = [
            SearchResult(id="1", text="user fact one", score=0.9),
            SearchResult(id="2", text="user fact two", score=0.8),
        ]
        vm = VectorMemory(mock_store)

        result = await vm.query_user("user456", "search term")

        assert result == "user fact one\nuser fact two"

    @pytest.mark.asyncio
    async def test_query_user_returns_empty_string_when_no_results(self):
        """无结果时 query_user 应返回空字符串。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        result = await vm.query_user("user456", "search term")

        assert result == ""

    @pytest.mark.asyncio
    async def test_query_user_default_top_k_is_5(self):
        """query_user 的 top_k 默认值应为 5。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query_user("user456", "search term")

        _, kwargs = mock_store.query.call_args
        assert kwargs["top_k"] == 5


# ─────────────────────────────────────────────────────────
# 6. 集合命名约定
# ─────────────────────────────────────────────────────────

class TestCollectionNaming:
    """验证集合命名约定。"""

    @pytest.mark.asyncio
    async def test_session_collection_prefix(self):
        """会话集合应使用 session_ 前缀。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add("abc123", "text")

        call_args = mock_store.add.call_args
        assert call_args.kwargs["collection"].startswith("session_")

    @pytest.mark.asyncio
    async def test_user_collection_prefix(self):
        """用户集合应使用 user_ 前缀。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add_user("user789", "text")

        call_args = mock_store.add.call_args
        assert call_args.kwargs["collection"].startswith("user_")

    @pytest.mark.asyncio
    async def test_session_id_preserved_in_collection_name(self):
        """session_id 应完整保留在集合名中。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        session_id = "my-special-session_123"
        await vm.add(session_id, "text")

        call_args = mock_store.add.call_args
        assert call_args.kwargs["collection"] == f"session_{session_id}"

    @pytest.mark.asyncio
    async def test_user_id_preserved_in_collection_name(self):
        """user_id 应完整保留在集合名中。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        user_id = "user-456_abc"
        await vm.add_user(user_id, "text")

        call_args = mock_store.add.call_args
        assert call_args.kwargs["collection"] == f"user_{user_id}"

    @pytest.mark.asyncio
    async def test_session_and_user_collections_are_separate(self):
        """相同 ID 的 session 和 user 集合应不同。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add("123", "session text")
        await vm.add_user("123", "user text")

        calls = mock_store.add.call_args_list
        session_collection = calls[0].kwargs["collection"]
        user_collection = calls[1].kwargs["collection"]
        assert session_collection != user_collection
        assert session_collection == "session_123"
        assert user_collection == "user_123"


# ─────────────────────────────────────────────────────────
# 7. 边界条件
# ─────────────────────────────────────────────────────────

class TestBoundaryConditions:
    """验证边界条件处理。"""

    @pytest.mark.asyncio
    async def test_add_empty_text(self):
        """应能添加空文本。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add("sess123", "")

        mock_store.add.assert_awaited_once_with(
            collection="session_sess123",
            text="",
            metadata=None,
        )

    @pytest.mark.asyncio
    async def test_query_empty_string(self):
        """应能用空字符串查询。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        result = await vm.query("sess123", "")

        assert result == ""
        mock_store.query.assert_awaited_once_with(
            collection="session_sess123",
            query="",
            top_k=5,
        )

    @pytest.mark.asyncio
    async def test_query_top_k_zero(self):
        """top_k=0 应正常传递。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query("sess123", "search", top_k=0)

        _, kwargs = mock_store.query.call_args
        assert kwargs["top_k"] == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_session_id(self):
        """session_id 中的特殊字符应被保留。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        session_id = "sess-123_abc.def"
        await vm.add(session_id, "text")

        call_args = mock_store.add.call_args
        assert call_args.kwargs["collection"] == f"session_{session_id}"

    @pytest.mark.asyncio
    async def test_special_characters_in_user_id(self):
        """user_id 中的特殊字符应被保留。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        user_id = "user-456_abc.def"
        await vm.add_user(user_id, "text")

        call_args = mock_store.add.call_args
        assert call_args.kwargs["collection"] == f"user_{user_id}"

    @pytest.mark.asyncio
    async def test_long_text(self):
        """应能处理长文本。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        long_text = "a" * 10000
        await vm.add("sess123", long_text)

        call_args = mock_store.add.call_args
        assert call_args.kwargs["text"] == long_text

    @pytest.mark.asyncio
    async def test_unicode_text(self):
        """应能处理 Unicode 文本。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        unicode_text = "你好世界 こんにちは 🌍"
        await vm.add("sess123", unicode_text)

        call_args = mock_store.add.call_args
        assert call_args.kwargs["text"] == unicode_text

    @pytest.mark.asyncio
    async def test_query_result_with_empty_text(self):
        """查询结果中包含空文本时应正确处理。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = [
            SearchResult(id="1", text="", score=0.9),
            SearchResult(id="2", text="valid text", score=0.8),
        ]
        vm = VectorMemory(mock_store)

        result = await vm.query("sess123", "search")

        assert result == "\nvalid text"

    @pytest.mark.asyncio
    async def test_multiple_adds_to_same_session(self):
        """同一会话应能多次添加记忆。"""
        mock_store = AsyncMock(spec=VectorStore)
        vm = VectorMemory(mock_store)

        await vm.add("sess123", "first memory")
        await vm.add("sess123", "second memory")
        await vm.add("sess123", "third memory")

        assert mock_store.add.await_count == 3
        for call_args in mock_store.add.call_args_list:
            assert call_args.kwargs["collection"] == "session_sess123"

    @pytest.mark.asyncio
    async def test_multiple_queries_to_same_session(self):
        """同一会话应能多次查询。"""
        mock_store = AsyncMock(spec=VectorStore)
        mock_store.query.return_value = []
        vm = VectorMemory(mock_store)

        await vm.query("sess123", "first query")
        await vm.query("sess123", "second query")

        assert mock_store.query.await_count == 2
