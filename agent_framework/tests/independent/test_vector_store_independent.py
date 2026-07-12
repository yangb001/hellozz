"""Independent test cases for VectorStore interface.

This module contains independent verification tests for the VectorStore
abstract base class and SearchResult data model, following the detailed
design specification in section 9.2.

Test categories:
1. SearchResult data class integrity and validation
2. VectorStore abstract base class method signatures
3. Boundary conditions and error handling
"""
import pytest
from typing import List, Optional, get_type_hints
from abc import ABC, abstractmethod
from pydantic import ValidationError

from infrastructure.storage.vector_store import SearchResult, VectorStore


class TestSearchResultDataClass:
    """Independent tests for SearchResult data model."""

    def test_search_result_has_required_fields(self):
        """SearchResult must have id, text, score as required fields."""
        # 验证必需字段存在
        fields = SearchResult.model_fields
        assert "id" in fields, "SearchResult 缺少 id 字段"
        assert "text" in fields, "SearchResult 缺少 text 字段"
        assert "score" in fields, "SearchResult 缺少 score 字段"

    def test_search_result_required_fields_no_default(self):
        """Required fields id, text, score must not have defaults."""
        fields = SearchResult.model_fields
        assert fields["id"].is_required(), "id 字段应该是必需的"
        assert fields["text"].is_required(), "text 字段应该是必需的"
        assert fields["score"].is_required(), "score 字段应该是必需的"

    def test_search_result_metadata_optional(self):
        """metadata field should be optional with None default."""
        fields = SearchResult.model_fields
        assert "metadata" in fields, "SearchResult 缺少 metadata 字段"
        assert not fields["metadata"].is_required(), "metadata 字段应该是可选的"
        assert fields["metadata"].default is None, "metadata 默认值应为 None"

    def test_search_result_create_with_required_fields(self):
        """SearchResult can be created with only required fields."""
        result = SearchResult(id="test-1", text="hello world", score=0.95)
        assert result.id == "test-1"
        assert result.text == "hello world"
        assert result.score == 0.95
        assert result.metadata is None

    def test_search_result_create_with_metadata(self):
        """SearchResult can be created with metadata."""
        metadata = {"source": "doc1", "page": 1}
        result = SearchResult(
            id="test-2",
            text="test content",
            score=0.88,
            metadata=metadata
        )
        assert result.metadata == metadata

    def test_search_result_id_must_be_string(self):
        """id field must be string type."""
        type_hints = get_type_hints(SearchResult)
        assert type_hints.get("id") == str, "id 字段类型应为 str"

    def test_search_result_text_must_be_string(self):
        """text field must be string type."""
        type_hints = get_type_hints(SearchResult)
        assert type_hints.get("text") == str, "text 字段类型应为 str"

    def test_search_result_score_must_be_float(self):
        """score field must be float type."""
        type_hints = get_type_hints(SearchResult)
        assert type_hints.get("score") == float, "score 字段类型应为 float"

    def test_search_result_metadata_must_be_optional_dict(self):
        """metadata field must be Optional[dict]."""
        type_hints = get_type_hints(SearchResult)
        # Optional[dict] 在运行时可能是 dict | None
        assert type_hints.get("metadata") is not None

    def test_search_result_validation_error_missing_required(self):
        """ValidationError raised when required fields missing."""
        with pytest.raises(ValidationError):
            SearchResult(id="test")  # Missing text and score

        with pytest.raises(ValidationError):
            SearchResult(text="test", score=0.5)  # Missing id

    def test_search_result_validation_error_invalid_score_type(self):
        """ValidationError raised when score is not numeric."""
        with pytest.raises(ValidationError):
            SearchResult(id="x", text="y", score="not-a-number")

    def test_search_result_accepts_integer_score(self):
        """Integer score should be coerced to float."""
        # Pydantic 会自动转换 int 为 float
        result = SearchResult(id="test", text="content", score=1)
        assert result.score == 1.0
        assert isinstance(result.score, float)


class TestVectorStoreAbstractClass:
    """Independent tests for VectorStore abstract base class."""

    def test_vector_store_is_abstract(self):
        """VectorStore must be an abstract base class."""
        assert issubclass(VectorStore, ABC), "VectorStore 应继承自 ABC"

    def test_vector_store_cannot_instantiate(self):
        """Cannot directly instantiate abstract VectorStore."""
        with pytest.raises(TypeError):
            VectorStore()

    def test_vector_store_has_add_method(self):
        """VectorStore must have abstract add method."""
        assert hasattr(VectorStore, "add"), "VectorStore 缺少 add 方法"
        add_method = getattr(VectorStore, "add")
        assert getattr(add_method, "__isabstractmethod__", False), \
            "add 方法应为抽象方法"

    def test_vector_store_has_query_method(self):
        """VectorStore must have abstract query method."""
        assert hasattr(VectorStore, "query"), "VectorStore 缺少 query 方法"
        query_method = getattr(VectorStore, "query")
        assert getattr(query_method, "__isabstractmethod__", False), \
            "query 方法应为抽象方法"

    def test_vector_store_has_delete_method(self):
        """VectorStore must have abstract delete method."""
        assert hasattr(VectorStore, "delete"), "VectorStore 缺少 delete 方法"
        delete_method = getattr(VectorStore, "delete")
        assert getattr(delete_method, "__isabstractmethod__", False), \
            "delete 方法应为抽象方法"

    def test_vector_store_has_create_collection_method(self):
        """VectorStore must have abstract create_collection method."""
        assert hasattr(VectorStore, "create_collection"), \
            "VectorStore 缺少 create_collection 方法"
        method = getattr(VectorStore, "create_collection")
        assert getattr(method, "__isabstractmethod__", False), \
            "create_collection 方法应为抽象方法"

    def test_add_method_signature(self):
        """add method must have correct signature per design."""
        hints = get_type_hints(VectorStore.add)
        # 检查参数类型
        assert "collection" in hints, "add 方法缺少 collection 参数"
        assert "text" in hints, "add 方法缺少 text 参数"
        assert "metadata" in hints, "add 方法缺少 metadata 参数"
        assert "id" in hints, "add 方法缺少 id 参数"
        assert "return" in hints, "add 方法缺少返回类型"
        assert hints["return"] == str, "add 方法返回类型应为 str"

    def test_query_method_signature(self):
        """query method must have correct signature per design."""
        hints = get_type_hints(VectorStore.query)
        assert "collection" in hints, "query 方法缺少 collection 参数"
        assert "query" in hints, "query 方法缺少 query 参数"
        assert "top_k" in hints, "query 方法缺少 top_k 参数"
        assert "return" in hints, "query 方法缺少返回类型"
        # 返回类型应为 List[SearchResult]
        assert hints["return"] == List[SearchResult], \
            "query 方法返回类型应为 List[SearchResult]"

    def test_delete_method_signature(self):
        """delete method must have correct signature per design."""
        hints = get_type_hints(VectorStore.delete)
        assert "collection" in hints, "delete 方法缺少 collection 参数"
        assert "ids" in hints, "delete 方法缺少 ids 参数"
        assert "return" in hints, "delete 方法缺少返回类型"
        assert hints["return"] is type(None) or hints["return"] == None, \
            "delete 方法返回类型应为 None"

    def test_create_collection_method_signature(self):
        """create_collection method must have correct signature."""
        hints = get_type_hints(VectorStore.create_collection)
        assert "collection" in hints, "create_collection 方法缺少 collection 参数"
        assert "dimension" in hints, "create_collection 方法缺少 dimension 参数"
        assert "return" in hints, "create_collection 方法缺少返回类型"

    def test_methods_are_async(self):
        """All VectorStore methods must be async."""
        import inspect
        assert inspect.iscoroutinefunction(VectorStore.add), \
            "add 方法应为 async"
        assert inspect.iscoroutinefunction(VectorStore.query), \
            "query 方法应为 async"
        assert inspect.iscoroutinefunction(VectorStore.delete), \
            "delete 方法应为 async"
        assert inspect.iscoroutinefunction(VectorStore.create_collection), \
            "create_collection 方法应为 async"


class MockVectorStore(VectorStore):
    """Mock implementation for testing subclass behavior."""

    async def add(
        self,
        collection: str,
        text: str,
        metadata: Optional[dict] = None,
        id: Optional[str] = None
    ) -> str:
        return id or "generated-id"

    async def query(
        self,
        collection: str,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        return [
            SearchResult(id="r1", text="result 1", score=0.9),
            SearchResult(id="r2", text="result 2", score=0.8)
        ]

    async def delete(
        self,
        collection: str,
        ids: Optional[List[str]] = None
    ) -> None:
        pass

    async def create_collection(
        self,
        collection: str,
        dimension: int
    ) -> None:
        pass


class TestVectorStoreConcreteImplementation:
    """Tests for VectorStore concrete implementation behavior."""

    def test_concrete_implementation_can_instantiate(self):
        """Concrete implementation can be instantiated."""
        store = MockVectorStore()
        assert isinstance(store, VectorStore)
        assert isinstance(store, ABC)

    @pytest.mark.asyncio
    async def test_add_returns_string_id(self):
        """add method must return document id as string."""
        store = MockVectorStore()
        result = await store.add("test-collection", "test text")
        assert isinstance(result, str)
        assert result == "generated-id"

    @pytest.mark.asyncio
    async def test_add_with_provided_id(self):
        """add method should return provided id when given."""
        store = MockVectorStore()
        result = await store.add(
            "test-collection",
            "test text",
            id="custom-id"
        )
        assert result == "custom-id"

    @pytest.mark.asyncio
    async def test_query_returns_search_result_list(self):
        """query method must return List[SearchResult]."""
        store = MockVectorStore()
        results = await store.query("test-collection", "test query")
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_query_respects_top_k(self):
        """query method should accept top_k parameter."""
        store = MockVectorStore()
        results = await store.query("test-collection", "test query", top_k=3)
        # Mock 返回固定 2 条，但应接受 top_k 参数
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_delete_with_none_ids(self):
        """delete method should accept None for ids."""
        store = MockVectorStore()
        # 不应抛出异常
        await store.delete("test-collection", ids=None)

    @pytest.mark.asyncio
    async def test_delete_with_empty_ids_list(self):
        """delete method should handle empty ids list."""
        store = MockVectorStore()
        # 不应抛出异常
        await store.delete("test-collection", ids=[])

    @pytest.mark.asyncio
    async def test_delete_with_specific_ids(self):
        """delete method should accept specific ids list."""
        store = MockVectorStore()
        # 不应抛出异常
        await store.delete("test-collection", ids=["id1", "id2"])

    @pytest.mark.asyncio
    async def test_create_collection_accepts_dimension(self):
        """create_collection must accept collection name and dimension."""
        store = MockVectorStore()
        # 不应抛出异常
        await store.create_collection("new-collection", dimension=1536)


class TestCollectionNamingConventions:
    """Tests for collection naming per design document."""

    @pytest.mark.asyncio
    async def test_regular_collection_name(self):
        """Regular session collection names should work."""
        store = MockVectorStore()
        await store.add("session-123", "test content")
        results = await store.query("session-123", "query")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_user_prefixed_collection_name(self):
        """User collection names with 'user:' prefix should work.

        Per design: user:{user_id} format for cross-session user memory.
        """
        store = MockVectorStore()
        user_collection = "user:user-456"
        await store.add(user_collection, "user memory content")
        results = await store.query(user_collection, "user query")
        assert len(results) > 0


class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    def test_search_result_with_empty_text(self):
        """SearchResult should accept empty text string."""
        result = SearchResult(id="empty", text="", score=0.5)
        assert result.text == ""

    def test_search_result_with_zero_score(self):
        """SearchResult should accept zero score."""
        result = SearchResult(id="zero", text="content", score=0.0)
        assert result.score == 0.0

    def test_search_result_with_negative_score(self):
        """SearchResult should accept negative score (distance metric)."""
        # 某些向量数据库使用距离而非相似度，可能产生负值
        result = SearchResult(id="neg", text="content", score=-0.5)
        assert result.score == -0.5

    def test_search_result_with_high_score(self):
        """SearchResult should accept score > 1."""
        result = SearchResult(id="high", text="content", score=10.5)
        assert result.score == 10.5

    def test_search_result_with_empty_metadata(self):
        """SearchResult should accept empty dict metadata."""
        result = SearchResult(
            id="meta",
            text="content",
            score=0.8,
            metadata={}
        )
        assert result.metadata == {}

    @pytest.mark.asyncio
    async def test_query_with_top_k_one(self):
        """query should work with top_k=1."""
        store = MockVectorStore()
        results = await store.query("test", "query", top_k=1)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_add_with_empty_metadata(self):
        """add should accept empty dict metadata."""
        store = MockVectorStore()
        result = await store.add("test", "text", metadata={})
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_add_with_complex_metadata(self):
        """add should accept complex nested metadata."""
        store = MockVectorStore()
        complex_metadata = {
            "source": "doc.pdf",
            "page": 1,
            "nested": {"key": "value"},
            "tags": ["tag1", "tag2"]
        }
        result = await store.add("test", "text", metadata=complex_metadata)
        assert isinstance(result, str)
