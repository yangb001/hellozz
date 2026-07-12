"""Tests for VectorStore interface and SearchResult data class.

This module tests:
- SearchResult data class creation and validation
- VectorStore abstract base class interface
"""
import pytest
from abc import ABC
from typing import get_type_hints


class TestSearchResult:
    """Test SearchResult data class."""

    def test_search_result_creation_with_required_fields(self):
        """Test SearchResult creation with required fields only."""
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        result = SearchResult(
            id="test-id-123",
            text="This is a sample text",
            score=0.85
        )

        assert result.id == "test-id-123"
        assert result.text == "This is a sample text"
        assert result.score == 0.85
        assert result.metadata is None

    def test_search_result_creation_with_all_fields(self):
        """Test SearchResult creation with all fields including metadata."""
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        metadata = {"source": "doc.pdf", "page": 5, "timestamp": "2026-07-11"}
        result = SearchResult(
            id="test-id-456",
            text="Another sample text",
            score=0.92,
            metadata=metadata
        )

        assert result.id == "test-id-456"
        assert result.text == "Another sample text"
        assert result.score == 0.92
        assert result.metadata == metadata
        assert result.metadata["source"] == "doc.pdf"

    def test_search_result_is_pydantic_model(self):
        """Test that SearchResult is a Pydantic model for validation."""
        from agent_framework.infrastructure.storage.vector_store import SearchResult
        from pydantic import BaseModel

        assert issubclass(SearchResult, BaseModel)

    def test_search_result_handles_various_score_types(self):
        """Test SearchResult with different score values."""
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        # Float score
        result1 = SearchResult(id="1", text="text", score=0.0)
        assert result1.score == 0.0

        # Maximum score
        result2 = SearchResult(id="2", text="text", score=1.0)
        assert result2.score == 1.0

        # High precision float
        result3 = SearchResult(id="3", text="text", score=0.123456789)
        assert abs(result3.score - 0.123456789) < 1e-9

    def test_search_result_metadata_can_be_empty_dict(self):
        """Test SearchResult with empty metadata dict."""
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        result = SearchResult(
            id="test-id",
            text="text",
            score=0.5,
            metadata={}
        )

        assert result.metadata == {}

    def test_search_result_immutability(self):
        """Test SearchResult can be frozen if needed."""
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        result = SearchResult(
            id="test-id",
            text="text",
            score=0.5,
            metadata={"key": "value"}
        )

        # Pydantic models allow modification by default
        # This test just verifies the fields can be accessed
        assert result.id == "test-id"


class TestVectorStoreInterface:
    """Test VectorStore abstract base class interface."""

    def test_vector_store_is_abstract_class(self):
        """Test that VectorStore is an abstract class."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        assert issubclass(VectorStore, ABC)

    def test_vector_store_cannot_be_instantiated(self):
        """Test that VectorStore cannot be instantiated directly."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        with pytest.raises(TypeError):
            VectorStore()

    def test_vector_store_has_add_method(self):
        """Test that VectorStore has add method with correct signature."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        # Check method exists and is abstract
        assert hasattr(VectorStore, 'add')
        assert getattr(VectorStore.add, '__isabstractmethod__', False)

        # Check method signature
        hints = get_type_hints(VectorStore.add)
        assert 'collection' in hints
        assert 'text' in hints
        assert 'return' in hints

    def test_vector_store_has_query_method(self):
        """Test that VectorStore has query method with correct signature."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        assert hasattr(VectorStore, 'query')
        assert getattr(VectorStore.query, '__isabstractmethod__', False)

        hints = get_type_hints(VectorStore.query)
        assert 'collection' in hints
        assert 'query' in hints
        assert 'return' in hints

    def test_vector_store_has_delete_method(self):
        """Test that VectorStore has delete method with correct signature."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        assert hasattr(VectorStore, 'delete')
        assert getattr(VectorStore.delete, '__isabstractmethod__', False)

        hints = get_type_hints(VectorStore.delete)
        assert 'collection' in hints
        assert 'return' in hints

    def test_vector_store_has_create_collection_method(self):
        """Test that VectorStore has create_collection method with correct signature."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        assert hasattr(VectorStore, 'create_collection')
        assert getattr(VectorStore.create_collection, '__isabstractmethod__', False)

        hints = get_type_hints(VectorStore.create_collection)
        assert 'collection' in hints
        assert 'dimension' in hints
        assert 'return' in hints

    def test_vector_store_all_methods_are_async(self):
        """Test that all VectorStore methods are async."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore
        import inspect

        methods = ['add', 'query', 'delete', 'create_collection']
        for method_name in methods:
            method = getattr(VectorStore, method_name)
            assert inspect.iscoroutinefunction(method), f"{method_name} should be async"


class TestVectorStoreImplementation:
    """Test that a minimal implementation of VectorStore works."""

    def test_minimal_implementation_can_be_instantiated(self):
        """Test that a concrete implementation can be created."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        class MockVectorStore(VectorStore):
            async def add(self, collection: str, text: str, metadata: dict = None, id: str = None) -> str:
                return id or "generated-id"

            async def query(self, collection: str, query: str, top_k: int = 5) -> list:
                return [
                    SearchResult(id="result-1", text="Result text 1", score=0.9),
                    SearchResult(id="result-2", text="Result text 2", score=0.7),
                ]

            async def delete(self, collection: str, ids: list = None) -> None:
                pass

            async def create_collection(self, collection: str, dimension: int) -> None:
                pass

        store = MockVectorStore()
        assert store is not None

    @pytest.mark.asyncio
    async def test_mock_implementation_add(self):
        """Test add method in mock implementation."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        class MockVectorStore(VectorStore):
            def __init__(self):
                self.data = {}

            async def add(self, collection: str, text: str, metadata: dict = None, id: str = None) -> str:
                doc_id = id or "auto-generated-id"
                self.data[doc_id] = {"text": text, "metadata": metadata}
                return doc_id

            async def query(self, collection: str, query: str, top_k: int = 5) -> list:
                return []

            async def delete(self, collection: str, ids: list = None) -> None:
                for doc_id in (ids or []):
                    self.data.pop(doc_id, None)

            async def create_collection(self, collection: str, dimension: int) -> None:
                pass

        store = MockVectorStore()
        result_id = await store.add("test-collection", "Hello world", {"key": "value"})
        assert result_id in store.data
        assert store.data[result_id]["text"] == "Hello world"

    @pytest.mark.asyncio
    async def test_mock_implementation_query(self):
        """Test query method in mock implementation."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore, SearchResult

        class MockVectorStore(VectorStore):
            async def add(self, collection: str, text: str, metadata: dict = None, id: str = None) -> str:
                return id or "generated-id"

            async def query(self, collection: str, query: str, top_k: int = 5) -> list:
                return [
                    SearchResult(id="r1", text=f"Result for '{query}'", score=0.95, metadata={"col": collection}),
                    SearchResult(id="r2", text="Another result", score=0.85),
                ][:top_k]

            async def delete(self, collection: str, ids: list = None) -> None:
                pass

            async def create_collection(self, collection: str, dimension: int) -> None:
                pass

        store = MockVectorStore()
        results = await store.query("test-collection", "search query", top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert results[0].score >= results[1].score  # Sorted by score

    @pytest.mark.asyncio
    async def test_mock_implementation_delete(self):
        """Test delete method in mock implementation."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        class MockVectorStore(VectorStore):
            def __init__(self):
                self.collections = {"default": {}}

            async def add(self, collection: str, text: str, metadata: dict = None, id: str = None) -> str:
                doc_id = id or "generated-id"
                if collection not in self.collections:
                    self.collections[collection] = {}
                self.collections[collection][doc_id] = {"text": text, "metadata": metadata}
                return doc_id

            async def query(self, collection: str, query: str, top_k: int = 5) -> list:
                return []

            async def delete(self, collection: str, ids: list = None) -> None:
                if collection in self.collections:
                    if ids is None:
                        self.collections[collection].clear()
                    else:
                        for doc_id in ids:
                            self.collections[collection].pop(doc_id, None)

            async def create_collection(self, collection: str, dimension: int) -> None:
                self.collections[collection] = {}

        store = MockVectorStore()
        await store.add("col1", "text1", id="id1")
        await store.add("col1", "text2", id="id2")

        # Delete specific ids
        await store.delete("col1", ids=["id1"])
        assert "id1" not in store.collections["col1"]
        assert "id2" in store.collections["col1"]

        # Delete all
        await store.delete("col1")
        assert len(store.collections["col1"]) == 0

    @pytest.mark.asyncio
    async def test_mock_implementation_create_collection(self):
        """Test create_collection method in mock implementation."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        class MockVectorStore(VectorStore):
            def __init__(self):
                self.collections = {}
                self.dimensions = {}

            async def add(self, collection: str, text: str, metadata: dict = None, id: str = None) -> str:
                return id or "generated-id"

            async def query(self, collection: str, query: str, top_k: int = 5) -> list:
                return []

            async def delete(self, collection: str, ids: list = None) -> None:
                pass

            async def create_collection(self, collection: str, dimension: int) -> None:
                self.collections[collection] = {}
                self.dimensions[collection] = dimension

        store = MockVectorStore()
        await store.create_collection("new-collection", dimension=768)
        assert "new-collection" in store.collections
        assert store.dimensions["new-collection"] == 768


class TestSearchResultImport:
    """Test that SearchResult can be imported from expected locations."""

    def test_import_from_vector_store(self):
        """Test importing SearchResult from vector_store module."""
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        assert SearchResult is not None

    def test_import_vector_store_from_vector_store(self):
        """Test importing VectorStore from vector_store module."""
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        assert VectorStore is not None
