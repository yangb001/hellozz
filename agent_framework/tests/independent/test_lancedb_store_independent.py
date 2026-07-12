"""Independent test cases for LanceDB VectorStore implementation.

This module contains independent verification tests for the LanceDBVectorStore
concrete implementation, following the detailed design specification.

Test categories:
1. LanceDBVectorStore inheritance and interface compliance
2. Collection management (create_collection, list_collections)
3. Document operations (add, query, delete)
4. Embedding integration
5. Error handling and boundary conditions
"""
import pytest
import tempfile
import os
import uuid
from pathlib import Path
from typing import List, Optional

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestLanceDBVectorStoreInheritance:
    """Independent tests for LanceDBVectorStore inheritance."""

    def test_lancedb_store_is_subclass_of_vector_store(self):
        """LanceDBVectorStore must inherit from VectorStore."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        assert issubclass(LanceDBVectorStore, VectorStore), \
            "LanceDBVectorStore should inherit from VectorStore"

    def test_lancedb_store_implements_all_abstract_methods(self):
        """LanceDBVectorStore must implement all abstract methods from VectorStore."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        from agent_framework.infrastructure.storage.vector_store import VectorStore

        # Check all abstract methods are implemented
        abstract_methods = [
            method for method in dir(VectorStore)
            if getattr(getattr(VectorStore, method), '__isabstractmethod__', False)
        ]

        for method in abstract_methods:
            assert hasattr(LanceDBVectorStore, method), \
                f"LanceDBVectorStore missing method: {method}"
            # Ensure method is no longer abstract
            method_obj = getattr(LanceDBVectorStore, method)
            assert not getattr(method_obj, '__isabstractmethod__', False), \
                f"Method {method} is still abstract"

    def test_lancedb_store_can_instantiate(self):
        """LanceDBVectorStore can be instantiated (not abstract)."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)
            assert store is not None
            assert isinstance(store, LanceDBVectorStore)


class TestLanceDBVectorStoreMethodSignatures:
    """Independent tests for method signatures."""

    def test_add_method_signature(self):
        """add method must have correct signature per design."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        import inspect

        sig = inspect.signature(LanceDBVectorStore.add)
        params = list(sig.parameters.keys())

        # Check parameters
        assert 'self' in params, "add method missing self parameter"
        assert 'collection' in params, "add method missing collection parameter"
        assert 'text' in params, "add method missing text parameter"
        assert 'metadata' in params, "add method missing metadata parameter"
        assert 'id' in params, "add method missing id parameter"

    def test_query_method_signature(self):
        """query method must have correct signature per design."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        import inspect

        sig = inspect.signature(LanceDBVectorStore.query)
        params = list(sig.parameters.keys())

        assert 'self' in params, "query method missing self parameter"
        assert 'collection' in params, "query method missing collection parameter"
        assert 'query' in params, "query method missing query parameter"
        assert 'top_k' in params, "query method missing top_k parameter"

    def test_delete_method_signature(self):
        """delete method must have correct signature per design."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        import inspect

        sig = inspect.signature(LanceDBVectorStore.delete)
        params = list(sig.parameters.keys())

        assert 'self' in params, "delete method missing self parameter"
        assert 'collection' in params, "delete method missing collection parameter"
        assert 'ids' in params, "delete method missing ids parameter"

    def test_create_collection_method_signature(self):
        """create_collection method must have correct signature."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        import inspect

        sig = inspect.signature(LanceDBVectorStore.create_collection)
        params = list(sig.parameters.keys())

        assert 'self' in params, "create_collection method missing self parameter"
        assert 'collection' in params, "create_collection method missing collection parameter"
        assert 'dimension' in params, "create_collection method missing dimension parameter"

    def test_list_collections_method_signature(self):
        """list_collections method must have correct signature."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        import inspect

        sig = inspect.signature(LanceDBVectorStore.list_collections)
        params = list(sig.parameters.keys())

        assert 'self' in params, "list_collections method missing self parameter"

    def test_all_methods_are_async(self):
        """All VectorStore methods must be async."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        import inspect

        async_methods = ['add', 'query', 'delete', 'create_collection', 'list_collections']

        for method_name in async_methods:
            method = getattr(LanceDBVectorStore, method_name)
            assert inspect.iscoroutinefunction(method), \
                f"{method_name} method should be async"


class TestLanceDBVectorStoreInitialization:
    """Independent tests for LanceDBVectorStore initialization."""

    def test_init_with_custom_path(self):
        """Test initialization with custom database path."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "custom_db")
            store = LanceDBVectorStore(db_path=db_path)

            assert store.db_path == db_path
            assert store._db is not None

    def test_init_with_default_path(self):
        """Test initialization with default path."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        store = LanceDBVectorStore()
        assert store.db_path == "~/.lancedb"

    def test_init_with_embedding_model(self):
        """Test initialization with embedding model parameter."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(
                db_path=db_path,
                embedding_model="all-MiniLM-L6-v2"
            )

            assert store.embedding_model == "all-MiniLM-L6-v2"

    def test_init_creates_database_connection(self):
        """Test that initialization creates LanceDB connection."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Verify database connection exists
            assert hasattr(store, '_db')
            assert store._db is not None

    def test_init_creates_collections_cache(self):
        """Test that initialization creates collections cache."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            assert hasattr(store, '_collections')
            assert isinstance(store._collections, dict)


class TestLanceDBVectorStoreCollectionManagement:
    """Independent tests for collection management."""

    @pytest.mark.asyncio
    async def test_create_collection_success(self):
        """Test successful collection creation."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            collections = await store.list_collections()
            assert "test_collection" in collections

    @pytest.mark.asyncio
    async def test_create_collection_duplicate_raises_error(self):
        """Test that creating duplicate collection raises error."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            with pytest.raises(Exception):
                await store.create_collection("test_collection", dimension=384)

    @pytest.mark.asyncio
    async def test_create_multiple_collections(self):
        """Test creating multiple collections."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("collection_1", dimension=384)
            await store.create_collection("collection_2", dimension=384)
            await store.create_collection("collection_3", dimension=384)

            collections = await store.list_collections()
            assert len(collections) == 3
            assert "collection_1" in collections
            assert "collection_2" in collections
            assert "collection_3" in collections

    @pytest.mark.asyncio
    async def test_list_collections_empty(self):
        """Test listing collections when none exist."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            collections = await store.list_collections()
            assert isinstance(collections, list)
            assert len(collections) == 0

    @pytest.mark.asyncio
    async def test_list_collections_returns_list(self):
        """Test that list_collections returns a list."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            collections = await store.list_collections()
            assert isinstance(collections, list)


class TestLanceDBVectorStoreAddDocuments:
    """Independent tests for adding documents."""

    @pytest.mark.asyncio
    async def test_add_document_returns_string_id(self):
        """add method must return document id as string."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            doc_id = await store.add(
                collection="test_collection",
                text="This is a test document."
            )

            assert isinstance(doc_id, str)
            assert len(doc_id) > 0

    @pytest.mark.asyncio
    async def test_add_document_with_custom_id(self):
        """add method should return provided id when given."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            custom_id = "custom-doc-id-123"
            doc_id = await store.add(
                collection="test_collection",
                text="Test document with custom ID.",
                id=custom_id
            )

            assert doc_id == custom_id

    @pytest.mark.asyncio
    async def test_add_document_with_metadata(self):
        """add method should accept and store metadata."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            metadata = {"source": "test", "page": 1}
            doc_id = await store.add(
                collection="test_collection",
                text="Document with metadata.",
                metadata=metadata
            )

            assert isinstance(doc_id, str)

    @pytest.mark.asyncio
    async def test_add_document_without_metadata(self):
        """add method should work without metadata."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            doc_id = await store.add(
                collection="test_collection",
                text="Document without metadata."
            )

            assert isinstance(doc_id, str)

    @pytest.mark.asyncio
    async def test_add_multiple_documents_unique_ids(self):
        """Each added document should get a unique ID."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            doc_id1 = await store.add(
                collection="test_collection",
                text="First document."
            )
            doc_id2 = await store.add(
                collection="test_collection",
                text="Second document."
            )
            doc_id3 = await store.add(
                collection="test_collection",
                text="Third document."
            )

            # All IDs should be unique
            assert doc_id1 != doc_id2
            assert doc_id2 != doc_id3
            assert doc_id1 != doc_id3

    @pytest.mark.asyncio
    async def test_add_to_nonexistent_collection_raises_error(self):
        """add should raise error when collection does not exist."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            with pytest.raises(Exception):
                await store.add(
                    collection="nonexistent",
                    text="This should fail."
                )


class TestLanceDBVectorStoreQueryDocuments:
    """Independent tests for querying documents."""

    @pytest.mark.asyncio
    async def test_query_returns_list_of_search_results(self):
        """query method must return List[SearchResult]."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add document
            await store.add(
                collection="test_collection",
                text="Machine learning is a subset of AI."
            )

            # Query
            results = await store.query(
                collection="test_collection",
                query="What is machine learning?",
                top_k=5
            )

            assert isinstance(results, list)
            assert len(results) > 0
            assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_query_result_has_required_fields(self):
        """SearchResult must have id, text, score fields."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            await store.add(
                collection="test_collection",
                text="Test document for field validation."
            )

            results = await store.query(
                collection="test_collection",
                query="test document",
                top_k=1
            )

            assert len(results) > 0
            result = results[0]

            assert hasattr(result, 'id'), "SearchResult missing id field"
            assert hasattr(result, 'text'), "SearchResult missing text field"
            assert hasattr(result, 'score'), "SearchResult missing score field"
            assert hasattr(result, 'metadata'), "SearchResult missing metadata field"

    @pytest.mark.asyncio
    async def test_query_respects_top_k(self):
        """query method should limit results by top_k."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add multiple documents
            for i in range(10):
                await store.add(
                    collection="test_collection",
                    text=f"Document number {i} about various topics."
                )

            # Query with top_k=3
            results = await store.query(
                collection="test_collection",
                query="document topics",
                top_k=3
            )

            assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_query_with_top_k_one(self):
        """query should work with top_k=1."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            await store.add(
                collection="test_collection",
                text="Single result test document."
            )

            results = await store.query(
                collection="test_collection",
                query="single result",
                top_k=1
            )

            assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_query_score_range(self):
        """query results should have score between 0 and 1."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            await store.add(
                collection="test_collection",
                text="Test document for score validation."
            )

            results = await store.query(
                collection="test_collection",
                query="test document",
                top_k=1
            )

            assert len(results) > 0
            for result in results:
                assert 0 <= result.score <= 1, \
                    f"Score should be between 0 and 1, got {result.score}"

    @pytest.mark.asyncio
    async def test_query_from_nonexistent_collection_raises_error(self):
        """query should raise error when collection does not exist."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            with pytest.raises(Exception):
                await store.query(
                    collection="nonexistent",
                    query="test query"
                )


class TestLanceDBVectorStoreDeleteDocuments:
    """Independent tests for deleting documents."""

    @pytest.mark.asyncio
    async def test_delete_specific_documents(self):
        """delete should remove specific documents by ID."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add documents
            doc_id1 = await store.add(
                collection="test_collection",
                text="Document to keep."
            )
            doc_id2 = await store.add(
                collection="test_collection",
                text="Document to delete."
            )

            # Delete specific document
            await store.delete(
                collection="test_collection",
                ids=[doc_id2]
            )

            # Query to verify deletion
            results = await store.query(
                collection="test_collection",
                query="document",
                top_k=10
            )

            # Only one document should remain
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_delete_all_documents(self):
        """delete with ids=None should delete all documents."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add multiple documents
            await store.add(
                collection="test_collection",
                text="First document."
            )
            await store.add(
                collection="test_collection",
                text="Second document."
            )

            # Delete all
            await store.delete(collection="test_collection")

            # Query to verify all deleted
            results = await store.query(
                collection="test_collection",
                query="document",
                top_k=10
            )

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_delete_with_empty_ids_list(self):
        """delete should handle empty ids list."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            await store.add(
                collection="test_collection",
                text="Test document."
            )

            # Empty list should not delete any documents
            await store.delete(
                collection="test_collection",
                ids=[]
            )

            results = await store.query(
                collection="test_collection",
                query="test",
                top_k=10
            )

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent_collection_raises_error(self):
        """delete should raise error when collection does not exist."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            with pytest.raises(Exception):
                await store.delete(collection="nonexistent")


class TestLanceDBVectorStoreCollectionIsolation:
    """Independent tests for collection isolation."""

    @pytest.mark.asyncio
    async def test_collections_are_isolated(self):
        """Documents in different collections should be isolated."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Create two collections
            await store.create_collection("collection_a", dimension=384)
            await store.create_collection("collection_b", dimension=384)

            # Add to collection_a
            await store.add(
                collection="collection_a",
                text="Document in collection A."
            )

            # Query collection_b should not return results from collection_a
            results_b = await store.query(
                collection="collection_b",
                query="Document",
                top_k=10
            )
            assert len(results_b) == 0

            # Query collection_a should return results
            results_a = await store.query(
                collection="collection_a",
                query="Document",
                top_k=10
            )
            assert len(results_a) == 1


class TestLanceDBVectorStoreMetadataHandling:
    """Independent tests for metadata handling."""

    @pytest.mark.asyncio
    async def test_metadata_stored_and_retrieved(self):
        """Metadata should be stored and retrievable."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add document with metadata
            metadata = {"source": "test.pdf", "page": 42, "tags": ["ai", "ml"]}
            doc_id = await store.add(
                collection="test_collection",
                text="Document with metadata.",
                metadata=metadata
            )

            # Query to verify metadata
            results = await store.query(
                collection="test_collection",
                query="document metadata",
                top_k=1
            )

            assert len(results) > 0
            result = results[0]
            assert result.metadata is not None
            assert result.metadata.get("source") == "test.pdf"
            assert result.metadata.get("page") == 42

    @pytest.mark.asyncio
    async def test_metadata_none_when_not_provided(self):
        """Metadata should be None when not provided."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add document without metadata
            await store.add(
                collection="test_collection",
                text="Document without metadata."
            )

            # Query to verify metadata
            results = await store.query(
                collection="test_collection",
                query="document",
                top_k=1
            )

            assert len(results) > 0
            result = results[0]
            # metadata should be None or empty dict
            assert result.metadata is None or result.metadata == {}


class TestLanceDBVectorStoreBoundaryConditions:
    """Independent tests for boundary conditions."""

    @pytest.mark.asyncio
    async def test_add_empty_text(self):
        """add should accept empty text string."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            doc_id = await store.add(
                collection="test_collection",
                text=""
            )

            assert isinstance(doc_id, str)

    @pytest.mark.asyncio
    async def test_add_long_text(self):
        """add should handle long text strings."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Create long text
            long_text = "This is a test document. " * 1000

            doc_id = await store.add(
                collection="test_collection",
                text=long_text
            )

            assert isinstance(doc_id, str)

    @pytest.mark.asyncio
    async def test_query_empty_collection(self):
        """query should return empty list for empty collection."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            results = await store.query(
                collection="test_collection",
                query="test query",
                top_k=5
            )

            assert isinstance(results, list)
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_special_characters_in_collection_name(self):
        """Collection names with special characters should work."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Use collection name with allowed special characters
            collection_name = "session_123-abc_def"
            await store.create_collection(collection_name, dimension=384)

            collections = await store.list_collections()
            assert collection_name in collections

    @pytest.mark.asyncio
    async def test_user_prefixed_collection_name(self):
        """User collection names with 'user_' prefix should work.

        Note: LanceDB does not support colon (:) in table names,
        so underscore is used instead for user collections.
        """
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Use user_ prefix (LanceDB does not support colon)
            user_collection = "user_user-456"
            await store.create_collection(user_collection, dimension=384)

            collections = await store.list_collections()
            assert user_collection in collections

    @pytest.mark.asyncio
    async def test_collection_name_with_colon_raises_error(self):
        """Collection names with colon should raise ValueError from LanceDB."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # LanceDB does not support colon in table names
            with pytest.raises(ValueError):
                await store.create_collection("user:user-456", dimension=384)


class TestLanceDBVectorStoreEmbeddingIntegration:
    """Independent tests for embedding integration."""

    def test_embedding_model_attribute(self):
        """LanceDBVectorStore should store embedding_model attribute."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(
                db_path=db_path,
                embedding_model="test-model"
            )

            assert store.embedding_model == "test-model"

    def test_embedding_model_default_none(self):
        """LanceDBVectorStore should default embedding_model to None."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            assert store.embedding_model is None

    @pytest.mark.asyncio
    async def test_get_embedding_returns_list_of_floats(self):
        """_get_embedding should return list of floats."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            embedding = await store._get_embedding("test text")

            assert isinstance(embedding, list)
            assert len(embedding) == 384  # Default dimension
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_get_embedding_deterministic(self):
        """_get_embedding should return same embedding for same text."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            embedding1 = await store._get_embedding("test text")
            embedding2 = await store._get_embedding("test text")

            assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_get_embedding_different_for_different_text(self):
        """_get_embedding should return different embeddings for different text."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            embedding1 = await store._get_embedding("hello world")
            embedding2 = await store._get_embedding("completely different text")

            assert embedding1 != embedding2


class TestLanceDBVectorStoreIntegrationWorkflow:
    """Independent integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete lifecycle: create, add, query, delete."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # 1. Create collection
            await store.create_collection("ai_docs", dimension=384)

            # 2. Add documents
            doc_id1 = await store.add(
                collection="ai_docs",
                text="Artificial intelligence is revolutionizing technology.",
                metadata={"topic": "AI", "year": 2024}
            )
            doc_id2 = await store.add(
                collection="ai_docs",
                text="Machine learning enables computers to learn from data.",
                metadata={"topic": "ML", "year": 2024}
            )
            doc_id3 = await store.add(
                collection="ai_docs",
                text="Deep learning uses neural networks for complex tasks.",
                metadata={"topic": "DL", "year": 2023}
            )

            # 3. Query
            results = await store.query(
                collection="ai_docs",
                query="How do computers learn?",
                top_k=2
            )
            assert len(results) == 2
            assert all(isinstance(r, SearchResult) for r in results)

            # 4. Delete one document
            await store.delete(collection="ai_docs", ids=[doc_id2])

            # 5. Query again
            results = await store.query(
                collection="ai_docs",
                query="machine learning",
                top_k=10
            )
            assert len(results) == 2  # Two documents remain

            # 6. Delete all
            await store.delete(collection="ai_docs")

            # 7. Verify empty
            results = await store.query(
                collection="ai_docs",
                query="artificial intelligence",
                top_k=10
            )
            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_multiple_collections_workflow(self):
        """Test workflow with multiple collections."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Create multiple collections
            await store.create_collection("session_1", dimension=384)
            await store.create_collection("session_2", dimension=384)
            await store.create_collection("user_alice", dimension=384)

            # Add documents to different collections
            await store.add(
                collection="session_1",
                text="Session 1 conversation about AI."
            )
            await store.add(
                collection="session_2",
                text="Session 2 discussion about ML."
            )
            await store.add(
                collection="user_alice",
                text="Alice's personal preferences."
            )

            # Verify collection list
            collections = await store.list_collections()
            assert len(collections) == 3

            # Verify each collection has content
            results_1 = await store.query(
                collection="session_1",
                query="AI",
                top_k=10
            )
            assert len(results_1) == 1

            results_2 = await store.query(
                collection="session_2",
                query="ML",
                top_k=10
            )
            assert len(results_2) == 1

            results_user = await store.query(
                collection="user_alice",
                query="preferences",
                top_k=10
            )
            assert len(results_user) == 1
