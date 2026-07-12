"""Tests for LanceDB VectorStore implementation.

This module tests:
- LanceDBVectorStore class creation and initialization
- add, query, delete, create_collection methods
- Integration with LanceDB backend
"""
import pytest
import tempfile
import os
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLanceDBVectorStoreInit:
    """Test LanceDBVectorStore initialization."""

    def test_import_lancedb_vector_store(self):
        """Test that LanceDBVectorStore can be imported."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        assert LanceDBVectorStore is not None

    def test_lancedb_vector_store_inherits_vector_store(self):
        """Test that LanceDBVectorStore inherits from VectorStore."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        from agent_framework.infrastructure.storage.vector_store import VectorStore
        assert issubclass(LanceDBVectorStore, VectorStore)

    def test_lancedb_vector_store_init_with_path(self):
        """Test LanceDBVectorStore initialization with database path."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)
            assert store is not None
            assert store.db_path == db_path

    def test_lancedb_vector_store_init_default_path(self):
        """Test LanceDBVectorStore initialization with default path."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        store = LanceDBVectorStore()
        assert store is not None


class TestLanceDBVectorStoreCreateCollection:
    """Test LanceDBVectorStore create_collection method."""

    @pytest.mark.asyncio
    async def test_create_collection(self):
        """Test creating a new collection."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Create a collection with dimension 384 (all-MiniLM-L6-v2 default)
            await store.create_collection("test_collection", dimension=384)

            # Verify collection was created
            assert "test_collection" in await store.list_collections()

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


class TestLanceDBVectorStoreAdd:
    """Test LanceDBVectorStore add method."""

    @pytest.mark.asyncio
    async def test_add_document(self):
        """Test adding a document to a collection."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Create collection first
            await store.create_collection("test_collection", dimension=384)

            # Add a document
            doc_id = await store.add(
                collection="test_collection",
                text="This is a test document about AI agents.",
                metadata={"source": "test"}
            )

            assert doc_id is not None
            assert isinstance(doc_id, str)

    @pytest.mark.asyncio
    async def test_add_document_with_custom_id(self):
        """Test adding a document with custom ID."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            doc_id = await store.add(
                collection="test_collection",
                text="Another test document.",
                id="custom-id-123"
            )

            assert doc_id == "custom-id-123"

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self):
        """Test adding multiple documents to a collection."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            doc_id1 = await store.add(
                collection="test_collection",
                text="First document about machine learning."
            )
            doc_id2 = await store.add(
                collection="test_collection",
                text="Second document about natural language processing."
            )

            assert doc_id1 != doc_id2

    @pytest.mark.asyncio
    async def test_add_to_nonexistent_collection_raises_error(self):
        """Test that adding to non-existent collection raises error."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            with pytest.raises(Exception):
                await store.add(
                    collection="nonexistent",
                    text="This should fail."
                )


class TestLanceDBVectorStoreQuery:
    """Test LanceDBVectorStore query method."""

    @pytest.mark.asyncio
    async def test_query_returns_results(self):
        """Test that query returns search results."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore
        from agent_framework.infrastructure.storage.vector_store import SearchResult

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add some documents
            await store.add(
                collection="test_collection",
                text="Machine learning is a subset of artificial intelligence."
            )
            await store.add(
                collection="test_collection",
                text="Natural language processing deals with text analysis."
            )

            # Query
            results = await store.query(
                collection="test_collection",
                query="What is machine learning?",
                top_k=2
            )

            assert isinstance(results, list)
            assert len(results) <= 2
            for result in results:
                assert isinstance(result, SearchResult)
                assert hasattr(result, 'id')
                assert hasattr(result, 'text')
                assert hasattr(result, 'score')

    @pytest.mark.asyncio
    async def test_query_top_k(self):
        """Test that top_k parameter limits results."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add multiple documents
            for i in range(5):
                await store.add(
                    collection="test_collection",
                    text=f"Document number {i} about various topics."
                )

            results = await store.query(
                collection="test_collection",
                query="document topics",
                top_k=3
            )

            assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_query_from_nonexistent_collection_raises_error(self):
        """Test that querying non-existent collection raises error."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            with pytest.raises(Exception):
                await store.query(
                    collection="nonexistent",
                    query="test query"
                )


class TestLanceDBVectorStoreDelete:
    """Test LanceDBVectorStore delete method."""

    @pytest.mark.asyncio
    async def test_delete_specific_documents(self):
        """Test deleting specific documents by ID."""
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
        """Test deleting all documents in a collection."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("test_collection", dimension=384)

            # Add documents
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
    async def test_delete_nonexistent_collection_raises_error(self):
        """Test that deleting from non-existent collection raises error."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            with pytest.raises(Exception):
                await store.delete(collection="nonexistent")


class TestLanceDBVectorStoreListCollections:
    """Test LanceDBVectorStore list_collections method."""

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
    async def test_list_collections_after_creation(self):
        """Test listing collections after creating some."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            await store.create_collection("collection1", dimension=384)
            await store.create_collection("collection2", dimension=384)

            collections = await store.list_collections()
            assert "collection1" in collections
            assert "collection2" in collections


class TestLanceDBVectorStoreIntegration:
    """Integration tests for LanceDBVectorStore."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow: create collection, add, query, delete."""
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
    async def test_multiple_collections_isolation(self):
        """Test that collections are isolated from each other."""
        from agent_framework.infrastructure.storage.lancedb_store import LanceDBVectorStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_db")
            store = LanceDBVectorStore(db_path=db_path)

            # Create two collections
            await store.create_collection("collection_a", dimension=384)
            await store.create_collection("collection_b", dimension=384)

            # Add to collection A
            await store.add(
                collection="collection_a",
                text="Document in collection A."
            )

            # Query collection B should not return results from A
            results = await store.query(
                collection="collection_b",
                query="Document",
                top_k=10
            )
            assert len(results) == 0

            # Query collection A should return results
            results = await store.query(
                collection="collection_a",
                query="Document",
                top_k=10
            )
            assert len(results) == 1
