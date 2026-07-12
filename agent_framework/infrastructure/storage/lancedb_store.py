"""LanceDB Vector Store implementation.

This module provides a concrete implementation of the VectorStore interface
using LanceDB as the vector database backend.
"""
import uuid
from typing import List, Optional

import lancedb
import pyarrow as pa

from .vector_store import SearchResult, VectorStore


class LanceDBVectorStore(VectorStore):
    """LanceDB implementation of the VectorStore interface.

    This class stores vectors using LanceDB, a serverless vector database
    that runs locally. It supports basic vector operations like add, query,
    delete, and collection management.

    Note: This implementation uses a simple random embedding for demonstration
    purposes. In production, you should integrate a proper embedding model
    (e.g., sentence-transformers, OpenAI embeddings) to generate meaningful
    vector representations of text.
    """

    def __init__(
        self,
        db_path: str = "~/.lancedb",
        embedding_model: Optional[str] = None
    ):
        """Initialize the LanceDB Vector Store.

        Args:
            db_path: Path to the LanceDB database directory.
            embedding_model: Name of the embedding model to use (optional).
                If not provided, uses a simple random embedding.
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self._db = lancedb.connect(db_path)
        self._collections: dict = {}  # Cache for table references

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text.

        This is a placeholder implementation that generates random embeddings.
        In production, replace with actual embedding model calls.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        import hashlib
        import numpy as np

        # Generate deterministic embedding based on text hash
        # This is for testing only - use real embeddings in production
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()

        # Convert to float array (384 dimensions for compatibility with all-MiniLM-L6-v2)
        np.random.seed(int.from_bytes(hash_bytes[:4], byteorder='big'))
        embedding = np.random.randn(384).tolist()

        return embedding

    async def create_collection(
        self,
        collection: str,
        dimension: int
    ) -> None:
        """Create a new collection in the vector store.

        Args:
            collection: Name of the collection to create.
            dimension: The dimension of the embedding vectors.

        Raises:
            Exception: If the collection already exists.
        """
        # Check if collection already exists
        existing_tables = self._db.table_names()
        if collection in existing_tables:
            raise Exception(f"Collection '{collection}' already exists.")

        # Create table with schema
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dimension)),
            pa.field("metadata", pa.string()),  # Store metadata as JSON string
        ])

        self._db.create_table(
            name=collection,
            schema=schema,
            mode="create"
        )

        self._collections[collection] = self._db.open_table(collection)

    async def add(
        self,
        collection: str,
        text: str,
        metadata: Optional[dict] = None,
        id: Optional[str] = None
    ) -> str:
        """Add a document to the vector store.

        Args:
            collection: Name of the collection to add the document to.
            text: The text content to store and embed.
            metadata: Optional metadata to associate with the document.
            id: Optional unique identifier for the document.

        Returns:
            The id of the added document.

        Raises:
            Exception: If the collection does not exist.
        """
        import json

        # Check if collection exists
        if collection not in self._db.table_names():
            raise Exception(f"Collection '{collection}' does not exist. "
                          f"Call create_collection first.")

        # Generate ID if not provided
        doc_id = id or str(uuid.uuid4())

        # Get embedding
        embedding = await self._get_embedding(text)

        # Prepare metadata as JSON string
        metadata_str = json.dumps(metadata) if metadata else "{}"

        # Insert data
        table = self._db.open_table(collection)
        data = [{
            "id": doc_id,
            "text": text,
            "vector": embedding,
            "metadata": metadata_str
        }]

        table.add(data)

        return doc_id

    async def query(
        self,
        collection: str,
        query: str,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Query the vector store for similar documents.

        Args:
            collection: Name of the collection to query.
            query: The query text to search for.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects, sorted by score (highest first).

        Raises:
            Exception: If the collection does not exist.
        """
        import json

        # Check if collection exists
        if collection not in self._db.table_names():
            raise Exception(f"Collection '{collection}' does not exist.")

        # Get query embedding
        query_embedding = await self._get_embedding(query)

        # Search
        table = self._db.open_table(collection)
        results = (
            table.search(query_embedding)
            .limit(top_k)
            .to_list()
        )

        # Convert to SearchResult objects
        search_results = []
        for row in results:
            metadata = json.loads(row.get("metadata", "{}"))
            # LanceDB returns _distance, convert to score (0-1 range)
            distance = row.get("_distance", 0)
            score = 1.0 / (1.0 + distance)  # Convert distance to similarity score

            search_results.append(SearchResult(
                id=row["id"],
                text=row["text"],
                score=score,
                metadata=metadata if metadata else None
            ))

        return search_results

    async def delete(
        self,
        collection: str,
        ids: Optional[List[str]] = None
    ) -> None:
        """Delete documents from the vector store.

        Args:
            collection: Name of the collection to delete from.
            ids: List of document ids to delete. If None, deletes all
                documents in the collection.

        Raises:
            Exception: If the collection does not exist.
        """
        # Check if collection exists
        if collection not in self._db.table_names():
            raise Exception(f"Collection '{collection}' does not exist.")

        table = self._db.open_table(collection)

        if ids is None:
            # Delete all documents by dropping and recreating the table
            # Get the schema first
            schema = table.schema
            self._db.drop_table(collection)
            self._db.create_table(
                name=collection,
                schema=schema,
                mode="create"
            )
        else:
            # Delete specific documents by ID
            # LanceDB uses SQL-like syntax for deletion
            id_list = "', '".join(ids)
            table.delete(f"id IN ('{id_list}')")

    async def list_collections(self) -> List[str]:
        """List all collections in the vector store.

        Returns:
            List of collection names.
        """
        return self._db.table_names()
