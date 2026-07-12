"""Vector Store interface for vector database operations.

This module provides abstract interfaces for vector storage operations,
supporting different backends like Chroma, LanceDB, Pinecone, etc.
"""
from abc import ABC, abstractmethod
from typing import Optional, List
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Represents a search result from vector store query.

    Attributes:
        id: Unique identifier for the document.
        text: The text content of the document.
        score: Similarity score (typically between 0 and 1, higher is better).
        metadata: Optional metadata associated with the document.
    """
    id: str
    text: str
    score: float
    metadata: Optional[dict] = None


class VectorStore(ABC):
    """Abstract base class for vector store operations.

    This interface defines the common operations for vector databases,
    allowing implementations to be swapped (e.g., Chroma, LanceDB, Pinecone).

    All methods are async to support both local and remote vector stores.
    """

    @abstractmethod
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
                If not provided, one will be generated.

        Returns:
            The id of the added document (either provided or generated).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            EmbeddingError: If text embedding fails.
        """
        ...

    @abstractmethod
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
            top_k: Maximum number of results to return. Defaults to 5.

        Returns:
            List of SearchResult objects, sorted by score (highest first).

        Raises:
            CollectionNotFoundError: If the collection does not exist.
            EmbeddingError: If query embedding fails.
        """
        ...

    @abstractmethod
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
            CollectionNotFoundError: If the collection does not exist.
        """
        ...

    @abstractmethod
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
            CollectionExistsError: If a collection with this name already exists.
        """
        ...
