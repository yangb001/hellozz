from abc import ABC, abstractmethod

from agent_framework.interfaces.session import SessionContext


class SessionStorage(ABC):
    """Abstract base class for session persistence.

    Implementations provide synchronous or asynchronous persistence
    of SessionContext objects to various storage backends.
    """

    @abstractmethod
    async def save(self, ctx: SessionContext) -> None:
        """Save or update a session context.

        Args:
            ctx: SessionContext to save.
        """
        ...

    @abstractmethod
    async def load(self, session_id: str) -> SessionContext | None:
        """Load a session context by ID.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            SessionContext if found, None otherwise.
        """
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete a session context.

        Args:
            session_id: Unique identifier for the session.
        """
        ...