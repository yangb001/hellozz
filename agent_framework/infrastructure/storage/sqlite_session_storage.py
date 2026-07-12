"""SQLite implementation of SessionStorage for persistent session storage."""
import aiosqlite
import json
from datetime import datetime
from typing import Optional

from agent_framework.interfaces.session import SessionContext, Message
from .session_storage import SessionStorage


class SQLiteSessionStorage(SessionStorage):
    """SQLite-based implementation of SessionStorage.

    Provides asynchronous persistence of SessionContext objects using
    aiosqlite for non-blocking database operations.
    """

    def __init__(self, db_path: str):
        """Initialize SQLite session storage.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            await self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            await self._conn.commit()
        return self._conn

    async def _serialize_session(self, ctx: SessionContext) -> str:
        """Serialize SessionContext to JSON string."""
        data = ctx.model_dump(mode="json")
        return json.dumps(data)

    async def _deserialize_session(self, data: str) -> SessionContext:
        """Deserialize JSON string to SessionContext."""
        parsed = json.loads(data)
        return SessionContext.model_validate(parsed)

    async def save(self, ctx: SessionContext) -> None:
        """Save or update a session context.

        Args:
            ctx: SessionContext to save.
        """
        conn = await self._get_connection()
        serialized = await self._serialize_session(ctx)
        updated_at = datetime.utcnow().isoformat()
        await conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, data, updated_at) VALUES (?, ?, ?)",
            (ctx.session_id, serialized, updated_at)
        )
        await conn.commit()

    async def load(self, session_id: str) -> Optional[SessionContext]:
        """Load a session context by ID.

        Args:
            session_id: Unique identifier for the session.

        Returns:
            SessionContext if found, None otherwise.
        """
        conn = await self._get_connection()
        cursor = await conn.execute(
            "SELECT data FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return await self._deserialize_session(row[0])

    async def delete(self, session_id: str) -> None:
        """Delete a session context.

        Args:
            session_id: Unique identifier for the session.
        """
        conn = await self._get_connection()
        await conn.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        await conn.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
