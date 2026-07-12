import pytest
from abc import ABC

import anyio

from agent_framework.infrastructure.storage.session_storage import SessionStorage
from agent_framework.interfaces.session import SessionContext, Message


class TestSessionStorageInterface:
    """Test suite for SessionStorage abstract base class."""

    def test_session_storage_is_abstract(self):
        """Test SessionStorage is an abstract class."""
        assert issubclass(SessionStorage, ABC)

    def test_save_is_abstract_method(self):
        """Test save is an abstract method."""
        assert hasattr(SessionStorage, 'save')
        assert getattr(SessionStorage.save, '__isabstractmethod__', False)

    def test_load_is_abstract_method(self):
        """Test load is an abstract method."""
        assert hasattr(SessionStorage, 'load')
        assert getattr(SessionStorage.load, '__isabstractmethod__', False)

    def test_delete_is_abstract_method(self):
        """Test delete is an abstract method."""
        assert hasattr(SessionStorage, 'delete')
        assert getattr(SessionStorage.delete, '__isabstractmethod__', False)

    def test_cannot_instantiate_directly(self):
        """Test SessionStorage cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SessionStorage()

    def test_subclass_must_implement_save(self):
        """Test that subclass without save raises TypeError."""
        class IncompleteStorage(SessionStorage):
            async def load(self, session_id: str) -> SessionContext:
                return None
            async def delete(self, session_id: str) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteStorage()

    def test_subclass_must_implement_load(self):
        """Test that subclass without load raises TypeError."""
        class IncompleteStorage(SessionStorage):
            async def save(self, ctx: SessionContext) -> None:
                pass
            async def delete(self, session_id: str) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteStorage()

    def test_subclass_must_implement_delete(self):
        """Test that subclass without delete raises TypeError."""
        class IncompleteStorage(SessionStorage):
            async def save(self, ctx: SessionContext) -> None:
                pass
            async def load(self, session_id: str) -> SessionContext:
                return None

        with pytest.raises(TypeError):
            IncompleteStorage()


class ConcreteSessionStorage(SessionStorage):
    """Concrete implementation for testing."""

    def __init__(self):
        self._storage = {}

    async def save(self, ctx: SessionContext) -> None:
        self._storage[ctx.session_id] = ctx

    async def load(self, session_id: str) -> SessionContext:
        return self._storage.get(session_id)

    async def delete(self, session_id: str) -> None:
        if session_id in self._storage:
            del self._storage[session_id]


class TestConcreteSessionStorage:
    """Test suite for concrete SessionStorage implementation."""

    @pytest.fixture
    def storage(self):
        """Create a fresh storage instance for each test."""
        return ConcreteSessionStorage()

    def test_save_and_load_roundtrip(self, storage):
        """Test save then load returns the same context."""
        async def _test():
            ctx = SessionContext(session_id="test_123")
            await storage.save(ctx)
            loaded = await storage.load("test_123")
            assert loaded is not None
            assert loaded.session_id == "test_123"
        anyio.run(_test)

    def test_load_nonexistent_returns_none(self, storage):
        """Test loading nonexistent session returns None."""
        async def _test():
            result = await storage.load("nonexistent")
            assert result is None
        anyio.run(_test)

    def test_delete_removes_session(self, storage):
        """Test delete removes session from storage."""
        async def _test():
            ctx = SessionContext(session_id="delete_me")
            await storage.save(ctx)
            await storage.delete("delete_me")
            result = await storage.load("delete_me")
            assert result is None
        anyio.run(_test)

    def test_delete_nonexistent_does_not_error(self, storage):
        """Test deleting nonexistent session does not raise."""
        async def _test():
            await storage.delete("nonexistent")
        anyio.run(_test)

    def test_save_updates_existing(self, storage):
        """Test saving same session_id updates existing."""
        async def _test():
            ctx1 = SessionContext(session_id="update_test", status="active")
            ctx2 = SessionContext(session_id="update_test", status="closed")
            await storage.save(ctx1)
            await storage.save(ctx2)
            loaded = await storage.load("update_test")
            assert loaded.status == "closed"
        anyio.run(_test)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])