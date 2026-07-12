"""Independent test cases for SessionManager with Actor model.

This module contains independent verification tests for the SessionManager
class defined in core/session_manager.py.

Test categories:
1. SessionManager initialization
2. create_session method
3. process_message method
4. get_session method
5. close_session method
6. resume_session method
7. Actor model behavior
8. Boundary conditions
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event
from agent_framework.core.session_manager import SessionManager, generate_id


# ============================================================================
# Mock Implementations
# ============================================================================


class MockStorage:
    """Mock session storage."""

    def __init__(self):
        self.saved: List[SessionContext] = []
        self.loaded: List[str] = []

    async def save(self, ctx: SessionContext):
        self.saved.append(ctx)

    async def load(self, session_id: str):
        self.loaded.append(session_id)
        return None

    async def delete(self, session_id: str):
        pass


class MockRuntime:
    """Mock agent runtime."""

    def __init__(self, events=None):
        self.events = events or [Event(type="final_answer", content="mock answer")]

    def run(self, ctx, user_input, memory, tools, planner, llm_gateway):
        async def _gen():
            for event in self.events:
                yield event
        return _gen()


class MockEventBus:
    """Mock event bus."""

    def __init__(self):
        self.published: List[tuple] = []

    async def publish(self, event_type, event):
        self.published.append((event_type, event))


class MockMemory:
    """Mock memory."""

    def __init__(self):
        self.extracted = []

    async def extract_long_term(self, session_id, force=False):
        self.extracted.append(session_id)


def _mock_memory_factory(session_id: str):
    return MockMemory()


# ============================================================================
# 1. Initialization
# ============================================================================


class TestSessionManagerInit:
    """Test SessionManager initialization."""

    def test_can_instantiate(self):
        """SessionManager can be instantiated with all dependencies."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        assert sm is not None

    def test_stores_dependencies(self):
        """SessionManager stores all dependency references."""
        runtime = MockRuntime()
        planner = MagicMock()
        tools = {"tool1": "mock"}
        event_bus = MockEventBus()
        storage = MockStorage()
        llm_gateway = MagicMock()

        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway,
        )
        assert sm.runtime is runtime
        assert sm.planner is planner
        assert sm.tools is tools
        assert sm.event_bus is event_bus
        assert sm.storage is storage
        assert sm.llm_gateway is llm_gateway

    def test_active_sessions_initially_empty(self):
        """_active_sessions should be empty initially."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        assert len(sm._active_sessions) == 0

    def test_session_queues_initially_empty(self):
        """_session_queues should be empty initially."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        assert len(sm._session_queues) == 0


# ============================================================================
# 2. generate_id
# ============================================================================


class TestGenerateId:
    """Test generate_id helper function."""

    def test_returns_string(self):
        """generate_id returns a string."""
        result = generate_id()
        assert isinstance(result, str)

    def test_returns_unique_values(self):
        """generate_id returns unique values."""
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_returns_non_empty(self):
        """generate_id returns non-empty string."""
        result = generate_id()
        assert len(result) > 0


# ============================================================================
# 3. create_session
# ============================================================================


class TestCreateSession:
    """Test create_session method."""

    @pytest.mark.asyncio
    async def test_returns_session_context(self):
        """create_session returns a SessionContext."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        assert isinstance(ctx, SessionContext)

    @pytest.mark.asyncio
    async def test_session_has_unique_id(self):
        """Each created session has a unique ID."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx1 = await sm.create_session("user-1")
        ctx2 = await sm.create_session("user-1")
        assert ctx1.session_id != ctx2.session_id

    @pytest.mark.asyncio
    async def test_default_session_type_private(self):
        """Default session_type should be 'private'."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        assert ctx.session_type == "private"

    @pytest.mark.asyncio
    async def test_group_session_type(self):
        """create_session accepts 'group' session type."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1", session_type="group")
        assert ctx.session_type == "group"

    @pytest.mark.asyncio
    async def test_participants_include_creator(self):
        """Participants must include the creating user."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        assert "user-1" in ctx.participants

    @pytest.mark.asyncio
    async def test_additional_participants(self):
        """Additional participants are included."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1", participants=["user-2", "user-3"])
        assert "user-1" in ctx.participants
        assert "user-2" in ctx.participants
        assert "user-3" in ctx.participants

    @pytest.mark.asyncio
    async def test_session_stored_in_active(self):
        """Created session is stored in _active_sessions."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        assert ctx.session_id in sm._active_sessions

    @pytest.mark.asyncio
    async def test_session_queue_created(self):
        """Created session has an asyncio.Queue."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        assert ctx.session_id in sm._session_queues
        assert isinstance(sm._session_queues[ctx.session_id], asyncio.Queue)

    @pytest.mark.asyncio
    async def test_session_saved_to_storage(self):
        """Created session is saved to storage."""
        storage = MockStorage()
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=storage,
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        assert len(storage.saved) == 1
        assert storage.saved[0].session_id == ctx.session_id


# ============================================================================
# 4. get_session
# ============================================================================


class TestGetSession:
    """Test get_session method."""

    @pytest.mark.asyncio
    async def test_get_existing_session(self):
        """get_session returns the session if it exists."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        result = await sm.get_session(ctx.session_id)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self):
        """get_session returns None for unknown session."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        result = await sm.get_session("nonexistent")
        assert result is None


# ============================================================================
# 5. close_session
# ============================================================================


class TestCloseSession:
    """Test close_session method."""

    @pytest.mark.asyncio
    async def test_close_sets_status(self):
        """close_session sets status to 'closed'."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        await sm.close_session(ctx.session_id)
        assert ctx.status == "closed"

    @pytest.mark.asyncio
    async def test_close_removes_from_active(self):
        """close_session removes session from _active_sessions."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        await sm.close_session(ctx.session_id)
        assert ctx.session_id not in sm._active_sessions

    @pytest.mark.asyncio
    async def test_close_removes_queue(self):
        """close_session removes session queue."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        await sm.close_session(ctx.session_id)
        assert ctx.session_id not in sm._session_queues

    @pytest.mark.asyncio
    async def test_close_saves_to_storage(self):
        """close_session saves the updated context to storage."""
        storage = MockStorage()
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=storage,
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        storage.saved.clear()  # Clear the create save
        await sm.close_session(ctx.session_id)
        assert len(storage.saved) == 1
        assert storage.saved[0].status == "closed"

    @pytest.mark.asyncio
    async def test_close_nonexistent_no_error(self):
        """close_session on unknown session does not raise."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        await sm.close_session("nonexistent")  # Should not raise


# ============================================================================
# 6. process_message
# ============================================================================


class TestProcessMessage:
    """Test process_message method."""

    @pytest.mark.asyncio
    async def test_process_returns_future(self):
        """process_message returns an asyncio.Future."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        future = await sm.process_message(ctx.session_id, {"content": "hello"})
        assert isinstance(future, asyncio.Future)

    @pytest.mark.asyncio
    async def test_process_nonexistent_session_raises(self):
        """process_message raises ValueError for unknown session."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        with pytest.raises(ValueError):
            await sm.process_message("nonexistent", {"content": "hello"})

    @pytest.mark.asyncio
    async def test_process_message_queues(self):
        """process_message puts message in session queue."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        future = await sm.process_message(ctx.session_id, {"content": "hello"})
        # Queue should have the message
        assert not sm._session_queues[ctx.session_id].empty()


# ============================================================================
# 7. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self):
        """Multiple sessions are independent."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx1 = await sm.create_session("user-1")
        ctx2 = await sm.create_session("user-2")
        assert ctx1.session_id != ctx2.session_id
        assert ctx1.session_id in sm._active_sessions
        assert ctx2.session_id in sm._active_sessions

    @pytest.mark.asyncio
    async def test_close_then_get_returns_none(self):
        """After closing, get_session returns None."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        await sm.close_session(ctx.session_id)
        result = await sm.get_session(ctx.session_id)
        assert result is None

    @pytest.mark.asyncio
    async def test_close_then_process_raises(self):
        """After closing, process_message raises ValueError."""
        sm = SessionManager(
            memory_factory=_mock_memory_factory,
            runtime=MockRuntime(),
            planner=MagicMock(),
            tools={},
            event_bus=MockEventBus(),
            storage=MockStorage(),
            llm_gateway=MagicMock(),
        )
        ctx = await sm.create_session("user-1")
        await sm.close_session(ctx.session_id)
        with pytest.raises(ValueError):
            await sm.process_message(ctx.session_id, {"content": "hello"})

    def test_generate_id_format(self):
        """generate_id returns UUID format string."""
        sid = generate_id()
        parts = sid.split("-")
        assert len(parts) == 5  # UUID format: 8-4-4-4-12
