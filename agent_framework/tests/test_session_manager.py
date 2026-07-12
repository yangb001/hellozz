"""Tests for core/session_manager.py SessionManager class."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, Callable, Optional

from agent_framework.core.session_manager import SessionManager
from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event


class MockStorage:
    """Mock storage implementation for testing."""
    def __init__(self):
        self.sessions = {}

    async def save(self, ctx: SessionContext):
        self.sessions[ctx.session_id] = ctx

    async def load(self, session_id: str) -> Optional[SessionContext]:
        return self.sessions.get(session_id)

    async def delete(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]


class MockEventBus:
    """Mock event bus implementation for testing."""
    def __init__(self):
        self.published_events = []

    async def publish(self, session_id: str, event: Event):
        self.published_events.append((session_id, event))


class MockRuntime:
    """Mock runtime implementation for testing."""
    def __init__(self, events_to_return=None):
        self.events_to_return = events_to_return or [Event(type="final_answer", content="test response")]
        self.run_called = False

    def run(self, ctx, user_input, memory, tools, planner, llm_gateway):
        self.run_called = True
        return self._generate_events()

    async def _generate_events(self):
        for event in self.events_to_return:
            yield event


class MockMemory:
    """Mock memory implementation for testing."""
    def __init__(self):
        self.saved_messages = []

    async def save(self, session_id: str, message: Message):
        self.saved_messages.append(message)

    async def retrieve(self, session_id: str, query: str, user_ids=None, top_k=5) -> str:
        return "mock retrieved memory"

    async def clear(self, session_id: str):
        pass

    async def extract_long_term(self, session_id: str, force=False):
        pass


class MockPlanner:
    """Mock planner implementation for testing."""
    name = "mock_planner"
    description = "A mock planner for testing"

    def __init__(self, events_to_return=None):
        self.events_to_return = events_to_return or [Event(type="final_answer", content="planned response")]

    async def plan_and_act(self, ctx, memory, tools, llm_call):
        for event in self.events_to_return:
            yield event


def create_mock_memory_factory():
    """Create a factory function that returns MockMemory instances."""
    def factory(session_id: str):
        return MockMemory()
    return factory


class TestSessionManagerCreation:
    """Tests for SessionManager creation and initialization."""

    def test_session_manager_can_be_created(self):
        """Test that SessionManager can be instantiated."""
        storage = MockStorage()
        event_bus = MockEventBus()
        runtime = MockRuntime()
        planner = MockPlanner()
        tools = {}
        llm_gateway = MagicMock()

        manager = SessionManager(
            memory_factory=create_mock_memory_factory(),
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        assert manager is not None

    def test_active_sessions_starts_empty(self):
        """Test that _active_sessions dictionary starts empty."""
        storage = MockStorage()
        event_bus = MockEventBus()
        runtime = MockRuntime()
        planner = MockPlanner()

        manager = SessionManager(
            memory_factory=create_mock_memory_factory(),
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=MagicMock()
        )

        assert len(manager._active_sessions) == 0


class TestCreateSession:
    """Tests for SessionManager.create_session method."""

    def test_create_session_returns_session_context(self):
        """Test that create_session returns a SessionContext."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            return ctx

        ctx = asyncio.run(run())
        assert isinstance(ctx, SessionContext)
        assert ctx.session_id is not None
        assert ctx.session_id != ""

    def test_create_session_adds_to_active_sessions(self):
        """Test that create_session adds session to _active_sessions."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            return ctx.session_id in manager._active_sessions

        result = asyncio.run(run())
        assert result

    def test_create_session_adds_participant(self):
        """Test that create_session adds user as participant."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            return "user123" in ctx.participants

        result = asyncio.run(run())
        assert result

    def test_create_session_stores_in_storage(self):
        """Test that create_session persists session to storage."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            loaded = await storage.load(ctx.session_id)
            return loaded is not None and loaded.session_id == ctx.session_id

        result = asyncio.run(run())
        assert result

    def test_create_multiple_sessions(self):
        """Test creating multiple sessions creates separate sessions."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx1 = await manager.create_session(user_id="user1")
            ctx2 = await manager.create_session(user_id="user2")

            return ctx1.session_id != ctx2.session_id and \
                   ctx1.session_id in manager._active_sessions and \
                   ctx2.session_id in manager._active_sessions

        result = asyncio.run(run())
        assert result


class TestGetSession:
    """Tests for SessionManager.get_session method."""

    def test_get_existing_session(self):
        """Test getting an existing session returns its context."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            created = await manager.create_session(user_id="user123")
            retrieved = await manager.get_session(created.session_id)

            return retrieved is not None and retrieved.session_id == created.session_id

        result = asyncio.run(run())
        assert result

    def test_get_nonexistent_session(self):
        """Test getting a non-existent session returns None."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            result = await manager.get_session("nonexistent-id")
            return result is None

        result = asyncio.run(run())
        assert result


class TestCloseSession:
    """Tests for SessionManager.close_session method."""

    def test_close_session_removes_from_active(self):
        """Test that close_session removes session from _active_sessions."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            await manager.close_session(ctx.session_id)
            return ctx.session_id not in manager._active_sessions

        result = asyncio.run(run())
        assert result

    def test_close_session_updates_status(self):
        """Test that close_session sets status to 'closed'."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            await manager.close_session(ctx.session_id)

            stored = await storage.load(ctx.session_id)
            return stored.status == "closed"

        result = asyncio.run(run())
        assert result


class TestProcessMessage:
    """Tests for SessionManager.process_message method."""

    def test_process_message_returns_future(self):
        """Test that process_message returns a Future."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            future = await manager.process_message(ctx.session_id, {"content": "hello"})
            return isinstance(future, asyncio.Future)

        result = asyncio.run(run())
        assert result

    def test_process_message_awaits_result(self):
        """Test that process_message result can be awaited."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime([Event(type="final_answer", content="response")])
            planner = MockPlanner([Event(type="final_answer", content="planned response")])

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            future = await manager.process_message(ctx.session_id, {"content": "hello"})
            events = await future
            return len(events) > 0

        result = asyncio.run(run())
        assert result

    def test_process_message_nonexistent_session_raises(self):
        """Test that processing message for non-existent session raises error."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            try:
                await manager.process_message("nonexistent-id", {"content": "hello"})
                return False  # Should have raised
            except ValueError as e:
                return "nonexistent-id" in str(e)

        result = asyncio.run(run())
        assert result


class TestSessionManagerActorModel:
    """Tests for SessionManager Actor model implementation."""

    def test_each_session_has_queue(self):
        """Test that each session gets its own queue."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx1 = await manager.create_session(user_id="user1")
            ctx2 = await manager.create_session(user_id="user2")

            return ctx1.session_id in manager._session_queues and \
                   ctx2.session_id in manager._session_queues and \
                   manager._session_queues[ctx1.session_id] is not manager._session_queues[ctx2.session_id]

        result = asyncio.run(run())
        assert result


class TestCrashRecovery:
    """Tests for SessionManager crash recovery logic."""

    def test_resume_session_restores_state(self):
        """Test that resume_session restores session from storage."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            ctx = await manager.create_session(user_id="user123")
            manager._active_sessions.pop(ctx.session_id)
            manager._session_queues.pop(ctx.session_id)

            resumed = await manager.resume_session(ctx.session_id)
            return resumed is not None and resumed.session_id == ctx.session_id

        result = asyncio.run(run())
        assert result

    def test_resume_nonexistent_session_returns_none(self):
        """Test resuming non-existent session returns None."""
        async def run():
            storage = MockStorage()
            event_bus = MockEventBus()
            runtime = MockRuntime()
            planner = MockPlanner()

            manager = SessionManager(
                memory_factory=create_mock_memory_factory(),
                runtime=runtime,
                planner=planner,
                tools={},
                event_bus=event_bus,
                storage=storage,
                llm_gateway=MagicMock()
            )

            result = await manager.resume_session("nonexistent-id")
            return result is None

        result = asyncio.run(run())
        assert result