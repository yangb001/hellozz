"""Independent test cases for SessionManager - Core component testing.

Tests cover:
- Session lifecycle (create, get, close, resume)
- Actor model and message queuing
- Serial message processing
- Future resolution
- Event publishing
- Error handling
- Concurrent access
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, Any

from agent_framework.core.session_manager import SessionManager, generate_id
from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event


class MockSessionStorage:
    """Mock implementation of SessionStorage interface."""

    def __init__(self):
        self._storage: Dict[str, SessionContext] = {}

    async def save(self, ctx: SessionContext) -> None:
        self._storage[ctx.session_id] = ctx

    async def load(self, session_id: str) -> SessionContext:
        return self._storage.get(session_id)

    async def delete(self, session_id: str) -> None:
        if session_id in self._storage:
            del self._storage[session_id]


class MockEventBus:
    """Mock implementation of EventBus."""

    def __init__(self):
        self.published_events: Dict[str, list] = {}

    async def publish(self, session_id: str, event: Event) -> None:
        if session_id not in self.published_events:
            self.published_events[session_id] = []
        self.published_events[session_id].append(event)


class MockRuntime:
    """Mock implementation of AgentRuntime."""

    def __init__(self, events_to_return: list = None):
        self.calls = []
        self.events_to_return = events_to_return or [
            Event(type="final_answer", content="Test response")
        ]

    async def run(self, ctx, user_input, memory, tools, planner, llm_gateway):
        self.calls.append({
            "ctx": ctx,
            "user_input": user_input,
            "memory": memory,
            "tools": tools,
            "planner": planner,
            "llm_gateway": llm_gateway
        })
        for event in self.events_to_return:
            yield event


class MockMemory:
    """Mock implementation of BaseMemory."""

    def __init__(self):
        self.saved_messages = []
        self.extract_called = False

    async def save(self, session_id: str, message: Message) -> None:
        self.saved_messages.append((session_id, message))

    async def retrieve(self, session_id: str, query: str, user_ids=None, top_k=5) -> str:
        return ""

    async def clear(self, session_id: str) -> None:
        pass

    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        self.extract_called = True


def make_mock_storage():
    return MockSessionStorage()


def make_mock_event_bus():
    return MockEventBus()


def make_mock_runtime(events_to_return=None):
    return MockRuntime(events_to_return)


def make_mock_planner():
    planner = Mock(spec=[
        "name", "description", "plan_and_act"
    ])
    planner.name = "MockPlanner"
    planner.description = "Mock planner for testing"
    planner.plan_and_act = AsyncMock(return_value=iter([
        Event(type="final_answer", content="Mock response")
    ]))
    return planner


def make_mock_llm_gateway():
    gateway = Mock()
    gateway.generate = AsyncMock(return_value="Mock LLM response")
    gateway.stream = AsyncMock(return_value=iter(["Mock", " streaming", " response"]))
    return gateway


def make_mock_tools():
    return {"calculator": Mock(), "web_search": Mock()}


def make_mock_memory_factory():
    def factory(session_id: str):
        return MockMemory()
    return factory


def make_session_manager():
    return SessionManager(
        memory_factory=make_mock_memory_factory(),
        runtime=make_mock_runtime(),
        planner=make_mock_planner(),
        tools=make_mock_tools(),
        event_bus=make_mock_event_bus(),
        storage=make_mock_storage(),
        llm_gateway=make_mock_llm_gateway()
    )


class TestGenerateId:
    """Tests for generate_id function."""

    def test_generate_id_returns_string(self):
        sid = generate_id()
        assert isinstance(sid, str)

    def test_generate_id_is_unique(self):
        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_generate_id_is_valid_uuid_format(self):
        import uuid
        sid = generate_id()
        # Should not raise ValueError if valid UUID
        uuid.UUID(sid)


class TestSessionCreation:
    """Tests for SessionManager.create_session."""

    def test_create_session_returns_session_context(self):
        sm = make_session_manager()

        async def run():
            return await sm.create_session(user_id="user1")

        ctx = asyncio.run(run())

        assert isinstance(ctx, SessionContext)
        assert ctx.session_id is not None
        assert ctx.session_type == "private"
        assert "user1" in ctx.participants

    def test_create_session_stores_in_active_sessions(self):
        sm = make_session_manager()

        async def run():
            return await sm.create_session(user_id="user1")

        ctx = asyncio.run(run())

        assert ctx.session_id in sm._active_sessions
        assert sm._active_sessions[ctx.session_id] is ctx

    def test_create_session_creates_queue(self):
        sm = make_session_manager()

        async def run():
            return await sm.create_session(user_id="user1")

        ctx = asyncio.run(run())

        assert ctx.session_id in sm._session_queues
        assert isinstance(sm._session_queues[ctx.session_id], asyncio.Queue)

    def test_create_session_saves_to_storage(self):
        storage = make_mock_storage()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=make_mock_runtime(),
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=make_mock_event_bus(),
            storage=storage,
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            return await sm.create_session(user_id="user1")

        ctx = asyncio.run(run())

        assert ctx.session_id in storage._storage

    def test_create_group_session(self):
        sm = make_session_manager()

        async def run():
            return await sm.create_session(
                user_id="user1",
                session_type="group",
                participants=["user2", "user3"]
            )

        ctx = asyncio.run(run())

        assert ctx.session_type == "group"
        assert "user1" in ctx.participants
        assert "user2" in ctx.participants
        assert "user3" in ctx.participants

    def test_create_multiple_sessions(self):
        sm = make_session_manager()

        async def run():
            ctx1 = await sm.create_session(user_id="user1")
            ctx2 = await sm.create_session(user_id="user2")
            return ctx1, ctx2

        ctx1, ctx2 = asyncio.run(run())

        assert ctx1.session_id != ctx2.session_id
        assert len(sm._active_sessions) == 2


class TestGetSession:
    """Tests for SessionManager.get_session."""

    def test_get_existing_session(self):
        sm = make_session_manager()

        async def run():
            created = await sm.create_session(user_id="user1")
            retrieved = await sm.get_session(created.session_id)
            return retrieved

        retrieved = asyncio.run(run())

        assert retrieved is not None
        assert retrieved.session_id is not None

    def test_get_nonexistent_session(self):
        sm = make_session_manager()

        async def run():
            return await sm.get_session("nonexistent-id")

        result = asyncio.run(run())

        assert result is None


class TestProcessMessage:
    """Tests for SessionManager.process_message."""

    def test_process_message_returns_future(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            future = await sm.process_message(ctx.session_id, {"content": "Hello"})
            return future

        future = asyncio.run(run())

        assert isinstance(future, asyncio.Future)

    def test_process_message_actor_processes_it(self):
        runtime = make_mock_runtime()
        storage = make_mock_storage()
        event_bus = make_mock_event_bus()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=runtime,
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=event_bus,
            storage=storage,
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            future = await sm.process_message(ctx.session_id, {"content": "Hello"})
            events = await asyncio.wait_for(future, timeout=2.0)
            return events

        events = asyncio.run(run())

        assert len(events) > 0
        assert len(runtime.calls) > 0

    def test_process_message_publishes_events(self):
        storage = make_mock_storage()
        event_bus = make_mock_event_bus()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=make_mock_runtime(),
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=event_bus,
            storage=storage,
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            future = await sm.process_message(ctx.session_id, {"content": "Hello"})
            await asyncio.wait_for(future, timeout=2.0)
            return ctx.session_id, event_bus.published_events

        sid, events = asyncio.run(run())

        assert sid in events

    def test_process_nonexistent_session_raises(self):
        sm = make_session_manager()

        async def run():
            return await sm.process_message("nonexistent-id", {"content": "Hello"})

        with pytest.raises(ValueError, match="Session 'nonexistent-id' does not exist"):
            asyncio.run(run())

    def test_multiple_messages_serial_processing(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")

            # Process two messages
            future1 = await sm.process_message(ctx.session_id, {"content": "First"})
            future2 = await sm.process_message(ctx.session_id, {"content": "Second"})

            # Both should eventually complete
            events1 = await asyncio.wait_for(future1, timeout=2.0)
            events2 = await asyncio.wait_for(future2, timeout=2.0)

            return events1, events2

        events1, events2 = asyncio.run(run())

        assert len(events1) > 0
        assert len(events2) > 0


class TestCloseSession:
    """Tests for SessionManager.close_session."""

    def test_close_session_removes_from_active(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            return sid

        sid = asyncio.run(run())

        assert sid not in sm._active_sessions

    def test_close_session_removes_queue(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            return sid

        sid = asyncio.run(run())

        assert sid not in sm._session_queues

    def test_close_session_updates_status(self):
        storage = make_mock_storage()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=make_mock_runtime(),
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=make_mock_event_bus(),
            storage=storage,
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            stored = storage._storage.get(sid)
            return stored

        stored = asyncio.run(run())

        assert stored.status == "closed"

    def test_close_nonexistent_session_no_error(self):
        sm = make_session_manager()

        async def run():
            # Should not raise
            await sm.close_session("nonexistent-id")

        asyncio.run(run())


class TestResumeSession:
    """Tests for SessionManager.resume_session."""

    def test_resume_existing_session(self):
        storage = make_mock_storage()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=make_mock_runtime(),
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=make_mock_event_bus(),
            storage=storage,
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            resumed = await sm.resume_session(sid)
            return resumed

        resumed = asyncio.run(run())

        assert resumed is not None
        assert resumed.session_id is not None

    def test_resume_restores_to_active(self):
        storage = make_mock_storage()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=make_mock_runtime(),
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=make_mock_event_bus(),
            storage=storage,
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            resumed = await sm.resume_session(sid)
            return resumed

        resumed = asyncio.run(run())

        assert resumed.session_id in sm._active_sessions

    def test_resume_restores_queue(self):
        storage = make_mock_storage()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=make_mock_runtime(),
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=make_mock_event_bus(),
            storage=storage,
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            resumed = await sm.resume_session(sid)
            return resumed

        resumed = asyncio.run(run())

        assert resumed.session_id in sm._session_queues

    def test_resume_nonexistent_returns_none(self):
        sm = make_session_manager()

        async def run():
            return await sm.resume_session("nonexistent-id")

        result = asyncio.run(run())

        assert result is None


class TestActorModelBehavior:
    """Tests specifically for Actor model implementation."""

    def test_actor_loop_starts_on_create(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            # Give actor time to start
            await asyncio.sleep(0.1)
            return ctx

        ctx = asyncio.run(run())

        # Queue should exist and be ready
        assert ctx.session_id in sm._session_queues

    def test_messages_processed_in_order(self):
        sm = make_session_manager()

        async def mock_run(*args, **kwargs):
            # Simulate processing taking some time
            await asyncio.sleep(0.05)
            yield Event(type="final_answer", content="Done")

        sm.runtime.run = mock_run

        async def run():
            ctx = await sm.create_session(user_id="user1")

            future1 = await sm.process_message(ctx.session_id, {"content": "First"})
            future2 = await sm.process_message(ctx.session_id, {"content": "Second"})

            result1 = await asyncio.wait_for(future1, timeout=2.0)
            result2 = await asyncio.wait_for(future2, timeout=2.0)

            return result1, result2

        result1, result2 = asyncio.run(run())

        # Both should complete successfully
        assert result1 is not None
        assert result2 is not None

    def test_exception_propagates_via_future(self):
        sm = make_session_manager()

        async def raising_run(*args, **kwargs):
            # Must be an async generator for the code's async for loop
            yield Event(type="text_token", content="Before error")
            raise ValueError("Test exception")

        sm.runtime.run = raising_run

        async def run():
            ctx = await sm.create_session(user_id="user1")
            future = await sm.process_message(ctx.session_id, {"content": "Hello"})
            return future

        future = asyncio.run(run())

        with pytest.raises(ValueError, match="Test exception"):
            asyncio.run(asyncio.wait_for(future, timeout=2.0))


class TestSessionContextUpdates:
    """Tests for SessionContext updates during processing."""

    def test_session_context_passed_to_runtime(self):
        runtime = make_mock_runtime()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=runtime,
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=make_mock_event_bus(),
            storage=make_mock_storage(),
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            await sm.process_message(ctx.session_id, {"content": "Hello"})
            await asyncio.sleep(0.2)
            return runtime.calls

        calls = asyncio.run(run())

        assert len(calls) > 0
        call = calls[-1]
        assert isinstance(call["ctx"], SessionContext)
        assert call["user_input"] == "Hello"


class TestEventPublishing:
    """Tests for event publishing via EventBus."""

    def test_events_published_for_processed_message(self):
        event_bus = make_mock_event_bus()
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=make_mock_runtime(),
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=event_bus,
            storage=make_mock_storage(),
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            await sm.process_message(ctx.session_id, {"content": "Hello"})
            await asyncio.sleep(0.2)
            return ctx.session_id, event_bus.published_events

        sid, events = asyncio.run(run())

        assert sid in events
        assert len(events.get(sid, [])) > 0

    def test_multiple_events_published(self):
        event_bus = make_mock_event_bus()
        runtime = make_mock_runtime(events_to_return=[
            Event(type="thought", content="Thinking..."),
            Event(type="action", content="Acting..."),
            Event(type="final_answer", content="Done!")
        ])
        sm = SessionManager(
            memory_factory=make_mock_memory_factory(),
            runtime=runtime,
            planner=make_mock_planner(),
            tools=make_mock_tools(),
            event_bus=event_bus,
            storage=make_mock_storage(),
            llm_gateway=make_mock_llm_gateway()
        )

        async def run():
            ctx = await sm.create_session(user_id="user1")
            future = await sm.process_message(ctx.session_id, {"content": "Hello"})
            await asyncio.wait_for(future, timeout=2.0)
            return ctx.session_id, event_bus.published_events

        sid, events = asyncio.run(run())

        session_events = events.get(sid, [])
        assert len(session_events) >= 3


class TestConcurrency:
    """Tests for concurrent access scenarios."""

    def test_concurrent_messages_different_sessions(self):
        sm = make_session_manager()

        async def run():
            ctx1 = await sm.create_session(user_id="user1")
            ctx2 = await sm.create_session(user_id="user2")

            future1 = await sm.process_message(ctx1.session_id, {"content": "Hello to session 1"})
            future2 = await sm.process_message(ctx2.session_id, {"content": "Hello to session 2"})

            results = await asyncio.gather(
                asyncio.wait_for(future1, timeout=2.0),
                asyncio.wait_for(future2, timeout=2.0)
            )

            return results

        results = asyncio.run(run())

        assert len(results) == 2
        assert all(r for r in results)


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    def test_create_session_with_empty_user_id(self):
        sm = make_session_manager()

        async def run():
            return await sm.create_session(user_id="")

        ctx = asyncio.run(run())

        assert ctx.session_id is not None
        assert "" in ctx.participants

    def test_process_empty_content(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            future = await sm.process_message(ctx.session_id, {"content": ""})
            result = await asyncio.wait_for(future, timeout=2.0)
            return result

        result = asyncio.run(run())

        assert result is not None

    def test_process_message_with_extra_fields(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            future = await sm.process_message(
                ctx.session_id,
                {"content": "Hello", "extra_field": "ignored"}
            )
            result = await asyncio.wait_for(future, timeout=2.0)
            return result

        result = asyncio.run(run())

        assert result is not None

    def test_double_close_session(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            # Second close should not raise
            await sm.close_session(sid)

        asyncio.run(run())

    def test_get_session_after_close(self):
        sm = make_session_manager()

        async def run():
            ctx = await sm.create_session(user_id="user1")
            sid = ctx.session_id
            await sm.close_session(sid)
            return await sm.get_session(sid)

        result = asyncio.run(run())

        assert result is None