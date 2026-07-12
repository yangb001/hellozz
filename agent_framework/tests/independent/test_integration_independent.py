"""Independent end-to-end integration tests for the Agent Framework.

This module contains independent verification tests for the complete
flow from session creation to response generation, verifying that all
components work together correctly.

Test categories:
1. Session lifecycle (create, get, close)
2. Message processing through SessionManager
3. AgentRuntime and Planner integration
4. Tool execution in agent flow
5. Memory integration (save, retrieve)
6. Event streaming and EventBus
7. Error handling across components
8. Multiple sessions and isolation
"""
import pytest
import asyncio
from typing import List, Dict, Any, AsyncIterator

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.runtime.agent_runtime import AgentRuntime
from agent_framework.core.session_manager import SessionManager


# ============================================================================
# Mock Implementations
# ============================================================================


class MockMemory(BaseMemory):
    """Mock memory that tracks all operations."""

    def __init__(self):
        self.saved: Dict[str, List[Message]] = {}
        self.retrieved: List[tuple] = []
        self.cleared: List[str] = []

    async def save(self, session_id: str, message: Message) -> None:
        if session_id not in self.saved:
            self.saved[session_id] = []
        self.saved[session_id].append(message)

    async def retrieve(self, session_id: str, query: str, user_ids=None, top_k=5) -> str:
        self.retrieved.append((session_id, query))
        return f"context_for:{query}"

    async def clear(self, session_id: str) -> None:
        self.cleared.append(session_id)
        self.saved.pop(session_id, None)

    async def extract_long_term(self, session_id: str, force=False) -> None:
        pass


class MockLLMGateway:
    """Mock LLM gateway."""

    def __init__(self, response: str = "mock llm response"):
        self.response = response
        self.calls: List[str] = []

    async def generate(self, prompt: str, model="default", **kwargs) -> str:
        self.calls.append(prompt)
        return self.response

    async def stream(self, prompt: str, model="default", **kwargs) -> AsyncIterator[str]:
        self.calls.append(prompt)
        for char in self.response:
            yield char


class MockStorage:
    """Mock session storage."""

    def __init__(self):
        self.saved: Dict[str, SessionContext] = {}

    async def save(self, ctx: SessionContext) -> None:
        self.saved[ctx.session_id] = ctx

    async def load(self, session_id: str):
        return self.saved.get(session_id)

    async def delete(self, session_id: str) -> None:
        self.saved.pop(session_id, None)


class MockEventBus:
    """Mock event bus."""

    def __init__(self):
        self.published: List[tuple] = []

    async def publish(self, session_id: str, event: Event) -> None:
        self.published.append((session_id, event))


class SimplePlanner(BasePlanner):
    """Simple planner that yields a fixed sequence of events."""

    def __init__(self, events=None, final_answer="The answer is 42."):
        self.events = events
        self.final_answer = final_answer
        self.calls: List[Dict] = []

    async def plan_and_act(self, ctx, memory, tools, llm_call):
        self.calls.append({"session_id": ctx.session_id, "msg_count": len(ctx.messages)})
        if self.events is not None:
            for event in self.events:
                yield event
        else:
            yield Event(type="thought", content="Processing request")
            yield Event(type="final_answer", content=self.final_answer)


class ToolUsingPlanner(BasePlanner):
    """Planner that uses a calculator tool."""

    async def plan_and_act(self, ctx, memory, tools, llm_call):
        yield Event(type="thought", content="Need to calculate")
        if "calculator" in tools:
            result = await tools["calculator"].run("2+2")
            yield Event(type="observation", content=f"Result: {result}")
        yield Event(type="final_answer", content="Calculation done")


class MockCalculator:
    """Mock calculator tool."""

    name = "calculator"
    description = "Basic calculator"
    call_count = 0

    async def run(self, input: str, session_id=None, **kwargs) -> str:
        MockCalculator.call_count += 1
        try:
            return str(eval(input))
        except Exception as e:
            return f"Error: {e}"


def _build_manager(memory=None, planner=None, tools=None, events=None):
    """Helper to build a fully wired SessionManager."""
    mem = memory or MockMemory()
    pl = planner or SimplePlanner(final_answer="default answer")
    mgr = SessionManager(
        memory_factory=lambda sid: mem,
        runtime=AgentRuntime(),
        planner=pl,
        tools=tools or {},
        event_bus=MockEventBus(),
        storage=MockStorage(),
        llm_gateway=MockLLMGateway(),
    )
    return mgr, mem, mgr.event_bus, mgr.storage


# ============================================================================
# 1. Session Lifecycle
# ============================================================================


class TestSessionLifecycle:
    """Test session creation, retrieval, and closure."""

    @pytest.mark.asyncio
    async def test_create_and_retrieve_session(self):
        """Create session then retrieve it by ID."""
        mgr, _, _, _ = _build_manager()
        ctx = await mgr.create_session("user-1")
        assert ctx.session_id is not None
        retrieved = await mgr.get_session(ctx.session_id)
        assert retrieved is ctx
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_create_group_session(self):
        """Create a group session with participants."""
        mgr, _, _, _ = _build_manager()
        ctx = await mgr.create_session("user-1", session_type="group", participants=["user-2", "user-3"])
        assert ctx.session_type == "group"
        assert "user-1" in ctx.participants
        assert "user-2" in ctx.participants
        assert "user-3" in ctx.participants
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_close_session_removes_from_active(self):
        """Closed session is no longer retrievable."""
        mgr, _, _, _ = _build_manager()
        ctx = await mgr.create_session("user-1")
        await mgr.close_session(ctx.session_id)
        assert await mgr.get_session(ctx.session_id) is None

    @pytest.mark.asyncio
    async def test_close_session_persists_status(self):
        """Closed session is saved with 'closed' status."""
        mgr, _, _, storage = _build_manager()
        ctx = await mgr.create_session("user-1")
        await mgr.close_session(ctx.session_id)
        saved = storage.saved.get(ctx.session_id)
        assert saved is not None
        assert saved.status == "closed"

    @pytest.mark.asyncio
    async def test_get_nonexistent_session(self):
        """Getting nonexistent session returns None."""
        mgr, _, _, _ = _build_manager()
        assert await mgr.get_session("no-such-id") is None


# ============================================================================
# 2. Message Processing
# ============================================================================


class TestMessageProcessing:
    """Test message processing through SessionManager."""

    @pytest.mark.asyncio
    async def test_process_message_returns_events(self):
        """Processing a message returns a list of events."""
        mgr, _, _, _ = _build_manager()
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "hello"})
        events = await asyncio.wait_for(future, timeout=5.0)
        assert len(events) > 0
        assert all(isinstance(e, Event) for e in events)
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_process_message_generates_final_answer(self):
        """Processing generates a final_answer event."""
        mgr, _, _, _ = _build_manager()
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "question"})
        events = await asyncio.wait_for(future, timeout=5.0)
        final = [e for e in events if e.type == "final_answer"]
        assert len(final) == 1
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_process_nonexistent_session_raises(self):
        """Processing message for nonexistent session raises ValueError."""
        mgr, _, _, _ = _build_manager()
        with pytest.raises(ValueError):
            await mgr.process_message("no-such-id", {"role": "user", "content": "hi"})

    @pytest.mark.asyncio
    async def test_process_message_saves_to_storage(self):
        """Processing saves the session to storage."""
        mgr, _, _, storage = _build_manager()
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "hi"})
        await asyncio.wait_for(future, timeout=5.0)
        assert ctx.session_id in storage.saved
        await mgr.close_session(ctx.session_id)


# ============================================================================
# 3. AgentRuntime and Planner Integration
# ============================================================================


class TestRuntimePlannerIntegration:
    """Test AgentRuntime and Planner working together."""

    @pytest.mark.asyncio
    async def test_planner_receives_context(self):
        """Planner receives the session context with user message."""
        planner = SimplePlanner()
        mgr, _, _, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "test"})
        await asyncio.wait_for(future, timeout=5.0)
        assert len(planner.calls) == 1
        assert planner.calls[0]["msg_count"] >= 1
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_planner_events_yielded_in_order(self):
        """Events from planner are yielded in correct order."""
        events = [
            Event(type="thought", content="step 1"),
            Event(type="thought", content="step 2"),
            Event(type="final_answer", content="done"),
        ]
        planner = SimplePlanner(events=events)
        mgr, _, _, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "q"})
        result = await asyncio.wait_for(future, timeout=5.0)
        assert len(result) == 3
        assert result[0].content == "step 1"
        assert result[1].content == "step 2"
        assert result[2].content == "done"
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_context_updated_after_processing(self):
        """Session context messages are updated after processing."""
        planner = SimplePlanner(final_answer="response")
        mgr, _, _, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        initial_count = len(ctx.messages)
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "hi"})
        await asyncio.wait_for(future, timeout=5.0)
        assert len(ctx.messages) > initial_count
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_user_and_assistant_messages_in_context(self):
        """Both user and assistant messages appear in context after processing."""
        planner = SimplePlanner(final_answer="reply")
        mgr, _, _, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "question"})
        await asyncio.wait_for(future, timeout=5.0)
        roles = [m.role for m in ctx.messages]
        assert "user" in roles
        assert "assistant" in roles
        await mgr.close_session(ctx.session_id)


# ============================================================================
# 4. Tool Execution
# ============================================================================


class TestToolExecution:
    """Test tool execution in agent flow."""

    @pytest.mark.asyncio
    async def test_tool_called_during_planning(self):
        """Tool is called when planner uses it."""
        MockCalculator.call_count = 0
        planner = ToolUsingPlanner()
        tools = {"calculator": MockCalculator()}
        mgr, _, _, _ = _build_manager(planner=planner, tools=tools)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "calc"})
        events = await asyncio.wait_for(future, timeout=5.0)
        assert MockCalculator.call_count > 0
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_tool_observation_in_events(self):
        """Tool result appears as observation event."""
        planner = ToolUsingPlanner()
        tools = {"calculator": MockCalculator()}
        mgr, _, _, _ = _build_manager(planner=planner, tools=tools)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "calc"})
        events = await asyncio.wait_for(future, timeout=5.0)
        observations = [e for e in events if e.type == "observation"]
        assert len(observations) > 0
        assert "4" in observations[0].content
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_planner_without_tools(self):
        """Planner works when no tools are available."""
        planner = SimplePlanner(final_answer="no tools needed")
        mgr, _, _, _ = _build_manager(planner=planner, tools={})
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "hi"})
        events = await asyncio.wait_for(future, timeout=5.0)
        assert any(e.type == "final_answer" for e in events)
        await mgr.close_session(ctx.session_id)


# ============================================================================
# 5. Memory Integration
# ============================================================================


class TestMemoryIntegration:
    """Test memory system integration."""

    @pytest.mark.asyncio
    async def test_user_message_saved_to_memory(self):
        """User message is saved to memory during processing."""
        memory = MockMemory()
        planner = SimplePlanner()
        mgr, _, _, _ = _build_manager(memory=memory, planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "hello"})
        await asyncio.wait_for(future, timeout=5.0)
        assert ctx.session_id in memory.saved
        user_msgs = [m for m in memory.saved[ctx.session_id] if m.role == "user"]
        assert any(m.content == "hello" for m in user_msgs)
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_assistant_message_saved_to_memory(self):
        """Assistant (final_answer) message is saved to memory."""
        memory = MockMemory()
        planner = SimplePlanner(final_answer="the response")
        mgr, _, _, _ = _build_manager(memory=memory, planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "q"})
        await asyncio.wait_for(future, timeout=5.0)
        assistant_msgs = [m for m in memory.saved[ctx.session_id] if m.role == "assistant"]
        assert any(m.content == "the response" for m in assistant_msgs)
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_memory_used_by_planner(self):
        """Planner can use memory for retrieval."""
        memory = MockMemory()

        class MemoryUsingPlanner(BasePlanner):
            async def plan_and_act(self, ctx, memory, tools, llm_call):
                context = await memory.retrieve(ctx.session_id, "prior context")
                yield Event(type="thought", content=f"Using: {context}")
                yield Event(type="final_answer", content="done")

        planner = MemoryUsingPlanner()
        mgr, _, _, _ = _build_manager(memory=memory, planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "q"})
        await asyncio.wait_for(future, timeout=5.0)
        assert len(memory.retrieved) > 0
        assert "prior context" in memory.retrieved[0][1]
        await mgr.close_session(ctx.session_id)


# ============================================================================
# 6. Event Streaming and EventBus
# ============================================================================


class TestEventStreaming:
    """Test event streaming and EventBus integration."""

    @pytest.mark.asyncio
    async def test_all_events_published_to_bus(self):
        """All events are published to the event bus."""
        planner = SimplePlanner(final_answer="answer")
        mgr, _, event_bus, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "q"})
        events = await asyncio.wait_for(future, timeout=5.0)
        assert len(event_bus.published) == len(events)
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_event_bus_session_id_consistency(self):
        """All published events have the correct session ID."""
        planner = SimplePlanner()
        mgr, _, event_bus, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "q"})
        await asyncio.wait_for(future, timeout=5.0)
        for sid, event in event_bus.published:
            assert sid == ctx.session_id
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_event_types_include_final_answer(self):
        """Published events include final_answer type."""
        planner = SimplePlanner(final_answer="done")
        mgr, _, event_bus, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "q"})
        await asyncio.wait_for(future, timeout=5.0)
        types = [e.type for _, e in event_bus.published]
        assert "final_answer" in types
        await mgr.close_session(ctx.session_id)


# ============================================================================
# 7. Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling across components."""

    @pytest.mark.asyncio
    async def test_planner_error_propagates(self):
        """Error in planner propagates to the future."""
        class FailingPlanner(BasePlanner):
            async def plan_and_act(self, ctx, memory, tools, llm_call):
                yield Event(type="thought", content="starting")
                raise RuntimeError("planner failure")

        planner = FailingPlanner()
        mgr, _, _, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        future = await mgr.process_message(ctx.session_id, {"role": "user", "content": "q"})
        with pytest.raises(RuntimeError, match="planner failure"):
            await asyncio.wait_for(future, timeout=5.0)
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_invalid_session_process_raises(self):
        """Processing message for invalid session raises ValueError."""
        mgr, _, _, _ = _build_manager()
        with pytest.raises(ValueError):
            await mgr.process_message("invalid-id", {"role": "user", "content": "hi"})

    @pytest.mark.asyncio
    async def test_close_nonexistent_session_no_error(self):
        """Closing a nonexistent session does not raise."""
        mgr, _, _, _ = _build_manager()
        await mgr.close_session("nonexistent")  # Should not raise


# ============================================================================
# 8. Multiple Sessions and Isolation
# ============================================================================


class TestMultipleSessions:
    """Test multiple sessions and their isolation."""

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent(self):
        """Multiple sessions are independent of each other."""
        planner = SimplePlanner(final_answer="ok")
        mgr, _, _, _ = _build_manager(planner=planner)
        ctx1 = await mgr.create_session("user-1")
        ctx2 = await mgr.create_session("user-2")
        assert ctx1.session_id != ctx2.session_id
        await mgr.close_session(ctx1.session_id)
        assert await mgr.get_session(ctx2.session_id) is ctx2
        await mgr.close_session(ctx2.session_id)

    @pytest.mark.asyncio
    async def test_multiple_messages_serial_per_session(self):
        """Multiple messages on same session are processed serially."""
        planner = SimplePlanner(final_answer="ok")
        mgr, _, _, _ = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        futures = []
        for i in range(3):
            f = await mgr.process_message(ctx.session_id, {"role": "user", "content": f"msg-{i}"})
            futures.append(f)
        for f in futures:
            events = await asyncio.wait_for(f, timeout=5.0)
            assert len(events) > 0
        assert len(ctx.messages) >= 6  # 3 user + 3 assistant
        await mgr.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_memory_isolation_between_sessions(self):
        """Memory is isolated between different sessions."""
        memory = MockMemory()
        planner = SimplePlanner(final_answer="ok")
        mgr, _, _, _ = _build_manager(memory=memory, planner=planner)
        ctx1 = await mgr.create_session("user-1")
        ctx2 = await mgr.create_session("user-2")
        f1 = await mgr.process_message(ctx1.session_id, {"role": "user", "content": "msg1"})
        f2 = await mgr.process_message(ctx2.session_id, {"role": "user", "content": "msg2"})
        await asyncio.wait_for(f1, timeout=5.0)
        await asyncio.wait_for(f2, timeout=5.0)
        assert ctx1.session_id in memory.saved
        assert ctx2.session_id in memory.saved
        assert memory.saved[ctx1.session_id] != memory.saved[ctx2.session_id]
        await mgr.close_session(ctx1.session_id)
        await mgr.close_session(ctx2.session_id)

    @pytest.mark.asyncio
    async def test_complete_flow_close_session(self):
        """Complete flow: create -> message -> verify -> close."""
        planner = SimplePlanner(final_answer="final response")
        mgr, memory, event_bus, storage = _build_manager(planner=planner)
        ctx = await mgr.create_session("user-1")
        sid = ctx.session_id
        future = await mgr.process_message(sid, {"role": "user", "content": "question"})
        events = await asyncio.wait_for(future, timeout=5.0)
        assert len(events) > 0
        assert any(e.type == "final_answer" for e in events)
        assert sid in memory.saved
        assert len(event_bus.published) == len(events)
        assert sid in storage.saved
        await mgr.close_session(sid)
        assert ctx.status == "closed"
        assert await mgr.get_session(sid) is None
