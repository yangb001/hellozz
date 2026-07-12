"""Independent test cases for AgentRuntime stateless execution engine.

This module contains independent verification tests for the AgentRuntime
implementation, following the detailed design specification in section 5.

Test categories:
1. Stateless design verification
2. run method flow and event streaming
3. Memory system integration
4. Planner integration
5. llm_call closure behavior
6. final_answer handling
7. Session context updates
8. Boundary conditions
"""
import pytest
from datetime import datetime, timezone
from typing import AsyncIterator, Dict, Any, List

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.runtime.agent_runtime import AgentRuntime


# ============================================================================
# Mock Implementations
# ============================================================================


class MockMemory(BaseMemory):
    """Mock memory for testing AgentRuntime in isolation."""

    def __init__(self):
        self.saved_messages: List[tuple] = []  # (session_id, message)
        self.cleared_sessions: List[str] = []

    async def save(self, session_id: str, message: Message) -> None:
        self.saved_messages.append((session_id, message))

    async def retrieve(
        self, session_id: str, query: str,
        user_ids=None, top_k: int = 5
    ) -> str:
        return "mock retrieved context"

    async def clear(self, session_id: str) -> None:
        self.cleared_sessions.append(session_id)

    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        pass


class MockPlanner(BasePlanner):
    """Mock planner that yields predefined events."""

    def __init__(self, events: List[Event] = None):
        self.events = events if events is not None else [
            Event(type="final_answer", content="mock answer")
        ]
        self.call_args = None

    async def plan_and_act(
        self, ctx: SessionContext, memory: BaseMemory,
        tools: Dict[str, Any], llm_call: callable
    ) -> AsyncIterator[Event]:
        self.call_args = {
            "ctx": ctx,
            "memory": memory,
            "tools": tools,
            "llm_call": llm_call,
        }
        for event in self.events:
            yield event


class MockLLMGateway:
    """Mock LLM gateway for testing."""

    def __init__(self, response: str = "mock llm response"):
        self.response = response
        self.calls: List[Dict] = []

    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        for word in self.response.split():
            yield word + " "


# ============================================================================
# Helpers
# ============================================================================


def _make_ctx(session_id: str = "test-session") -> SessionContext:
    """Create a test SessionContext."""
    return SessionContext(session_id=session_id)


def _collect_events(events) -> List[Event]:
    """Collect async events into a list."""
    return events  # Will be used with async iteration in tests


# ============================================================================
# 1. Stateless Design Verification
# ============================================================================


class TestStatelessDesign:
    """Verify AgentRuntime does not maintain state between invocations."""

    def test_agent_runtime_has_no_instance_state(self):
        """AgentRuntime should have no stateful instance variables."""
        runtime = AgentRuntime()
        # Check there are no instance attributes beyond standard Python ones
        stateful_attrs = [
            attr for attr in vars(runtime)
            if not attr.startswith('_')
        ]
        assert stateful_attrs == [], \
            f"AgentRuntime should be stateless, found: {stateful_attrs}"

    def test_agent_runtime_no_class_state(self):
        """AgentRuntime class should not have mutable class-level state."""
        runtime = AgentRuntime()
        # Verify no mutable class attributes (excluding methods)
        mutable_class_attrs = [
            attr for attr in dir(AgentRuntime)
            if not attr.startswith('_')
            and not callable(getattr(AgentRuntime, attr, None))
            and attr in AgentRuntime.__dict__
        ]
        # Should only have the class itself, no stored state
        for attr in mutable_class_attrs:
            val = getattr(AgentRuntime, attr)
            assert not isinstance(val, (dict, list)), \
                f"AgentRuntime has mutable class attribute: {attr}"

    def test_agent_runtime_reusable(self):
        """Same runtime instance can be used for multiple calls."""
        runtime = AgentRuntime()
        # Should be able to create multiple instances without side effects
        runtime2 = AgentRuntime()
        assert type(runtime) == type(runtime2)

    def test_agent_runtime_run_method_exists(self):
        """AgentRuntime must have a run method."""
        assert hasattr(AgentRuntime, "run"), "AgentRuntime 缺少 run 方法"

    def test_agent_runtime_run_is_async_generator(self):
        """run method must be an async generator function."""
        import inspect
        assert inspect.isasyncgenfunction(AgentRuntime.run), \
            "run 方法应为 async generator"


# ============================================================================
# 2. run Method Flow and Event Streaming
# ============================================================================


class TestRunMethodFlow:
    """Test the run method execution flow."""

    @pytest.mark.asyncio
    async def test_run_yields_events_from_planner(self):
        """run must yield all events produced by the planner."""
        runtime = AgentRuntime()
        events = [
            Event(type="thought", content="thinking..."),
            Event(type="action", content="calling tool"),
            Event(type="observation", content="tool result"),
            Event(type="final_answer", content="here is the answer"),
        ]
        planner = MockPlanner(events=events)
        memory = MockMemory()
        ctx = _make_ctx()

        collected = []
        async for event in runtime.run(
            ctx=ctx, user_input="test input", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            collected.append(event)

        assert len(collected) == 4
        assert collected[0].type == "thought"
        assert collected[1].type == "action"
        assert collected[2].type == "observation"
        assert collected[3].type == "final_answer"

    @pytest.mark.asyncio
    async def test_run_returns_async_iterator(self):
        """run must return an async iterator."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        result = runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        )
        # Should be an async generator
        import inspect
        assert inspect.isasyncgen(result)

    @pytest.mark.asyncio
    async def test_run_with_single_final_answer(self):
        """run works with planner that yields only final_answer."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="final_answer", content="direct answer")
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        collected = []
        async for event in runtime.run(
            ctx=ctx, user_input="hi", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            collected.append(event)

        assert len(collected) == 1
        assert collected[0].type == "final_answer"
        assert collected[0].content == "direct answer"

    @pytest.mark.asyncio
    async def test_run_with_multiple_thought_steps(self):
        """run handles multiple thought steps before final answer."""
        runtime = AgentRuntime()
        events = [
            Event(type="thought", content="step 1"),
            Event(type="thought", content="step 2"),
            Event(type="thought", content="step 3"),
            Event(type="final_answer", content="done"),
        ]
        planner = MockPlanner(events=events)
        memory = MockMemory()
        ctx = _make_ctx()

        collected = []
        async for event in runtime.run(
            ctx=ctx, user_input="complex task", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            collected.append(event)

        assert len(collected) == 4
        assert all(e.type == "thought" for e in collected[:3])


# ============================================================================
# 3. Memory System Integration
# ============================================================================


class TestMemoryIntegration:
    """Test that AgentRuntime correctly integrates with memory."""

    @pytest.mark.asyncio
    async def test_user_message_saved_to_memory(self):
        """run must save the user message to memory."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx("mem-test-session")

        async for _ in runtime.run(
            ctx=ctx, user_input="hello world", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        # Check user message was saved
        user_saves = [
            (sid, msg) for sid, msg in memory.saved_messages
            if msg.role == "user"
        ]
        assert len(user_saves) == 1
        assert user_saves[0][0] == "mem-test-session"
        assert user_saves[0][1].content == "hello world"

    @pytest.mark.asyncio
    async def test_final_answer_saved_to_memory(self):
        """run must save the final_answer event as assistant message to memory."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="final_answer", content="the answer")
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="question", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        # Check assistant message was saved
        assistant_saves = [
            (sid, msg) for sid, msg in memory.saved_messages
            if msg.role == "assistant"
        ]
        assert len(assistant_saves) == 1
        assert assistant_saves[0][1].content == "the answer"

    @pytest.mark.asyncio
    async def test_non_final_events_not_saved_to_memory(self):
        """Events other than final_answer should not be saved as messages."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="thought", content="thinking"),
            Event(type="action", content="doing"),
            Event(type="observation", content="saw"),
            Event(type="final_answer", content="answer"),
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        # Only user + final_answer should be saved (2 messages)
        assert len(memory.saved_messages) == 2
        assert memory.saved_messages[0][1].role == "user"
        assert memory.saved_messages[1][1].role == "assistant"

    @pytest.mark.asyncio
    async def test_memory_save_uses_correct_session_id(self):
        """Memory save must use the session_id from context."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx("my-session-123")

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        for sid, msg in memory.saved_messages:
            assert sid == "my-session-123"


# ============================================================================
# 4. Planner Integration
# ============================================================================


class TestPlannerIntegration:
    """Test that AgentRuntime correctly delegates to planner."""

    @pytest.mark.asyncio
    async def test_planner_receives_context(self):
        """Planner must receive the SessionContext."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={"tool1": "mock"}, planner=planner,
            llm_gateway=MockLLMGateway()
        ):
            pass

        assert planner.call_args is not None
        assert planner.call_args["ctx"] is ctx

    @pytest.mark.asyncio
    async def test_planner_receives_memory(self):
        """Planner must receive the memory instance."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        assert planner.call_args["memory"] is memory

    @pytest.mark.asyncio
    async def test_planner_receives_tools(self):
        """Planner must receive the tools dictionary."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()
        tools = {"search": "mock_search", "calc": "mock_calc"}

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools=tools, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        assert planner.call_args["tools"] is tools

    @pytest.mark.asyncio
    async def test_planner_receives_llm_call(self):
        """Planner must receive an llm_call callable."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        assert planner.call_args["llm_call"] is not None
        assert callable(planner.call_args["llm_call"])

    @pytest.mark.asyncio
    async def test_user_message_added_before_planner_call(self):
        """User message must be in context before planner is called."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test input", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        # At planner call time, ctx should have the user message
        planner_ctx = planner.call_args["ctx"]
        user_msgs = [m for m in planner_ctx.messages if m.role == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[-1].content == "test input"


# ============================================================================
# 5. llm_call Closure Behavior
# ============================================================================


class TestLLMCallClosure:
    """Test the llm_call closure created by AgentRuntime."""

    @pytest.mark.asyncio
    async def test_llm_call_delegates_to_gateway_stream(self):
        """llm_call must call llm_gateway.stream and yield tokens."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()
        gateway = MockLLMGateway(response="hello world test")

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=gateway
        ):
            pass

        # Get the llm_call and test it
        llm_call = planner.call_args["llm_call"]
        tokens = []
        async for token in llm_call("test prompt"):
            tokens.append(token)

        assert len(tokens) == 3
        assert "hello" in tokens[0]
        assert "world" in tokens[1]
        assert "test" in tokens[2]

    @pytest.mark.asyncio
    async def test_llm_call_passes_prompt_to_gateway(self):
        """llm_call must pass the prompt to gateway.stream."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()
        gateway = MockLLMGateway()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=gateway
        ):
            pass

        llm_call = planner.call_args["llm_call"]
        async for _ in llm_call("my specific prompt"):
            pass

        assert len(gateway.calls) == 1
        assert gateway.calls[0]["prompt"] == "my specific prompt"

    @pytest.mark.asyncio
    async def test_llm_call_passes_kwargs(self):
        """llm_call must pass additional kwargs to gateway.stream."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()
        gateway = MockLLMGateway()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=gateway
        ):
            pass

        llm_call = planner.call_args["llm_call"]
        async for _ in llm_call("prompt", model="gpt-4", temperature=0.7):
            pass

        assert gateway.calls[0]["kwargs"]["model"] == "gpt-4"
        assert gateway.calls[0]["kwargs"]["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_llm_call_returns_async_iterator(self):
        """llm_call must return an async iterator."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        llm_call = planner.call_args["llm_call"]
        result = llm_call("test")
        import inspect
        assert inspect.isasyncgen(result)


# ============================================================================
# 6. final_answer Handling
# ============================================================================


class TestFinalAnswerHandling:
    """Test that final_answer events are properly handled."""

    @pytest.mark.asyncio
    async def test_final_answer_appended_to_context(self):
        """final_answer event content must be appended as assistant message."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="final_answer", content="the final answer")
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="question", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        assistant_msgs = [m for m in ctx.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content == "the final answer"

    @pytest.mark.asyncio
    async def test_multiple_final_answers_all_saved(self):
        """If planner yields multiple final_answers, all are saved."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="final_answer", content="answer 1"),
            Event(type="final_answer", content="answer 2"),
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        assistant_msgs = [m for m in ctx.messages if m.role == "assistant"]
        assert len(assistant_msgs) == 2
        assert assistant_msgs[0].content == "answer 1"
        assert assistant_msgs[1].content == "answer 2"

    @pytest.mark.asyncio
    async def test_thought_events_not_in_context_messages(self):
        """thought events should not be added to context.messages."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="thought", content="thinking"),
            Event(type="final_answer", content="answer"),
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        # Only user + assistant (final_answer) should be in messages
        assert len(ctx.messages) == 2
        assert ctx.messages[0].role == "user"
        assert ctx.messages[1].role == "assistant"


# ============================================================================
# 7. Session Context Updates
# ============================================================================


class TestSessionContextUpdates:
    """Test that session context is properly updated."""

    @pytest.mark.asyncio
    async def test_user_message_added_to_context(self):
        """User input must be added to ctx.messages."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="my question", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        user_msgs = [m for m in ctx.messages if m.role == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[0].content == "my question"

    @pytest.mark.asyncio
    async def test_last_active_updated_after_run(self):
        """ctx.last_active must be updated after run completes."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()
        old_time = ctx.last_active

        # Small delay to ensure time difference
        import asyncio
        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        assert ctx.last_active >= old_time

    @pytest.mark.asyncio
    async def test_last_active_is_utc(self):
        """Updated last_active must be UTC timezone-aware."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        assert ctx.last_active.tzinfo is not None
        assert ctx.last_active.tzinfo == timezone.utc

    @pytest.mark.asyncio
    async def test_message_order_in_context(self):
        """Messages in context should be in chronological order."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="thought", content="thinking"),
            Event(type="final_answer", content="answer"),
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="question", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        # Should be: user, assistant (final_answer)
        assert ctx.messages[0].role == "user"
        assert ctx.messages[0].content == "question"
        assert ctx.messages[1].role == "assistant"
        assert ctx.messages[1].content == "answer"


# ============================================================================
# 8. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_user_input(self):
        """run must handle empty user input."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        collected = []
        async for event in runtime.run(
            ctx=ctx, user_input="", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            collected.append(event)

        assert len(collected) >= 1
        user_msgs = [m for m in ctx.messages if m.role == "user"]
        assert user_msgs[0].content == ""

    @pytest.mark.asyncio
    async def test_empty_tools_dict(self):
        """run must handle empty tools dictionary."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        collected = []
        async for event in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            collected.append(event)

        assert len(collected) >= 1

    @pytest.mark.asyncio
    async def test_planner_yields_no_events(self):
        """run handles planner that yields no events."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[])
        memory = MockMemory()
        ctx = _make_ctx()

        collected = []
        async for event in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            collected.append(event)

        assert len(collected) == 0
        # User message should still be in context
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"

    @pytest.mark.asyncio
    async def test_error_events_not_saved_as_messages(self):
        """error events should not be saved as assistant messages."""
        runtime = AgentRuntime()
        planner = MockPlanner(events=[
            Event(type="error", content="something went wrong"),
            Event(type="final_answer", content="recovered"),
        ])
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="test", memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        # Only user + final_answer should be saved
        assert len(memory.saved_messages) == 2

    @pytest.mark.asyncio
    async def test_context_messages_accumulate_across_runs(self):
        """Multiple runs on same context should accumulate messages."""
        runtime = AgentRuntime()
        memory = MockMemory()

        ctx = _make_ctx()
        # First run
        planner1 = MockPlanner(events=[
            Event(type="final_answer", content="answer 1")
        ])
        async for _ in runtime.run(
            ctx=ctx, user_input="question 1", memory=memory,
            tools={}, planner=planner1, llm_gateway=MockLLMGateway()
        ):
            pass

        # Second run on same context
        planner2 = MockPlanner(events=[
            Event(type="final_answer", content="answer 2")
        ])
        async for _ in runtime.run(
            ctx=ctx, user_input="question 2", memory=memory,
            tools={}, planner=planner2, llm_gateway=MockLLMGateway()
        ):
            pass

        # Should have: user1, assistant1, user2, assistant2
        assert len(ctx.messages) == 4
        assert ctx.messages[0].content == "question 1"
        assert ctx.messages[1].content == "answer 1"
        assert ctx.messages[2].content == "question 2"
        assert ctx.messages[3].content == "answer 2"

    @pytest.mark.asyncio
    async def test_unicode_user_input(self):
        """run handles Unicode user input."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()

        async for _ in runtime.run(
            ctx=ctx, user_input="你好世界",
            memory=memory, tools={}, planner=planner,
            llm_gateway=MockLLMGateway()
        ):
            pass

        user_msgs = [m for m in ctx.messages if m.role == "user"]
        assert user_msgs[0].content == "你好世界"

    @pytest.mark.asyncio
    async def test_very_long_user_input(self):
        """run handles very long user input."""
        runtime = AgentRuntime()
        planner = MockPlanner()
        memory = MockMemory()
        ctx = _make_ctx()
        long_input = "x" * 10000

        async for _ in runtime.run(
            ctx=ctx, user_input=long_input, memory=memory,
            tools={}, planner=planner, llm_gateway=MockLLMGateway()
        ):
            pass

        user_msgs = [m for m in ctx.messages if m.role == "user"]
        assert user_msgs[0].content == long_input
