"""Tests for AgentRuntime stateless engine.

This module tests:
- AgentRuntime class creation and initialization
- run method with proper event streaming
- Integration with SessionContext, Memory, Tools, Planner, LLM Gateway
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event


async def async_iter(items):
    """Helper to create async iterable from list."""
    for item in items:
        yield item


class TestAgentRuntimeInit:
    """Test AgentRuntime initialization."""

    def test_import_agent_runtime(self):
        """Test that AgentRuntime can be imported."""
        from agent_framework.runtime.agent_runtime import AgentRuntime
        assert AgentRuntime is not None

    def test_agent_runtime_init(self):
        """Test AgentRuntime initialization."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()
        assert runtime is not None


class TestAgentRuntimeRun:
    """Test AgentRuntime run method."""

    @pytest.mark.asyncio
    async def test_run_returns_async_iterator(self):
        """Test that run returns an async iterator."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        # Create mocks
        ctx = SessionContext(session_id="test-session")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            return
            yield  # Make it an async generator

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act

        llm_gateway = AsyncMock()

        # Call run
        result = runtime.run(
            ctx=ctx,
            user_input="Hello",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        )

        # Should be an async iterator
        assert hasattr(result, '__aiter__')
        assert hasattr(result, '__anext__')

    @pytest.mark.asyncio
    async def test_run_adds_user_message_to_context(self):
        """Test that run adds user message to context."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        ctx = SessionContext(session_id="test-session")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            return
            yield  # Make it an async generator

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act
        llm_gateway = AsyncMock()

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="Hello Agent",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # User message should be added to context
        assert len(ctx.messages) >= 1
        user_msg = ctx.messages[0]
        assert user_msg.role == "user"
        assert user_msg.content == "Hello Agent"

    @pytest.mark.asyncio
    async def test_run_saves_user_message_to_memory(self):
        """Test that run saves user message to memory."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        ctx = SessionContext(session_id="test-session")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            return
            yield  # Make it an async generator

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act
        llm_gateway = AsyncMock()

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="Test message",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # Memory save should be called at least once (for user message)
        assert memory.save.call_count >= 1
        first_call = memory.save.call_args_list[0]
        assert first_call[0][0] == "test-session"  # session_id
        assert first_call[0][1].role == "user"
        assert first_call[0][1].content == "Test message"

    @pytest.mark.asyncio
    async def test_run_yields_events_from_planner(self):
        """Test that run yields events from planner."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        ctx = SessionContext(session_id="test-session")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        # Create planner that yields events
        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="thought", content="Thinking...")
            yield Event(type="action", content="Searching...")
            yield Event(type="final_answer", content="Here is the answer.")

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act

        llm_gateway = AsyncMock()

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="Question?",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # Should have all events from planner
        assert len(events) == 3
        assert events[0].type == "thought"
        assert events[1].type == "action"
        assert events[2].type == "final_answer"

    @pytest.mark.asyncio
    async def test_run_saves_final_answer_to_context(self):
        """Test that run saves final_answer to context."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        ctx = SessionContext(session_id="test-session")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="final_answer", content="The answer is 42.")

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act

        llm_gateway = AsyncMock()

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="What is the answer?",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # Context should have user message + assistant message
        assert len(ctx.messages) == 2
        assert ctx.messages[0].role == "user"
        assert ctx.messages[0].content == "What is the answer?"
        assert ctx.messages[1].role == "assistant"
        assert ctx.messages[1].content == "The answer is 42."

    @pytest.mark.asyncio
    async def test_run_saves_final_answer_to_memory(self):
        """Test that run saves final_answer to memory."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        ctx = SessionContext(session_id="test-session")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="final_answer", content="Done!")

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act

        llm_gateway = AsyncMock()

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="Do it",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # Memory save should be called twice (user message + final answer)
        assert memory.save.call_count == 2
        second_call = memory.save.call_args_list[1]
        assert second_call[0][1].role == "assistant"
        assert second_call[0][1].content == "Done!"

    @pytest.mark.asyncio
    async def test_run_updates_last_active_time(self):
        """Test that run updates session last_active time."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        # Set initial time in the past
        past_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        ctx = SessionContext(session_id="test-session")
        ctx.last_active = past_time

        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="final_answer", content="Done!")

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act

        llm_gateway = AsyncMock()

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="Hello",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # last_active should be updated
        assert ctx.last_active > past_time

    @pytest.mark.asyncio
    async def test_run_provides_llm_call_to_planner(self):
        """Test that run provides llm_call closure to planner."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        ctx = SessionContext(session_id="test-session")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {}

        captured_llm_call = None

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            nonlocal captured_llm_call
            captured_llm_call = llm_call
            yield Event(type="final_answer", content="Done!")

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act

        llm_gateway = AsyncMock()
        llm_gateway.stream = AsyncMock(return_value=async_iter(["Hello", " World"]))

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="Test",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # llm_call should be provided to planner
        assert captured_llm_call is not None
        assert callable(captured_llm_call)


class TestAgentRuntimeIntegration:
    """Integration tests for AgentRuntime."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow with multiple events."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        ctx = SessionContext(session_id="integration-test")
        memory = AsyncMock()
        memory.save = AsyncMock()

        tools = {"search": MagicMock()}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="thought", content="I need to search for information.")
            yield Event(type="action", content="Calling search tool...")
            yield Event(type="observation", content="Found relevant results.")
            yield Event(type="final_answer", content="Based on my search, the answer is X.")

        planner = AsyncMock()
        planner.plan_and_act = mock_plan_and_act

        llm_gateway = AsyncMock()

        # Collect events
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="What is X?",
            memory=memory,
            tools=tools,
            planner=planner,
            llm_gateway=llm_gateway
        ):
            events.append(event)

        # Should have all events
        assert len(events) == 4
        assert events[0].type == "thought"
        assert events[1].type == "action"
        assert events[2].type == "observation"
        assert events[3].type == "final_answer"

        # Context should have user + assistant messages
        assert len(ctx.messages) == 2
        assert ctx.messages[0].role == "user"
        assert ctx.messages[1].role == "assistant"
        assert ctx.messages[1].content == "Based on my search, the answer is X."

        # Memory should be called twice
        assert memory.save.call_count == 2
