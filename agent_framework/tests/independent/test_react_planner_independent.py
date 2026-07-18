"""Independent test cases for ToolCallPlanner (formerly ReActPlanner) implementation.

This module contains independent verification tests for the ToolCallPlanner,
following the detailed design specification in section 7.

Test categories:
1. ToolCallPlanner inheritance and initialization
2. plan_and_act method flow (streaming with tool_calls)
3. Tool call loop logic
4. Event stream generation
5. Tool call handling
6. _build_prompt method (legacy)
7. Boundary conditions
8. Integration tests
"""
import pytest
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncIterator, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.core.planner_context import PlannerContext

# Import directly from module to avoid __init__.py export issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "react_planner",
    Path(__file__).parent.parent.parent / "planners" / "react_planner.py"
)
react_planner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(react_planner_module)

ToolCallPlanner = react_planner_module.ToolCallPlanner
ReActPlanner = react_planner_module.ReActPlanner
ToolCall = react_planner_module.ToolCall
FunctionCall = react_planner_module.FunctionCall
ChatResponse = react_planner_module.ChatResponse
ChatMessage = react_planner_module.ChatMessage


class TestToolCallPlannerInheritance:
    """Independent tests for ToolCallPlanner inheritance."""

    def test_planner_is_subclass_of_base_planner(self):
        """ToolCallPlanner must inherit from BasePlanner."""
        from agent_framework.interfaces.base_planner import BasePlanner
        assert issubclass(ToolCallPlanner, BasePlanner)

    def test_planner_implements_plan_and_act(self):
        """ToolCallPlanner must implement plan_and_act method."""
        assert hasattr(ToolCallPlanner, 'plan_and_act')
        assert callable(getattr(ToolCallPlanner, 'plan_and_act'))

    def test_planner_can_instantiate(self):
        """ToolCallPlanner can be instantiated."""
        planner = ToolCallPlanner()
        assert planner is not None
        assert isinstance(planner, ToolCallPlanner)

    def test_react_planner_is_alias(self):
        """ReActPlanner should be an alias for ToolCallPlanner."""
        assert ReActPlanner is ToolCallPlanner


class TestToolCallPlannerInitialization:
    """Independent tests for ToolCallPlanner initialization."""

    def test_default_name(self):
        """ToolCallPlanner should default to 'tool_call' name."""
        planner = ToolCallPlanner()
        assert planner.name == "tool_call"

    def test_default_description(self):
        """ToolCallPlanner should have default description."""
        planner = ToolCallPlanner()
        assert planner.description is not None
        assert len(planner.description) > 0

    def test_custom_name(self):
        """ToolCallPlanner should accept custom name."""
        planner = ToolCallPlanner(name="custom_planner")
        assert planner.name == "custom_planner"

    def test_custom_description(self):
        """ToolCallPlanner should accept custom description."""
        planner = ToolCallPlanner(description="My custom planner")
        assert planner.description == "My custom planner"

    def test_default_max_iterations(self):
        """ToolCallPlanner should have default max_iterations."""
        planner = ToolCallPlanner()
        assert planner.max_iterations == 10

    def test_custom_max_iterations(self):
        """ToolCallPlanner should accept custom max_iterations."""
        planner = ToolCallPlanner(max_iterations=5)
        assert planner.max_iterations == 5

    def test_has_build_system_message_method(self):
        """ToolCallPlanner must have _build_system_message method."""
        assert hasattr(ToolCallPlanner, '_build_system_message')

    def test_has_handle_chat_response_method(self):
        """ToolCallPlanner must have _handle_chat_response method."""
        assert hasattr(ToolCallPlanner, '_handle_chat_response')


class TestToolCallPlannerPlanAndAct:
    """Independent tests for plan_and_act method with streaming tool_calls format."""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="Memory context")
        return mock

    @pytest.fixture
    def mock_tool(self):
        """Create mock tool."""
        mock = AsyncMock(spec=BaseTool)
        mock.name = "test_tool"
        mock.description = "A test tool"
        mock.run = AsyncMock(return_value="Tool result")
        return mock

    @pytest.fixture
    def session_context(self):
        """Create test session context."""
        return SessionContext(
            session_id="test-session",
            session_type="private",
            participants=["user1"],
            messages=[
                Message(role="user", content="What is Python?", sender_id="user1")
            ]
        )

    @pytest.fixture
    def planner(self):
        """Create ToolCallPlanner instance."""
        return ToolCallPlanner()

    def test_plan_and_act_returns_async_iterator(self, planner, session_context, mock_memory):
        """plan_and_act should return an async iterator."""
        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Test")

        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools={},
            messages=session_context.messages,
            memory=mock_memory
        )
        result = planner.plan_and_act(ctx, mock_llm_call)
        assert hasattr(result, '__aiter__')

    def test_plan_and_act_yields_events(self, planner, session_context, mock_memory):
        """plan_and_act should yield Event objects."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Test")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools={},
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert len(events) > 0
        assert all(isinstance(e, Event) for e in events)

    def test_plan_and_act_immediate_final_answer(self, planner, session_context, mock_memory):
        """plan_and_act should handle immediate final answer via streaming."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Python is a programming language.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools={},
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        final_events = [e for e in events if e.type == "final_answer"]
        assert len(final_events) == 1
        assert "Python is a programming language" in final_events[0].content

    def test_plan_and_act_with_tool_call(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should handle tool calls via streaming events."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test query"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test query"}'})
            else:
                yield Event(type="final_answer", content="Based on the tool result.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools=tools,
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "action" for e in events)
        assert any(e.type == "observation" for e in events)
        assert any(e.type == "final_answer" for e in events)

        # Tool should have been called
        mock_tool.run.assert_called_once()

    def test_plan_and_act_multiple_tool_calls(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should handle multiple sequential tool call iterations."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "first"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "first"}'})
            elif call_count == 2:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_2",
                                      "arguments": '{"input": "second"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_2",
                                      "arguments": '{"input": "second"}'})
            else:
                yield Event(type="final_answer", content="Done.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools=tools,
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert mock_tool.run.call_count == 2
        observation_events = [e for e in events if e.type == "observation"]
        assert len(observation_events) == 2

    def test_plan_and_act_unknown_tool_error(self, planner, session_context, mock_memory):
        """plan_and_act should yield error for unknown tool."""

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "unknown_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "query"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "unknown_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "query"}'})

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools={},
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "error" for e in events)

    def test_plan_and_act_tool_execution_error(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should yield error when tool execution fails."""
        mock_tool.run = AsyncMock(side_effect=RuntimeError("Tool failed"))
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "query"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "query"}'})

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools=tools,
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "error" for e in events)

    def test_plan_and_act_max_iterations(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should stop at max_iterations."""
        planner.max_iterations = 3
        tools = {"test_tool": mock_tool}
        llm_call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal llm_call_count
            llm_call_count += 1
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "test_tool",
                                  "tool_call_id": f"call_{llm_call_count}",
                                  "arguments": '{"input": "query"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "test_tool",
                                  "tool_call_id": f"call_{llm_call_count}",
                                  "arguments": '{"input": "query"}'})

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools=tools,
                messages=session_context.messages,
                memory=mock_memory,
                max_iterations=planner.max_iterations
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert llm_call_count == 3
        assert any(e.type == "error" for e in events)

    def test_plan_and_act_empty_messages(self, planner, mock_memory):
        """plan_and_act should handle empty message history."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Hello!")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="empty",
                tools={},
                messages=[],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "final_answer" for e in events)

    def test_plan_and_act_no_tools(self, planner, session_context, mock_memory):
        """plan_and_act should work with no tools."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="I can answer directly.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools={},
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "final_answer" for e in events)


class TestToolCallPlannerEventStream:
    """Independent tests for event stream generation."""

    @pytest.fixture
    def planner(self):
        """Create ToolCallPlanner instance."""
        return ToolCallPlanner()

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    @pytest.fixture
    def session_context(self):
        """Create test session context."""
        return SessionContext(
            session_id="test",
            messages=[Message(role="user", content="test", sender_id="user1")]
        )

    def test_final_answer_event_content(self, planner, session_context, mock_memory):
        """final_answer event should contain the answer text."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="The answer is 42.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools={},
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        final_events = [e for e in events if e.type == "final_answer"]
        assert len(final_events) == 1
        assert "42" in final_events[0].content

    def test_action_event_content(self, planner, session_context, mock_memory):
        """action event should contain tool name."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value="result")
        tools = {"my_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "my_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "my_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
            else:
                yield Event(type="final_answer", content="Done.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools=tools,
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        action_events = [e for e in events if e.type == "action"]
        assert len(action_events) >= 1
        assert "my_tool" in action_events[0].content

    def test_observation_event_content(self, planner, session_context, mock_memory):
        """observation event should contain tool result."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value="tool result here")
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
            else:
                yield Event(type="final_answer", content="Done.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools=tools,
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        observation_events = [e for e in events if e.type == "observation"]
        assert len(observation_events) == 1
        assert "tool result here" in observation_events[0].content

    def test_events_have_timestamps(self, planner, session_context, mock_memory):
        """All events should have timestamps."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Test.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools={},
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        for event in events:
            assert event.timestamp is not None


class TestToolCallPlannerBoundaryConditions:
    """Independent tests for boundary conditions."""

    @pytest.fixture
    def planner(self):
        """Create ToolCallPlanner instance."""
        return ToolCallPlanner()

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    def test_llm_call_returns_final_answer_directly(self, planner, mock_memory):
        """plan_and_act should handle LLM returning final answer directly."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Direct answer.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools={},
                messages=[],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert len(events) > 0
        assert any(e.type == "final_answer" for e in events)

    def test_long_conversation_history(self, planner, mock_memory):
        """plan_and_act should handle long conversation history."""
        messages = [
            Message(role="user", content=f"Message {i}", sender_id="user1")
            for i in range(50)
        ]

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Handled long history.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools={},
                messages=messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "final_answer" for e in events)

    def test_special_characters_in_messages(self, planner, mock_memory):
        """plan_and_act should handle special characters."""

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Special chars handled.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools={},
                messages=[Message(role="user", content="Hello! @#$%^&*()", sender_id="user1")],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "final_answer" for e in events)

    def test_tool_returns_non_string(self, planner, mock_memory):
        """plan_and_act should handle tool returning non-string."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value=42)
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
            else:
                yield Event(type="final_answer", content="Done.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools=tools,
                messages=[],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "observation" for e in events)

    def test_memory_retrieval_fails(self, planner):
        """plan_and_act should handle memory retrieval failure gracefully."""
        mock_memory = AsyncMock(spec=BaseMemory)
        mock_memory.retrieve = AsyncMock(side_effect=Exception("Memory error"))

        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Memory error handled.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools={},
                messages=[Message(role="user", content="Hello", sender_id="user1")],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "final_answer" for e in events)

    def test_handles_llm_exception(self, planner, mock_memory):
        """plan_and_act should handle LLM call exceptions gracefully."""

        async def mock_llm_call(messages, tools):
            raise RuntimeError("LLM service unavailable")
            yield  # Make it an async generator

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools={},
                messages=[],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) > 0

    def test_json_parse_error_in_arguments(self, planner, mock_memory):
        """Should handle malformed JSON in tool arguments gracefully."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value="result")
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": "not valid json {"})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": "not valid json {"})

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools=tools,
                messages=[],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        # Should handle gracefully without crashing
        assert len(events) >= 0


class TestToolCallPlannerIntegration:
    """Independent integration tests for ToolCallPlanner."""

    @pytest.fixture
    def planner(self):
        """Create ToolCallPlanner instance."""
        return ToolCallPlanner(max_iterations=5)

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="User likes Python")
        return mock

    def test_full_tool_call_loop(self, planner, mock_memory):
        """Test complete tool call loop with tool execution and final answer."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.description = "Searches for information"
        mock_tool.run = AsyncMock(return_value="Python is a programming language")
        tools = {"search": mock_tool}

        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "search", "tool_call_id": "call_1",
                                      "arguments": '{"input": "What is Python"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "search", "tool_call_id": "call_1",
                                      "arguments": '{"input": "What is Python"}'})
            else:
                yield Event(type="final_answer", content="Python is a programming language.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools=tools,
                messages=[Message(role="user", content="What is Python?", sender_id="user1")],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        event_types = [e.type for e in events]
        assert "action" in event_types
        assert "observation" in event_types
        assert "final_answer" in event_types

        mock_tool.run.assert_called_once()

    def test_multiple_tools_workflow(self, planner, mock_memory):
        """Test workflow with multiple different tools."""
        tool1 = AsyncMock(spec=BaseTool)
        tool1.description = "Search tool"
        tool1.run = AsyncMock(return_value="Search results")

        tool2 = AsyncMock(spec=BaseTool)
        tool2.description = "Calculator tool"
        tool2.run = AsyncMock(return_value="42")

        tools = {"search": tool1, "calculator": tool2}

        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "search", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "search", "tool_call_id": "call_1",
                                      "arguments": '{"input": "query"}'})
            elif call_count == 2:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "calculator", "tool_call_id": "call_2",
                                      "arguments": '{"input": "2+2"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "calculator", "tool_call_id": "call_2",
                                      "arguments": '{"input": "2+2"}'})
            else:
                yield Event(type="final_answer", content="Combined results.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools=tools,
                messages=[Message(role="user", content="Complex query", sender_id="user1")],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        tool1.run.assert_called_once()
        tool2.run.assert_called_once()

    def test_tool_call_with_chat_response_path(self, planner, mock_memory):
        """Test the non-streaming ChatResponse path."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value="result")
        tools = {"test_tool": mock_tool}

        call_count = 0

        def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="test_tool", arguments='{"input": "test"}')
                    )]
                )
            else:
                return ChatResponse(content="Final answer via ChatResponse.")

        events = []

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools=tools,
                messages=[Message(role="user", content="Test", sender_id="user1")],
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        assert any(e.type == "observation" for e in events)
        assert any(e.type == "final_answer" for e in events)

    def test_data_types_integrity(self):
        """Test that ToolCall, FunctionCall, ChatResponse, ChatMessage are proper dataclasses."""
        import dataclasses

        assert dataclasses.is_dataclass(FunctionCall)
        assert dataclasses.is_dataclass(ToolCall)
        assert dataclasses.is_dataclass(ChatResponse)
        assert dataclasses.is_dataclass(ChatMessage)

        fc = FunctionCall(name="test", arguments='{"key": "value"}')
        assert fc.name == "test"

        tc = ToolCall(id="call_1", type="function", function=fc)
        assert tc.id == "call_1"
        assert tc.function.name == "test"

        resp = ChatResponse(content="Hello", tool_calls=[tc])
        assert resp.has_tool_calls is True
        assert resp.content == "Hello"

        empty_resp = ChatResponse(content="No tools")
        assert empty_resp.has_tool_calls is False

        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.tool_call_id is None

        tool_msg = ChatMessage(role="tool", content="result", tool_call_id="call_1")
        assert tool_msg.tool_call_id == "call_1"
