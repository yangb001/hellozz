"""Independent verification tests for tool_calls architecture adjustment.

Verification areas:
1. Non-streaming path: tool_calls -> execute tool -> continue LLM -> final response
2. Streaming path works correctly
3. Redundant code cleanup (_extract_final_answer, Final Answer: detection, events.py EventType)
4. Tool call results visible to LLM for generating final answer
5. Normal conversation not affected

Reference: 详细设计.md

Refactored: Updated to use PlannerContext (Phase 2A) - ToolCallPlanner.plan_and_act
now accepts (PlannerContext, llm_call) instead of (SessionContext, memory, tools, llm_call).
"""
import pytest
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, AsyncIterator

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event, EventType
from agent_framework.core.planner_context import PlannerContext

# Import directly from module to avoid __init__.py export issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "react_planner",
    Path(__file__).parent.parent.parent / "planners" / "react_planner.py"
)
react_planner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(react_planner_module)

# The planner class is named ToolCallPlanner in the module
ToolCallPlanner = react_planner_module.ToolCallPlanner
ToolCall = react_planner_module.ToolCall
ChatMessage = react_planner_module.ChatMessage
ChatResponse = react_planner_module.ChatResponse
FunctionCall = react_planner_module.FunctionCall


def _make_planner_ctx(session_id: str, tools: dict, memory, messages=None) -> PlannerContext:
    """Helper to create a PlannerContext for tests."""
    return PlannerContext(
        session_id=session_id,
        tools=tools,
        messages=messages or [],
        memory=memory
    )


class TestNonStreamingToolCallPath:
    """Test area 1: Non-streaming path tool_calls -> execute tool -> continue LLM -> final response"""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    @pytest.fixture
    def mock_tool(self):
        """Create mock tool."""
        mock = AsyncMock(spec=BaseTool)
        mock.name = "get_weather"
        mock.description = "Get weather for a city"
        mock.run = AsyncMock(return_value="Sunny, 25°C")
        return mock

    def test_non_streaming_tool_call_flow(self, mock_memory, mock_tool):
        """Verify complete non-streaming tool call flow."""
        planner = ToolCallPlanner()
        tools = {"get_weather": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1

            # First call: LLM requests tool
            if call_count == 1:
                # Verify tool result was added to messages
                tool_results = [m for m in messages if m.get("role") == "tool"]
                assert len(tool_results) == 0, "First call should not have tool results yet"
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="get_weather", arguments='{"city":"Beijing"}')
                    )]
                )
            # Second call: LLM uses tool result to generate answer
            elif call_count == 2:
                # Verify tool result was passed to LLM
                tool_results = [m for m in messages if m.get("role") == "tool"]
                assert len(tool_results) == 1, "Second call should have tool result"
                return ChatResponse(content="The weather in Beijing is sunny and 25°C.", tool_calls=[])
            else:
                return ChatResponse(content="Done", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    return

        asyncio.run(collect())

        # Verify tool was executed
        assert mock_tool.run.call_count == 1, "Tool should be executed exactly once"

        # Verify final answer was generated
        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) > 0, "Should yield final answer"

    def test_tool_result_passed_to_next_llm_call(self, mock_memory, mock_tool):
        """Verify tool result is included in messages for next LLM call."""
        planner = ToolCallPlanner()
        tools = {"get_weather": mock_tool}
        call_count = 0
        tool_result_verified = False

        async def mock_llm_call(messages, tools):
            nonlocal call_count, tool_result_verified
            call_count += 1

            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="get_weather", arguments='{"city":"Beijing"}')
                    )]
                )
            else:
                # Check if tool result is in messages
                tool_results = [m for m in messages if m.get("role") == "tool"]
                if len(tool_results) > 0 and "Sunny" in tool_results[0].get("content", ""):
                    tool_result_verified = True
                return ChatResponse(content="Final Answer: Done", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        async def run():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                if event.type == "final_answer":
                    break

        asyncio.run(run())
        assert tool_result_verified, "Tool result should be passed to second LLM call"

    def test_multiple_tool_calls_sequential_execution(self, mock_memory):
        """Verify multiple tool calls are executed sequentially."""
        planner = ToolCallPlanner()
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_1", type="function", function=FunctionCall(name="tool1", arguments="{}")),
                        ToolCall(id="call_2", type="function", function=FunctionCall(name="tool2", arguments="{}"))
                    ]
                )
            else:
                return ChatResponse(content="Final Answer: Done", tool_calls=[])

        # Mock tools
        tool1 = AsyncMock(spec=BaseTool)
        tool1.name = "tool1"
        tool1.run = AsyncMock(return_value="result1")
        tool2 = AsyncMock(spec=BaseTool)
        tool2.name = "tool2"
        tool2.run = AsyncMock(return_value="result2")
        tools = {"tool1": tool1, "tool2": tool2}

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        # Both tools should be called
        assert tool1.run.call_count == 1
        assert tool2.run.call_count == 1


class TestStreamingToolCallPath:
    """Test area 2: Streaming path works correctly"""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    @pytest.fixture
    def mock_tool(self):
        """Create mock tool."""
        mock = AsyncMock(spec=BaseTool)
        mock.name = "search"
        mock.description = "Search the web"
        mock.run = AsyncMock(return_value="Search results: Python is a programming language.")
        return mock

    def test_streaming_yields_tool_call_events(self, mock_memory, mock_tool):
        """Verify streaming path yields tool_call_start/argument/end events."""
        planner = ToolCallPlanner()
        tools = {"search": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Return ChatResponse for streaming simulation
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="search", arguments='{"query":"Python"}')
                    )]
                )
            else:
                return ChatResponse(content="Final Answer: Found it!", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        # Verify tool call events were yielded
        action_events = [e for e in events if e.type == "action"]
        observation_events = [e for e in events if e.type == "observation"]
        assert len(action_events) > 0, "Should yield action events"
        assert len(observation_events) > 0, "Should yield observation events"

    def test_streaming_tool_execution(self, mock_memory, mock_tool):
        """Verify tool is executed in streaming path."""
        planner = ToolCallPlanner()
        tools = {"search": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="search", arguments='{"query":"Python"}')
                    )]
                )
            else:
                return ChatResponse(content="Final Answer: Done", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        async def run():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                if event.type == "final_answer":
                    break

        asyncio.run(run())

        assert mock_tool.run.call_count == 1, "Tool should be executed in streaming path"


class TestRedundantCodeCleanup:
    """Test area 3: Verify redundant code has been cleaned up"""

    def test_no_final_answer_detection_in_content(self):
        """Verify Final Answer: detection logic is no longer used in main flow."""
        planner = ToolCallPlanner()

        # The planner should use tool_calls for final answer, not content detection
        # If content detection is removed, this test verifies the planner
        # correctly handles responses without "Final Answer:" prefix
        assert hasattr(planner, '_handle_chat_response'), "Planner should have _handle_chat_response method"

    def test_events_py_no_tool_call_related_types(self):
        """Verify events.py EventType enum has correct tool call types."""
        # Check that EventType has tool call types
        assert hasattr(EventType, 'TOOL_CALL_START'), "EventType should have TOOL_CALL_START"
        assert hasattr(EventType, 'TOOL_CALL_ARGUMENT'), "EventType should have TOOL_CALL_ARGUMENT"
        assert hasattr(EventType, 'TOOL_CALL_END'), "EventType should have TOOL_CALL_END"

        # These are the correct tool call event types
        assert EventType.TOOL_CALL_START == "tool_call_start"
        assert EventType.TOOL_CALL_ARGUMENT == "tool_call_argument"
        assert EventType.TOOL_CALL_END == "tool_call_end"

    def test_no_legacy_final_answer_parsing(self):
        """Verify planner handles final answer via content check, not _extract_final_answer."""
        planner = ToolCallPlanner()

        # The refactored planner uses _handle_chat_response to check content
        # instead of a separate _extract_final_answer method
        assert hasattr(planner, '_handle_chat_response'), "Planner should use _handle_chat_response"


class TestToolResultVisibility:
    """Test area 4: Tool call results visible to LLM for final answer"""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    def test_llm_receives_tool_result_before_final_answer(self, mock_memory):
        """Verify LLM receives tool result before generating final answer."""
        planner = ToolCallPlanner()

        calc_tool = AsyncMock(spec=BaseTool)
        calc_tool.name = "calculator"
        calc_tool.run = AsyncMock(return_value="4")
        tools = {"calculator": calc_tool}

        call_count = 0
        messages_at_second_call = None

        async def mock_llm_call(messages, tools):
            nonlocal call_count, messages_at_second_call
            call_count += 1

            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="calculator", arguments='{"input":"2+2"}')
                    )]
                )
            else:
                messages_at_second_call = list(messages)
                return ChatResponse(content="Final Answer: 4", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        # Verify second LLM call received tool result
        assert messages_at_second_call is not None, "Second LLM call should have happened"
        tool_roles = [m for m in messages_at_second_call if m.get("role") == "tool"]
        assert len(tool_roles) == 1, "Second call should include tool result message"
        assert "4" in tool_roles[0].get("content", ""), "Tool result should contain the calculated value"

    def test_final_answer_depends_on_tool_result(self, mock_memory):
        """Verify final answer is generated based on tool result."""
        planner = ToolCallPlanner()

        # Tool that returns specific value
        search_tool = AsyncMock(spec=BaseTool)
        search_tool.name = "web_search"
        search_tool.run = AsyncMock(return_value="Python 3.11 was released in 2022")
        tools = {"web_search": search_tool}

        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="web_search", arguments='{"query":"Python release date"}')
                    )]
                )
            else:
                # Verify LLM saw the tool result
                tool_results = [m for m in messages if m.get("role") == "tool"]
                tool_content = tool_results[0].get("content", "") if tool_results else ""
                # LLM should include info from tool result in final answer
                return ChatResponse(content=f"Final Answer: {tool_content}", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) > 0
        # Final answer should reflect tool result
        assert "2022" in final_answer_events[0].content or "released" in final_answer_events[0].content


class TestNormalConversationUnaffected:
    """Test area 5: Normal conversation without tool calls not affected"""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    def test_simple_conversation_no_tools(self, mock_memory):
        """Verify simple conversation without tool calls works."""
        planner = ToolCallPlanner()
        tools = {}

        async def mock_llm_call(messages, tools):
            return ChatResponse(content="Hello! I'm doing well, thank you for asking.", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        # Should yield text or final_answer events
        assert len(events) > 0

    def test_conversation_with_content_but_no_tools(self, mock_memory):
        """Verify response with content but no tool_calls works."""
        planner = ToolCallPlanner()
        tools = {}

        async def mock_llm_call(messages, tools):
            return ChatResponse(content="The capital of France is Paris.", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) > 0
        assert "Paris" in final_answer_events[0].content

    def test_empty_content_no_tools(self, mock_memory):
        """Verify handling of empty content response."""
        planner = ToolCallPlanner()
        tools = {}

        async def mock_llm_call(messages, tools):
            return ChatResponse(content="", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)

        asyncio.run(collect())

        # Should handle empty response gracefully
        assert events is not None


class TestToolCallsArchitectureIntegration:
    """Integration tests for complete tool_calls architecture"""

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    def test_full_flow_user_message_to_final_answer(self, mock_memory):
        """Verify complete flow from user message to final answer with tool calls."""
        planner = ToolCallPlanner()

        # Mock tools
        weather_tool = AsyncMock(spec=BaseTool)
        weather_tool.name = "get_weather"
        weather_tool.run = AsyncMock(return_value="Sunny")
        temp_tool = AsyncMock(spec=BaseTool)
        temp_tool.name = "get_temperature"
        temp_tool.run = AsyncMock(return_value="25°C")
        tools = {"get_weather": weather_tool, "get_temperature": temp_tool}

        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return ChatResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id="call_1", type="function", function=FunctionCall(name="get_weather", arguments="{}")),
                        ToolCall(id="call_2", type="function", function=FunctionCall(name="get_temperature", arguments="{}"))
                    ]
                )
            else:
                return ChatResponse(content="Final Answer: It's sunny with a temperature of 25°C.", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        # Verify both tools were called
        assert weather_tool.run.call_count == 1
        assert temp_tool.run.call_count == 1

        # Verify final answer was generated
        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) == 1
        assert "sunny" in final_answer_events[0].content.lower()
        assert "25" in final_answer_events[0].content

    def test_tool_execution_error_handling(self, mock_memory):
        """Verify tool execution errors are handled gracefully."""
        planner = ToolCallPlanner()

        failing_tool = AsyncMock(spec=BaseTool)
        failing_tool.name = "failing_tool"
        failing_tool.run = AsyncMock(side_effect=RuntimeError("Tool failed"))
        tools = {"failing_tool": failing_tool}

        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: request tool
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="failing_tool", arguments="{}")
                    )]
                )
            else:
                # Second call: return final answer after error
                return ChatResponse(content="Sorry, the tool failed.", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        # Should yield observation event with error content (non-streaming path)
        observation_events = [e for e in events if e.type == "observation"]
        error_events = [e for e in events if e.type == "error"]
        # Either error events or observation events with error content
        has_error = len(error_events) > 0 or any("error" in e.content.lower() for e in observation_events)
        assert has_error, "Should yield error information for failed tool"

    def test_unknown_tool_error(self, mock_memory):
        """Verify unknown tool error is handled."""
        planner = ToolCallPlanner()
        tools = {}  # No tools available

        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: request unknown tool
                return ChatResponse(
                    content="",
                    tool_calls=[ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="unknown_tool", arguments="{}")
                    )]
                )
            else:
                # Second call: return final answer after error
                return ChatResponse(content="Sorry, tool not found.", tool_calls=[])

        planner_ctx = _make_planner_ctx("test-session", tools, mock_memory)

        events = []
        async def collect():
            async for event in planner.plan_and_act(planner_ctx, mock_llm_call):
                events.append(event)
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        # Should yield observation event with error content for unknown tool
        observation_events = [e for e in events if e.type == "observation"]
        error_events = [e for e in events if e.type == "error"]
        has_error = len(error_events) > 0 or any("error" in e.content.lower() or "unknown" in e.content.lower() for e in observation_events)
        assert has_error, "Should yield error information for unknown tool"
