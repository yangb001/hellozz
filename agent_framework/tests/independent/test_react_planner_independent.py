"""Independent test cases for ReActPlanner implementation.

This module contains independent verification tests for the ReActPlanner
and Action classes, following the detailed design specification in section 7.

Test categories:
1. Action data class integrity
2. ReActPlanner inheritance and initialization
3. plan_and_act method flow
4. ReAct loop logic
5. Event stream generation
6. Tool call handling
7. Boundary conditions
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncIterator, Dict, Any

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.planners.react_planner import ReActPlanner, Action


class TestActionDataClass:
    """Independent tests for Action data class."""

    def test_action_has_type_field(self):
        """Action must have type field."""
        action = Action(type="tool_call")
        assert action.type == "tool_call"

    def test_action_has_tool_field(self):
        """Action must have tool field."""
        action = Action(type="tool_call", tool="web_search")
        assert action.tool == "web_search"

    def test_action_has_input_field(self):
        """Action must have input field."""
        action = Action(type="tool_call", tool="web_search", input="query")
        assert action.input == "query"

    def test_action_has_content_field(self):
        """Action must have content field."""
        action = Action(type="final_answer", content="The answer is 42")
        assert action.content == "The answer is 42"

    def test_action_tool_defaults_to_none(self):
        """Action tool should default to None."""
        action = Action(type="thought")
        assert action.tool is None

    def test_action_input_defaults_to_none(self):
        """Action input should default to None."""
        action = Action(type="thought")
        assert action.input is None

    def test_action_content_defaults_to_none(self):
        """Action content should default to None."""
        action = Action(type="tool_call", tool="test")
        assert action.content is None

    def test_action_is_dataclass(self):
        """Action should be a dataclass."""
        import dataclasses
        assert dataclasses.is_dataclass(Action)

    def test_action_tool_call_type(self):
        """Action with type='tool_call' represents tool invocation."""
        action = Action(type="tool_call", tool="search", input="query")
        assert action.type == "tool_call"
        assert action.tool is not None

    def test_action_final_answer_type(self):
        """Action with type='final_answer' represents final response."""
        action = Action(type="final_answer", content="answer")
        assert action.type == "final_answer"
        assert action.content is not None

    def test_action_thought_type(self):
        """Action with type='thought' represents reasoning step."""
        action = Action(type="thought", content="thinking...")
        assert action.type == "thought"


class TestReActPlannerInheritance:
    """Independent tests for ReActPlanner inheritance."""

    def test_react_planner_is_subclass_of_base_planner(self):
        """ReActPlanner must inherit from BasePlanner."""
        from agent_framework.interfaces.base_planner import BasePlanner
        assert issubclass(ReActPlanner, BasePlanner)

    def test_react_planner_implements_plan_and_act(self):
        """ReActPlanner must implement plan_and_act method."""
        assert hasattr(ReActPlanner, 'plan_and_act')
        assert callable(getattr(ReActPlanner, 'plan_and_act'))

    def test_react_planner_can_instantiate(self):
        """ReActPlanner can be instantiated."""
        planner = ReActPlanner()
        assert planner is not None
        assert isinstance(planner, ReActPlanner)


class TestReActPlannerInitialization:
    """Independent tests for ReActPlanner initialization."""

    def test_default_name(self):
        """ReActPlanner should default to 'react' name."""
        planner = ReActPlanner()
        assert planner.name == "react"

    def test_default_description(self):
        """ReActPlanner should have default description."""
        planner = ReActPlanner()
        assert planner.description is not None
        assert len(planner.description) > 0

    def test_custom_name(self):
        """ReActPlanner should accept custom name."""
        planner = ReActPlanner(name="custom_react")
        assert planner.name == "custom_react"

    def test_custom_description(self):
        """ReActPlanner should accept custom description."""
        planner = ReActPlanner(description="My custom planner")
        assert planner.description == "My custom planner"

    def test_default_max_iterations(self):
        """ReActPlanner should have default max_iterations."""
        planner = ReActPlanner()
        assert planner.max_iterations == 10

    def test_custom_max_iterations(self):
        """ReActPlanner should accept custom max_iterations."""
        planner = ReActPlanner(max_iterations=5)
        assert planner.max_iterations == 5

    def test_has_build_prompt_method(self):
        """ReActPlanner must have _build_prompt method."""
        assert hasattr(ReActPlanner, '_build_prompt')

    def test_has_parse_action_method(self):
        """ReActPlanner must have _parse_action method."""
        assert hasattr(ReActPlanner, '_parse_action')


class TestReActPlannerParseAction:
    """Independent tests for _parse_action method."""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner()

    def test_parse_final_answer(self, planner):
        """_parse_action should parse 'Final Answer:' pattern."""
        text = "Thought: I know the answer.\nFinal Answer: The answer is 42."
        action = planner._parse_action(text)

        assert action.type == "final_answer"
        assert "The answer is 42" in action.content

    def test_parse_tool_call(self, planner):
        """_parse_action should parse 'Action:' and 'Action Input:' pattern."""
        text = "Thought: I need to search.\nAction: web_search\nAction Input: Python tutorials"
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    def test_parse_thought_only(self, planner):
        """_parse_action should handle thought-only text."""
        text = "Thought: I'm thinking about the problem..."
        action = planner._parse_action(text)

        # Should be treated as thought or final_answer
        assert action.type in ["thought", "final_answer"]

    def test_parse_case_insensitive(self, planner):
        """_parse_action should be case-insensitive."""
        text = "ACTION: test_tool\nACTION INPUT: query"
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert action.tool == "test_tool"

    def test_parse_with_extra_whitespace(self, planner):
        """_parse_action should handle extra whitespace."""
        text = "  Final Answer:   The answer  "
        action = planner._parse_action(text)

        assert action.type == "final_answer"
        assert action.content == "The answer"

    def test_parse_empty_action_input(self, planner):
        """_parse_action should handle empty action input."""
        text = "Action: test_tool\nAction Input:"
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert action.tool == "test_tool"

    def test_parse_no_pattern_matches(self, planner):
        """_parse_action should default to final_answer for unmatched text."""
        text = "Just a plain response without any markers."
        action = planner._parse_action(text)

        # Should default to final_answer
        assert action.type == "final_answer"

    def test_parse_with_colons_in_input(self, planner):
        """_parse_action should handle colons in action input."""
        text = "Action: web_search\nAction Input: http://example.com"
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert "http://example.com" in action.input


class TestReActPlannerPlanAndAct:
    """Independent tests for plan_and_act method."""

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
        """Create ReActPlanner instance."""
        return ReActPlanner()

    @pytest.mark.asyncio
    async def test_plan_and_act_returns_async_iterator(self, planner, session_context, mock_memory):
        """plan_and_act should return an async iterator."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Test"

        result = planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call)
        assert hasattr(result, '__aiter__')

    @pytest.mark.asyncio
    async def test_plan_and_act_yields_events(self, planner, session_context, mock_memory):
        """plan_and_act should yield Event objects."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Test"

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call):
            events.append(event)

        assert len(events) > 0
        assert all(isinstance(e, Event) for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_immediate_final_answer(self, planner, session_context, mock_memory):
        """plan_and_act should handle immediate final answer."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Thought: I can answer directly.\n"
            yield "Final Answer: Python is a programming language."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call):
            events.append(event)

        # Should have text_token and final_answer events
        assert any(e.type == "text_token" for e in events)
        assert any(e.type == "final_answer" for e in events)

        # Final answer content should be correct
        final_events = [e for e in events if e.type == "final_answer"]
        assert len(final_events) == 1
        assert "Python is a programming language" in final_events[0].content

    @pytest.mark.asyncio
    async def test_plan_and_act_with_tool_call(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should handle tool calls."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Thought: I need to use the tool.\n"
                yield "Action: test_tool\n"
                yield "Action Input: test query"
            else:
                yield "Thought: Got the result.\n"
                yield "Final Answer: Based on the tool result."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, tools, mock_llm_call):
            events.append(event)

        # Should have action, observation, and final_answer events
        assert any(e.type == "action" for e in events)
        assert any(e.type == "observation" for e in events)
        assert any(e.type == "final_answer" for e in events)

        # Tool should have been called
        mock_tool.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_and_act_multiple_tool_calls(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should handle multiple sequential tool calls."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Action: test_tool\nAction Input: first"
            elif call_count == 2:
                yield "Action: test_tool\nAction Input: second"
            else:
                yield "Final Answer: Done."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, tools, mock_llm_call):
            events.append(event)

        # Tool should be called twice
        assert mock_tool.run.call_count == 2

        # Should have two observation events
        observation_events = [e for e in events if e.type == "observation"]
        assert len(observation_events) == 2

    @pytest.mark.asyncio
    async def test_plan_and_act_unknown_tool_error(self, planner, session_context, mock_memory):
        """plan_and_act should yield error for unknown tool."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Action: unknown_tool\nAction Input: query"

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call):
            events.append(event)

        # Should have error event
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_tool_execution_error(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should yield error when tool execution fails."""
        mock_tool.run.side_effect = RuntimeError("Tool failed")
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Action: test_tool\nAction Input: query"

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, tools, mock_llm_call):
            events.append(event)

        # Should have error event
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_max_iterations(self, planner, session_context, mock_memory, mock_tool):
        """plan_and_act should stop at max_iterations."""
        planner.max_iterations = 3
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Action: test_tool\nAction Input: query"

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, tools, mock_llm_call):
            events.append(event)

        # Should have error event for max iterations
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_empty_messages(self, planner, mock_memory):
        """plan_and_act should handle empty message history."""
        ctx = SessionContext(session_id="empty", messages=[])

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Hello!"

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, {}, mock_llm_call):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_no_tools(self, planner, session_context, mock_memory):
        """plan_and_act should work with no tools."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: I can answer directly."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)


class TestReActPlannerEventStream:
    """Independent tests for event stream generation."""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner()

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

    @pytest.mark.asyncio
    async def test_text_token_events(self, planner, session_context, mock_memory):
        """plan_and_act should yield text_token events for each token."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Hello"
            yield " World"
            yield "\nFinal Answer: Done."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call):
            events.append(event)

        text_token_events = [e for e in events if e.type == "text_token"]
        assert len(text_token_events) >= 3

    @pytest.mark.asyncio
    async def test_final_answer_event_content(self, planner, session_context, mock_memory):
        """final_answer event should contain the answer text."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: The answer is 42."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call):
            events.append(event)

        final_events = [e for e in events if e.type == "final_answer"]
        assert len(final_events) == 1
        assert "42" in final_events[0].content

    @pytest.mark.asyncio
    async def test_action_event_content(self, planner, session_context, mock_memory):
        """action event should contain tool name."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value="result")
        tools = {"my_tool": mock_tool}

        call_count = 0
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Action: my_tool\nAction Input: query"
            else:
                yield "Final Answer: Done."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, tools, mock_llm_call):
            events.append(event)

        action_events = [e for e in events if e.type == "action"]
        assert len(action_events) == 1
        assert "my_tool" in action_events[0].content

    @pytest.mark.asyncio
    async def test_observation_event_content(self, planner, session_context, mock_memory):
        """observation event should contain tool result."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value="tool result here")
        tools = {"test_tool": mock_tool}

        call_count = 0
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Action: test_tool\nAction Input: query"
            else:
                yield "Final Answer: Done."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, tools, mock_llm_call):
            events.append(event)

        observation_events = [e for e in events if e.type == "observation"]
        assert len(observation_events) == 1
        assert "tool result here" in observation_events[0].content

    @pytest.mark.asyncio
    async def test_events_have_timestamps(self, planner, session_context, mock_memory):
        """All events should have timestamps."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Test."

        events = []
        async for event in planner.plan_and_act(session_context, mock_memory, {}, mock_llm_call):
            events.append(event)

        for event in events:
            assert event.timestamp is not None


class TestReActPlannerBuildPrompt:
    """Independent tests for _build_prompt method."""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner()

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="Memory context")
        return mock

    @pytest.mark.asyncio
    async def test_build_prompt_includes_tools(self, planner, mock_memory):
        """_build_prompt should include available tools."""
        mock_tool = MagicMock()
        mock_tool.description = "Searches the web"
        tools = {"web_search": mock_tool}

        ctx = SessionContext(session_id="test", messages=[])
        prompt = await planner._build_prompt(ctx, mock_memory, tools)

        assert "web_search" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_includes_memory(self, planner, mock_memory):
        """_build_prompt should include memory context."""
        ctx = SessionContext(
            session_id="test",
            messages=[Message(role="user", content="Hello", sender_id="user1")]
        )
        prompt = await planner._build_prompt(ctx, mock_memory, {})

        assert "Memory context" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_includes_message_history(self, planner, mock_memory):
        """_build_prompt should include conversation history."""
        ctx = SessionContext(
            session_id="test",
            messages=[
                Message(role="user", content="Hello", sender_id="user1"),
                Message(role="assistant", content="Hi!", sender_id="assistant"),
            ]
        )
        prompt = await planner._build_prompt(ctx, mock_memory, {})

        assert "Hello" in prompt
        assert "Hi!" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_empty_context(self, planner, mock_memory):
        """_build_prompt should handle empty context."""
        ctx = SessionContext(session_id="test", messages=[])
        prompt = await planner._build_prompt(ctx, mock_memory, {})

        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestReActPlannerBoundaryConditions:
    """Independent tests for boundary conditions."""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner()

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="")
        return mock

    @pytest.mark.asyncio
    async def test_llm_call_returns_empty(self, planner, mock_memory):
        """plan_and_act should handle empty LLM response."""
        ctx = SessionContext(session_id="test", messages=[])

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield ""

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, {}, mock_llm_call):
            events.append(event)

        # Should still yield events
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_long_conversation_history(self, planner, mock_memory):
        """plan_and_act should handle long conversation history."""
        messages = [
            Message(role="user", content=f"Message {i}", sender_id="user1")
            for i in range(50)
        ]
        ctx = SessionContext(session_id="test", messages=messages)

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Handled long history."

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, {}, mock_llm_call):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)

    @pytest.mark.asyncio
    async def test_special_characters_in_messages(self, planner, mock_memory):
        """plan_and_act should handle special characters."""
        ctx = SessionContext(
            session_id="test",
            messages=[Message(role="user", content="Hello! @#$%^&*()", sender_id="user1")]
        )

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Special chars handled."

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, {}, mock_llm_call):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)

    @pytest.mark.asyncio
    async def test_tool_returns_non_string(self, planner, mock_memory):
        """plan_and_act should handle tool returning non-string."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value=42)
        tools = {"test_tool": mock_tool}

        ctx = SessionContext(session_id="test", messages=[])
        call_count = 0

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Action: test_tool\nAction Input: query"
            else:
                yield "Final Answer: Done."

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, tools, mock_llm_call):
            events.append(event)

        # Should handle non-string result
        assert any(e.type == "observation" for e in events)

    @pytest.mark.asyncio
    async def test_memory_retrieval_fails(self, planner):
        """plan_and_act should handle memory retrieval failure."""
        mock_memory = AsyncMock(spec=BaseMemory)
        mock_memory.retrieve = AsyncMock(side_effect=Exception("Memory error"))

        ctx = SessionContext(
            session_id="test",
            messages=[Message(role="user", content="Hello", sender_id="user1")]
        )

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Memory error handled."

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, {}, mock_llm_call):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)


class TestReActPlannerIntegration:
    """Independent integration tests for ReActPlanner."""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner(max_iterations=5)

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="User likes Python")
        return mock

    @pytest.mark.asyncio
    async def test_full_react_loop(self, planner, mock_memory):
        """Test complete ReAct loop with tool call and final answer."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.description = "Searches for information"
        mock_tool.run = AsyncMock(return_value="Python is a programming language")
        tools = {"search": mock_tool}

        ctx = SessionContext(
            session_id="test",
            messages=[Message(role="user", content="What is Python?", sender_id="user1")]
        )

        call_count = 0
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Thought: I need to search for information.\n"
                yield "Action: search\n"
                yield "Action Input: What is Python"
            else:
                yield "Thought: I found the answer.\n"
                yield "Final Answer: Python is a programming language."

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, tools, mock_llm_call):
            events.append(event)

        # Verify event sequence
        event_types = [e.type for e in events]
        assert "text_token" in event_types
        assert "action" in event_types
        assert "observation" in event_types
        assert "final_answer" in event_types

        # Verify tool was called correctly
        mock_tool.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_multiple_tools_workflow(self, planner, mock_memory):
        """Test workflow with multiple different tools."""
        tool1 = AsyncMock(spec=BaseTool)
        tool1.description = "Search tool"
        tool1.run = AsyncMock(return_value="Search results")

        tool2 = AsyncMock(spec=BaseTool)
        tool2.description = "Calculator tool"
        tool2.run = AsyncMock(return_value="42")

        tools = {"search": tool1, "calculator": tool2}

        ctx = SessionContext(
            session_id="test",
            messages=[Message(role="user", content="Complex query", sender_id="user1")]
        )

        call_count = 0
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Action: search\nAction Input: query"
            elif call_count == 2:
                yield "Action: calculator\nAction Input: 2+2"
            else:
                yield "Final Answer: Combined results."

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, tools, mock_llm_call):
            events.append(event)

        # Both tools should be called
        tool1.run.assert_called_once()
        tool2.run.assert_called_once()
