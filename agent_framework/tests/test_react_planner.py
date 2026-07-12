"""Tests for ReActPlanner - TDD implementation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, AsyncIterator

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.planners.react_planner import ReActPlanner, Action


class TestAction:
    """Test Action data class."""

    def test_action_creation_tool_call(self):
        """Test creating a tool_call Action."""
        action = Action(
            type="tool_call",
            tool="web_search",
            input="Python best practices"
        )
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python best practices"
        assert action.content is None

    def test_action_creation_final_answer(self):
        """Test creating a final_answer Action."""
        action = Action(
            type="final_answer",
            content="The answer is 42"
        )
        assert action.type == "final_answer"
        assert action.content == "The answer is 42"
        assert action.tool is None
        assert action.input is None

    def test_action_creation_thought(self):
        """Test creating a thought Action."""
        action = Action(
            type="thought",
            content="I need to search for more information"
        )
        assert action.type == "thought"
        assert action.content == "I need to search for more information"


class TestReActPlanner:
    """Test ReActPlanner class."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory instance."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="Relevant memory context")
        return mock

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool instance."""
        mock = AsyncMock(spec=BaseTool)
        mock.name = "test_tool"
        mock.description = "A test tool"
        mock.run = AsyncMock(return_value="Tool execution result")
        return mock

    @pytest.fixture
    def mock_tools(self, mock_tool):
        """Create a dictionary of mock tools."""
        return {"test_tool": mock_tool}

    @pytest.fixture
    def session_context(self):
        """Create a test session context."""
        return SessionContext(
            session_id="test-session-123",
            session_type="private",
            participants=["user1"],
            messages=[
                Message(role="user", content="What is Python?", sender_id="user1")
            ]
        )

    @pytest.fixture
    def planner(self):
        """Create a ReActPlanner instance."""
        return ReActPlanner()

    def test_planner_initialization(self, planner):
        """Test ReActPlanner initialization."""
        assert planner.name == "react"
        assert "ReAct" in planner.description

    def test_planner_custom_name(self):
        """Test ReActPlanner with custom name."""
        planner = ReActPlanner(name="custom_react", description="Custom ReAct planner")
        assert planner.name == "custom_react"
        assert planner.description == "Custom ReAct planner"

    @pytest.mark.asyncio
    async def test_plan_and_act_final_answer(self, planner, session_context, mock_memory, mock_tools):
        """Test plan_and_act with immediate final answer."""
        # Mock LLM call that returns a final answer
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Thought: I can answer this directly.\n"
            yield "Final Answer: Python is a programming language."

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # Should have text_token events and a final_answer event
        assert len(events) > 0
        assert any(e.type == "text_token" for e in events)
        assert any(e.type == "final_answer" for e in events)

        # Check final answer content
        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) == 1
        assert "Python is a programming language" in final_answer_events[0].content

    @pytest.mark.asyncio
    async def test_plan_and_act_tool_call(self, planner, session_context, mock_memory, mock_tools, mock_tool):
        """Test plan_and_act with tool call."""
        call_count = 0

        # Mock LLM call that first calls a tool, then gives final answer
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Thought: I need to use the test tool.\n"
                yield "Action: test_tool\n"
                yield "Action Input: test query"
            else:
                yield "Thought: Now I have the information.\n"
                yield "Final Answer: Based on the tool result, the answer is test."

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # Should have action, observation, and final_answer events
        assert any(e.type == "action" for e in events)
        assert any(e.type == "observation" for e in events)
        assert any(e.type == "final_answer" for e in events)

        # Verify tool was called
        mock_tool.run.assert_called_once_with("test query", session_id="test-session-123")

    @pytest.mark.asyncio
    async def test_plan_and_act_multiple_tool_calls(self, planner, session_context, mock_memory, mock_tools, mock_tool):
        """Test plan_and_act with multiple sequential tool calls."""
        call_count = 0

        # Mock LLM call that makes two tool calls before final answer
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Thought: First tool call.\n"
                yield "Action: test_tool\n"
                yield "Action Input: first query"
            elif call_count == 2:
                yield "Thought: Second tool call.\n"
                yield "Action: test_tool\n"
                yield "Action Input: second query"
            else:
                yield "Final Answer: Combined results from both calls."

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # Tool should be called twice
        assert mock_tool.run.call_count == 2

        # Should have two observation events
        observation_events = [e for e in events if e.type == "observation"]
        assert len(observation_events) == 2

    @pytest.mark.asyncio
    async def test_plan_and_act_unknown_tool(self, planner, session_context, mock_memory, mock_tools):
        """Test plan_and_act with unknown tool name."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Thought: I'll use a tool.\n"
            yield "Action: unknown_tool\n"
            yield "Action Input: some input"

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # Should have an error event for unknown tool
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_tool_execution_error(self, planner, session_context, mock_memory, mock_tools, mock_tool):
        """Test plan_and_act when tool execution fails."""
        # Make tool raise an exception
        mock_tool.run.side_effect = RuntimeError("Tool execution failed")

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Thought: I'll use the tool.\n"
            yield "Action: test_tool\n"
            yield "Action Input: failing query"

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # Should have an error event
        assert any(e.type == "error" for e in events)

    @pytest.mark.asyncio
    async def test_build_prompt_basic(self, planner, session_context, mock_memory, mock_tools):
        """Test _build_prompt method."""
        prompt = await planner._build_prompt(session_context, mock_memory, mock_tools)

        # Prompt should contain session context
        assert "test-session-123" in prompt or "user1" in prompt

        # Should contain tool information
        assert "test_tool" in prompt

        # Should contain memory context
        assert "Relevant memory context" in prompt

    @pytest.mark.asyncio
    async def test_build_prompt_with_message_history(self, planner, mock_memory, mock_tools):
        """Test _build_prompt with message history."""
        ctx = SessionContext(
            session_id="test-session",
            messages=[
                Message(role="user", content="Hello", sender_id="user1"),
                Message(role="assistant", content="Hi there!", sender_id="assistant"),
                Message(role="user", content="How are you?", sender_id="user1")
            ]
        )

        prompt = await planner._build_prompt(ctx, mock_memory, mock_tools)

        # Should include message history
        assert "Hello" in prompt
        assert "Hi there!" in prompt
        assert "How are you?" in prompt

    def test_parse_action_final_answer(self, planner):
        """Test _parse_action with final answer."""
        text = "Thought: I have enough information.\nFinal Answer: The answer is 42."
        action = planner._parse_action(text)

        assert action.type == "final_answer"
        assert "The answer is 42" in action.content

    def test_parse_action_tool_call(self, planner):
        """Test _parse_action with tool call."""
        text = "Thought: I need to search.\nAction: web_search\nAction Input: Python tutorials"
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    def test_parse_action_no_action(self, planner):
        """Test _parse_action with no clear action."""
        text = "Just thinking about the problem..."
        action = planner._parse_action(text)

        # Should default to thought or handle gracefully
        assert action.type in ["thought", "final_answer"]

    def test_parse_action_multiple_final_answers(self, planner):
        """Test _parse_action with multiple final answer markers (takes last)."""
        text = "Final Answer: First attempt.\nFinal Answer: Second attempt."
        action = planner._parse_action(text)

        assert action.type == "final_answer"
        assert "Second attempt" in action.content

    def test_parse_action_case_insensitive(self, planner):
        """Test _parse_action with case variations."""
        text = "thought: thinking...\nACTION: test_tool\nACTION INPUT: query"
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert action.tool == "test_tool"

    @pytest.mark.asyncio
    async def test_plan_and_act_empty_context(self, planner, mock_memory, mock_tools):
        """Test plan_and_act with empty session context."""
        ctx = SessionContext(session_id="empty-session", messages=[])

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Hello! How can I help?"

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, mock_tools, mock_llm_call):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_no_tools(self, planner, session_context, mock_memory):
        """Test plan_and_act with no tools available."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Thought: No tools available.\nFinal Answer: I can answer directly."

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, {}, mock_llm_call
        ):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_with_memory_retrieval(self, planner, session_context, mock_memory, mock_tools):
        """Test that plan_and_act retrieves memory context."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Based on memory."

        await planner._build_prompt(session_context, mock_memory, mock_tools)

        # Memory retrieve should be called
        mock_memory.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_and_act_event_metadata(self, planner, session_context, mock_memory, mock_tools):
        """Test that events contain proper metadata."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Thought: thinking\nFinal Answer: answer"

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # All events should have timestamps
        for event in events:
            assert event.timestamp is not None

    @pytest.mark.asyncio
    async def test_plan_and_act_text_token_events(self, planner, session_context, mock_memory, mock_tools):
        """Test that text tokens are yielded as events."""
        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Hello"
            yield " World"
            yield "\nFinal Answer: Done."

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # Should have multiple text_token events
        text_token_events = [e for e in events if e.type == "text_token"]
        assert len(text_token_events) >= 3

    @pytest.mark.asyncio
    async def test_plan_and_act_tool_not_async(self, planner, session_context, mock_memory):
        """Test plan_and_act with synchronous tool."""
        # Create a tool with synchronous run method
        sync_tool = MagicMock(spec=BaseTool)
        sync_tool.name = "sync_tool"
        sync_tool.run = MagicMock(return_value="Sync result")

        tools = {"sync_tool": sync_tool}

        call_count = 0

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Thought: Using sync tool.\n"
                yield "Action: sync_tool\n"
                yield "Action Input: query"
            else:
                yield "\nFinal Answer: Done."

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, tools, mock_llm_call
        ):
            events.append(event)

        # Should handle sync tool gracefully
        assert any(e.type == "observation" for e in events)

    @pytest.mark.asyncio
    async def test_plan_and_act_max_iterations(self, planner, session_context, mock_memory, mock_tools, mock_tool):
        """Test plan_and_act respects max iterations limit."""
        # Set a low max iterations
        planner.max_iterations = 3

        call_count = 0

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            yield f"Thought: Iteration {call_count}\n"
            yield "Action: test_tool\n"
            yield "Action Input: query"

        events = []
        async for event in planner.plan_and_act(
            session_context, mock_memory, mock_tools, mock_llm_call
        ):
            events.append(event)

        # Should stop after max_iterations
        assert call_count <= 3

    @pytest.mark.asyncio
    async def test_plan_and_act_with_system_message(self, planner, mock_memory, mock_tools):
        """Test plan_and_act with system message in context."""
        ctx = SessionContext(
            session_id="test-session",
            messages=[
                Message(role="system", content="You are a helpful assistant.", sender_id="system"),
                Message(role="user", content="Hello", sender_id="user1")
            ]
        )

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            yield "Final Answer: Hello! How can I help?"

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, mock_tools, mock_llm_call):
            events.append(event)

        assert any(e.type == "final_answer" for e in events)

    def test_parse_action_with_extra_whitespace(self, planner):
        """Test _parse_action handles extra whitespace."""
        text = "  Thought: thinking...  \n  Action: tool_name  \n  Action Input: input  "
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert action.tool == "tool_name"
        assert action.input == "input"

    def test_parse_action_with_colons_in_input(self, planner):
        """Test _parse_action with colons in action input."""
        text = "Action: web_search\nAction Input: http://example.com"
        action = planner._parse_action(text)

        assert action.type == "tool_call"
        assert action.input == "http://example.com"

    @pytest.mark.asyncio
    async def test_plan_and_act_preserves_conversation_flow(self, planner, mock_memory, mock_tools, mock_tool):
        """Test that plan_and_act maintains conversation flow."""
        ctx = SessionContext(
            session_id="test-session",
            messages=[
                Message(role="user", content="What's the weather?", sender_id="user1"),
                Message(role="assistant", content="I'll check that for you.", sender_id="assistant"),
                Message(role="user", content="In New York", sender_id="user1")
            ]
        )

        call_count = 0

        async def mock_llm_call(prompt: str) -> AsyncIterator[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield "Thought: I need to check the weather.\nAction: test_tool\nAction Input: weather New York"
            else:
                yield "Thought: Got the result.\nFinal Answer: The weather in New York is sunny."

        events = []
        async for event in planner.plan_and_act(ctx, mock_memory, mock_tools, mock_llm_call):
            events.append(event)

        # Should complete successfully
        assert any(e.type == "final_answer" for e in events)
        assert "New York" in [e.content for e in events if e.type == "final_answer"][0]