"""Tests for optimized ReActPlanner - TDD implementation for tool calling improvements."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncIterator

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.planners.react_planner import ReActPlanner, Action


class TestOptimizedParseAction:
    """Test optimized _parse_action method with various LLM response formats."""

    @pytest.fixture
    def planner(self):
        """Create a ReActPlanner instance."""
        return ReActPlanner()

    # Standard format tests
    def test_parse_standard_final_answer(self, planner):
        """Test parsing standard Final Answer format."""
        text = "Thought: I have enough information.\nFinal Answer: The answer is 42."
        action = planner._parse_action(text)
        assert action.type == "final_answer"
        assert "42" in action.content

    def test_parse_standard_tool_call(self, planner):
        """Test parsing standard Action/Action Input format."""
        text = "Thought: I need to search.\nAction: web_search\nAction Input: Python tutorials"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    # Alternative format tests - "Use tool" pattern
    def test_parse_use_tool_format(self, planner):
        """Test parsing 'Use tool' format."""
        text = "I need to search for information.\nUse tool: web_search\nInput: Python tutorials"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    # Alternative format tests - "Tool:" pattern
    def test_parse_tool_colon_format(self, planner):
        """Test parsing 'Tool:' format."""
        text = "Let me search for this.\nTool: web_search\nQuery: Python best practices"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python best practices"

    # Alternative format tests - "Call" pattern
    def test_parse_call_tool_format(self, planner):
        """Test parsing 'Call tool' format."""
        text = "I'll search for that.\nCall web_search with query: Python tutorials"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    # Alternative format tests - JSON-like format
    def test_parse_json_like_format(self, planner):
        """Test parsing JSON-like format."""
        text = '{"tool": "web_search", "input": "Python tutorials"}'
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    # Alternative format tests - Function call format
    def test_parse_function_call_format(self, planner):
        """Test parsing function call format."""
        text = "web_search('Python tutorials')"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    # Alternative format tests - I will format
    def test_parse_i_will_format(self, planner):
        """Test parsing 'I will' format."""
        text = "I will use web_search to find: Python tutorials"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"
        assert action.input == "Python tutorials"

    # Alternative final answer formats
    def test_parse_answer_is_format(self, planner):
        """Test parsing 'The answer is' format."""
        text = "Based on my research, the answer is Python is a programming language."
        action = planner._parse_action(text)
        assert action.type == "final_answer"
        assert "Python is a programming language" in action.content

    def test_parse_in_conclusion_format(self, planner):
        """Test parsing 'In conclusion' format."""
        text = "In conclusion, Python is a versatile programming language."
        action = planner._parse_action(text)
        assert action.type == "final_answer"
        assert "Python is a versatile programming language" in action.content

    def test_parse_to_summarize_format(self, planner):
        """Test parsing 'To summarize' format."""
        text = "To summarize, Python is widely used in data science."
        action = planner._parse_action(text)
        assert action.type == "final_answer"
        assert "Python is widely used in data science" in action.content

    # Edge cases
    def test_parse_empty_text(self, planner):
        """Test parsing empty text."""
        text = ""
        action = planner._parse_action(text)
        assert action.type == "final_answer"
        assert action.content == ""

    def test_parse_only_whitespace(self, planner):
        """Test parsing whitespace-only text."""
        text = "   \n\t  "
        action = planner._parse_action(text)
        assert action.type == "final_answer"

    def test_parse_multiple_actions_takes_last(self, planner):
        """Test that multiple actions takes the last one."""
        text = "Action: first_tool\nAction Input: query1\nAction: second_tool\nAction Input: query2"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "second_tool"
        assert action.input == "query2"

    def test_parse_final_answer_overrides_action(self, planner):
        """Test that Final Answer overrides Action."""
        text = "Action: web_search\nAction Input: query\nFinal Answer: The answer is 42."
        action = planner._parse_action(text)
        assert action.type == "final_answer"
        assert "42" in action.content

    def test_parse_tool_name_with_underscores(self, planner):
        """Test parsing tool name with underscores."""
        text = "Action: web_search_tool\nAction Input: query"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search_tool"

    def test_parse_tool_name_with_hyphens(self, planner):
        """Test parsing tool name with hyphens."""
        text = "Action: web-search\nAction Input: query"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web-search"

    def test_parse_input_with_colons(self, planner):
        """Test parsing input with colons (e.g., URLs)."""
        text = "Action: web_search\nAction Input: http://example.com"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.input == "http://example.com"

    def test_parse_input_with_newlines(self, planner):
        """Test parsing input with newlines."""
        text = "Action: web_search\nAction Input: line1\nline2\nline3"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert "line1" in action.input

    def test_parse_case_insensitive_action(self, planner):
        """Test case-insensitive action parsing."""
        text = "ACTION: web_search\nACTION INPUT: query"
        action = planner._parse_action(text)
        assert action.type == "tool_call"
        assert action.tool == "web_search"

    def test_parse_case_insensitive_final_answer(self, planner):
        """Test case-insensitive final answer parsing."""
        text = "FINAL ANSWER: The answer is 42."
        action = planner._parse_action(text)
        assert action.type == "final_answer"
        assert "42" in action.content


class TestOptimizedBuildPrompt:
    """Test optimized _build_prompt method with tool call examples."""

    @pytest.fixture
    def mock_memory(self):
        """Create a mock memory instance."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="Relevant memory context")
        return mock

    @pytest.fixture
    def mock_tool_web_search(self):
        """Create a mock web search tool."""
        mock = AsyncMock(spec=BaseTool)
        mock.name = "web_search"
        mock.description = "Search the web for information"
        return mock

    @pytest.fixture
    def mock_tool_calculator(self):
        """Create a mock calculator tool."""
        mock = AsyncMock(spec=BaseTool)
        mock.name = "calculator"
        mock.description = "Perform mathematical calculations"
        return mock

    @pytest.fixture
    def planner(self):
        """Create a ReActPlanner instance."""
        return ReActPlanner()

    @pytest.mark.asyncio
    async def test_prompt_includes_tool_examples(self, planner, mock_memory, mock_tool_web_search):
        """Test that prompt includes tool call examples."""
        tools = {"web_search": mock_tool_web_search}
        ctx = SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Hello", sender_id="user1")]
        )

        prompt = await planner._build_prompt(ctx, mock_memory, tools)

        # Should include example format
        assert "Example" in prompt or "example" in prompt
        assert "Action:" in prompt
        assert "Action Input:" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_tool_names(self, planner, mock_memory, mock_tool_web_search, mock_tool_calculator):
        """Test that prompt includes all tool names."""
        tools = {
            "web_search": mock_tool_web_search,
            "calculator": mock_tool_calculator
        }
        ctx = SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Hello", sender_id="user1")]
        )

        prompt = await planner._build_prompt(ctx, mock_memory, tools)

        assert "web_search" in prompt
        assert "calculator" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_tool_descriptions(self, planner, mock_memory, mock_tool_web_search):
        """Test that prompt includes tool descriptions."""
        tools = {"web_search": mock_tool_web_search}
        ctx = SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Hello", sender_id="user1")]
        )

        prompt = await planner._build_prompt(ctx, mock_memory, tools)

        assert "Search the web for information" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_format_reminder(self, planner, mock_memory, mock_tool_web_search):
        """Test that prompt includes format reminder."""
        tools = {"web_search": mock_tool_web_search}
        ctx = SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Hello", sender_id="user1")]
        )

        prompt = await planner._build_prompt(ctx, mock_memory, tools)

        # Should remind about the format
        assert "Thought:" in prompt
        assert "Final Answer:" in prompt

    @pytest.mark.asyncio
    async def test_prompt_with_no_tools(self, planner, mock_memory):
        """Test prompt when no tools are available."""
        tools = {}
        ctx = SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Hello", sender_id="user1")]
        )

        prompt = await planner._build_prompt(ctx, mock_memory, tools)

        # Should still be a valid prompt
        assert len(prompt) > 0
        assert "ReAct" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_conversation_history(self, planner, mock_memory, mock_tool_web_search):
        """Test that prompt includes conversation history."""
        tools = {"web_search": mock_tool_web_search}
        ctx = SessionContext(
            session_id="test-session",
            messages=[
                Message(role="user", content="What is Python?", sender_id="user1"),
                Message(role="assistant", content="Python is a programming language.", sender_id="assistant"),
                Message(role="user", content="Tell me more about it.", sender_id="user1")
            ]
        )

        prompt = await planner._build_prompt(ctx, mock_memory, tools)

        assert "What is Python?" in prompt
        assert "Tell me more about it." in prompt


class TestToolCallingSuccessRate:
    """Test tool calling success rate with various formats."""

    @pytest.fixture
    def planner(self):
        """Create a ReActPlanner instance."""
        return ReActPlanner()

    def test_various_tool_call_formats_recognized(self, planner):
        """Test that various tool call formats are recognized."""
        formats = [
            # Standard format
            "Action: web_search\nAction Input: query",
            # Use tool format
            "Use tool: web_search\nInput: query",
            # Tool colon format
            "Tool: web_search\nQuery: query",
            # Call format
            "Call web_search with query: query",
            # JSON-like format
            '{"tool": "web_search", "input": "query"}',
            # Function call format
            "web_search('query')",
            # I will format
            "I will use web_search to find: query",
        ]

        for fmt in formats:
            action = planner._parse_action(fmt)
            assert action.type == "tool_call", f"Failed to recognize format: {fmt}"
            assert action.tool == "web_search", f"Wrong tool name for format: {fmt}"

    def test_various_final_answer_formats_recognized(self, planner):
        """Test that various final answer formats are recognized."""
        formats = [
            # Standard format
            "Final Answer: The answer is 42.",
            # Answer is format
            "The answer is 42.",
            # In conclusion format
            "In conclusion, the answer is 42.",
            # To summarize format
            "To summarize, the answer is 42.",
        ]

        for fmt in formats:
            action = planner._parse_action(fmt)
            assert action.type == "final_answer", f"Failed to recognize format: {fmt}"
            assert "42" in action.content, f"Wrong content for format: {fmt}"

    def test_edge_case_formats(self, planner):
        """Test edge case formats."""
        # Tool name with numbers
        action = planner._parse_action("Action: tool123\nAction Input: query")
        assert action.type == "tool_call"
        assert action.tool == "tool123"

        # Input with special characters
        action = planner._parse_action("Action: web_search\nAction Input: hello@world.com")
        assert action.type == "tool_call"
        assert "hello@world.com" in action.input

        # Empty input
        action = planner._parse_action("Action: web_search\nAction Input: ")
        assert action.type == "tool_call"
        assert action.tool == "web_search"