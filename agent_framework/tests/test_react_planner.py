"""Tests for ToolCallPlanner - Updated implementation.

NOTE: This test file was updated to match the ToolCallPlanner implementation.
The old Action-based tests have been replaced with ToolCall-based tests.

Reference: ToolCallPlanner uses chat completions API with tool_calls support
and PlannerContext for explicit state management.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, AsyncIterator

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.planners.react_planner import ToolCallPlanner, ReActPlanner, ChatMessage, ChatResponse
from agent_framework.interfaces.llm_types import ToolCall, FunctionCall
from agent_framework.core.planner_context import PlannerContext


class TestToolCall:
    """Test ToolCall data class (replaces old Action class)."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        tc = ToolCall(
            id="call_abc123",
            type="function",
            function=FunctionCall(
                name="web_search",
                arguments='{"query": "Python best practices"}'
            )
        )
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "web_search"
        assert tc.function.arguments == '{"query": "Python best practices"}'


class TestChatMessage:
    """Test ChatMessage data class."""

    def test_chat_message_creation(self):
        """Test creating a ChatMessage."""
        msg = ChatMessage(
            role="user",
            content="What is Python?"
        )
        assert msg.role == "user"
        assert msg.content == "What is Python?"
        assert msg.name is None
        assert msg.tool_call_id is None


class TestChatResponse:
    """Test ChatResponse data class."""

    def test_chat_response_content_only(self):
        """Test ChatResponse with only content."""
        resp = ChatResponse(content="The answer is 42")
        assert resp.content == "The answer is 42"
        assert len(resp.tool_calls) == 0
        assert not resp.has_tool_calls

    def test_chat_response_with_tool_calls(self):
        """Test ChatResponse with tool calls."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="calc", arguments="1+1")
        )
        resp = ChatResponse(content="", tool_calls=[tc])
        assert resp.has_tool_calls
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "calc"


class TestReActPlanner:
    """Test ModernReActPlanner class."""

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
        """Create a ToolCallPlanner instance."""
        return ToolCallPlanner()

    def test_planner_initialization(self, planner):
        """Test ToolCallPlanner initialization."""
        assert planner.name == "tool_call"
        assert "Tool-call" in planner.description
        assert planner.max_iterations == 10

    def test_planner_custom_name(self):
        """Test ToolCallPlanner with custom name."""
        planner = ToolCallPlanner(name="custom_planner", description="Custom planner", max_iterations=5)
        assert planner.name == "custom_planner"
        assert planner.description == "Custom planner"
        assert planner.max_iterations == 5

    @pytest.mark.asyncio
    async def test_build_system_message_basic(self, planner, mock_memory, mock_tools):
        """Test _build_system_message returns ChatMessage using PlannerContext."""
        ctx = PlannerContext(
            session_id="test-session-123",
            tools=mock_tools,
            memory=mock_memory
        )
        msg = await planner._build_system_message(ctx)

        assert isinstance(msg, ChatMessage)
        assert msg.role == "system"

    def test_build_system_message_content(self, planner, mock_tools):
        """Test _build_system_message_content creates proper system message string with tool info."""
        system_msg = planner._build_system_message_content(mock_tools)

        assert isinstance(system_msg, str)
        assert "test_tool" in system_msg
        assert "A test tool" in system_msg