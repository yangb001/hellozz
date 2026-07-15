"""Tests for ModernReActPlanner - Updated implementation.

NOTE: This test file was updated to match the ModernReAct implementation.
The old Action-based tests have been replaced with ToolCall-based tests.

If you're looking for the old-style ReAct tests, they have been moved to
test_react_planner_legacy.py or deleted.

Reference: ModernReAct uses chat completions API with tool_calls support.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, AsyncIterator

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.planners.react_planner import ReActPlanner, ChatMessage, ChatResponse
from agent_framework.interfaces.llm_types import ToolCall, FunctionCall


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
        """Create a ReActPlanner instance."""
        return ReActPlanner()

    def test_planner_initialization(self, planner):
        """Test ReActPlanner initialization."""
        assert planner.name == "react"
        assert "ReAct" in planner.description
        assert planner.max_iterations == 10

    def test_planner_custom_name(self):
        """Test ReActPlanner with custom name."""
        planner = ReActPlanner(name="custom_react", description="Custom ReAct planner", max_iterations=5)
        assert planner.name == "custom_react"
        assert planner.description == "Custom ReAct planner"
        assert planner.max_iterations == 5

    @pytest.mark.asyncio
    async def test_build_messages_basic(self, planner, session_context, mock_memory, mock_tools):
        """Test _build_messages creates proper messages array."""
        messages = await planner._build_messages(session_context, mock_memory, mock_tools)

        assert isinstance(messages, list)
        assert len(messages) > 0
        # First message should be from system or user
        assert messages[0]["role"] in ["system", "user"]

    @pytest.mark.asyncio
    async def test_build_tools_dict(self, planner, mock_tools):
        """Test _build_tools_dict creates proper OpenAI tools format."""
        tools = planner._build_tools_dict(mock_tools)

        assert isinstance(tools, list)
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert "function" in tools[0]
        assert tools[0]["function"]["name"] == "test_tool"