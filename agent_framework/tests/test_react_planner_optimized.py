"""Tests for optimized ToolCallPlanner (formerly ReActPlanner).

This module tests:
1. _build_system_message method (modern)
2. Tool calling via ChatResponse and streaming formats
3. JSON serialization of LLM payloads
"""
import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock
from typing import AsyncIterator

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.core.planner_context import PlannerContext

# Import directly from module to avoid __init__.py export issues
import sys
from pathlib import Path
import importlib.util

sys.path.insert(0, str(Path(__file__).parent.parent))

spec = importlib.util.spec_from_file_location(
    "react_planner",
    Path(__file__).parent.parent / "planners" / "react_planner.py"
)
react_planner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(react_planner_module)

ToolCallPlanner = react_planner_module.ToolCallPlanner
ReActPlanner = react_planner_module.ReActPlanner
ToolCall = react_planner_module.ToolCall
FunctionCall = react_planner_module.FunctionCall
ChatResponse = react_planner_module.ChatResponse
ChatMessage = react_planner_module.ChatMessage


class TestBuildMessages:
    """Test _build_system_message method (modern chat completions API)."""

    @pytest.fixture
    def planner(self):
        """Create ToolCallPlanner instance."""
        return ToolCallPlanner()

    @pytest.fixture
    def mock_memory(self):
        """Create mock memory."""
        mock = AsyncMock(spec=BaseMemory)
        mock.retrieve = AsyncMock(return_value="Relevant context from memory")
        return mock

    @pytest.fixture
    def session_context(self):
        """Create test session context."""
        return SessionContext(
            session_id="test-session-123",
            session_type="private",
            participants=["user1", "user2"],
            messages=[
                Message(role="user", content="What is Python?", sender_id="user1")
            ]
        )

    @pytest.fixture
    def mock_tools(self):
        """Create mock tools."""
        tool1 = MagicMock()
        tool1.description = "A search tool"
        tool2 = MagicMock()
        tool2.description = "A calculator tool"
        return {"search": tool1, "calc": tool2}

    def test_build_system_message_returns_chat_message(self, planner, session_context, mock_memory, mock_tools):
        """_build_system_message should return a ChatMessage."""
        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools=mock_tools,
            messages=session_context.messages,
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        assert isinstance(msg, ChatMessage)
        assert msg.role == "system"

    def test_build_system_message_has_correct_content(self, planner, session_context, mock_memory, mock_tools):
        """System message should contain tool instructions."""
        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools=mock_tools,
            messages=session_context.messages,
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        assert msg.role == "system"
        assert "helpful assistant" in msg.content.lower()

    def test_build_system_message_includes_tool_definitions(self, planner, session_context, mock_memory, mock_tools):
        """System message should include available tool definitions."""
        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools=mock_tools,
            messages=session_context.messages,
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        assert "search" in msg.content
        assert "calc" in msg.content

    def test_build_system_message_adds_memory_context(self, planner, session_context, mock_memory, mock_tools):
        """Should inject memory context into the system message when available."""
        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools=mock_tools,
            messages=session_context.messages,
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        # Memory context is injected into the system message
        assert "Relevant context" in msg.content or "Memory" in msg.content

    def test_build_system_message_conversation_history_in_ctx(self, planner, mock_memory, mock_tools):
        """After refactor, _build_system_message returns only system ChatMessage.
        Conversation history lives in ctx.messages and is combined in plan_and_act."""
        ctx = PlannerContext(
            session_id="test",
            tools=mock_tools,
            messages=[
                Message(role="user", content="First question", sender_id="user1"),
                Message(role="assistant", content="First answer", sender_id="assistant"),
                Message(role="user", content="Follow-up", sender_id="user1"),
            ],
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        # _build_system_message returns only system message
        assert isinstance(msg, ChatMessage)
        assert msg.role == "system"
        # Conversation history remains in ctx.messages
        assert len(ctx.messages) == 3

    def test_build_system_message_returns_only_system_message(self, planner, mock_memory, mock_tools):
        """After refactor, _build_system_message returns only the system ChatMessage."""
        ctx = PlannerContext(
            session_id="test",
            tools=mock_tools,
            messages=[
                Message(role="system", content="System prompt", sender_id="system"),
                Message(role="user", content="User question", sender_id="user1"),
            ],
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        # Should only have one system message (the built one)
        assert isinstance(msg, ChatMessage)
        assert msg.role == "system"

    def test_build_system_message_preserves_ctx_messages(self, planner, mock_memory, mock_tools):
        """After refactor, ctx.messages should be preserved."""
        original_messages = [
            Message(role="user", content="Hello", sender_id="user1"),
            Message(role="tool", content="Tool result", tool_call_id="call_123"),
        ]
        ctx = PlannerContext(
            session_id="test",
            tools=mock_tools,
            messages=original_messages,
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        # _build_system_message returns only system ChatMessage
        assert isinstance(msg, ChatMessage)
        assert msg.role == "system"
        # ctx.messages still has original messages
        assert len(ctx.messages) == 2

    def test_build_system_message_no_longer_stores_side_channel_state(self, planner, session_context, mock_memory, mock_tools):
        """_build_system_message should NOT store tools/session_id as side-channel state.

        After Phase 2A refactor, _build_system_message takes PlannerContext directly.
        No side-channel state should be stored on the planner instance.
        """
        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools=mock_tools,
            messages=session_context.messages,
            memory=mock_memory
        )
        asyncio.run(planner._build_system_message(ctx))
        assert not hasattr(planner, '_tools_ref'), "_tools_ref should not be stored as side-channel state"
        assert not hasattr(planner, '_session_id_ref'), "_session_id_ref should not be stored as side-channel state"

    def test_handle_chat_response_accepts_planner_context(self, planner):
        """_handle_chat_response should accept PlannerContext as parameter."""
        import inspect
        sig = inspect.signature(planner._handle_chat_response)
        param_names = list(sig.parameters.keys())
        assert 'ctx' in param_names, "_handle_chat_response must accept 'ctx' parameter (PlannerContext)"


class TestToolCallLoopStreaming:
    """Test tool call loop with streaming events."""

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
    def mock_tool(self):
        """Create mock tool."""
        mock = AsyncMock(spec=BaseTool)
        mock.name = "test_tool"
        mock.description = "A test tool"
        mock.run = AsyncMock(return_value="Tool executed successfully")
        return mock

    @pytest.fixture
    def session_context(self):
        """Create test session context."""
        return SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Use the tool", sender_id="user1")]
        )

    def test_tool_call_yields_action_event(self, planner, session_context, mock_memory, mock_tool):
        """Should yield 'action' event when LLM requests tool call."""
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})

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
        assert len(action_events) > 0
        assert "test_tool" in action_events[0].content

    def test_tool_call_executes_tool(self, planner, session_context, mock_memory, mock_tool):
        """Should execute the tool when LLM requests tool call."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Done")

        async def run():
            ctx = PlannerContext(
                session_id=session_context.session_id,
                tools=tools,
                messages=session_context.messages,
                memory=mock_memory
            )
            async for event in planner.plan_and_act(ctx, mock_llm_call):
                pass

        asyncio.run(run())

        assert mock_tool.run.call_count == 1

    def test_tool_call_yields_observation_event(self, planner, session_context, mock_memory, mock_tool):
        """Should yield 'observation' event with tool result."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Done")

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
        assert len(observation_events) > 0
        assert "Tool executed successfully" in observation_events[0].content

    def test_unknown_tool_yields_error_event(self, planner, session_context, mock_memory):
        """Should yield error event when tool name is unknown."""

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "unknown_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "unknown_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})

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

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) > 0

    def test_async_tool_execution(self, planner, session_context, mock_memory):
        """Both async and sync tools should execute correctly."""
        async_tool = AsyncMock(spec=BaseTool)
        async_tool.name = "async_tool"
        async_tool.run = AsyncMock(return_value="async result")

        tools = {"async_tool": async_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "async_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "async_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="")

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

        assert async_tool.run.called

    def test_tool_execution_exception_handling(self, planner, session_context, mock_memory):
        """Should handle tool execution exceptions gracefully."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.name = "failing_tool"
        mock_tool.run = AsyncMock(side_effect=RuntimeError("Tool execution failed"))
        tools = {"failing_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "failing_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "failing_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})

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

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) > 0


class TestToolCallLoopChatResponse:
    """Test tool call loop with non-streaming ChatResponse."""

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
            session_id="test-session",
            messages=[Message(role="user", content="Test", sender_id="user1")]
        )

    def test_chat_response_with_tool_calls(self, planner, session_context, mock_memory):
        """Should handle ChatResponse with tool_calls."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.run = AsyncMock(return_value="Tool result")
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
                return ChatResponse(content="Final answer after tool call.")

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

        assert any(e.type == "observation" for e in events)
        assert any(e.type == "final_answer" for e in events)
        assert mock_tool.run.call_count == 1

    def test_chat_response_final_answer_only(self, planner, session_context, mock_memory):
        """Should handle ChatResponse with content only (no tool calls)."""

        def mock_llm_call(messages, tools):
            return ChatResponse(content="The answer is 42.")

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


class TestMaxIterationsProtection:
    """Test max_iterations protection."""

    @pytest.fixture
    def planner(self):
        """Create ToolCallPlanner with max_iterations=3."""
        return ToolCallPlanner(max_iterations=3)

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
        mock.name = "test_tool"
        mock.run = AsyncMock(return_value="result")
        return mock

    @pytest.fixture
    def session_context(self):
        """Create test session context."""
        return SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Test", sender_id="user1")]
        )

    def test_max_iterations_stops_loop(self, planner, session_context, mock_memory, mock_tool):
        """Should stop loop when max_iterations reached."""
        tools = {"test_tool": mock_tool}
        llm_call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal llm_call_count
            llm_call_count += 1
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "test_tool",
                                  "tool_call_id": f"call_{llm_call_count}",
                                  "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "test_tool",
                                  "tool_call_id": f"call_{llm_call_count}",
                                  "arguments": '{"input": "test"}'})

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

        assert llm_call_count == planner.max_iterations

    def test_max_iterations_yields_error(self, planner, session_context, mock_memory, mock_tool):
        """Should yield error event when max_iterations reached without final answer."""
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                  "arguments": '{"input": "test"}'})

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

        error_events = [e for e in events if e.type == "error"]
        assert len(error_events) > 0

    def test_custom_max_iterations_respected(self, mock_memory, mock_tool):
        """Custom max_iterations should be respected."""
        custom_planner = ToolCallPlanner(max_iterations=5)
        tools = {"test_tool": mock_tool}
        llm_call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal llm_call_count
            llm_call_count += 1
            yield Event(type="tool_call_start", content="",
                        metadata={"tool_name": "test_tool",
                                  "tool_call_id": f"call_{llm_call_count}",
                                  "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="",
                        metadata={"tool_name": "test_tool",
                                  "tool_call_id": f"call_{llm_call_count}",
                                  "arguments": '{"input": "test"}'})

        async def collect():
            ctx = PlannerContext(
                session_id="test",
                tools=tools,
                messages=[Message(role="user", content="Test", sender_id="user1")],
                memory=mock_memory,
                max_iterations=custom_planner.max_iterations
            )
            async for event in custom_planner.plan_and_act(ctx, mock_llm_call):
                pass

        asyncio.run(collect())

        assert llm_call_count == 5

    def test_final_answer_exits_before_max_iterations(self, planner, session_context, mock_memory, mock_tool):
        """Should exit loop as soon as final answer received."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="",
                            metadata={"tool_name": "test_tool", "tool_call_id": "call_1",
                                      "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Done")

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

        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) > 0
        assert call_count == 2


class TestJSONSerialization:
    """Test JSON serialization boundaries for LLM chat payload."""

    @pytest.fixture
    def planner(self):
        """Create ToolCallPlanner instance."""
        return ToolCallPlanner()

    def test_chat_message_serializable(self):
        """ChatMessage objects should be serializable via json.dumps()."""
        msg = ChatMessage(role="user", content="Hello, world!")
        result = json.dumps(msg.__dict__)
        assert "user" in result
        assert "Hello, world!" in result

    def test_chat_message_with_tool_call_id_serializable(self):
        """ChatMessage with tool_call_id should be serializable."""
        msg = ChatMessage(role="tool", content="result", tool_call_id="call_123")
        result = json.dumps(msg.__dict__)
        parsed = json.loads(result)
        assert parsed["tool_call_id"] == "call_123"

    def test_messages_list_serializable(self):
        """List of ChatMessage objects should be serializable."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hello!"),
            ChatMessage(role="assistant", content="Hi there!"),
            ChatMessage(role="tool", content="tool result", tool_call_id="call_1"),
        ]
        messages_dicts = [{"role": m.role, "content": m.content, "name": m.name, "tool_call_id": m.tool_call_id} for m in messages]
        result = json.dumps(messages_dicts)
        parsed = json.loads(result)
        assert len(parsed) == 4
        assert parsed[0]["role"] == "system"
        assert parsed[3]["tool_call_id"] == "call_1"

    def test_tool_call_serializable(self):
        """ToolCall objects should be serializable."""
        tc = ToolCall(id="call_1", type="function",
                      function=FunctionCall(name="get_weather", arguments='{"city":"Beijing"}'))
        tc_dict = {
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        }
        result = json.dumps(tc_dict)
        parsed = json.loads(result)
        assert parsed["id"] == "call_1"
        assert parsed["function"]["name"] == "get_weather"

    def test_full_payload_serializable(self):
        """Full LLM chat payload should be serializable."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "A calculator tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        payload = {
            "model": "gpt-4",
            "messages": messages,
            "tools": tools,
        }
        result = json.dumps(payload)
        parsed = json.loads(result)
        assert parsed["model"] == "gpt-4"
        assert len(parsed["messages"]) == 2
        assert len(parsed["tools"]) == 1

    def test_llm_chat_payload_with_chat_messages(self):
        """LLM chat payload built from ChatMessage objects should serialize."""
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Use the calculator"),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "A calculator tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]

        planner = ToolCallPlanner()
        messages_dicts = [planner._chat_message_to_dict(m) for m in messages]

        payload = {
            "model": "gpt-4",
            "messages": messages_dicts,
            "tools": tools,
        }

        result = json.dumps(payload)
        parsed = json.loads(result)
        assert len(parsed["messages"]) == 2
        assert parsed["messages"][0]["role"] == "system"

    def test_unicode_content_serializable(self):
        """Unicode content should serialize correctly."""
        messages = [
            ChatMessage(role="user", content="你好世界！"),
            ChatMessage(role="assistant", content="Hello, 世界!"),
        ]
        planner = ToolCallPlanner()
        messages_dicts = [planner._chat_message_to_dict(m) for m in messages]
        result = json.dumps(messages_dicts)
        parsed = json.loads(result)
        assert parsed[0]["content"] == "你好世界！"
        assert "世界" in parsed[1]["content"]

    def test_special_characters_in_arguments_serializable(self):
        """Special characters in tool arguments should serialize correctly."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="search",
                arguments='{"query": "test with \\"quotes\\" and \\\\backslashes\\\\"}'
            )
        )
        tc_dict = {
            "id": tc.id,
            "type": tc.type,
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        }
        result = json.dumps(tc_dict)
        parsed = json.loads(result)
        assert "quotes" in parsed["function"]["arguments"]

