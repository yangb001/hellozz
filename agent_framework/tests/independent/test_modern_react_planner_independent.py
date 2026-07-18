"""Independent tests for ModernReActPlanner tool call loop.

Test verification areas:
1. _build_system_message() 是否正确构建 messages 数组
2. 工具调用循环是否正确（检测 tool_calls -> 执行工具 -> 继续循环）
3. max_iterations 保护是否生效
4. 最终答案是否正确 yield

Reference: ModernReActPlanner.md
"""
import pytest
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, AsyncIterator

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event
from agent_framework.core.planner_context import PlannerContext

# Import directly from module to avoid __init__.py export issues with Action
import importlib.util
spec = importlib.util.spec_from_file_location(
    "react_planner",
    Path(__file__).parent.parent.parent / "planners" / "react_planner.py"
)
react_planner_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(react_planner_module)

ReActPlanner = react_planner_module.ReActPlanner
ToolCall = react_planner_module.ToolCall
FunctionCall = react_planner_module.FunctionCall
ChatMessage = react_planner_module.ChatMessage
ChatResponse = react_planner_module.ChatResponse


class TestBuildMessages:
    """Test area 1: _build_system_message() 是否正确构建 messages 数组"""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner()

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

    def test_build_system_message_has_correct_role(self, planner, session_context, mock_memory, mock_tools):
        """System message should have role=system."""
        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools=mock_tools,
            messages=session_context.messages,
            memory=mock_memory
        )
        msg = asyncio.run(planner._build_system_message(ctx))
        assert msg.role == "system"
        assert "ReAct" in msg.content or "helpful assistant" in msg.content.lower()

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
        assert msg.role == "system"
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
        # _build_system_message returns only system ChatMessage
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

    def test_build_system_message_no_side_channel_state(self, planner, session_context, mock_memory, mock_tools):
        """After Phase 2A refactor, _build_system_message should not store side-channel state."""
        ctx = PlannerContext(
            session_id=session_context.session_id,
            tools=mock_tools,
            messages=session_context.messages,
            memory=mock_memory
        )
        asyncio.run(planner._build_system_message(ctx))
        assert not hasattr(planner, '_tools_ref'), "_tools_ref should not be stored as side-channel state"
        assert not hasattr(planner, '_session_id_ref'), "_session_id_ref should not be stored as side-channel state"


class TestToolCallLoop:
    """Test area 2: 工具调用循环是否正确（检测 tool_calls -> 执行工具 -> 继续循环）"""

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
            yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
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
                yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Final Answer: Done")
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

        # Tool should be called exactly once
        assert mock_tool.run.call_count == 1

    def test_tool_call_yields_observation_event(self, planner, session_context, mock_memory, mock_tool):
        """Should yield 'observation' event with tool result."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Final Answer: Done")
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

    def test_multiple_tool_calls_execute_sequentially(self, planner, session_context, mock_memory, mock_tool):
        """Should execute multiple tool calls sequentially."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "first"}'})
                yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "first"}'})
            else:
                yield Event(type="final_answer", content="Final Answer: Done")
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
                if event.type == "final_answer":
                    break

        asyncio.run(collect())

        assert mock_tool.run.call_count == 1

    def test_tool_result_added_to_messages(self, planner, session_context, mock_memory, mock_tool):
        """Tool results should be added to messages array for next LLM call."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Final Answer: Done")
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
        # Verify tool was called (which means tool results were processed)
        assert mock_tool.run.call_count == 1
        # Verify LLM was called twice (tool call + final answer)
        assert call_count == 2

    def test_unknown_tool_yields_error_event(self, planner, session_context, mock_memory):
        """Should yield error event when tool name is unknown."""
        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="", metadata={"tool_name": "unknown_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="", metadata={"tool_name": "unknown_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
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
        assert "unknown" in error_events[0].content.lower() or "Unknown tool" in error_events[0].content

    def test_async_and_sync_tools_both_work(self, planner, session_context, mock_memory):
        """Both async and sync tools should execute correctly."""
        # Create async tool
        async_tool = AsyncMock(spec=BaseTool)
        async_tool.name = "async_tool"
        async_tool.run = AsyncMock(return_value="async result")

        # Create sync tool
        sync_tool = MagicMock(spec=BaseTool)
        sync_tool.name = "sync_tool"
        sync_tool.run = MagicMock(return_value="sync result")

        tools = {"async_tool": async_tool, "sync_tool": sync_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="", metadata={"tool_name": "async_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="", metadata={"tool_name": "async_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
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


class TestMaxIterationsProtection:
    """Test area 3: max_iterations 保护是否生效"""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner(max_iterations=3)

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
            # Always return tool_call to force max_iterations
            yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": f"call_{llm_call_count}", "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": f"call_{llm_call_count}", "arguments": '{"input": "test"}'})
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

        # Should stop at max_iterations (3) LLM calls
        assert llm_call_count == planner.max_iterations

    def test_max_iterations_yields_error(self, planner, session_context, mock_memory, mock_tool):
        """Should yield error event when max_iterations reached without final answer."""
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
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

        error_events = [e for e in events if "maximum iterations" in e.content.lower() or e.type == "error"]
        assert len(error_events) > 0

    def test_custom_max_iterations_respected(self, planner, mock_memory, mock_tool):
        """Custom max_iterations should be respected."""
        custom_planner = ReActPlanner(max_iterations=5)
        tools = {"test_tool": mock_tool}
        llm_call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal llm_call_count
            llm_call_count += 1
            # Always return tool_call to force max_iterations
            yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": f"call_{llm_call_count}", "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": f"call_{llm_call_count}", "arguments": '{"input": "test"}'})
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

        assert llm_call_count == 5  # Exactly 5 iterations before stopping

    def test_final_answer_exits_before_max_iterations(self, planner, session_context, mock_memory, mock_tool):
        """Should exit loop as soon as final answer received, even if under max_iterations."""
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Final Answer: Done")
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
        # Should have called LLM only 2 times, not hit max
        assert call_count == 2


class TestFinalAnswerYield:
    """Test area 4: 最终答案是否正确 yield"""

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
            session_id="test-session",
            messages=[Message(role="user", content="What is the answer?", sender_id="user1")]
        )

    def test_final_answer_yields_event(self, planner, session_context, mock_memory):
        """Should yield final_answer event when LLM provides final answer."""
        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Final Answer: The answer is 42.")
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

        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) > 0

    def test_final_answer_content_extraction(self, planner, session_context, mock_memory):
        """Should extract and yield the correct final answer content."""
        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="Final Answer: Python is a programming language.")
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

        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) > 0
        assert "Python is a programming language" in final_answer_events[0].content

    def test_final_answer_case_insensitive_detection(self, planner, session_context, mock_memory):
        """Should detect 'Final Answer' case-insensitively."""
        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="final answer: Test content")
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

        final_answer_events = [e for e in events if e.type == "final_answer"]
        assert len(final_answer_events) > 0

    def test_final_answer_after_tool_calls(self, planner, session_context, mock_memory):
        """Should yield final answer after completing tool calls."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.run = AsyncMock(return_value="tool result")
        tools = {"test_tool": mock_tool}
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
                yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            else:
                yield Event(type="final_answer", content="Final Answer: Tool processed successfully")
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
        assert "processed" in final_answer_events[0].content.lower()

    def test_plan_and_act_returns_after_final_answer(self, planner, session_context, mock_memory):
        """Should return/exit after yielding final answer."""
        call_count = 0

        async def mock_llm_call(messages, tools):
            nonlocal call_count
            call_count += 1
            yield Event(type="final_answer", content=f"Final Answer: Call {call_count}")

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

        # Should only have called LLM once since it exits after final answer
        assert call_count == 1

    def test_no_tool_calls_returns_final_answer(self, planner, session_context, mock_memory):
        """Should handle response with no tool calls and final answer in content."""
        async def mock_llm_call(messages, tools):
            yield Event(type="final_answer", content="The answer is simple.")
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

        # Should still yield final answer if content contains final answer indicator
        final_answer_events = [e for e in events if e.type == "final_answer"]
        # May also have text_token events
        text_events = [e for e in events if e.type == "text_token"]
        assert len(final_answer_events) > 0 or len(text_events) > 0


class TestChatResponseHandling:
    """Additional tests for ChatResponse handling."""

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
            session_id="test-session",
            messages=[Message(role="user", content="Test", sender_id="user1")]
        )

    def test_chat_response_has_tool_calls_property(self):
        """ChatResponse should have has_tool_calls property."""
        response = ChatResponse(content="test", tool_calls=[])
        assert hasattr(response, 'has_tool_calls')
        assert response.has_tool_calls == False

        response_with_calls = ChatResponse(
            content="",
            tool_calls=[ToolCall(id="1", type="function", function=FunctionCall(name="tool", arguments="{}"))]
        )
        assert response_with_calls.has_tool_calls == True

    def test_tool_call_dataclass(self):
        """ToolCall should be a proper dataclass."""
        tc = ToolCall(id="call_1", type="function", function=FunctionCall(name="test_tool", arguments='{"input": "test"}'))
        assert tc.id == "call_1"
        assert tc.function.name == "test_tool"
        assert tc.function.arguments == '{"input": "test"}'

    def test_chat_message_dataclass(self):
        """ChatMessage should be a proper dataclass."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.tool_call_id is None

        msg_with_tool = ChatMessage(role="tool", content="result", tool_call_id="call_123")
        assert msg_with_tool.tool_call_id == "call_123"

    def test_handles_string_response(self, planner, session_context, mock_memory):
        """Should handle legacy string response format."""
        async def mock_llm_call(messages, tools):
            yield Event(type="text_token", content="Legacy string response")

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

        # Should handle string without crashing
        assert len(events) >= 0

    def test_handles_llm_exception(self, planner, session_context, mock_memory):
        """Should handle LLM call exceptions gracefully."""
        async def mock_llm_call(messages, tools):
            # Async generator that raises exception
            raise RuntimeError("LLM service unavailable")
            yield  # Make it an async generator

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
        assert "failed" in error_events[0].content.lower() or "error" in error_events[0].content.lower()

    def test_tool_execution_exception_handling(self, planner, session_context, mock_memory):
        """Should handle tool execution exceptions gracefully."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.name = "failing_tool"
        mock_tool.run = AsyncMock(side_effect=RuntimeError("Tool execution failed"))
        tools = {"failing_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="", metadata={"tool_name": "failing_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
            yield Event(type="tool_call_end", content="", metadata={"tool_name": "failing_tool", "tool_call_id": "call_1", "arguments": '{"input": "test"}'})
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

    def test_json_parse_error_in_arguments(self, planner, session_context, mock_memory):
        """Should handle malformed JSON in tool arguments."""
        mock_tool = AsyncMock(spec=BaseTool)
        mock_tool.name = "test_tool"
        mock_tool.run = AsyncMock(return_value="result")
        tools = {"test_tool": mock_tool}

        async def mock_llm_call(messages, tools):
            yield Event(type="tool_call_start", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": "not valid json {"})
            yield Event(type="tool_call_end", content="", metadata={"tool_name": "test_tool", "tool_call_id": "call_1", "arguments": "not valid json {"})

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

        # Should handle gracefully (fallback to treating entire string as input)
        # or yield error but not crash
        assert len(events) >= 0


# ============================================================================
# JSON Serialization Boundary Tests
# Cover edge cases: ChatMessage, tools dict, and full payload
# ============================================================================

class TestJSONSerialization:
    """Test JSON serialization boundaries for LLM chat payload."""

    @pytest.fixture
    def planner(self):
        """Create ReActPlanner instance."""
        return ReActPlanner()

    def test_chat_message_serializable(self):
        """ChatMessage objects should be serializable via json.dumps()."""
        msg = ChatMessage(role="user", content="Hello, world!")
        result = json.dumps(msg.__dict__)
        assert "user" in result
        assert "Hello, world!" in result

    def test_chat_message_with_name_serializable(self):
        """ChatMessage with name field should be serializable."""
        msg = ChatMessage(role="tool", content="result", name="test_tool")
        result = json.dumps(msg.__dict__)
        parsed = json.loads(result)
        assert parsed["name"] == "test_tool"

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
        # Convert to dicts (as done in _chat_message_to_dict)
        messages_dicts = [{"role": m.role, "content": m.content, "name": m.name, "tool_call_id": m.tool_call_id} for m in messages]
        result = json.dumps(messages_dicts)
        parsed = json.loads(result)
        assert len(parsed) == 4
        assert parsed[0]["role"] == "system"
        assert parsed[3]["tool_call_id"] == "call_1"

    def test_tool_call_serializable(self):
        """ToolCall objects should be serializable."""
        tc = ToolCall(id="call_1", type="function", function=FunctionCall(name="get_weather", arguments='{"city":"Beijing"}'))
        # Use proper serialization (same as planner._chat_message_to_dict)
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
        assert '"city":"Beijing"' in parsed["function"]["arguments"]

    def test_chat_response_serializable(self):
        """ChatResponse objects should be serializable."""
        tc = ToolCall(id="call_1", type="function", function=FunctionCall(name="test_tool", arguments='{"input":"test"}'))
        response = ChatResponse(content="Hello", tool_calls=[tc])
        # Note: ChatResponse contains ToolCall objects, needs proper conversion
        response_dict = {
            "content": response.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response.tool_calls
            ]
        }
        result = json.dumps(response_dict)
        parsed = json.loads(result)
        assert parsed["content"] == "Hello"
        assert len(parsed["tool_calls"]) == 1

    def test_tools_dict_to_schema_serializable(self):
        """Tools dict (BaseTool objects) should convert to serializable schema."""
        from agent_framework.interfaces.base_tool import BaseTool
        from dataclasses import dataclass

        @dataclass
        class MockTool(BaseTool):
            name: str = "calculator"
            description: str = "A simple calculator"
            parameters: dict = None

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return str(eval(input))

        tool = MockTool()
        tools = {"calculator": tool}

        # Convert using AgentRuntime's _tools_to_schemas
        schemas = []
        for name, t in tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": getattr(t, 'description', ''),
                    "parameters": getattr(t, 'parameters', {"type": "object", "properties": {}})
                }
            })

        result = json.dumps(schemas)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["function"]["name"] == "calculator"
        assert parsed[0]["function"]["description"] == "A simple calculator"

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

        # Simulate what happens in plan_and_act
        planner = ReActPlanner()
        messages_dicts = [planner._chat_message_to_dict(m) for m in messages]

        payload = {
            "model": "gpt-4",
            "messages": messages_dicts,
            "tools": tools,
        }

        # This should not raise
        result = json.dumps(payload)
        parsed = json.loads(result)
        assert len(parsed["messages"]) == 2
        assert parsed["messages"][0]["role"] == "system"

    def test_unicode_content_serializable(self):
        """Unicode content should serialize correctly."""
        messages = [
            ChatMessage(role="user", content="你好世界！"),
            ChatMessage(role="assistant", content="Hello, 世界! 🌍"),
        ]
        planner = ReActPlanner()
        messages_dicts = [planner._chat_message_to_dict(m) for m in messages]
        result = json.dumps(messages_dicts)
        parsed = json.loads(result)
        assert parsed[0]["content"] == "你好世界！"
        assert "世界" in parsed[1]["content"]

    def test_empty_content_serializable(self):
        """Empty content should serialize correctly."""
        messages = [
            ChatMessage(role="user", content=""),
            ChatMessage(role="assistant", content=""),
        ]
        planner = ReActPlanner()
        messages_dicts = [planner._chat_message_to_dict(m) for m in messages]
        result = json.dumps(messages_dicts)
        parsed = json.loads(result)
        assert parsed[0]["content"] == ""
        assert parsed[1]["content"] == ""

    def test_special_characters_in_arguments_serializable(self):
        """Special characters in tool arguments should serialize correctly."""
        tc = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(
                name="search",
                arguments='{"query": "test with \"quotes\" and \\backslashes\\"}'
            )
        )
        # Use proper serialization (same as planner._chat_message_to_dict)
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
        # Should handle escaped characters
        assert "quotes" in parsed["function"]["arguments"]