"""Test ToolCall planner loop - verify tool result is sent back to LLM.

This test verifies that after executing a tool call, the planner
correctly continues the loop and makes a second LLM call with the tool result.
"""
import asyncio
import pytest
from typing import AsyncIterator, List, Dict, Any
from unittest.mock import AsyncMock, MagicMock

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event, EventType
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.planners.react_planner import ToolCallPlanner
from agent_framework.infrastructure.llm_gateway import StreamChatResponse, ChatResponseType, ToolCall, FunctionCall


class MockTool:
    """Mock tool for testing."""
    name = "calculator"
    description = "Evaluate mathematical expressions"

    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        """Echo back the input with a prefix."""
        return f"Result: {input}"


class MockLLM:
    """Mock LLM that returns tool call on first call, then final answer on second."""

    def __init__(self):
        self.call_count = 0
        self.messages_received = []

    async def __call__(self, messages: List[Dict], tools: List[Dict] = None) -> AsyncIterator[Event]:
        """Return tool call on first call, final answer on second."""
        self.call_count += 1
        self.messages_received.append(messages)

        if self.call_count == 1:
            # First call: return a tool call
            yield Event(
                type=EventType.TOOL_CALL_START,
                content="",
                metadata={
                    "tool_call_id": "call_123",
                    "tool_name": "calculator",
                    "arguments": '{"input": "100+100"}',
                    "is_complete": False
                }
            )
            yield Event(
                type=EventType.TOOL_CALL_END,
                content="",
                metadata={
                    "tool_call_id": "call_123",
                    "tool_name": "calculator",
                    "arguments": '{"input": "100+100"}',
                    "is_complete": True
                }
            )
            yield Event(type=EventType.STREAMING_END, content="", metadata={"finish_reason": "tool_calls"})
        else:
            # Second call: return final answer
            yield Event(type=EventType.TEXT_TOKEN, content="The ")
            yield Event(type=EventType.TEXT_TOKEN, content="result ")
            yield Event(type=EventType.TEXT_TOKEN, content="is ")
            yield Event(type=EventType.TEXT_TOKEN, content="200")
            yield Event(type=EventType.FINAL_ANSWER, content="The result is 200")
            yield Event(type=EventType.STREAMING_END, content="", metadata={"finish_reason": "stop"})


class MockMemory(BaseMemory):
    """Mock memory for testing."""

    async def save(self, session_id: str, message: Message) -> None:
        pass

    async def retrieve(self, session_id: str, query: str, user_ids: List[str] = None, top_k: int = 5) -> str:
        return ""

    async def clear(self, session_id: str) -> None:
        pass

    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        pass


@pytest.mark.asyncio
async def test_tool_call_loop_continues_after_tool_result():
    """Test that planner makes second LLM call after tool execution.

    This verifies the fix for the bug where tool results were not sent back to LLM.
    """
    # Setup
    planner = ToolCallPlanner(max_iterations=5)
    ctx = SessionContext(session_id="test_session")
    ctx.messages.append(Message(role="user", content="Calculate 100+100"))
    memory = MockMemory()
    tools = {"calculator": MockTool()}
    mock_llm = MockLLM()

    # Execute
    events = []
    event_types = []
    async for event in planner.plan_and_act(ctx, memory, tools, mock_llm):
        events.append(event)
        event_types.append(event.type)
        print(f"Event: {event.type} | content: {event.content[:50] if event.content else ''} | metadata: {event.metadata}")

    print(f"\n=== Event types received: {event_types} ===")
    print(f"=== LLM call count: {mock_llm.call_count} ===")

    # Verify LLM was called twice (once for tool, once for final answer)
    assert mock_llm.call_count == 2, f"Expected 2 LLM calls, got {mock_llm.call_count}"

    # Verify second LLM call received tool result
    second_messages = mock_llm.messages_received[1]
    tool_result_found = False
    for msg in second_messages:
        if msg.get("role") == "tool":
            tool_result_found = True
            assert "100+100" in msg.get("content", "") or "Result" in msg.get("content", ""), \
                f"Tool result should contain expression or result, got: {msg.get('content')}"
            break

    assert tool_result_found, f"Tool result not found in second LLM call messages: {second_messages}"

    # Verify final answer was received
    final_answer_events = [e for e in events if e.type == "final_answer"]
    assert len(final_answer_events) == 1, f"Expected 1 final answer, got {len(final_answer_events)}"
    assert "200" in final_answer_events[0].content, f"Final answer should contain 200, got: {final_answer_events[0].content}"

    print(f"[PASS] Test passed: LLM called {mock_llm.call_count} times")
    print(f"[PASS] Tool result sent back to LLM in second call")
    print(f"[PASS] Final answer received: {final_answer_events[0].content}")


@pytest.mark.asyncio
async def test_tool_call_iteration_count():
    """Test that planner logs iteration correctly."""
    planner = ToolCallPlanner(max_iterations=5)
    ctx = SessionContext(session_id="test_session")
    ctx.messages.append(Message(role="user", content="Calculate 100+100"))
    memory = MockMemory()
    tools = {"calculator": MockTool()}
    mock_llm = MockLLM()

    iteration_events = []
    async for event in planner.plan_and_act(ctx, memory, tools, mock_llm):
        if event.type == "action" and "iteration" in str(event.content):
            iteration_events.append(event)

    # Should have completed without hitting max iterations
    assert mock_llm.call_count <= 5, "Should not exceed max iterations"


if __name__ == "__main__":
    asyncio.run(test_tool_call_loop_continues_after_tool_result())
    asyncio.run(test_tool_call_iteration_count())