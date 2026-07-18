"""End-to-end integration tests for Modern ReAct flow.

This module tests the complete integration:
1. AgentRuntime + ModernReActPlanner + LLMGateway (stream_chat)
2. Complete flow: user message -> LLM response -> tool call -> result
3. WebSocket/REST API session processing

Tests use real components where possible, with mocks for external services.
"""
import pytest
import asyncio
import json
from typing import AsyncIterator, Dict, Any, List

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.llm_types import ChatResponse, ToolCall, FunctionCall
from typing import Optional
from agent_framework.runtime.agent_runtime import AgentRuntime
from agent_framework.core.session_manager import SessionManager


# ============== Helper Classes ==============

class SimpleMemory(BaseMemory):
    """Simple in-memory memory implementation for testing."""

    def __init__(self):
        self.saved_messages: Dict[str, List[Message]] = {}

    async def save(self, session_id: str, message: Message) -> None:
        if session_id not in self.saved_messages:
            self.saved_messages[session_id] = []
        self.saved_messages[session_id].append(message)

    async def retrieve(
        self,
        session_id: str,
        query: str,
        user_ids=None,
        top_k: int = 5
    ) -> str:
        return ""

    async def clear(self, session_id: str) -> None:
        if session_id in self.saved_messages:
            del self.saved_messages[session_id]

    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        pass


class SimpleStorage:
    """Simple in-memory session storage for testing."""

    def __init__(self):
        self.saved_sessions: Dict[str, SessionContext] = {}

    async def save(self, ctx: SessionContext) -> None:
        self.saved_sessions[ctx.session_id] = ctx

    async def load(self, session_id: str) -> SessionContext:
        return self.saved_sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        if session_id in self.saved_sessions:
            del self.saved_sessions[session_id]


class SimpleEventBus:
    """Simple event bus for testing."""

    def __init__(self):
        self.published_events: List[tuple] = []

    async def publish(self, session_id: str, event: Event) -> None:
        self.published_events.append((session_id, event))


class CalculatorTool(BaseTool):
    """Calculator tool for testing."""

    name = "calculator"
    description = "Performs basic arithmetic calculations"

    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        try:
            result = eval(input)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class MockLLMGatewayStreamChat:
    """Mock LLM gateway that implements stream_chat returning ChatResponse."""

    def __init__(self, responses: List[ChatResponse] = None):
        self.responses = responses or []
        self.call_count = 0
        self.calls: List[Dict] = []

    async def stream_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
        **kwargs
    ) -> AsyncIterator:
        """Stream ChatResponse objects."""
        self.call_count += 1
        self.calls.append({
            "messages": messages,
            "tools": tools,
            "model": model
        })

        if self.responses:
            response = self.responses[self.call_count - 1] if self.call_count <= len(self.responses) else self.responses[-1]
            yield response


# ============== Test Cases ==============

class TestModernReActIntegration:
    """Test ModernReActPlanner integration with AgentRuntime."""

    @pytest.fixture
    def setup(self):
        """Set up test dependencies."""
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGatewayStreamChat()
        tools = {"calculator": CalculatorTool()}

        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner(max_iterations=5)

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        return {
            "storage": storage,
            "event_bus": event_bus,
            "memory": memory,
            "runtime": runtime,
            "llm_gateway": llm_gateway,
            "planner": planner,
            "manager": manager,
            "tools": tools
        }

    @pytest.mark.asyncio
    async def test_agent_runtime_with_stream_chat(self, setup):
        """Test that AgentRuntime works with stream_chat interface."""
        runtime = setup["runtime"]
        llm_gateway = setup["llm_gateway"]
        memory = setup["memory"]
        tools = setup["tools"]

        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        ctx = SessionContext(
            session_id="test-session",
            messages=[Message(role="user", content="Hello")]
        )

        # Create a simple response
        responses = [
            ChatResponse(content="Hello! How can I help you?", tool_calls=[])
        ]
        llm_gateway.responses = responses

        events = []
        async for event in runtime.run(ctx, "Test message", memory, tools, planner, llm_gateway):
            events.append(event)

        assert len(events) > 0
        print(f"Events generated: {[e.type for e in events]}")

    @pytest.mark.asyncio
    async def test_complete_flow_with_tool_call(self, setup):
        """Test complete flow: message -> LLM -> tool call -> result -> final answer."""
        storage = setup["storage"]
        event_bus = setup["event_bus"]
        memory = setup["memory"]
        runtime = setup["runtime"]
        llm_gateway = setup["llm_gateway"]
        planner = setup["planner"]
        manager = setup["manager"]
        tools = setup["tools"]

        # First LLM call returns tool call
        # Second LLM call returns final answer
        responses = [
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="calculator",
                            arguments='{"input":"2+2"}'
                        )
                    )
                ]
            ),
            ChatResponse(content="Final Answer: The result of 2+2 is 4.", tool_calls=[])
        ]
        llm_gateway.responses = responses

        # Create session
        ctx = await manager.create_session(user_id="test-user")
        session_id = ctx.session_id

        # Process message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "What is 2+2?"}
        )

        events = await asyncio.wait_for(future, timeout=10.0)

        # Verify events were generated
        assert len(events) > 0

        # Print events for debugging
        print(f"\nTotal events: {len(events)}")
        for event in events:
            print(f"  [{event.type}]: {event.content[:80]}...")

        # Verify event types
        event_types = [e.type for e in events]
        assert "final_answer" in event_types, f"No final_answer in events: {event_types}"

        # Verify messages were saved to memory
        assert session_id in memory.saved_messages
        print(f"\nMessages saved: {len(memory.saved_messages[session_id])}")

        # Verify events were published to bus
        assert len(event_bus.published_events) == len(events)
        print(f"\nEvents published to bus: {len(event_bus.published_events)}")

        await manager.close_session(session_id)


class TestLLMGatewayChatIntegration:
    """Test LLM Gateway chat interface integration."""

    @pytest.mark.asyncio
    async def test_stream_chat_returns_chat_response(self):
        """Test that stream_chat yields ChatResponse objects."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        from agent_framework.infrastructure.llm_gateway import StreamChatResponse, ChatResponseType

        config = OpenAIConfig(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key"
        )
        llm = OpenAILLM(config)

        # Mock the client
        from unittest.mock import AsyncMock, MagicMock

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":" World"},"finish_reason":"stop"}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        # Test stream_chat
        responses = []
        async for resp in llm.stream_chat([{"role": "user", "content": "test"}]):
            responses.append(resp)
            assert hasattr(resp, 'type')
            assert hasattr(resp, 'content')

        print(f"\nstream_chat responses: {len(responses)}")
        for resp in responses:
            print(f"  Type: {resp.type}, Content: {resp.content}")

        assert len(responses) >= 2


class TestPlannerRuntimeIntegration:
    """Test Planner and Runtime integration."""

    @pytest.mark.asyncio
    async def test_planner_receives_correct_messages(self):
        """Test that planner receives correctly formatted messages from runtime."""
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGatewayStreamChat()
        tools = {"calculator": CalculatorTool()}

        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        responses = [
            ChatResponse(content="Final Answer: Test completed.", tool_calls=[])
        ]
        llm_gateway.responses = responses

        ctx = await manager.create_session(user_id="test-user")
        session_id = ctx.session_id

        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Run test"}
        )

        events = await asyncio.wait_for(future, timeout=10.0)

        # Verify LLM was called with correct messages
        assert llm_gateway.call_count > 0
        assert len(llm_gateway.calls) > 0

        call = llm_gateway.calls[0]
        print(f"\nLLM call messages: {len(call['messages'])}")
        for msg in call['messages']:
            print(f"  {msg.get('role')}: {msg.get('content')[:50]}...")

        # Verify messages are in correct format (dicts, not objects)
        for msg in call['messages']:
            assert isinstance(msg, dict), f"Message should be dict, got {type(msg)}"
            assert "role" in msg
            assert "content" in msg

        # Verify tools were passed correctly
        assert call['tools'] is not None
        assert len(call['tools']) == 1
        assert call['tools'][0]['function']['name'] == 'calculator'

        await manager.close_session(session_id)


class TestEndToEndWithRealComponents:
    """End-to-end tests with real components (using mock LLM)."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test complete session lifecycle."""
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGatewayStreamChat()

        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()
        tools = {"calculator": CalculatorTool()}

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        responses = [
            ChatResponse(content="Final Answer: Session created.", tool_calls=[])
        ]
        llm_gateway.responses = responses

        # Create session
        ctx = await manager.create_session(user_id="user-001")
        assert ctx is not None
        assert ctx.session_id is not None
        print(f"\nSession created: {ctx.session_id}")

        # Process message
        future = await manager.process_message(
            session_id=ctx.session_id,
            user_msg={"role": "user", "content": "Test message"}
        )
        events = await asyncio.wait_for(future, timeout=10.0)
        assert len(events) > 0
        print(f"Events generated: {len(events)}")

        # Close session
        await manager.close_session(ctx.session_id)
        assert ctx.status == "closed"
        print(f"Session closed: {ctx.session_id}")

        # Verify storage
        assert ctx.session_id in storage.saved_sessions
        print(f"Session saved to storage")

    @pytest.mark.asyncio
    async def test_multiple_turns(self):
        """Test multiple conversation turns."""
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGatewayStreamChat()

        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()
        tools = {"calculator": CalculatorTool()}

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Each response is used for one turn
        responses = [
            ChatResponse(content="Turn 1 response.", tool_calls=[]),
            ChatResponse(content="Turn 2 response.", tool_calls=[]),
            ChatResponse(content="Turn 3 response.", tool_calls=[]),
        ]
        llm_gateway.responses = responses

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        # Turn 1
        future1 = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Message 1"}
        )
        events1 = await asyncio.wait_for(future1, timeout=10.0)
        assert len(events1) > 0

        # Turn 2
        future2 = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Message 2"}
        )
        events2 = await asyncio.wait_for(future2, timeout=10.0)
        assert len(events2) > 0

        # Turn 3
        future3 = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Message 3"}
        )
        events3 = await asyncio.wait_for(future3, timeout=10.0)
        assert len(events3) > 0

        print(f"\nTurn 1 events: {len(events1)}")
        print(f"Turn 2 events: {len(events2)}")
        print(f"Turn 3 events: {len(events3)}")

        # Verify memory has all messages
        saved = memory.saved_messages.get(session_id, [])
        print(f"Total messages in memory: {len(saved)}")

        await manager.close_session(session_id)

    @pytest.mark.asyncio
    async def test_tool_execution_in_loop(self):
        """Test tool execution in ReAct loop."""
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGatewayStreamChat()
        tools = {"calculator": CalculatorTool()}

        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner(max_iterations=5)

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # First call: tool call, Second call: final answer
        responses = [
            ChatResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="calculator",
                            arguments='{"input":"10+20"}'
                        )
                    )
                ]
            ),
            ChatResponse(content="Final Answer: 10+20=30", tool_calls=[])
        ]
        llm_gateway.responses = responses

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Calculate 10+20"}
        )

        events = await asyncio.wait_for(future, timeout=10.0)

        print(f"\nTotal events: {len(events)}")
        for event in events:
            print(f"  [{event.type}]: {event.content[:60]}...")

        # Verify action and observation events
        event_types = [e.type for e in events]
        assert "action" in event_types or "observation" in event_types, \
            f"Expected tool events, got: {event_types}"
        assert "final_answer" in event_types

        await manager.close_session(session_id)