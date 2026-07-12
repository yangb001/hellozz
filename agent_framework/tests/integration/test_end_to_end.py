"""End-to-end integration tests for the Agent Framework.

This module tests the complete flow from session creation to response generation,
verifying that all components work together correctly.

Test coverage:
- Session creation and management
- Message processing through SessionManager
- AgentRuntime event generation
- Planner integration (ReAct pattern)
- Tool execution
- Memory storage and retrieval
- Event streaming
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any, AsyncIterator

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.runtime.agent_runtime import AgentRuntime
from agent_framework.core.session_manager import SessionManager


# ============== Helper Functions ==============

async def async_iter(items):
    """Helper to create async iterable from list."""
    for item in items:
        yield item


# ============== Mock Implementations ==============

class MockMemory(BaseMemory):
    """Mock memory implementation for testing."""

    def __init__(self):
        self.saved_messages: Dict[str, List[Message]] = {}
        self.retrieved_queries: List[tuple] = []

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
        self.retrieved_queries.append((session_id, query))
        return f"Retrieved context for: {query}"

    async def clear(self, session_id: str) -> None:
        if session_id in self.saved_messages:
            del self.saved_messages[session_id]

    async def extract_long_term(self, session_id: str, force: bool = False) -> None:
        pass


class MockLLMGateway:
    """Mock LLM gateway for testing."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.calls: List[str] = []

    async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
        self.calls.append(prompt)
        for key, response in self.responses.items():
            if key in prompt:
                return response
        return "Default LLM response"

    async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
        self.calls.append(prompt)
        response = await self.generate(prompt, model, **kwargs)
        for char in response:
            yield char


class MockPlanner(BasePlanner):
    """Mock planner that follows a simple thought-action-observation pattern."""

    def __init__(self, final_answer: str = "The answer is 42."):
        self.final_answer = final_answer
        self.plan_calls: List[Dict] = []

    async def plan_and_act(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any],
        llm_call: callable,
    ) -> AsyncIterator[Event]:
        self.plan_calls.append({
            "session_id": ctx.session_id,
            "message_count": len(ctx.messages)
        })

        # Simulate thought
        yield Event(type="thought", content="I need to process this request.")

        # Simulate action (tool call) if calculator is available
        if "calculator" in tools:
            yield Event(type="action", content="Calling calculator tool...")
            try:
                result = await tools["calculator"].run("2 + 2")
                yield Event(type="observation", content=f"Calculator result: {result}")
            except Exception as e:
                yield Event(type="observation", content=f"Tool error: {e}")

        # Final answer
        yield Event(type="final_answer", content=self.final_answer)


class MockCalculatorTool:
    """Mock calculator tool for testing."""

    name = "calculator"
    description = "Performs basic arithmetic calculations"

    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        try:
            # Simple eval for testing (not safe for production)
            result = eval(input)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class MockStorage:
    """Mock session storage for testing."""

    def __init__(self):
        self.saved_sessions: Dict[str, SessionContext] = {}

    async def save(self, ctx: SessionContext) -> None:
        self.saved_sessions[ctx.session_id] = ctx

    async def load(self, session_id: str) -> SessionContext:
        return self.saved_sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        if session_id in self.saved_sessions:
            del self.saved_sessions[session_id]


class MockEventBus:
    """Mock event bus for testing."""

    def __init__(self):
        self.published_events: List[tuple] = []

    async def publish(self, session_id: str, event: Event) -> None:
        self.published_events.append((session_id, event))


# ============== Test Cases ==============

class TestEndToEndSessionCreation:
    """Test end-to-end session creation flow."""

    @pytest.mark.asyncio
    async def test_create_session_success(self):
        """Test that a session can be created successfully."""
        # Setup dependencies
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        # Create session manager
        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Create session
        ctx = await manager.create_session(user_id="user-001")

        # Verify session was created
        assert ctx is not None
        assert ctx.session_id is not None
        assert ctx.session_type == "private"
        assert "user-001" in ctx.participants
        assert ctx.status == "active"

        # Verify session was saved to storage
        assert ctx.session_id in storage.saved_sessions

        # Cleanup
        await manager.close_session(ctx.session_id)

    @pytest.mark.asyncio
    async def test_create_group_session(self):
        """Test that a group session can be created."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Create group session
        ctx = await manager.create_session(
            user_id="user-001",
            session_type="group",
            participants=["user-002", "user-003"]
        )

        # Verify group session properties
        assert ctx.session_type == "group"
        assert "user-001" in ctx.participants
        assert "user-002" in ctx.participants
        assert "user-003" in ctx.participants

        await manager.close_session(ctx.session_id)


class TestEndToEndMessageProcessing:
    """Test end-to-end message processing flow."""

    @pytest.mark.asyncio
    async def test_process_message_generates_events(self):
        """Test that processing a message generates proper events."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner(final_answer="The sum is 4.")
        llm_gateway = MockLLMGateway()
        tools = {"calculator": MockCalculatorTool()}

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

        # Create session
        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        # Process message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "What is 2 + 2?"}
        )

        # Wait for processing to complete
        events = await asyncio.wait_for(future, timeout=5.0)

        # Verify events were generated
        assert len(events) > 0

        # Check event types
        event_types = [e.type for e in events]
        assert "thought" in event_types
        assert "final_answer" in event_types

        # Verify final answer
        final_events = [e for e in events if e.type == "final_answer"]
        assert len(final_events) == 1
        assert final_events[0].content == "The sum is 4."

        # Verify events were published to event bus
        assert len(event_bus.published_events) > 0
        published_session_ids = [sid for sid, _ in event_bus.published_events]
        assert session_id in published_session_ids

        # Verify message was saved to memory
        assert session_id in memory.saved_messages
        saved_messages = memory.saved_messages[session_id]
        assert any(m.role == "user" for m in saved_messages)
        assert any(m.role == "assistant" for m in saved_messages)

        await manager.close_session(session_id)

    @pytest.mark.asyncio
    async def test_process_message_with_tool_execution(self):
        """Test that tools are properly called during message processing."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner(final_answer="Calculation complete.")
        llm_gateway = MockLLMGateway()

        # Create a spy tool to track calls
        class SpyCalculator:
            name = "calculator"
            description = "Calculator tool"
            call_count = 0

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                SpyCalculator.call_count += 1
                return "42"

        spy_tool = SpyCalculator()
        tools = {"calculator": spy_tool}

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

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        # Process message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Calculate something"}
        )

        events = await asyncio.wait_for(future, timeout=5.0)

        # Verify tool was called
        assert spy_tool.call_count > 0

        # Verify observation event exists
        event_types = [e.type for e in events]
        assert "observation" in event_types

        await manager.close_session(session_id)


class TestEndToEndMemoryIntegration:
    """Test end-to-end memory integration."""

    @pytest.mark.asyncio
    async def test_messages_saved_to_memory(self):
        """Test that all messages are properly saved to memory."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        # Process a message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Hello, how are you?"}
        )

        await asyncio.wait_for(future, timeout=5.0)

        # Verify messages were saved
        assert session_id in memory.saved_messages
        messages = memory.saved_messages[session_id]

        # Should have at least user message and assistant response
        assert len(messages) >= 2

        # Verify message roles
        roles = [m.role for m in messages]
        assert "user" in roles
        assert "assistant" in roles

        # Verify user message content
        user_messages = [m for m in messages if m.role == "user"]
        assert any("Hello, how are you?" in m.content for m in user_messages)

        await manager.close_session(session_id)

    @pytest.mark.asyncio
    async def test_memory_retrieval_during_processing(self):
        """Test that memory is retrieved during message processing."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()

        # Create a planner that uses memory
        class MemoryUsingPlanner(BasePlanner):
            async def plan_and_act(self, ctx, memory, tools, llm_call):
                # Retrieve memory
                context = await memory.retrieve(ctx.session_id, "previous context")
                yield Event(type="thought", content=f"Using context: {context}")
                yield Event(type="final_answer", content="Response with context.")

        planner = MemoryUsingPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Tell me about previous context"}
        )

        events = await asyncio.wait_for(future, timeout=5.0)

        # Verify memory was retrieved
        assert len(memory.retrieved_queries) > 0
        assert any("previous context" in q[1] for q in memory.retrieved_queries)

        await manager.close_session(session_id)


class TestEndToEndPlannerIntegration:
    """Test end-to-end planner integration."""

    @pytest.mark.asyncio
    async def test_react_planner_pattern(self):
        """Test that ReAct planner pattern works end-to-end."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGateway()
        tools = {"calculator": MockCalculatorTool()}

        # Create a more realistic ReAct-style planner
        class ReActPlanner(BasePlanner):
            async def plan_and_act(self, ctx, memory, tools, llm_call):
                # Thought
                yield Event(type="thought", content="I need to calculate 2+2")

                # Action
                yield Event(type="action", content="Using calculator tool")
                result = await tools["calculator"].run("2+2")

                # Observation
                yield Event(type="observation", content=f"Result: {result}")

                # Final answer
                yield Event(type="final_answer", content=f"The answer is {result}")

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

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "What is 2+2?"}
        )

        events = await asyncio.wait_for(future, timeout=5.0)

        # Verify ReAct pattern events
        event_types = [e.type for e in events]
        assert event_types == ["thought", "action", "observation", "final_answer"]

        # Verify final answer contains calculation result
        final_event = [e for e in events if e.type == "final_answer"][0]
        assert "4" in final_event.content

        await manager.close_session(session_id)


class TestEndToEndEventStreaming:
    """Test end-to-end event streaming."""

    @pytest.mark.asyncio
    async def test_events_published_to_bus(self):
        """Test that all events are published to the event bus."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Test message"}
        )

        events = await asyncio.wait_for(future, timeout=5.0)

        # Verify all events were published
        assert len(event_bus.published_events) == len(events)

        # Verify session ID consistency
        for sid, event in event_bus.published_events:
            assert sid == session_id

        await manager.close_session(session_id)

    @pytest.mark.asyncio
    async def test_session_context_updated(self):
        """Test that session context is updated after processing."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        # Record initial state
        initial_message_count = len(ctx.messages)

        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Update my context"}
        )

        await asyncio.wait_for(future, timeout=5.0)

        # Verify context was updated
        assert len(ctx.messages) > initial_message_count

        # Verify session was saved to storage
        saved_ctx = storage.saved_sessions.get(session_id)
        assert saved_ctx is not None
        assert len(saved_ctx.messages) > initial_message_count

        await manager.close_session(session_id)


class TestEndToEndMultipleMessages:
    """Test end-to-end multiple message handling."""

    @pytest.mark.asyncio
    async def test_multiple_messages_processed_serially(self):
        """Test that multiple messages are processed serially per session."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        # Send multiple messages
        futures = []
        for i in range(3):
            future = await manager.process_message(
                session_id=session_id,
                user_msg={"role": "user", "content": f"Message {i+1}"}
            )
            futures.append(future)

        # Wait for all to complete
        all_events = []
        for future in futures:
            events = await asyncio.wait_for(future, timeout=5.0)
            all_events.append(events)

        # Verify all messages were processed
        assert len(all_events) == 3

        # Verify context has all messages
        assert len(ctx.messages) >= 6  # 3 user + 3 assistant

        await manager.close_session(session_id)


class TestEndToEndErrorHandling:
    """Test end-to-end error handling."""

    @pytest.mark.asyncio
    async def test_planner_error_propagated(self):
        """Test that planner errors are properly propagated."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGateway()

        # Create a planner that raises an error
        class ErrorPlanner(BasePlanner):
            async def plan_and_act(self, ctx, memory, tools, llm_call):
                yield Event(type="thought", content="Starting...")
                raise ValueError("Planner error occurred")

        planner = ErrorPlanner()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        ctx = await manager.create_session(user_id="user-001")
        session_id = ctx.session_id

        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Cause an error"}
        )

        # Should raise the planner error
        with pytest.raises(ValueError, match="Planner error occurred"):
            await asyncio.wait_for(future, timeout=5.0)

        await manager.close_session(session_id)

    @pytest.mark.asyncio
    async def test_invalid_session_id_raises_error(self):
        """Test that processing message for invalid session raises error."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        planner = MockPlanner()
        llm_gateway = MockLLMGateway()

        def memory_factory(sid):
            return memory

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Try to process message for non-existent session
        with pytest.raises(ValueError, match="does not exist"):
            await manager.process_message(
                session_id="non-existent-session",
                user_msg={"role": "user", "content": "Hello"}
            )


class TestEndToEndCompleteFlow:
    """Test complete end-to-end flow with all components."""

    @pytest.mark.asyncio
    async def test_complete_agent_flow(self):
        """Test complete flow: session -> message -> planning -> tools -> response."""
        storage = MockStorage()
        event_bus = MockEventBus()
        memory = MockMemory()
        runtime = AgentRuntime()
        llm_gateway = MockLLMGateway()

        # Create a realistic planner that uses tools
        class RealisticPlanner(BasePlanner):
            async def plan_and_act(self, ctx, memory, tools, llm_call):
                # Get the user's question
                user_msg = ctx.messages[-1].content if ctx.messages else ""

                # Thought
                yield Event(type="thought", content=f"User asked: {user_msg}")

                # If calculator is available and question involves math
                if "calculator" in tools and any(op in user_msg for op in ['+', '-', '*', '/']):
                    yield Event(type="action", content="Using calculator...")
                    # Extract simple math expression (very basic)
                    result = await tools["calculator"].run("2 + 2")
                    yield Event(type="observation", content=f"Result: {result}")
                    yield Event(type="final_answer", content=f"The calculation result is {result}")
                else:
                    # Use LLM for other questions
                    yield Event(type="action", content="Consulting LLM...")
                    llm_response = ""
                    async for token in llm_call("Answer the question"):
                        llm_response += token
                    yield Event(type="observation", content="LLM responded")
                    yield Event(type="final_answer", content="Here is my response based on LLM.")

        planner = RealisticPlanner()
        tools = {"calculator": MockCalculatorTool()}

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

        # Step 1: Create session
        ctx = await manager.create_session(
            user_id="user-001",
            session_type="private"
        )
        session_id = ctx.session_id

        # Step 2: Send message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "What is 2 + 2?"}
        )

        # Step 3: Wait for processing
        events = await asyncio.wait_for(future, timeout=5.0)

        # Step 4: Verify complete flow
        # - Events were generated
        assert len(events) > 0

        # - Event types follow ReAct pattern
        event_types = [e.type for e in events]
        assert "thought" in event_types
        assert "action" in event_types
        assert "observation" in event_types
        assert "final_answer" in event_types

        # - Final answer contains calculation result
        final_event = [e for e in events if e.type == "final_answer"][0]
        assert "4" in final_event.content

        # - Messages were saved to memory
        assert session_id in memory.saved_messages
        assert len(memory.saved_messages[session_id]) >= 2

        # - Events were published to event bus
        assert len(event_bus.published_events) == len(events)

        # - Session was saved to storage
        assert session_id in storage.saved_sessions

        # Step 5: Close session
        await manager.close_session(session_id)

        # Verify session is closed
        assert ctx.status == "closed"
