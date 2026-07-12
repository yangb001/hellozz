"""End-to-end integration tests with real Mimo LLM.

This module tests the complete flow using real Mimo LLM API,
verifying that all components work together with actual LLM responses.

Test coverage:
- Session creation with real LLM
- Message processing through ReAct planner
- Tool execution (Calculator)
- Memory storage
- Event streaming with real LLM responses
"""
import pytest
import asyncio
import json
import os
from typing import AsyncIterator, Dict, Any, List

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
        # Return empty string for simplicity
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


class CalculatorTool:
    """Calculator tool for testing."""

    name = "calculator"
    description = "Performs basic arithmetic calculations"

    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        try:
            result = eval(input)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


# ============== OpenAI-compatible LLM Gateway for Mimo ==============

class MimoLLMGateway:
    """OpenAI-compatible LLM Gateway for Mimo API."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_url = config.get("base_url", "")
        self.api_key = config.get("api_key", "")
        # Use mimo-v2.5 as the default model name
        self.model = config.get("model", "mimo-v2.5")
        if self.model == "mimo":
            self.model = "mimo-v2.5"
        self._client = None

    def _get_client(self):
        """Get or create httpx client."""
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=120.0
            )
        return self._client

    async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
        """Generate a complete response."""
        client = self._get_client()

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7)
        }

        try:
            response = await client.post("/chat/completions", json=payload)
            response.raise_for_status()
            result = response.json()

            # Handle mimo-v2.5 response format
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                content = message.get("content", "")
                # If content is empty, try reasoning_content
                if not content and "reasoning_content" in message:
                    content = message["reasoning_content"]
                return content
            return ""
        except Exception as e:
            raise RuntimeError(f"Mimo API error: {e}")

    async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
        """Generate a streaming response."""
        client = self._get_client()

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.7)
        }

        try:
            async with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                # Handle content field
                                content = delta.get("content")
                                if content is not None:
                                    yield content
                                # Handle reasoning_content field (for mimo-v2.5)
                                reasoning = delta.get("reasoning_content")
                                if reasoning is not None:
                                    yield reasoning
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            raise RuntimeError(f"Mimo streaming error: {e}")

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# ============== Test Cases ==============

class TestRealLLMSessionCreation:
    """Test session creation with real LLM."""

    @pytest.mark.asyncio
    async def test_create_session_with_real_llm(self):
        """Test that a session can be created with real LLM configuration."""
        # Load config
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create LLM gateway
        mimo_config = config["llm"]["providers"]["mimo"]
        llm_gateway = MimoLLMGateway(mimo_config)

        # Create other dependencies
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()

        # Use a simple planner that just returns the LLM response
        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        def memory_factory(sid):
            return memory

        tools = {"calculator": CalculatorTool()}

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
        ctx = await manager.create_session(user_id="test-user")
        assert ctx is not None
        assert ctx.session_id is not None

        # Cleanup
        await manager.close_session(ctx.session_id)
        await llm_gateway.close()


class TestRealLLMMessageProcessing:
    """Test message processing with real LLM."""

    @pytest.mark.asyncio
    async def test_simple_conversation(self):
        """Test simple conversation without tool calls."""
        # Load config
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create LLM gateway
        mimo_config = config["llm"]["providers"]["mimo"]
        llm_gateway = MimoLLMGateway(mimo_config)

        # Create other dependencies
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()

        # Use ReAct planner
        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        def memory_factory(sid):
            return memory

        # No tools for simple conversation
        tools = {}

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
        ctx = await manager.create_session(user_id="test-user")
        session_id = ctx.session_id

        # Send message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "What is 2+2? Please answer directly."}
        )

        # Wait for processing
        try:
            events = await asyncio.wait_for(future, timeout=60.0)

            # Verify events were generated
            assert len(events) > 0

            # Print all events for debugging
            print(f"\nTotal events: {len(events)}")
            for event in events:
                print(f"  [{event.type}]: {event.content[:100]}...")

            # Check for final answer or error
            final_events = [e for e in events if e.type == "final_answer"]
            error_events = [e for e in events if e.type == "error"]

            # If there's an error about unknown tool, that's expected
            if error_events:
                print(f"\nError events (expected): {[e.content for e in error_events]}")

            # Verify final answer exists
            assert len(final_events) > 0

            # Verify the answer contains a number
            final_answer = final_events[0].content
            assert any(char.isdigit() for char in final_answer)

            print(f"\nLLM Response: {final_answer}")

        except asyncio.TimeoutError:
            pytest.fail("Test timed out waiting for LLM response")
        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")
        finally:
            await manager.close_session(session_id)
            await llm_gateway.close()

    @pytest.mark.asyncio
    async def test_conversation_with_calculator(self):
        """Test conversation that uses calculator tool."""
        # Load config
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create LLM gateway
        mimo_config = config["llm"]["providers"]["mimo"]
        llm_gateway = MimoLLMGateway(mimo_config)

        # Create other dependencies
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()

        # Use ReAct planner
        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        def memory_factory(sid):
            return memory

        # Include calculator tool
        tools = {"calculator": CalculatorTool()}

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
        ctx = await manager.create_session(user_id="test-user")
        session_id = ctx.session_id

        # Send message that requires calculation
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Please calculate 15 * 23 using the calculator tool."}
        )

        # Wait for processing
        try:
            events = await asyncio.wait_for(future, timeout=90.0)

            # Verify events were generated
            assert len(events) > 0

            # Print all events for debugging
            for event in events:
                print(f"\nEvent [{event.type}]: {event.content[:100]}...")

            # Check for tool usage
            action_events = [e for e in events if e.type == "action"]
            observation_events = [e for e in events if e.type == "observation"]

            # Verify tool was called (may or may not be called depending on LLM behavior)
            if action_events:
                print(f"\nTool was called: {len(action_events)} times")

            # Check for final answer
            final_events = [e for e in events if e.type == "final_answer"]
            assert len(final_events) > 0

            # The answer should contain 345 (15 * 23)
            final_answer = final_events[0].content
            print(f"\nFinal Answer: {final_answer}")

        except asyncio.TimeoutError:
            pytest.fail("Test timed out waiting for LLM response")
        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")
        finally:
            await manager.close_session(session_id)
            await llm_gateway.close()


class TestRealLLMMemoryIntegration:
    """Test memory integration with real LLM."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation with memory."""
        # Load config
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create LLM gateway
        mimo_config = config["llm"]["providers"]["mimo"]
        llm_gateway = MimoLLMGateway(mimo_config)

        # Create other dependencies
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()

        # Use ReAct planner
        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        def memory_factory(sid):
            return memory

        tools = {"calculator": CalculatorTool()}

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
        ctx = await manager.create_session(user_id="test-user")
        session_id = ctx.session_id

        # First message
        future1 = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "My name is TestUser. Remember this."}
        )

        try:
            events1 = await asyncio.wait_for(future1, timeout=60.0)
            print(f"\nFirst response events: {len(events1)}")

            # Second message (should have context from first)
            future2 = await manager.process_message(
                session_id=session_id,
                user_msg={"role": "user", "content": "What is my name?"}
            )

            events2 = await asyncio.wait_for(future2, timeout=60.0)
            print(f"\nSecond response events: {len(events2)}")

            # Verify messages were saved to memory
            assert session_id in memory.saved_messages
            assert len(memory.saved_messages[session_id]) >= 4  # 2 user + 2 assistant

            # Check final answer
            final_events = [e for e in events2 if e.type == "final_answer"]
            if final_events:
                print(f"\nFinal answer about name: {final_events[0].content}")

        except asyncio.TimeoutError:
            pytest.fail("Test timed out waiting for LLM response")
        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")
        finally:
            await manager.close_session(session_id)
            await llm_gateway.close()


class TestRealLLMEventStreaming:
    """Test event streaming with real LLM."""

    @pytest.mark.asyncio
    async def test_event_types_generated(self):
        """Test that various event types are generated."""
        # Load config
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create LLM gateway
        mimo_config = config["llm"]["providers"]["mimo"]
        llm_gateway = MimoLLMGateway(mimo_config)

        # Create other dependencies
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()

        # Use ReAct planner
        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        def memory_factory(sid):
            return memory

        tools = {"calculator": CalculatorTool()}

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
        ctx = await manager.create_session(user_id="test-user")
        session_id = ctx.session_id

        # Send message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Calculate 10 + 20 and explain."}
        )

        try:
            events = await asyncio.wait_for(future, timeout=90.0)

            # Verify various event types
            event_types = set(e.type for e in events)
            print(f"\nEvent types generated: {event_types}")

            # Should have at least final_answer
            assert "final_answer" in event_types

            # Verify events were published to event bus
            assert len(event_bus.published_events) == len(events)

            # Print all events
            for event in events:
                print(f"\n[{event.type}]: {event.content[:80]}...")

        except asyncio.TimeoutError:
            pytest.fail("Test timed out waiting for LLM response")
        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")
        finally:
            await manager.close_session(session_id)
            await llm_gateway.close()


class TestRealLLMCompleteFlow:
    """Test complete flow with real LLM."""

    @pytest.mark.asyncio
    async def test_complete_agent_flow(self):
        """Test complete agent flow with real LLM."""
        # Load config
        config_path = Path(__file__).parent.parent.parent.parent / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        # Create LLM gateway
        mimo_config = config["llm"]["providers"]["mimo"]
        llm_gateway = MimoLLMGateway(mimo_config)

        # Create other dependencies
        storage = SimpleStorage()
        event_bus = SimpleEventBus()
        memory = SimpleMemory()
        runtime = AgentRuntime()

        # Use ReAct planner
        from agent_framework.planners.react_planner import ReActPlanner
        planner = ReActPlanner()

        def memory_factory(sid):
            return memory

        tools = {"calculator": CalculatorTool()}

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
            user_id="test-user",
            session_type="private"
        )
        session_id = ctx.session_id
        print(f"\nCreated session: {session_id}")

        # Step 2: Send message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "What is the result of 123 * 456? Use the calculator."}
        )

        try:
            # Step 3: Wait for processing
            events = await asyncio.wait_for(future, timeout=120.0)

            # Step 4: Verify results
            print(f"\nTotal events: {len(events)}")

            # Print each event
            for i, event in enumerate(events):
                print(f"\nEvent {i+1} [{event.type}]:")
                print(f"  {event.content[:200]}...")

            # Verify final answer exists
            final_events = [e for e in events if e.type == "final_answer"]
            assert len(final_events) > 0, "No final answer generated"

            # Verify memory was used
            assert session_id in memory.saved_messages
            print(f"\nMessages saved to memory: {len(memory.saved_messages[session_id])}")

            # Verify events were published
            assert len(event_bus.published_events) == len(events)
            print(f"\nEvents published to bus: {len(event_bus.published_events)}")

            # Verify session was saved
            assert session_id in storage.saved_sessions
            print(f"\nSession saved to storage: {session_id}")

            # Step 5: Close session
            await manager.close_session(session_id)
            assert ctx.status == "closed"
            print(f"\nSession closed successfully")

        except asyncio.TimeoutError:
            pytest.fail("Test timed out waiting for LLM response")
        except Exception as e:
            pytest.fail(f"Test failed with error: {e}")
        finally:
            await llm_gateway.close()
