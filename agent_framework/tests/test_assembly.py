"""Assembly tests - Verify component assembly and dependency injection.

This module tests that components can be properly assembled together,
verifying dependency injection and component interactions work correctly.

参考：CLAUDE.md Testing Strategy - 组装测试 > 单元测试
"""
import pytest
import asyncio
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.runtime.agent_runtime import AgentRuntime
from agent_framework.core.session_manager import SessionManager


class TestComponentAssembly:
    """Test that components can be assembled together."""

    def test_session_manager_with_real_runtime(self):
        """Test SessionManager can be assembled with AgentRuntime."""
        from unittest.mock import AsyncMock, MagicMock

        # Create real runtime
        runtime = AgentRuntime()

        # Create mock dependencies
        memory = AsyncMock(spec=BaseMemory)
        planner = AsyncMock(spec=BasePlanner)
        storage = AsyncMock()
        event_bus = AsyncMock()
        llm_gateway = AsyncMock()

        def memory_factory(sid):
            return memory

        # Assemble SessionManager
        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Verify assembly
        assert manager is not None
        assert manager.runtime is runtime
        assert manager.planner is planner
        assert manager.memory_factory is memory_factory

    def test_session_manager_with_real_planner(self):
        """Test SessionManager can be assembled with ReActPlanner."""
        from unittest.mock import AsyncMock
        from agent_framework.planners.react_planner import ReActPlanner

        # Create real components
        runtime = AgentRuntime()
        planner = ReActPlanner()

        # Create mock dependencies
        memory = AsyncMock(spec=BaseMemory)
        storage = AsyncMock()
        event_bus = AsyncMock()
        llm_gateway = AsyncMock()

        def memory_factory(sid):
            return memory

        # Assemble SessionManager
        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Verify assembly
        assert manager is not None
        assert isinstance(manager.planner, ReActPlanner)

    def test_session_manager_with_tools(self):
        """Test SessionManager can be assembled with tools."""
        from unittest.mock import AsyncMock

        # Create real components
        runtime = AgentRuntime()

        # Create mock tool
        calculator = AsyncMock()
        calculator.name = "calculator"

        # Create mock dependencies
        memory = AsyncMock(spec=BaseMemory)
        planner = AsyncMock(spec=BasePlanner)
        storage = AsyncMock()
        event_bus = AsyncMock()
        llm_gateway = AsyncMock()

        def memory_factory(sid):
            return memory

        tools = {"calculator": calculator}

        # Assemble SessionManager
        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Verify assembly
        assert "calculator" in manager.tools
        assert manager.tools["calculator"] is calculator


class TestDependencyInjection:
    """Test dependency injection patterns."""

    def test_memory_factory_creates_instances(self):
        """Test that memory factory creates correct instances."""
        from unittest.mock import AsyncMock, MagicMock
        from agent_framework.memory.vector_memory import VectorMemory
        from agent_framework.memory.buffer_memory import BufferMemory

        # Create mock vector store
        vector_store = MagicMock()

        # Create memory factory
        def memory_factory(sid):
            return VectorMemory(vector_store=vector_store)

        # Test factory
        memory1 = memory_factory("session-1")
        memory2 = memory_factory("session-2")

        # Verify different instances
        assert memory1 is not memory2
        assert isinstance(memory1, VectorMemory)
        assert isinstance(memory2, VectorMemory)

    def test_dependency_injection_with_config(self):
        """Test dependency injection with configuration."""
        from unittest.mock import AsyncMock
        from agent_framework.core.config import MemoryConfig

        # Create config
        config = MemoryConfig(short_term_size=50)

        # Verify config can be used for dependency injection
        assert config.short_term_size == 50


class TestComponentInteraction:
    """Test component interactions."""

    @pytest.mark.asyncio
    async def test_runtime_planner_interaction(self):
        """Test AgentRuntime and planner interaction."""
        from unittest.mock import AsyncMock

        # Create real runtime
        runtime = AgentRuntime()

        # Create mock planner that yields events
        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="thought", content="Testing")
            yield Event(type="final_answer", content="Done")

        planner = AsyncMock(spec=BasePlanner)
        planner.plan_and_act = mock_plan_and_act

        # Create other mocks
        memory = AsyncMock(spec=BaseMemory)
        memory.save = AsyncMock()

        ctx = SessionContext(session_id="test")

        # Run runtime
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="test",
            memory=memory,
            tools={},
            planner=planner,
            llm_gateway=AsyncMock()
        ):
            events.append(event)

        # Verify interaction
        assert len(events) == 2
        assert events[0].type == "thought"
        assert events[1].type == "final_answer"

    @pytest.mark.asyncio
    async def test_session_manager_memory_interaction(self):
        """Test SessionManager and memory interaction."""
        from unittest.mock import AsyncMock

        # Create mock memory
        memory = AsyncMock(spec=BaseMemory)
        memory.save = AsyncMock()
        memory.extract_long_term = AsyncMock()

        # Create mock storage
        storage = AsyncMock()
        storage.save = AsyncMock()

        # Create runtime
        runtime = AgentRuntime()

        # Create planner
        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="final_answer", content="Response")

        planner = AsyncMock(spec=BasePlanner)
        planner.plan_and_act = mock_plan_and_act

        def memory_factory(sid):
            return memory

        # Create SessionManager
        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=AsyncMock(),
            storage=storage,
            llm_gateway=AsyncMock()
        )

        # Create session
        ctx = await manager.create_session(user_id="test-user")
        session_id = ctx.session_id

        # Process message
        future = await manager.process_message(
            session_id=session_id,
            user_msg={"role": "user", "content": "Hello"}
        )

        # Wait for processing
        events = await asyncio.wait_for(future, timeout=5.0)

        # Verify memory was called
        assert memory.save.call_count >= 1

        # Cleanup
        await manager.close_session(session_id)
