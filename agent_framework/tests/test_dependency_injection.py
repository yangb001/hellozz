"""Dependency injection tests - Verify dependency injection correctness.

This module tests dependency injection patterns, ensuring components
receive correct dependencies and can interact properly.

参考：CLAUDE.md Testing Strategy - 验证依赖注入正确性
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework.interfaces.events import Event


class TestGatewayDependencies:
    """Test gateway dependency injection."""

    def test_get_session_manager(self):
        """Test get_session_manager function."""
        from agent_framework.gateway.dependencies import (
            get_session_manager,
            set_session_manager,
            clear_session_manager
        )

        # Initially should be None
        clear_session_manager()
        assert get_session_manager() is None

        # Set a mock manager
        mock_manager = MagicMock()
        set_session_manager(mock_manager)

        # Verify it's set
        assert get_session_manager() is mock_manager

        # Clear
        clear_session_manager()
        assert get_session_manager() is None

    def test_set_session_manager_replaces_existing(self):
        """Test set_session_manager replaces existing manager."""
        from agent_framework.gateway.dependencies import (
            get_session_manager,
            set_session_manager,
            clear_session_manager
        )

        clear_session_manager()

        # Set first manager
        manager1 = MagicMock()
        set_session_manager(manager1)
        assert get_session_manager() is manager1

        # Set second manager (should replace)
        manager2 = MagicMock()
        set_session_manager(manager2)
        assert get_session_manager() is manager2
        assert get_session_manager() is not manager1

        clear_session_manager()


class TestSessionManagerDependencies:
    """Test SessionManager dependency injection."""

    def test_session_manager_receives_all_dependencies(self):
        """Test SessionManager receives all required dependencies."""
        from agent_framework.core.session_manager import SessionManager

        # Create mock dependencies
        runtime = MagicMock()
        planner = MagicMock()
        memory = AsyncMock()
        storage = AsyncMock()
        event_bus = AsyncMock()
        llm_gateway = AsyncMock()
        tools = {"calculator": MagicMock()}

        def memory_factory(sid):
            return memory

        # Create SessionManager
        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools=tools,
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Verify all dependencies are injected
        assert manager.runtime is runtime
        assert manager.planner is planner
        assert manager.tools is tools
        assert manager.event_bus is event_bus
        assert manager.storage is storage
        assert manager.llm_gateway is llm_gateway
        assert manager.memory_factory is memory_factory

    def test_session_manager_memory_factory_is_stored(self):
        """Test memory factory is stored in SessionManager."""
        from agent_framework.core.session_manager import SessionManager

        # Track created instances
        created_instances = []

        def memory_factory(sid):
            instance = MagicMock()
            instance.session_id = sid
            created_instances.append(instance)
            return instance

        storage = AsyncMock()
        storage.save = AsyncMock()

        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=MagicMock(),
            planner=MagicMock(),
            tools={},
            event_bus=AsyncMock(),
            storage=storage,
            llm_gateway=AsyncMock()
        )

        # Verify factory is stored
        assert manager.memory_factory is memory_factory

        # Verify factory creates instances
        instance1 = memory_factory("session1")
        instance2 = memory_factory("session2")

        assert len(created_instances) == 2
        assert created_instances[0] is instance1
        assert created_instances[1] is instance2


class TestRuntimeDependencies:
    """Test AgentRuntime dependency injection."""

    @pytest.mark.asyncio
    async def test_runtime_receives_planner(self):
        """Test runtime uses injected planner."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        # Create mock planner
        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="final_answer", content="test")

        planner = MagicMock()
        planner.plan_and_act = mock_plan_and_act

        # Create context
        ctx = MagicMock()
        ctx.session_id = "test"
        ctx.messages = []

        # Run runtime
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="test",
            memory=AsyncMock(),
            tools={},
            planner=planner,
            llm_gateway=AsyncMock()
        ):
            events.append(event)

        # Verify planner was used
        assert len(events) == 1
        assert events[0].content == "test"

    @pytest.mark.asyncio
    async def test_runtime_receives_memory(self):
        """Test runtime uses injected memory."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        # Create mock memory
        memory = AsyncMock()
        memory.save = AsyncMock()

        # Create planner
        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            yield Event(type="final_answer", content="test")

        planner = MagicMock()
        planner.plan_and_act = mock_plan_and_act

        # Create context
        ctx = MagicMock()
        ctx.session_id = "test"
        ctx.messages = []

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

        # Verify memory was called
        assert memory.save.call_count >= 1

    @pytest.mark.asyncio
    async def test_runtime_receives_tools(self):
        """Test runtime passes tools to planner."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()

        # Track received tools
        received_tools = {}

        async def mock_plan_and_act(ctx, memory, tools, llm_call):
            received_tools.update(tools)
            yield Event(type="final_answer", content="test")

        planner = MagicMock()
        planner.plan_and_act = mock_plan_and_act

        # Create tools
        calculator = MagicMock()
        tools = {"calculator": calculator}

        # Create context
        ctx = MagicMock()
        ctx.session_id = "test"
        ctx.messages = []

        # Run runtime
        events = []
        async for event in runtime.run(
            ctx=ctx,
            user_input="test",
            memory=AsyncMock(),
            tools=tools,
            planner=planner,
            llm_gateway=AsyncMock()
        ):
            events.append(event)

        # Verify tools were passed
        assert "calculator" in received_tools
        assert received_tools["calculator"] is calculator


class TestPlannerDependencies:
    """Test planner dependency injection."""

    @pytest.mark.asyncio
    async def test_planner_receives_llm_call(self):
        """Test planner receives llm_call function."""
        from agent_framework.planners.react_planner import ReActPlanner

        planner = ReActPlanner()

        # Track received llm_call
        received_llm_call = None

        # Override _build_prompt to avoid complexity
        original_build_prompt = planner._build_prompt

        async def mock_build_prompt(ctx, memory, tools):
            return "test prompt"

        planner._build_prompt = mock_build_prompt

        # Create mock llm_call
        async def mock_llm_call(prompt):
            yield "Final Answer: test response"

        # Create context
        ctx = MagicMock()
        ctx.session_id = "test"
        ctx.messages = []

        # Run planner
        events = []
        async for event in planner.plan_and_act(
            ctx=ctx,
            memory=AsyncMock(),
            tools={},
            llm_call=mock_llm_call
        ):
            events.append(event)

        # Verify planner ran
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_planner_receives_memory(self):
        """Test planner receives memory for context retrieval."""
        from agent_framework.planners.react_planner import ReActPlanner

        planner = ReActPlanner()

        # Create mock memory
        memory = AsyncMock()
        memory.retrieve = AsyncMock(return_value="test context")

        # Create mock llm_call
        async def mock_llm_call(prompt):
            yield "Final Answer: test"

        # Create context
        ctx = MagicMock()
        ctx.session_id = "test"
        ctx.messages = [MagicMock(role="user", content="test")]

        # Run planner
        events = []
        async for event in planner.plan_and_act(
            ctx=ctx,
            memory=memory,
            tools={},
            llm_call=mock_llm_call
        ):
            events.append(event)

        # Verify memory was called
        memory.retrieve.assert_called_once()
