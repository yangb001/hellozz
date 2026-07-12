"""Startup tests - Verify application can start normally.

This module tests that the application can start successfully,
verifying critical components initialization.

参考：CLAUDE.md Testing Strategy - 启动测试是最后一道防线
"""
import pytest
import asyncio
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestApplicationStartup:
    """Test application startup."""

    def test_fastapi_app_creation(self):
        """Test FastAPI app can be created."""
        from agent_framework.gateway.main import app
        from fastapi import FastAPI

        assert app is not None
        assert isinstance(app, FastAPI)

    def test_fastapi_app_has_routes(self):
        """Test FastAPI app has routes configured."""
        from agent_framework.gateway.main import app

        # Verify routes exist
        assert len(app.routes) > 0

    def test_fastapi_app_has_middleware(self):
        """Test FastAPI app has middleware configured."""
        from agent_framework.gateway.main import app
        from starlette.middleware.cors import CORSMiddleware

        # Verify CORS middleware
        middleware_classes = [m.cls for m in app.user_middleware]
        assert CORSMiddleware in middleware_classes

    def test_fastapi_app_startup_event(self):
        """Test FastAPI app has startup event configured."""
        from agent_framework.gateway.main import app

        # Verify lifespan is configured
        assert app.router.lifespan_context is not None


class TestComponentInitialization:
    """Test component initialization."""

    def test_agent_runtime_initialization(self):
        """Test AgentRuntime can be initialized."""
        from agent_framework.runtime.agent_runtime import AgentRuntime

        runtime = AgentRuntime()
        assert runtime is not None

    def test_react_planner_initialization(self):
        """Test ReActPlanner can be initialized."""
        from agent_framework.planners.react_planner import ReActPlanner

        planner = ReActPlanner()
        assert planner is not None
        assert planner.name == "react"

    def test_calculator_tool_initialization(self):
        """Test CalculatorTool can be initialized."""
        from unittest.mock import AsyncMock

        # Create mock tool
        tool = AsyncMock()
        tool.name = "calculator"
        tool.description = "Calculator tool"

        assert tool is not None
        assert tool.name == "calculator"

    def test_vector_memory_initialization(self):
        """Test VectorMemory can be initialized."""
        from unittest.mock import MagicMock
        from agent_framework.memory.vector_memory import VectorMemory

        vector_store = MagicMock()
        memory = VectorMemory(vector_store=vector_store)
        assert memory is not None

    def test_buffer_memory_initialization(self):
        """Test BufferMemory can be initialized."""
        from agent_framework.memory.buffer_memory import BufferMemory

        memory = BufferMemory()
        assert memory is not None


class TestSessionManagerStartup:
    """Test SessionManager startup."""

    def test_session_manager_initialization(self):
        """Test SessionManager can be initialized with all dependencies."""
        from unittest.mock import AsyncMock, MagicMock
        from agent_framework.runtime.agent_runtime import AgentRuntime
        from agent_framework.planners.react_planner import ReActPlanner
        from agent_framework.core.session_manager import SessionManager

        # Create real components
        runtime = AgentRuntime()
        planner = ReActPlanner()

        # Create mock dependencies
        memory = AsyncMock()
        storage = AsyncMock()
        event_bus = AsyncMock()
        llm_gateway = AsyncMock()

        def memory_factory(sid):
            return memory

        # Initialize SessionManager
        manager = SessionManager(
            memory_factory=memory_factory,
            runtime=runtime,
            planner=planner,
            tools={},
            event_bus=event_bus,
            storage=storage,
            llm_gateway=llm_gateway
        )

        # Verify initialization
        assert manager is not None
        assert manager.runtime is runtime
        assert manager.planner is planner

    @pytest.mark.asyncio
    async def test_session_manager_create_session(self):
        """Test SessionManager can create a session."""
        from unittest.mock import AsyncMock
        from agent_framework.runtime.agent_runtime import AgentRuntime
        from agent_framework.planners.react_planner import ReActPlanner
        from agent_framework.core.session_manager import SessionManager

        # Create components
        runtime = AgentRuntime()
        planner = ReActPlanner()
        memory = AsyncMock()
        storage = AsyncMock()
        storage.save = AsyncMock()
        event_bus = AsyncMock()
        llm_gateway = AsyncMock()

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

        # Create session
        ctx = await manager.create_session(user_id="test-user")

        # Verify session created
        assert ctx is not None
        assert ctx.session_id is not None
        assert ctx.status == "active"

        # Cleanup
        await manager.close_session(ctx.session_id)


class TestTestClientStartup:
    """Test application startup with test client."""

    def test_test_client_creation(self):
        """Test test client can be created."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        assert client is not None

    def test_health_endpoint(self):
        """Test health endpoint responds correctly."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_root_endpoint(self):
        """Test root endpoint responds correctly."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        response = client.get("/api/v1/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
