"""Independent test cases for FastAPI Gateway application.

This module contains independent verification tests for the FastAPI
application skeleton, following the detailed design specification in section 8.

Test categories:
1. FastAPI application creation
2. CORS middleware configuration
3. Lifespan events (startup/shutdown)
4. Route structure
5. Dependency injection
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestFastAPIAppCreation:
    """Independent tests for FastAPI application creation."""

    def test_create_app_returns_fastapi_instance(self):
        """create_app should return a FastAPI instance."""
        from fastapi import FastAPI
        from agent_framework.gateway.main import create_app

        app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_has_title(self):
        """FastAPI app should have a title."""
        from agent_framework.gateway.main import create_app

        app = create_app()
        assert app.title is not None
        assert len(app.title) > 0

    def test_app_has_description(self):
        """FastAPI app should have a description."""
        from agent_framework.gateway.main import create_app

        app = create_app()
        assert app.description is not None

    def test_app_has_version(self):
        """FastAPI app should have a version."""
        from agent_framework.gateway.main import create_app

        app = create_app()
        assert app.version is not None
        assert len(app.version) > 0

    def test_app_instance_exists(self):
        """Module-level app instance should exist."""
        from agent_framework.gateway.main import app

        from fastapi import FastAPI
        assert isinstance(app, FastAPI)


class TestCORSMiddleware:
    """Independent tests for CORS middleware configuration."""

    def test_cors_middleware_is_configured(self):
        """FastAPI app should have CORS middleware configured."""
        from agent_framework.gateway.main import create_app

        app = create_app()

        # Check if CORS middleware is in the middleware stack
        cors_found = False
        for middleware in app.user_middleware:
            if 'CORSMiddleware' in str(middleware.cls):
                cors_found = True
                break

        assert cors_found, "CORS middleware should be configured"

    def test_cors_allows_all_origins(self):
        """CORS should allow all origins by default."""
        from agent_framework.gateway.main import create_app

        app = create_app()

        # Find CORS middleware and check configuration
        for middleware in app.user_middleware:
            if 'CORSMiddleware' in str(middleware.cls):
                kwargs = middleware.kwargs
                assert kwargs.get('allow_origins') == ["*"]
                break

    def test_cors_allows_credentials(self):
        """CORS should allow credentials."""
        from agent_framework.gateway.main import create_app

        app = create_app()

        for middleware in app.user_middleware:
            if 'CORSMiddleware' in str(middleware.cls):
                kwargs = middleware.kwargs
                assert kwargs.get('allow_credentials') is True
                break

    def test_cors_allows_all_methods(self):
        """CORS should allow all HTTP methods."""
        from agent_framework.gateway.main import create_app

        app = create_app()

        for middleware in app.user_middleware:
            if 'CORSMiddleware' in str(middleware.cls):
                kwargs = middleware.kwargs
                assert kwargs.get('allow_methods') == ["*"]
                break

    def test_cors_allows_all_headers(self):
        """CORS should allow all headers."""
        from agent_framework.gateway.main import create_app

        app = create_app()

        for middleware in app.user_middleware:
            if 'CORSMiddleware' in str(middleware.cls):
                kwargs = middleware.kwargs
                assert kwargs.get('allow_headers') == ["*"]
                break


class TestLifespanEvents:
    """Independent tests for lifespan events."""

    @pytest.mark.asyncio
    async def test_lifespan_is_async_context_manager(self):
        """lifespan should be an async context manager."""
        from agent_framework.gateway.main import lifespan
        from fastapi import FastAPI

        app = FastAPI()

        # Should be usable as async context manager
        async with lifespan(app):
            pass

    @pytest.mark.asyncio
    async def test_lifespan_startup_prints_message(self, capsys):
        """lifespan should log startup message to stderr."""
        from agent_framework.gateway.main import lifespan
        from fastapi import FastAPI

        app = FastAPI()

        async with lifespan(app):
            captured = capsys.readouterr()
            # Logging goes to stderr via StreamHandler
            assert "Starting" in captured.err or "Agent Framework" in captured.err

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_clears_session_manager(self):
        """lifespan should clear session manager on shutdown."""
        from agent_framework.gateway.main import lifespan
        from agent_framework.gateway.dependencies import get_session_manager
        from fastapi import FastAPI

        app = FastAPI()

        async with lifespan(app):
            # During lifespan, session manager should be set (created by lifespan itself)
            assert get_session_manager() is not None

        # After lifespan ends, session manager should be cleared
        assert get_session_manager() is None

    def test_create_app_uses_lifespan(self):
        """create_app should configure lifespan handler."""
        from agent_framework.gateway.main import create_app

        app = create_app()

        # App should have a lifespan handler configured
        assert app.router.lifespan_context is not None


class TestRouteStructure:
    """Independent tests for route structure."""

    def test_app_has_routes(self):
        """FastAPI app should have routes configured."""
        from agent_framework.gateway.main import create_app

        app = create_app()
        routes = app.routes

        assert len(routes) > 0

    def test_health_check_endpoint_exists(self):
        """App should have /api/v1/health endpoint."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        # Test health endpoint by making a request
        response = client.get("/api/v1/health")
        assert response.status_code == 200, "Health check endpoint should exist and return 200"

    def test_root_endpoint_exists(self):
        """App should have /api/v1/ root endpoint."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import create_app

        app = create_app()
        client = TestClient(app)

        # Test root endpoint by making a request
        response = client.get("/api/v1/")
        assert response.status_code == 200, "Root endpoint should exist and return 200"

    def test_rest_router_prefix(self):
        """REST router should have /api/v1 prefix."""
        from agent_framework.gateway.api.rest import router

        assert router.prefix == "/api/v1"

    def test_health_endpoint_method(self):
        """Health endpoint should accept GET requests."""
        from agent_framework.gateway.api.rest import router

        # Check routes in the router
        for route in router.routes:
            if hasattr(route, 'path') and route.path == '/health':
                assert 'GET' in route.methods
                break

    def test_root_endpoint_method(self):
        """Root endpoint should accept GET requests."""
        from agent_framework.gateway.api.rest import router

        for route in router.routes:
            if hasattr(route, 'path') and route.path == '/':
                assert 'GET' in route.methods
                break


class TestDependencyInjection:
    """Independent tests for dependency injection."""

    def test_get_session_manager_returns_none_initially(self):
        """get_session_manager should return None when not set."""
        from agent_framework.gateway.dependencies import get_session_manager, clear_session_manager

        # Clear any existing session manager
        clear_session_manager()

        result = get_session_manager()
        assert result is None

    def test_set_session_manager(self):
        """set_session_manager should set the session manager."""
        from agent_framework.gateway.dependencies import set_session_manager, get_session_manager, clear_session_manager

        # Clear any existing session manager
        clear_session_manager()

        mock_manager = MagicMock()
        set_session_manager(mock_manager)

        result = get_session_manager()
        assert result is mock_manager

        # Cleanup
        clear_session_manager()

    def test_clear_session_manager(self):
        """clear_session_manager should clear the session manager."""
        from agent_framework.gateway.dependencies import set_session_manager, get_session_manager, clear_session_manager

        # Set a mock manager
        mock_manager = MagicMock()
        set_session_manager(mock_manager)

        # Clear it
        clear_session_manager()

        result = get_session_manager()
        assert result is None

    def test_set_session_manager_replaces_existing(self):
        """set_session_manager should replace existing manager."""
        from agent_framework.gateway.dependencies import set_session_manager, get_session_manager, clear_session_manager

        # Clear any existing session manager
        clear_session_manager()

        mock_manager1 = MagicMock()
        mock_manager2 = MagicMock()

        set_session_manager(mock_manager1)
        assert get_session_manager() is mock_manager1

        set_session_manager(mock_manager2)
        assert get_session_manager() is mock_manager2

        # Cleanup
        clear_session_manager()

    def test_get_session_manager_function_exists(self):
        """get_session_manager function should be importable."""
        from agent_framework.gateway.dependencies import get_session_manager
        assert callable(get_session_manager)

    def test_set_session_manager_function_exists(self):
        """set_session_manager function should be importable."""
        from agent_framework.gateway.dependencies import set_session_manager
        assert callable(set_session_manager)

    def test_clear_session_manager_function_exists(self):
        """clear_session_manager function should be importable."""
        from agent_framework.gateway.dependencies import clear_session_manager
        assert callable(clear_session_manager)


class TestRESTEndpoints:
    """Independent tests for REST endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import create_app

        app = create_app()
        return TestClient(app)

    def test_health_check_returns_200(self, client):
        """Health check endpoint should return 200."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_check_returns_status_ok(self, client):
        """Health check should return status ok."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_check_returns_service_name(self, client):
        """Health check should return service name."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "service" in data
        assert data["service"] == "agent-framework"

    def test_root_endpoint_returns_200(self, client):
        """Root endpoint should return 200."""
        response = client.get("/api/v1/")
        assert response.status_code == 200

    def test_root_endpoint_returns_welcome_message(self, client):
        """Root endpoint should return welcome message."""
        response = client.get("/api/v1/")
        data = response.json()
        assert "message" in data
        assert "Welcome" in data["message"] or "Agent Framework" in data["message"]

    def test_nonexistent_endpoint_returns_404(self, client):
        """Non-existent endpoint should return 404."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404


class TestAppConfiguration:
    """Independent tests for app configuration."""

    def test_app_has_openapi_url(self):
        """FastAPI app should have OpenAPI URL configured."""
        from agent_framework.gateway.main import create_app

        app = create_app()
        assert app.openapi_url is not None

    def test_app_has_docs_url(self):
        """FastAPI app should have docs URL configured."""
        from agent_framework.gateway.main import create_app

        app = create_app()
        assert app.docs_url is not None

    def test_app_has_redoc_url(self):
        """FastAPI app should have ReDoc URL configured."""
        from agent_framework.gateway.main import create_app

        app = create_app()
        assert app.redoc_url is not None


class TestGatewayIntegration:
    """Independent integration tests for gateway."""

    def test_create_app_is_callable(self):
        """create_app function should be callable."""
        from agent_framework.gateway.main import create_app
        assert callable(create_app)

    def test_app_module_exports(self):
        """Gateway module should export necessary components."""
        from agent_framework.gateway import main

        assert hasattr(main, 'create_app')
        assert hasattr(main, 'app')
        assert hasattr(main, 'lifespan')

    def test_dependencies_module_exports(self):
        """Dependencies module should export necessary functions."""
        from agent_framework.gateway import dependencies

        assert hasattr(dependencies, 'get_session_manager')
        assert hasattr(dependencies, 'set_session_manager')
        assert hasattr(dependencies, 'clear_session_manager')

    def test_rest_module_exports(self):
        """REST module should export router."""
        from agent_framework.gateway.api import rest

        assert hasattr(rest, 'router')

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test full application lifecycle."""
        from agent_framework.gateway.main import create_app, lifespan
        from agent_framework.gateway.dependencies import set_session_manager, get_session_manager, clear_session_manager

        app = create_app()

        # Clear any existing state
        clear_session_manager()

        # Simulate startup
        async with lifespan(app):
            # During lifespan, app should be ready
            assert app is not None

        # After shutdown, session manager should be cleared
        assert get_session_manager() is None
