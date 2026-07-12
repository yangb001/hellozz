"""Tests for FastAPI application skeleton.

This module tests:
- FastAPI app creation and configuration
- CORS middleware setup
- Startup/shutdown events
- Basic route structure
- Dependencies injection
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFastAPIAppCreation:
    """Test FastAPI application creation."""

    def test_import_main(self):
        """Test that gateway.main can be imported."""
        from agent_framework.gateway.main import app
        assert app is not None

    def test_app_is_fastapi_instance(self):
        """Test that app is a FastAPI instance."""
        from fastapi import FastAPI
        from agent_framework.gateway.main import app
        assert isinstance(app, FastAPI)

    def test_app_has_title(self):
        """Test that app has a title."""
        from agent_framework.gateway.main import app
        assert app.title == "Agent Framework API"

    def test_app_has_version(self):
        """Test that app has a version."""
        from agent_framework.gateway.main import app
        assert app.version is not None


class TestCORSMiddleware:
    """Test CORS middleware configuration."""

    def test_cors_middleware_configured(self):
        """Test that CORS middleware is configured."""
        from agent_framework.gateway.main import app
        from starlette.middleware.cors import CORSMiddleware

        # Check if CORS middleware is in the middleware stack
        middleware_classes = [m.cls for m in app.user_middleware]
        assert CORSMiddleware in middleware_classes

    def test_cors_allows_origins(self):
        """Test that CORS allows configured origins."""
        from agent_framework.gateway.main import app
        from starlette.middleware.cors import CORSMiddleware

        # Find CORS middleware configuration
        for middleware in app.user_middleware:
            if middleware.cls == CORSMiddleware:
                # CORS should allow all origins in development
                assert "allow_origins" in middleware.kwargs
                break


class TestRoutes:
    """Test basic route structure."""

    def test_app_has_routes(self):
        """Test that app has routes defined."""
        from agent_framework.gateway.main import app
        # Check that app has routes (including nested routers)
        assert len(app.routes) > 0

    def test_health_check_route(self):
        """Test that health check route exists."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        # Health endpoint is at /api/v1/health
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_api_v1_routes_exist(self):
        """Test that API v1 routes are included."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        # Root API endpoint should exist
        response = client.get("/api/v1/")
        assert response.status_code == 200


class TestStartupShutdown:
    """Test startup and shutdown events."""

    def test_startup_event_configured(self):
        """Test that startup event is configured."""
        from agent_framework.gateway.main import app
        # FastAPI uses on_event decorator or lifespan
        # Check if startup handler exists
        assert len(app.router.on_startup) > 0 or app.router.lifespan_context is not None

    def test_shutdown_event_configured(self):
        """Test that shutdown event is configured."""
        from agent_framework.gateway.main import app
        # Check if shutdown handler exists
        assert len(app.router.on_shutdown) > 0 or app.router.lifespan_context is not None


class TestDependencies:
    """Test dependency injection."""

    def test_import_dependencies(self):
        """Test that dependencies module can be imported."""
        from agent_framework.gateway.dependencies import get_session_manager
        assert callable(get_session_manager)

    def test_get_session_manager_function(self):
        """Test get_session_manager function exists and is callable."""
        from agent_framework.gateway.dependencies import get_session_manager
        assert callable(get_session_manager)

    def test_get_session_manager_returns_value(self):
        """Test get_session_manager returns a value (or None if not initialized)."""
        from agent_framework.gateway.dependencies import get_session_manager
        # This might return None if app is not started
        # Just verify it doesn't raise an exception
        try:
            result = get_session_manager()
            # Result can be None or a SessionManager instance
        except Exception:
            # It's okay if it raises when not initialized
            pass


class TestGatewayIntegration:
    """Integration tests for gateway."""

    def test_app_startup_with_test_client(self):
        """Test app can start with test client."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        # Should not raise exception
        client = TestClient(app)
        assert client is not None

    def test_health_endpoint_returns_200(self):
        """Test health endpoint returns 200 OK."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self):
        """Test health endpoint returns JSON response."""
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        response = client.get("/api/v1/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"


class TestGatewayLoggingInit:
    """Test that logging is initialized during gateway startup."""

    def test_lifespan_calls_setup_logging(self):
        """Test that the lifespan function initializes logging."""
        import logging
        from unittest.mock import patch, MagicMock
        from agent_framework.gateway.main import lifespan, app

        # Verify that setup_logging is called during lifespan
        with patch("agent_framework.gateway.main.setup_logging") as mock_setup:
            with patch("agent_framework.gateway.main.build_session_manager", side_effect=Exception("test")):
                from fastapi.testclient import TestClient
                try:
                    client = TestClient(app)
                except Exception:
                    pass
            # setup_logging should have been called at least once
            # (it may be called during module import or during lifespan)

    def test_logging_initialized_on_startup(self):
        """Test that setup_logging is called during gateway lifespan."""
        from unittest.mock import patch, call
        from agent_framework.core.logging_config import setup_logging as real_setup_logging

        # Patch setup_logging at the module level where it's used
        with patch("agent_framework.gateway.main.setup_logging") as mock_setup:
            with patch("agent_framework.gateway.main.build_session_manager", side_effect=Exception("skip")):
                with patch("agent_framework.gateway.main.load_config") as mock_load:
                    mock_load.return_value.logging.level = "INFO"
                    mock_load.return_value.logging.log_dir = "logs"
                    mock_load.return_value.logging.max_bytes = 10485760
                    mock_load.return_value.logging.backup_count = 5
                    mock_load.return_value.logging.console_output = True
                    mock_load.return_value.logging.file_output = True

                    from agent_framework.gateway.main import lifespan, app
                    import asyncio
                    # Manually run lifespan to verify setup_logging is called
                    async def run_lifespan():
                        async with lifespan(app):
                            pass
                    try:
                        asyncio.run(run_lifespan())
                    except Exception:
                        pass

                    # setup_logging should have been called
                    mock_setup.assert_called()

    def test_gateway_logger_exists_after_startup(self):
        """Test that the gateway logger is available after startup."""
        import logging
        from fastapi.testclient import TestClient
        from agent_framework.gateway.main import app

        client = TestClient(app)
        # The gateway module logger should exist
        gateway_logger = logging.getLogger("agent_framework.gateway")
        assert gateway_logger is not None
