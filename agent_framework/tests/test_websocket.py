"""Tests for WebSocket endpoint - /ws/chat."""

import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_framework.gateway.api.websocket import router
from agent_framework.interfaces.events import Event


def _make_awaitable(result):
    """Create an awaitable that resolves to result without needing an event loop."""
    class _Awaitable:
        def __await__(self):
            yield
            return result
    return _Awaitable()


@pytest.fixture
def app():
    """Create a test FastAPI app with WebSocket router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def mock_session_manager():
    """Create a mock SessionManager."""
    sm = AsyncMock()
    sm.get_session = AsyncMock(return_value=MagicMock(session_id="test-session"))
    return sm


@pytest.fixture
def mock_event():
    """Create a mock Event."""
    return Event(type="text_token", content="Hello", metadata={})


class TestWebSocketEndpoint:
    """Test cases for WebSocket /ws/chat endpoint."""

    def test_websocket_connect_and_disconnect(self, app, mock_session_manager):
        """Should accept WebSocket connection and handle disconnect."""
        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test-token") as ws:
                # Connection established, then disconnect
                pass
            # If we get here without error, connect/disconnect works

    def test_websocket_receives_user_message(self, app, mock_session_manager, mock_event):
        """Should receive user message and process it."""
        # process_message returns an awaitable (future); await future returns events
        mock_session_manager.process_message = AsyncMock(
            return_value=_make_awaitable([mock_event])
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test-token") as ws:
                ws.send_json({"type": "user_message", "content": "Hello"})
                response = ws.receive_json()
                assert response["type"] == "text_token"
                assert response["content"] == "Hello"

    def test_websocket_sends_multiple_events(self, app, mock_session_manager):
        """Should send multiple events from processing."""
        events = [
            Event(type="thought", content="Thinking..."),
            Event(type="text_token", content="Hello"),
            Event(type="final_answer", content="Done"),
        ]
        mock_session_manager.process_message = AsyncMock(
            return_value=_make_awaitable(events)
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test-token") as ws:
                ws.send_json({"type": "user_message", "content": "Hello"})
                for expected in events:
                    response = ws.receive_json()
                    assert response["type"] == expected.type
                    assert response["content"] == expected.content

    def test_websocket_handles_processing_error(self, app, mock_session_manager):
        """Should send error event when processing fails."""
        mock_session_manager.process_message = AsyncMock(
            side_effect=RuntimeError("Processing failed")
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test-token") as ws:
                ws.send_json({"type": "user_message", "content": "Hello"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "Processing failed" in response["content"]

    def test_websocket_handles_session_not_found(self, app, mock_session_manager):
        """Should send error when session does not exist."""
        mock_session_manager.process_message = AsyncMock(
            side_effect=ValueError("Session 'bad-session' does not exist")
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=bad-session&token=test-token") as ws:
                ws.send_json({"type": "user_message", "content": "Hello"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "does not exist" in response["content"]

    def test_websocket_handles_no_session_manager(self, app):
        """Should send error when SessionManager is not initialized."""
        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=None
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test-token") as ws:
                ws.send_json({"type": "user_message", "content": "Hello"})
                response = ws.receive_json()
                assert response["type"] == "error"
                assert "not initialized" in response["content"]

    def test_websocket_multiple_messages(self, app, mock_session_manager):
        """Should handle multiple messages in sequence."""
        event1 = Event(type="text_token", content="Response 1")
        event2 = Event(type="text_token", content="Response 2")

        mock_session_manager.process_message = AsyncMock(
            side_effect=[_make_awaitable([event1]), _make_awaitable([event2])]
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test-token") as ws:
                ws.send_json({"type": "user_message", "content": "First"})
                response1 = ws.receive_json()
                assert response1["content"] == "Response 1"

                ws.send_json({"type": "user_message", "content": "Second"})
                response2 = ws.receive_json()
                assert response2["content"] == "Response 2"
