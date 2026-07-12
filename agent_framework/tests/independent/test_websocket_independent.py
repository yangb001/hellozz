"""Independent test cases for WebSocket endpoint.

This module contains independent verification tests for the WebSocket
chat endpoint defined in gateway/api/websocket.py.

Test categories:
1. WebSocket connection establishment
2. Message receiving and processing
3. Event stream pushing
4. Error handling
5. Connection disconnect handling
6. Boundary conditions
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from typing import List

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event
from agent_framework.gateway.api.websocket import router
from agent_framework.gateway import dependencies


# ============================================================================
# Mock Implementations
# ============================================================================


class MockSessionManager:
    """Mock SessionManager for testing WebSocket endpoint."""

    def __init__(self, events=None):
        self.events = events or [Event(type="final_answer", content="mock response")]
        self.processed_messages: List[dict] = []

    async def process_message(self, session_id: str, user_msg: dict):
        """Mock process_message that returns a future with events."""
        self.processed_messages.append({"session_id": session_id, "msg": user_msg})
        future = asyncio.Future()
        future.set_result(self.events)
        return future


def _create_app(sm=None):
    """Create a test FastAPI app with WebSocket router."""
    app = FastAPI()
    app.include_router(router)
    dependencies.clear_session_manager()
    if sm is not None:
        dependencies.set_session_manager(sm)
    return app


# ============================================================================
# 1. WebSocket Connection Establishment
# ============================================================================


class TestWebSocketConnection:
    """Test WebSocket connection establishment."""

    def test_websocket_connects_successfully(self):
        """WebSocket can connect with valid parameters."""
        app = _create_app(MockSessionManager())
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            # Connection established successfully
            assert ws is not None

    def test_websocket_accepts_connection(self):
        """WebSocket accepts the connection (no rejection)."""
        app = _create_app(MockSessionManager())
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            # Should be able to send/receive after accept
            ws.send_json({"type": "user_message", "content": "hello"})
            response = ws.receive_json()
            assert response is not None

    def test_websocket_requires_session_id(self):
        """WebSocket endpoint requires session_id parameter."""
        app = _create_app(MockSessionManager())
        client = TestClient(app)
        # Missing session_id should cause an error
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat?token=test") as ws:
                pass

    def test_websocket_requires_token(self):
        """WebSocket endpoint requires token parameter."""
        app = _create_app(MockSessionManager())
        client = TestClient(app)
        # Missing token should cause an error
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/chat?session_id=s1") as ws:
                pass


# ============================================================================
# 2. Message Receiving and Processing
# ============================================================================


class TestMessageProcessing:
    """Test message receiving and processing."""

    def test_receives_user_message(self):
        """WebSocket processes user_message type."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            response = ws.receive_json()
            # Should get at least one response
            assert response is not None

    def test_passes_content_to_session_manager(self):
        """WebSocket passes message content to SessionManager."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "test message"})
            ws.receive_json()  # consume response
            assert len(sm.processed_messages) == 1
            assert sm.processed_messages[0]["msg"]["content"] == "test message"

    def test_passes_session_id_to_session_manager(self):
        """WebSocket passes session_id to SessionManager."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=session-id-123&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            ws.receive_json()
            # session_id comes from query param
            assert sm.processed_messages[0]["session_id"] == "session-id-123"

    def test_sets_sender_id_to_websocket_user(self):
        """WebSocket sets sender_id to 'websocket_user'."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            ws.receive_json()
            assert sm.processed_messages[0]["msg"]["sender_id"] == "websocket_user"

    def test_sets_role_to_user(self):
        """WebSocket sets role to 'user'."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            ws.receive_json()
            assert sm.processed_messages[0]["msg"]["role"] == "user"

    def test_handles_empty_content(self):
        """WebSocket handles message with empty content."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": ""})
            ws.receive_json()
            assert sm.processed_messages[0]["msg"]["content"] == ""

    def test_handles_missing_content(self):
        """WebSocket handles message with missing content field."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message"})
            ws.receive_json()
            assert sm.processed_messages[0]["msg"]["content"] == ""


# ============================================================================
# 3. Event Stream Pushing
# ============================================================================


class TestEventStreamPushing:
    """Test event stream pushing to client."""

    def test_pushes_single_event(self):
        """WebSocket pushes single event back to client."""
        events = [Event(type="final_answer", content="the answer")]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "question"})
            response = ws.receive_json()
            assert response["type"] == "final_answer"
            assert response["content"] == "the answer"

    def test_pushes_multiple_events(self):
        """WebSocket pushes multiple events back to client."""
        events = [
            Event(type="thought", content="thinking..."),
            Event(type="action", content="calling tool"),
            Event(type="final_answer", content="done"),
        ]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "do something"})
            responses = []
            for _ in range(3):
                responses.append(ws.receive_json())
            assert responses[0]["type"] == "thought"
            assert responses[1]["type"] == "action"
            assert responses[2]["type"] == "final_answer"

    def test_event_contains_type(self):
        """Pushed event contains type field."""
        events = [Event(type="observation", content="result")]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "q"})
            response = ws.receive_json()
            assert "type" in response

    def test_event_contains_content(self):
        """Pushed event contains content field."""
        events = [Event(type="final_answer", content="test content")]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "q"})
            response = ws.receive_json()
            assert "content" in response
            assert response["content"] == "test content"

    def test_event_contains_metadata(self):
        """Pushed event contains metadata field."""
        events = [Event(type="final_answer", content="x", metadata={"key": "value"})]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "q"})
            response = ws.receive_json()
            assert "metadata" in response
            assert response["metadata"] == {"key": "value"}

    def test_event_contains_timestamp(self):
        """Pushed event contains timestamp field as ISO string."""
        events = [Event(type="final_answer", content="x")]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "q"})
            response = ws.receive_json()
            assert "timestamp" in response
            # Should be ISO format string
            assert "T" in response["timestamp"]


# ============================================================================
# 4. Error Handling
# ============================================================================


class TestErrorHandling:
    """Test error handling in WebSocket endpoint."""

    def test_invalid_message_type_returns_error(self):
        """WebSocket returns error for invalid message type."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "invalid_type", "content": "hello"})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "Invalid message type" in response["content"]

    def test_missing_type_field_returns_error(self):
        """WebSocket returns error when type field is missing."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"content": "hello"})
            response = ws.receive_json()
            assert response["type"] == "error"

    def test_session_manager_not_initialized(self):
        """WebSocket returns error when SessionManager is None."""
        app = _create_app(None)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "not initialized" in response["content"].lower()

    def test_session_manager_process_error(self):
        """WebSocket returns error when process_message raises."""
        sm = MockSessionManager()

        async def raise_error(session_id, msg):
            future = asyncio.Future()
            future.set_exception(ValueError("Session not found"))
            return future

        sm.process_message = raise_error
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "Session not found" in response["content"]

    def test_general_exception_handling(self):
        """WebSocket returns error for unexpected exceptions."""
        sm = MockSessionManager()

        async def raise_unexpected(session_id, msg):
            future = asyncio.Future()
            future.set_exception(RuntimeError("Unexpected failure"))
            return future

        sm.process_message = raise_unexpected
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            response = ws.receive_json()
            assert response["type"] == "error"
            assert "Processing error" in response["content"]

    def test_continues_after_error(self):
        """WebSocket continues processing after an error."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            # First: invalid message
            ws.send_json({"type": "invalid", "content": "bad"})
            error = ws.receive_json()
            assert error["type"] == "error"

            # Second: valid message should still work
            ws.send_json({"type": "user_message", "content": "hello"})
            response = ws.receive_json()
            assert response["type"] == "final_answer"


# ============================================================================
# 5. Connection Disconnect Handling
# ============================================================================


class TestDisconnectHandling:
    """Test connection disconnect handling."""

    def test_disconnect_no_crash(self):
        """Server does not crash when client disconnects."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            ws.receive_json()
            # Disconnect happens when exiting context manager
        # If we get here, no crash occurred

    def test_multiple_messages_before_disconnect(self):
        """Server handles multiple messages before disconnect."""
        events = [Event(type="final_answer", content="ok")]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            for i in range(5):
                ws.send_json({"type": "user_message", "content": f"msg-{i}"})
                ws.receive_json()
        assert len(sm.processed_messages) == 5


# ============================================================================
# 6. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_long_message_content(self):
        """WebSocket handles very long message content."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        long_content = "x" * 10000
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": long_content})
            ws.receive_json()
            assert sm.processed_messages[0]["msg"]["content"] == long_content

    def test_unicode_message_content(self):
        """WebSocket handles Unicode message content."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "你好世界"})
            ws.receive_json()
            assert sm.processed_messages[0]["msg"]["content"] == "你好世界"

    def test_special_characters_in_content(self):
        """WebSocket handles special characters in content."""
        sm = MockSessionManager()
        app = _create_app(sm)
        client = TestClient(app)
        special = 'test "quotes" and \n newlines'
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": special})
            ws.receive_json()
            assert sm.processed_messages[0]["msg"]["content"] == special

    def test_rapid_successive_messages(self):
        """WebSocket handles rapid successive messages."""
        events = [Event(type="final_answer", content="ok")]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            for i in range(10):
                ws.send_json({"type": "user_message", "content": f"msg-{i}"})
                ws.receive_json()
        assert len(sm.processed_messages) == 10

    def test_empty_event_list(self):
        """WebSocket handles empty event list from SessionManager."""
        sm = MockSessionManager(events=[])
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "hello"})
            # No events to receive, but no crash either
        assert len(sm.processed_messages) == 1

    def test_event_with_empty_metadata(self):
        """WebSocket handles event with empty metadata."""
        events = [Event(type="final_answer", content="x", metadata={})]
        sm = MockSessionManager(events=events)
        app = _create_app(sm)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat?session_id=s1&token=test") as ws:
            ws.send_json({"type": "user_message", "content": "q"})
            response = ws.receive_json()
            assert response["metadata"] == {}

    def test_router_is_apirouter(self):
        """WebSocket router must be an APIRouter."""
        from fastapi import APIRouter
        assert isinstance(router, APIRouter)

    def test_endpoint_path(self):
        """WebSocket endpoint must be at /ws/chat."""
        routes = [r.path for r in router.routes]
        assert "/ws/chat" in routes
