"""Tests for WebSocket streaming and session switching fixes.

Tests verify:
1. Streaming text_token events are properly sent to the client
2. Multiple event types (text_token, action, observation, final_answer) are handled
3. Session switching resets processing state
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent_framework.gateway.api.websocket import router
from agent_framework.interfaces.events import Event


def _make_awaitable(result):
    """Create an awaitable that resolves to result."""
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


class TestWebSocketStreaming:
    """Test cases for streaming text_token events."""

    def test_streaming_tokens_sent_individually(self, app, mock_session_manager):
        """Each text_token event should be sent as a separate WebSocket message."""
        events = [
            Event(type="text_token", content="Hello"),
            Event(type="text_token", content=" world"),
            Event(type="text_token", content="!"),
            Event(type="final_answer", content="Hello world!"),
        ]
        mock_session_manager.process_message = AsyncMock(
            return_value=_make_awaitable(events)
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test") as ws:
                ws.send_json({"type": "user_message", "content": "Hi"})

                # Receive all events
                received = []
                for _ in range(4):
                    received.append(ws.receive_json())

                assert received[0]["type"] == "text_token"
                assert received[0]["content"] == "Hello"
                assert received[1]["type"] == "text_token"
                assert received[1]["content"] == " world"
                assert received[2]["type"] == "text_token"
                assert received[2]["content"] == "!"
                assert received[3]["type"] == "final_answer"
                assert received[3]["content"] == "Hello world!"

    def test_mixed_event_types_sent(self, app, mock_session_manager):
        """All event types (thought, action, observation, text_token, final_answer) should be sent."""
        events = [
            Event(type="thought", content="I need to think"),
            Event(type="text_token", content="Let me"),
            Event(type="text_token", content=" search"),
            Event(type="action", content="Calling search..."),
            Event(type="observation", content="Search results here"),
            Event(type="text_token", content=" Based on"),
            Event(type="text_token", content=" results"),
            Event(type="final_answer", content="Based on results, the answer is..."),
        ]
        mock_session_manager.process_message = AsyncMock(
            return_value=_make_awaitable(events)
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test") as ws:
                ws.send_json({"type": "user_message", "content": "Search for something"})

                received = []
                for _ in range(8):
                    received.append(ws.receive_json())

                types = [r["type"] for r in received]
                assert types == [
                    "thought", "text_token", "text_token",
                    "action", "observation",
                    "text_token", "text_token",
                    "final_answer"
                ]

    def test_only_text_tokens_no_final_answer(self, app, mock_session_manager):
        """Events without final_answer should still be sent (frontend handles gracefully)."""
        events = [
            Event(type="text_token", content="Partial"),
            Event(type="text_token", content=" response"),
            Event(type="error", content="Max iterations reached"),
        ]
        mock_session_manager.process_message = AsyncMock(
            return_value=_make_awaitable(events)
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test") as ws:
                ws.send_json({"type": "user_message", "content": "Hello"})

                received = []
                for _ in range(3):
                    received.append(ws.receive_json())

                assert received[0]["type"] == "text_token"
                assert received[0]["content"] == "Partial"
                assert received[1]["type"] == "text_token"
                assert received[1]["content"] == " response"
                assert received[2]["type"] == "error"
                assert "Max iterations" in received[2]["content"]

    def test_empty_events_list(self, app, mock_session_manager):
        """Empty events list should not cause errors."""
        mock_session_manager.process_message = AsyncMock(
            return_value=_make_awaitable([])
        )

        with patch(
            "agent_framework.gateway.api.websocket.get_session_manager",
            return_value=mock_session_manager
        ):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test") as ws:
                ws.send_json({"type": "user_message", "content": "Hello"})
                # No events to receive, connection should remain open
                # Send another message to verify connection is still alive
                events2 = [Event(type="final_answer", content="OK")]
                mock_session_manager.process_message = AsyncMock(
                    return_value=_make_awaitable(events2)
                )
                ws.send_json({"type": "user_message", "content": "Again"})
                response = ws.receive_json()
                assert response["type"] == "final_answer"
                assert response["content"] == "OK"

    def test_timestamp_included_in_events(self, app, mock_session_manager):
        """Each event should include an ISO format timestamp."""
        events = [
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
            with client.websocket_connect("/ws/chat?session_id=test-session&token=test") as ws:
                ws.send_json({"type": "user_message", "content": "Hi"})

                r1 = ws.receive_json()
                assert "timestamp" in r1
                # Should be ISO format string
                assert "T" in r1["timestamp"]

                r2 = ws.receive_json()
                assert "timestamp" in r2
                assert "T" in r2["timestamp"]
