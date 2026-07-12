"""Tests for REST API endpoints - TDD implementation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any
from datetime import datetime, timezone

from fastapi.testclient import TestClient
from httpx import AsyncClient

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.gateway.api.rest import (
    router,
    CreateSessionRequest,
    CreateSessionResponse,
    SessionInfoResponse,
    SendMessageRequest,
    SendMessageResponse,
    MessageHistoryResponse,
    ErrorResponse
)


class TestPydanticModels:
    """Test Pydantic request/response models."""

    def test_create_session_request(self):
        """Test CreateSessionRequest model."""
        request = CreateSessionRequest(
            user_id="user123",
            session_type="private",
            participants=["user456"]
        )
        assert request.user_id == "user123"
        assert request.session_type == "private"
        assert request.participants == ["user456"]

    def test_create_session_request_defaults(self):
        """Test CreateSessionRequest with default values."""
        request = CreateSessionRequest(user_id="user123")
        assert request.session_type == "private"
        assert request.participants is None

    def test_create_session_response(self):
        """Test CreateSessionResponse model."""
        response = CreateSessionResponse(
            session_id="sess-123",
            session_type="private",
            participants=["user123", "user456"],
            status="active",
            created_at=datetime.now(timezone.utc)
        )
        assert response.session_id == "sess-123"
        assert response.status == "active"

    def test_session_info_response(self):
        """Test SessionInfoResponse model."""
        response = SessionInfoResponse(
            session_id="sess-123",
            session_type="group",
            participants=["user1", "user2"],
            status="active",
            message_count=5,
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        assert response.session_id == "sess-123"
        assert response.message_count == 5

    def test_send_message_request(self):
        """Test SendMessageRequest model."""
        request = SendMessageRequest(
            content="Hello, how are you?",
            sender_id="user123"
        )
        assert request.content == "Hello, how are you?"
        assert request.sender_id == "user123"

    def test_send_message_request_without_sender(self):
        """Test SendMessageRequest without sender_id."""
        request = SendMessageRequest(content="Hello")
        assert request.sender_id is None

    def test_send_message_response(self):
        """Test SendMessageResponse model."""
        response = SendMessageResponse(
            session_id="sess-123",
            events=[{"type": "text_token", "content": "Hello"}, {"type": "final_answer", "content": "Hi!"}],
            message_count=2
        )
        assert response.session_id == "sess-123"
        assert len(response.events) == 2

    def test_message_history_response(self):
        """Test MessageHistoryResponse model."""
        response = MessageHistoryResponse(
            session_id="sess-123",
            messages=[
                {"role": "user", "content": "Hello", "sender_id": "user1"},
                {"role": "assistant", "content": "Hi!", "sender_id": "assistant"}
            ],
            total_count=2
        )
        assert response.session_id == "sess-123"
        assert response.total_count == 2

    def test_error_response(self):
        """Test ErrorResponse model."""
        response = ErrorResponse(
            error="Session not found",
            detail="Session with ID 'sess-123' does not exist"
        )
        assert response.error == "Session not found"


class TestCreateSessionEndpoint:
    """Test POST /sessions endpoint."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        mock = AsyncMock()
        mock.create_session = AsyncMock()
        return mock

    @pytest.fixture
    def client(self, mock_session_manager):
        """Create a test client with mocked dependencies."""
        from fastapi import FastAPI
        from agent_framework.gateway.dependencies import set_session_manager, clear_session_manager

        app = FastAPI()
        app.include_router(router)

        # Set up mock session manager
        set_session_manager(mock_session_manager)

        yield TestClient(app)

        # Cleanup
        clear_session_manager()

    def test_create_session_success(self, client, mock_session_manager):
        """Test successful session creation."""
        # Mock the session manager response
        mock_ctx = SessionContext(
            session_id="new-session-123",
            session_type="private",
            participants=["user123"],
            status="active",
            created_at=datetime.now(timezone.utc)
        )
        mock_session_manager.create_session.return_value = mock_ctx

        # Make request
        response = client.post(
            "/api/v1/sessions",
            json={"user_id": "user123", "session_type": "private"}
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "new-session-123"
        assert data["session_type"] == "private"
        assert "user123" in data["participants"]

    def test_create_session_with_participants(self, client, mock_session_manager):
        """Test session creation with participants."""
        mock_ctx = SessionContext(
            session_id="group-session-456",
            session_type="group",
            participants=["user1", "user2", "user3"],
            status="active",
            created_at=datetime.now(timezone.utc)
        )
        mock_session_manager.create_session.return_value = mock_ctx

        response = client.post(
            "/api/v1/sessions",
            json={
                "user_id": "user1",
                "session_type": "group",
                "participants": ["user2", "user3"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_type"] == "group"
        assert len(data["participants"]) == 3

    def test_create_session_missing_user_id(self, client):
        """Test session creation with missing user_id."""
        response = client.post(
            "/api/v1/sessions",
            json={"session_type": "private"}
        )

        assert response.status_code == 422  # Validation error

    def test_create_session_manager_error(self, client, mock_session_manager):
        """Test session creation when manager raises error."""
        mock_session_manager.create_session.side_effect = RuntimeError("Failed to create session")

        response = client.post(
            "/api/v1/sessions",
            json={"user_id": "user123"}
        )

        assert response.status_code == 500
        data = response.json()
        # FastAPI wraps HTTPException detail in a 'detail' key
        assert "detail" in data
        assert "error" in data["detail"]


class TestGetSessionEndpoint:
    """Test GET /sessions/{session_id} endpoint."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        mock = AsyncMock()
        mock.get_session = AsyncMock()
        return mock

    @pytest.fixture
    def client(self, mock_session_manager):
        """Create a test client with mocked dependencies."""
        from fastapi import FastAPI
        from agent_framework.gateway.dependencies import set_session_manager, clear_session_manager

        app = FastAPI()
        app.include_router(router)

        set_session_manager(mock_session_manager)

        yield TestClient(app)

        clear_session_manager()

    def test_get_session_success(self, client, mock_session_manager):
        """Test successful session retrieval."""
        mock_ctx = SessionContext(
            session_id="sess-123",
            session_type="private",
            participants=["user123"],
            status="active",
            messages=[
                Message(role="user", content="Hello", sender_id="user123"),
                Message(role="assistant", content="Hi!", sender_id="assistant")
            ],
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        mock_session_manager.get_session.return_value = mock_ctx

        response = client.get("/api/v1/sessions/sess-123")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-123"
        assert data["message_count"] == 2

    def test_get_session_not_found(self, client, mock_session_manager):
        """Test session not found."""
        mock_session_manager.get_session.return_value = None

        response = client.get("/api/v1/sessions/nonexistent")

        assert response.status_code == 404
        data = response.json()
        # FastAPI wraps HTTPException detail in a 'detail' key
        assert "detail" in data
        assert "error" in data["detail"]

    def test_get_session_manager_error(self, client, mock_session_manager):
        """Test session retrieval when manager raises error."""
        mock_session_manager.get_session.side_effect = RuntimeError("Database error")

        response = client.get("/api/v1/sessions/sess-123")

        assert response.status_code == 500


class TestSendMessageEndpoint:
    """Test POST /sessions/{session_id}/messages endpoint."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        mock = AsyncMock()
        mock.process_message = AsyncMock()
        return mock

    @pytest.fixture
    def client(self, mock_session_manager):
        """Create a test client with mocked dependencies."""
        from fastapi import FastAPI
        from agent_framework.gateway.dependencies import set_session_manager, clear_session_manager

        app = FastAPI()
        app.include_router(router)

        set_session_manager(mock_session_manager)

        yield TestClient(app)

        clear_session_manager()

    def test_send_message_success(self, client, mock_session_manager):
        """Test successful message sending."""
        # Create mock events
        from agent_framework.interfaces.events import Event
        from datetime import datetime, timezone
        import asyncio

        mock_events = [
            Event(type="text_token", content="Thinking...", timestamp=datetime.now(timezone.utc)),
            Event(type="final_answer", content="Hello! How can I help?", timestamp=datetime.now(timezone.utc))
        ]

        # Create a mock future that returns events
        async def mock_process_message(session_id, user_msg):
            # Create and return a pre-resolved future
            future = asyncio.get_event_loop().create_future()
            future.set_result(mock_events)
            return future

        mock_session_manager.process_message = mock_process_message

        # Mock get_session for message count
        mock_ctx = SessionContext(
            session_id="sess-123",
            session_type="private",
            participants=["user123"],
            status="active",
            messages=[
                Message(role="user", content="Hello", sender_id="user123"),
                Message(role="assistant", content="Hello! How can I help?", sender_id="assistant")
            ],
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        mock_session_manager.get_session.return_value = mock_ctx

        response = client.post(
            "/api/v1/sessions/sess-123/messages",
            json={"content": "Hello", "sender_id": "user123"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-123"
        assert len(data["events"]) > 0

    def test_send_message_session_not_found(self, client, mock_session_manager):
        """Test sending message to non-existent session."""
        # Make process_message raise ValueError
        async def mock_process_message_error(session_id, user_msg):
            raise ValueError(f"Session '{session_id}' does not exist")

        mock_session_manager.process_message = mock_process_message_error

        response = client.post(
            "/api/v1/sessions/nonexistent/messages",
            json={"content": "Hello"}
        )

        assert response.status_code == 404
        data = response.json()
        # FastAPI wraps HTTPException detail in a 'detail' key
        assert "detail" in data
        assert "error" in data["detail"]

    def test_send_message_missing_content(self, client, mock_session_manager):
        """Test sending message with missing content."""
        response = client.post(
            "/api/v1/sessions/sess-123/messages",
            json={"sender_id": "user123"}
        )

        assert response.status_code == 422  # Validation error

    def test_send_message_manager_error(self, client, mock_session_manager):
        """Test sending message when manager raises error."""
        mock_session_manager.process_message.side_effect = RuntimeError("Processing failed")

        response = client.post(
            "/api/v1/sessions/sess-123/messages",
            json={"content": "Hello"}
        )

        assert response.status_code == 500


class TestGetMessageHistoryEndpoint:
    """Test GET /sessions/{session_id}/messages endpoint."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        mock = AsyncMock()
        mock.get_session = AsyncMock()
        return mock

    @pytest.fixture
    def client(self, mock_session_manager):
        """Create a test client with mocked dependencies."""
        from fastapi import FastAPI
        from agent_framework.gateway.dependencies import set_session_manager, clear_session_manager

        app = FastAPI()
        app.include_router(router)

        set_session_manager(mock_session_manager)

        yield TestClient(app)

        clear_session_manager()

    def test_get_messages_success(self, client, mock_session_manager):
        """Test successful message history retrieval."""
        mock_ctx = SessionContext(
            session_id="sess-123",
            session_type="private",
            participants=["user123"],
            status="active",
            messages=[
                Message(role="user", content="Hello", sender_id="user123", timestamp=datetime.now(timezone.utc)),
                Message(role="assistant", content="Hi!", sender_id="assistant", timestamp=datetime.now(timezone.utc)),
                Message(role="user", content="How are you?", sender_id="user123", timestamp=datetime.now(timezone.utc))
            ],
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        mock_session_manager.get_session.return_value = mock_ctx

        response = client.get("/api/v1/sessions/sess-123/messages")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-123"
        assert data["total_count"] == 3
        assert len(data["messages"]) == 3

    def test_get_messages_with_limit(self, client, mock_session_manager):
        """Test message history retrieval with limit."""
        mock_ctx = SessionContext(
            session_id="sess-123",
            session_type="private",
            participants=["user123"],
            status="active",
            messages=[
                Message(role="user", content="Hello", sender_id="user123", timestamp=datetime.now(timezone.utc)),
                Message(role="assistant", content="Hi!", sender_id="assistant", timestamp=datetime.now(timezone.utc)),
                Message(role="user", content="How are you?", sender_id="user123", timestamp=datetime.now(timezone.utc))
            ],
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        mock_session_manager.get_session.return_value = mock_ctx

        response = client.get("/api/v1/sessions/sess-123/messages?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2

    def test_get_messages_session_not_found(self, client, mock_session_manager):
        """Test message history for non-existent session."""
        mock_session_manager.get_session.return_value = None

        response = client.get("/api/v1/sessions/nonexistent/messages")

        assert response.status_code == 404
        data = response.json()
        # FastAPI wraps HTTPException detail in a 'detail' key
        assert "detail" in data
        assert "error" in data["detail"]

    def test_get_messages_empty_history(self, client, mock_session_manager):
        """Test message history for session with no messages."""
        mock_ctx = SessionContext(
            session_id="empty-session",
            session_type="private",
            participants=["user123"],
            status="active",
            messages=[],
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        mock_session_manager.get_session.return_value = mock_ctx

        response = client.get("/api/v1/sessions/empty-session/messages")

        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 0
        assert data["messages"] == []


class TestEndpointIntegration:
    """Integration tests for REST endpoints."""

    @pytest.fixture
    def mock_session_manager(self):
        """Create a mock session manager."""
        mock = AsyncMock()
        mock.create_session = AsyncMock()
        mock.get_session = AsyncMock()
        mock.process_message = AsyncMock()
        return mock

    @pytest.fixture
    def client(self, mock_session_manager):
        """Create a test client with mocked dependencies."""
        from fastapi import FastAPI
        from agent_framework.gateway.dependencies import set_session_manager, clear_session_manager

        app = FastAPI()
        app.include_router(router)

        set_session_manager(mock_session_manager)

        yield TestClient(app)

        clear_session_manager()

    def test_full_workflow(self, client, mock_session_manager):
        """Test full workflow: create session, send message, get history."""
        from agent_framework.interfaces.events import Event

        # Step 1: Create session
        mock_ctx = SessionContext(
            session_id="new-session",
            session_type="private",
            participants=["user123"],
            status="active",
            created_at=datetime.now(timezone.utc)
        )
        mock_session_manager.create_session.return_value = mock_ctx

        create_response = client.post(
            "/api/v1/sessions",
            json={"user_id": "user123"}
        )
        assert create_response.status_code == 200
        session_id = create_response.json()["session_id"]

        # Step 2: Get session info
        mock_ctx_with_messages = SessionContext(
            session_id=session_id,
            session_type="private",
            participants=["user123"],
            status="active",
            messages=[],
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        mock_session_manager.get_session.return_value = mock_ctx_with_messages

        get_response = client.get(f"/api/v1/sessions/{session_id}")
        assert get_response.status_code == 200

        # Step 3: Send message
        import asyncio

        mock_events = [
            Event(type="final_answer", content="Hello!", timestamp=datetime.now(timezone.utc))
        ]

        async def mock_process_message(session_id, user_msg):
            # Create and return a pre-resolved future
            future = asyncio.get_event_loop().create_future()
            future.set_result(mock_events)
            return future

        mock_session_manager.process_message = mock_process_message

        # Update mock for get_session after message sent
        mock_ctx_after_send = SessionContext(
            session_id=session_id,
            session_type="private",
            participants=["user123"],
            status="active",
            messages=[
                Message(role="user", content="Hello", sender_id="user123", timestamp=datetime.now(timezone.utc)),
                Message(role="assistant", content="Hello!", sender_id="assistant", timestamp=datetime.now(timezone.utc))
            ],
            created_at=datetime.now(timezone.utc),
            last_active=datetime.now(timezone.utc)
        )
        mock_session_manager.get_session.return_value = mock_ctx_after_send

        send_response = client.post(
            f"/api/v1/sessions/{session_id}/messages",
            json={"content": "Hello", "sender_id": "user123"}
        )
        assert send_response.status_code == 200

        # Step 4: Get message history
        history_response = client.get(f"/api/v1/sessions/{session_id}/messages")
        assert history_response.status_code == 200
        assert history_response.json()["total_count"] == 2

    def test_error_handling_consistency(self, client, mock_session_manager):
        """Test that error responses are consistent across endpoints."""
        # Session not found for GET
        mock_session_manager.get_session.return_value = None
        response1 = client.get("/api/v1/sessions/nonexistent")
        assert response1.status_code == 404
        data1 = response1.json()
        assert "detail" in data1
        assert "error" in data1["detail"]

        # Session not found for POST messages
        async def mock_process_message_error(session_id, user_msg):
            raise ValueError("Session not found")

        mock_session_manager.process_message = mock_process_message_error

        response2 = client.post(
            "/api/v1/sessions/nonexistent/messages",
            json={"content": "Hello"}
        )
        assert response2.status_code == 404
        data2 = response2.json()
        assert "detail" in data2
        assert "error" in data2["detail"]

        # Session not found for GET messages
        response3 = client.get("/api/v1/sessions/nonexistent/messages")
        assert response3.status_code == 404
        data3 = response3.json()
        assert "detail" in data3
        assert "error" in data3["detail"]