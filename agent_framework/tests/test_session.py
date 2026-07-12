import pytest
from datetime import datetime
from pydantic import ValidationError

from agent_framework.interfaces.session import Message, SessionContext


class TestMessage:
    """Test suite for Message data class."""

    def test_message_creation_with_required_fields(self):
        """Test Message creation with only required fields."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.sender_id is None
        assert isinstance(msg.timestamp, datetime)

    def test_message_creation_with_all_fields(self):
        """Test Message creation with all fields specified."""
        ts = datetime(2026, 1, 1, 12, 0, 0)
        msg = Message(
            role="assistant",
            content="Hi there",
            sender_id="user123",
            timestamp=ts
        )
        assert msg.role == "assistant"
        assert msg.content == "Hi there"
        assert msg.sender_id == "user123"
        assert msg.timestamp == ts

    def test_message_role_types(self):
        """Test Message accepts valid role values."""
        for role in ["user", "assistant", "system", "tool"]:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_message_content_types(self):
        """Test Message accepts various content types."""
        msg = Message(role="user", content="Simple string")
        assert msg.content == "Simple string"

    def test_message_sender_id_optional(self):
        """Test sender_id is optional."""
        msg = Message(role="user", content="test")
        assert msg.sender_id is None


class TestSessionContext:
    """Test suite for SessionContext data class."""

    def test_session_context_creation_with_required_fields(self):
        """Test SessionContext creation with only required fields."""
        ctx = SessionContext(session_id="sess_123")
        assert ctx.session_id == "sess_123"
        assert ctx.session_type == "private"
        assert ctx.participants == []
        assert ctx.status == "active"
        assert ctx.messages == []
        assert ctx.agent_state == {}
        assert ctx.tool_instances == {}
        assert ctx.metadata == {}
        assert isinstance(ctx.created_at, datetime)
        assert isinstance(ctx.last_active, datetime)

    def test_session_context_creation_with_all_fields(self):
        """Test SessionContext creation with all fields specified."""
        ts_created = datetime(2026, 1, 1, 10, 0, 0)
        ts_active = datetime(2026, 1, 1, 12, 0, 0)
        msg = Message(role="user", content="Hello")

        ctx = SessionContext(
            session_id="sess_456",
            session_type="group",
            participants=["user1", "user2"],
            status="paused",
            messages=[msg],
            agent_state={"step": 5},
            tool_instances={"calc": {}},
            metadata={"key": "value"},
            created_at=ts_created,
            last_active=ts_active
        )

        assert ctx.session_id == "sess_456"
        assert ctx.session_type == "group"
        assert ctx.participants == ["user1", "user2"]
        assert ctx.status == "paused"
        assert len(ctx.messages) == 1
        assert ctx.agent_state == {"step": 5}
        assert ctx.tool_instances == {"calc": {}}
        assert ctx.metadata == {"key": "value"}
        assert ctx.created_at == ts_created
        assert ctx.last_active == ts_active

    def test_session_context_default_values(self):
        """Test SessionContext default value settings."""
        ctx = SessionContext(session_id="sess_789")

        assert ctx.session_type == "private"
        assert ctx.status == "active"
        assert ctx.messages == []

    def test_session_context_private_session(self):
        """Test private session context."""
        ctx = SessionContext(
            session_id="private_1",
            session_type="private",
            participants=["user1"]
        )
        assert ctx.session_type == "private"
        assert ctx.participants == ["user1"]

    def test_session_context_group_session(self):
        """Test group session context."""
        ctx = SessionContext(
            session_id="group_1",
            session_type="group",
            participants=["user1", "user2", "user3"]
        )
        assert ctx.session_type == "group"
        assert len(ctx.participants) == 3

    def test_session_context_status_values(self):
        """Test SessionContext accepts valid status values."""
        for status in ["active", "paused", "closed"]:
            ctx = SessionContext(session_id="sess", status=status)
            assert ctx.status == status

    def test_session_context_add_message(self):
        """Test adding messages to session context."""
        ctx = SessionContext(session_id="sess_msg")
        msg1 = Message(role="user", content="First")
        msg2 = Message(role="assistant", content="Second")

        ctx.messages.append(msg1)
        ctx.messages.append(msg2)

        assert len(ctx.messages) == 2
        assert ctx.messages[0].content == "First"
        assert ctx.messages[1].content == "Second"

    def test_session_context_agent_state(self):
        """Test agent_state dictionary manipulation."""
        ctx = SessionContext(session_id="sess_state")
        ctx.agent_state["current_step"] = 1
        ctx.agent_state["total_steps"] = 3

        assert ctx.agent_state["current_step"] == 1
        assert ctx.agent_state["total_steps"] == 3

    def test_session_context_tool_instances(self):
        """Test tool_instances dictionary manipulation."""
        ctx = SessionContext(session_id="sess_tools")
        ctx.tool_instances["calculator"] = {"last_result": 42}

        assert ctx.tool_instances["calculator"]["last_result"] == 42

    def test_session_context_metadata(self):
        """Test metadata dictionary manipulation."""
        ctx = SessionContext(session_id="sess_meta")
        ctx.metadata["custom_key"] = "custom_value"

        assert ctx.metadata["custom_key"] == "custom_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])