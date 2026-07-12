"""Independent tests for interfaces/session.py - Based on 详细设计.md specification."""
import pytest
from datetime import datetime, timezone
from agent_framework.interfaces.session import Message, SessionContext


class TestMessageFromSpec:
    """Test Message data class according to spec."""

    def test_required_fields_role_and_content(self):
        """Spec: role: str, content: str are required"""
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_optional_sender_id(self):
        """Spec: sender_id: Optional[str] = None"""
        msg_without = Message(role="user", content="test")
        msg_with = Message(role="assistant", content="reply", sender_id="user123")

        assert msg_without.sender_id is None
        assert msg_with.sender_id == "user123"

    def test_timestamp_exists(self):
        """Spec: timestamp: datetime = datetime.utcnow()"""
        msg = Message(role="user", content="test")
        assert hasattr(msg, 'timestamp')
        assert isinstance(msg.timestamp, datetime)


class TestMessageRolesFromSpec:
    """Test valid role values according to spec."""

    def test_user_role(self):
        """Spec: role: str = 'user'"""
        msg = Message(role="user", content="hi")
        assert msg.role == "user"

    def test_assistant_role(self):
        """Spec: role: str = 'assistant'"""
        msg = Message(role="assistant", content="hi")
        assert msg.role == "assistant"

    def test_system_role(self):
        """Spec: role: str = 'system'"""
        msg = Message(role="system", content="hi")
        assert msg.role == "system"


class TestSessionContextFromSpec:
    """Test SessionContext data class according to spec."""

    def test_required_field_session_id(self):
        """Spec: session_id: str is required"""
        ctx = SessionContext(session_id="sess_123")
        assert ctx.session_id == "sess_123"

    def test_default_session_type_is_private(self):
        """Spec: session_type: str = 'private'"""
        ctx = SessionContext(session_id="test")
        assert ctx.session_type == "private"

    def test_default_participants_is_empty_list(self):
        """Spec: participants: List[str] = []"""
        ctx = SessionContext(session_id="test")
        assert ctx.participants == []

    def test_default_status_is_active(self):
        """Spec: status: str = 'active'"""
        ctx = SessionContext(session_id="test")
        assert ctx.status == "active"

    def test_default_messages_is_empty_list(self):
        """Spec: messages: List[Message] = []"""
        ctx = SessionContext(session_id="test")
        assert ctx.messages == []

    def test_default_agent_state_is_empty_dict(self):
        """Spec: agent_state: Dict[str, Any] = {}"""
        ctx = SessionContext(session_id="test")
        assert ctx.agent_state == {}

    def test_default_tool_instances_is_empty_dict(self):
        """Spec: tool_instances: Dict[str, Any] = {}"""
        ctx = SessionContext(session_id="test")
        assert ctx.tool_instances == {}

    def test_default_metadata_is_empty_dict(self):
        """Spec: metadata: Dict[str, Any] = {}"""
        ctx = SessionContext(session_id="test")
        assert ctx.metadata == {}

    def test_default_created_at_exists(self):
        """Spec: created_at: datetime = datetime.utcnow()"""
        ctx = SessionContext(session_id="test")
        assert hasattr(ctx, 'created_at')
        assert isinstance(ctx.created_at, datetime)

    def test_default_last_active_exists(self):
        """Spec: last_active: datetime = datetime.utcnow()"""
        ctx = SessionContext(session_id="test")
        assert hasattr(ctx, 'last_active')
        assert isinstance(ctx.last_active, datetime)


class TestSessionContextGroupType:
    """Test session_type field according to spec."""

    def test_private_session_type(self):
        """Spec: session_type: str = 'private'"""
        ctx = SessionContext(session_id="test", session_type="private")
        assert ctx.session_type == "private"

    def test_group_session_type(self):
        """Spec: session_type supports 'group' for group chats"""
        ctx = SessionContext(session_id="test", session_type="group")
        assert ctx.session_type == "group"


class TestSessionContextParticipants:
    """Test participants field according to spec."""

    def test_single_participant(self):
        """Spec: participants: List[str] for multi-user sessions"""
        ctx = SessionContext(session_id="test", participants=["user1"])
        assert ctx.participants == ["user1"]

    def test_multiple_participants(self):
        """Group session with multiple participants"""
        ctx = SessionContext(
            session_id="test",
            participants=["user1", "user2", "user3"]
        )
        assert len(ctx.participants) == 3


class TestSessionContextMessages:
    """Test messages field according to spec."""

    def test_can_append_messages(self):
        """Messages should be appendable"""
        ctx = SessionContext(session_id="test")
        msg = Message(role="user", content="hello")
        ctx.messages.append(msg)
        assert len(ctx.messages) == 1

    def test_messages_list_type(self):
        """Spec: messages: List[Message]"""
        ctx = SessionContext(session_id="test")
        assert isinstance(ctx.messages, list)


class TestSessionContextAgentState:
    """Test agent_state field according to spec."""

    def test_agent_state_is_dict(self):
        """Spec: agent_state: Dict[str, Any]"""
        ctx = SessionContext(session_id="test")
        assert isinstance(ctx.agent_state, dict)

    def test_can_store_arbitrary_state(self):
        """Should store any key-value pairs"""
        ctx = SessionContext(session_id="test")
        ctx.agent_state["step"] = 1
        ctx.agent_state["memory_size"] = 1000
        assert ctx.agent_state["step"] == 1


class TestSessionContextTimestamps:
    """Test timestamp fields according to spec."""

    def test_timestamps_are_timezone_aware(self):
        """Implementation uses timezone.utc"""
        ctx = SessionContext(session_id="test")
        assert ctx.created_at.tzinfo is not None
        assert ctx.last_active.tzinfo is not None

    def test_custom_timestamps(self):
        """Custom timestamps should be accepted"""
        custom_time = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)
        ctx = SessionContext(
            session_id="test",
            created_at=custom_time,
            last_active=custom_time
        )
        assert ctx.created_at == custom_time
        assert ctx.last_active == custom_time


class TestSessionContextEdgeCases:
    """Boundary and edge case tests."""

    def test_empty_session_id_is_allowed_by_spec(self):
        """Spec does not forbid empty session_id, so it is accepted"""
        ctx = SessionContext(session_id="")
        assert ctx.session_id == ""

    def test_messages_can_be_initialized_with_list(self):
        """Can provide initial messages list"""
        msgs = [
            Message(role="user", content="first"),
            Message(role="assistant", content="second")
        ]
        ctx = SessionContext(session_id="test", messages=msgs)
        assert len(ctx.messages) == 2

    def test_metadata_can_be_custom_dict(self):
        """Custom metadata should be accepted"""
        meta = {"version": "1.0", "env": "test"}
        ctx = SessionContext(session_id="test", metadata=meta)
        assert ctx.metadata == meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])