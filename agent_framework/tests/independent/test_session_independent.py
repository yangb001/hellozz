"""Independent test cases for Message and SessionContext data classes.

This module contains independent verification tests for Message and
SessionContext defined in interfaces/session.py.

Test categories:
1. Message field definitions, types, and defaults
2. Message creation and validation
3. SessionContext field definitions, types, and defaults
4. SessionContext creation and validation
5. Boundary conditions
"""
import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from agent_framework.interfaces.session import Message, SessionContext


# ============================================================================
# 1. Message Field Definitions
# ============================================================================


class TestMessageFields:
    """Test Message data class field definitions."""

    def test_has_role_field(self):
        """Message must have role field."""
        fields = Message.model_fields
        assert "role" in fields

    def test_has_content_field(self):
        """Message must have content field."""
        fields = Message.model_fields
        assert "content" in fields

    def test_has_sender_id_field(self):
        """Message must have sender_id field."""
        fields = Message.model_fields
        assert "sender_id" in fields

    def test_has_timestamp_field(self):
        """Message must have timestamp field."""
        fields = Message.model_fields
        assert "timestamp" in fields

    def test_role_is_required(self):
        """role field must be required."""
        assert Message.model_fields["role"].is_required()

    def test_content_is_required(self):
        """content field must be required."""
        assert Message.model_fields["content"].is_required()

    def test_sender_id_is_optional(self):
        """sender_id field must be optional."""
        assert not Message.model_fields["sender_id"].is_required()


# ============================================================================
# 2. Message Creation and Validation
# ============================================================================


class TestMessageCreation:
    """Test Message creation and validation."""

    def test_create_with_required_fields(self):
        """Message can be created with only required fields."""
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_create_with_all_fields(self):
        """Message can be created with all fields."""
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        msg = Message(role="assistant", content="hi", sender_id="bot-1", timestamp=ts)
        assert msg.role == "assistant"
        assert msg.content == "hi"
        assert msg.sender_id == "bot-1"
        assert msg.timestamp == ts

    def test_default_sender_id_is_none(self):
        """Default sender_id should be None."""
        msg = Message(role="user", content="test")
        assert msg.sender_id is None

    def test_default_timestamp_is_utc(self):
        """Default timestamp should be UTC timezone-aware."""
        msg = Message(role="user", content="test")
        assert msg.timestamp.tzinfo is not None
        assert msg.timestamp.tzinfo == timezone.utc

    def test_default_timestamp_is_recent(self):
        """Default timestamp should be close to current time."""
        before = datetime.now(timezone.utc)
        msg = Message(role="user", content="test")
        after = datetime.now(timezone.utc)
        assert before <= msg.timestamp <= after

    def test_missing_role_raises(self):
        """Message raises ValidationError when role is missing."""
        with pytest.raises(ValidationError):
            Message(content="hello")

    def test_missing_content_raises(self):
        """Message raises ValidationError when content is missing."""
        with pytest.raises(ValidationError):
            Message(role="user")

    def test_accepts_known_roles(self):
        """Message accepts known role strings."""
        for role in ["user", "assistant", "system"]:
            msg = Message(role=role, content="test")
            assert msg.role == role

    def test_accepts_unknown_role(self):
        """Message accepts unknown role strings (no enum constraint)."""
        msg = Message(role="custom_role", content="test")
        assert msg.role == "custom_role"

    def test_empty_content(self):
        """Message accepts empty content string."""
        msg = Message(role="user", content="")
        assert msg.content == ""

    def test_is_pydantic_model(self):
        """Message must be a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert issubclass(Message, BaseModel)

    def test_serialization(self):
        """Message can be serialized to dict."""
        msg = Message(role="user", content="hello")
        data = msg.model_dump()
        assert data["role"] == "user"
        assert data["content"] == "hello"

    def test_timestamp_independent_per_instance(self):
        """Each Message instance should have independent timestamp."""
        import time
        m1 = Message(role="user", content="a")
        m2 = Message(role="user", content="b")
        # Both should have timestamps (may be same due to speed)
        assert m1.timestamp is not None
        assert m2.timestamp is not None


# ============================================================================
# 3. SessionContext Field Definitions
# ============================================================================


class TestSessionContextFields:
    """Test SessionContext data class field definitions."""

    def test_has_session_id(self):
        """SessionContext must have session_id field."""
        assert "session_id" in SessionContext.model_fields

    def test_has_session_type(self):
        """SessionContext must have session_type field."""
        assert "session_type" in SessionContext.model_fields

    def test_has_participants(self):
        """SessionContext must have participants field."""
        assert "participants" in SessionContext.model_fields

    def test_has_status(self):
        """SessionContext must have status field."""
        assert "status" in SessionContext.model_fields

    def test_has_messages(self):
        """SessionContext must have messages field."""
        assert "messages" in SessionContext.model_fields

    def test_has_agent_state(self):
        """SessionContext must have agent_state field."""
        assert "agent_state" in SessionContext.model_fields

    def test_has_tool_instances(self):
        """SessionContext must have tool_instances field."""
        assert "tool_instances" in SessionContext.model_fields

    def test_has_metadata(self):
        """SessionContext must have metadata field."""
        assert "metadata" in SessionContext.model_fields

    def test_has_created_at(self):
        """SessionContext must have created_at field."""
        assert "created_at" in SessionContext.model_fields

    def test_has_last_active(self):
        """SessionContext must have last_active field."""
        assert "last_active" in SessionContext.model_fields

    def test_session_id_is_required(self):
        """session_id must be required."""
        assert SessionContext.model_fields["session_id"].is_required()


# ============================================================================
# 4. SessionContext Creation and Validation
# ============================================================================


class TestSessionContextCreation:
    """Test SessionContext creation and validation."""

    def test_create_with_session_id_only(self):
        """SessionContext can be created with only session_id."""
        ctx = SessionContext(session_id="s1")
        assert ctx.session_id == "s1"

    def test_default_session_type(self):
        """Default session_type should be 'private'."""
        ctx = SessionContext(session_id="s1")
        assert ctx.session_type == "private"

    def test_default_participants_empty(self):
        """Default participants should be empty list."""
        ctx = SessionContext(session_id="s1")
        assert ctx.participants == []

    def test_default_status(self):
        """Default status should be 'active'."""
        ctx = SessionContext(session_id="s1")
        assert ctx.status == "active"

    def test_default_messages_empty(self):
        """Default messages should be empty list."""
        ctx = SessionContext(session_id="s1")
        assert ctx.messages == []

    def test_default_agent_state_empty(self):
        """Default agent_state should be empty dict."""
        ctx = SessionContext(session_id="s1")
        assert ctx.agent_state == {}

    def test_default_tool_instances_empty(self):
        """Default tool_instances should be empty dict."""
        ctx = SessionContext(session_id="s1")
        assert ctx.tool_instances == {}

    def test_default_metadata_empty(self):
        """Default metadata should be empty dict."""
        ctx = SessionContext(session_id="s1")
        assert ctx.metadata == {}

    def test_default_created_at_is_utc(self):
        """Default created_at should be UTC timezone-aware."""
        ctx = SessionContext(session_id="s1")
        assert ctx.created_at.tzinfo == timezone.utc

    def test_default_last_active_is_utc(self):
        """Default last_active should be UTC timezone-aware."""
        ctx = SessionContext(session_id="s1")
        assert ctx.last_active.tzinfo == timezone.utc

    def test_create_with_custom_values(self):
        """SessionContext accepts custom values."""
        ctx = SessionContext(
            session_id="s1",
            session_type="group",
            participants=["u1", "u2"],
            status="paused",
        )
        assert ctx.session_type == "group"
        assert ctx.participants == ["u1", "u2"]
        assert ctx.status == "paused"

    def test_missing_session_id_raises(self):
        """SessionContext raises ValidationError when session_id is missing."""
        with pytest.raises(ValidationError):
            SessionContext()

    def test_add_message(self):
        """Messages can be appended to context."""
        ctx = SessionContext(session_id="s1")
        msg = Message(role="user", content="hello")
        ctx.messages.append(msg)
        assert len(ctx.messages) == 1
        assert ctx.messages[0].content == "hello"

    def test_messages_independent_per_instance(self):
        """Each SessionContext should have independent messages list."""
        ctx1 = SessionContext(session_id="s1")
        ctx2 = SessionContext(session_id="s2")
        ctx1.messages.append(Message(role="user", content="a"))
        assert len(ctx2.messages) == 0

    def test_agent_state_independent_per_instance(self):
        """Each SessionContext should have independent agent_state."""
        ctx1 = SessionContext(session_id="s1")
        ctx2 = SessionContext(session_id="s2")
        ctx1.agent_state["key"] = "value"
        assert "key" not in ctx2.agent_state


# ============================================================================
# 5. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_message_with_unicode(self):
        """Message handles Unicode content."""
        msg = Message(role="user", content="你好世界")
        assert msg.content == "你好世界"

    def test_message_with_very_long_content(self):
        """Message handles very long content."""
        content = "x" * 100000
        msg = Message(role="user", content=content)
        assert len(msg.content) == 100000

    def test_session_context_with_many_messages(self):
        """SessionContext handles many messages."""
        ctx = SessionContext(session_id="s1")
        for i in range(1000):
            ctx.messages.append(Message(role="user", content=f"msg-{i}"))
        assert len(ctx.messages) == 1000

    def test_session_context_with_many_participants(self):
        """SessionContext handles many participants."""
        participants = [f"user-{i}" for i in range(100)]
        ctx = SessionContext(session_id="s1", participants=participants)
        assert len(ctx.participants) == 100

    def test_session_context_with_nested_metadata(self):
        """SessionContext handles nested metadata."""
        ctx = SessionContext(session_id="s1")
        ctx.metadata["nested"] = {"key": {"deep": "value"}}
        assert ctx.metadata["nested"]["key"]["deep"] == "value"

    def test_message_with_whitespace_content(self):
        """Message handles whitespace-only content."""
        msg = Message(role="user", content="   ")
        assert msg.content == "   "

    def test_message_with_newlines(self):
        """Message handles content with newlines."""
        msg = Message(role="user", content="line1\nline2\nline3")
        assert "\n" in msg.content
