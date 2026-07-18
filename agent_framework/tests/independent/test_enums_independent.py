"""Independent test cases for enumeration types.

This module contains independent verification tests for SessionStatus,
EventType, and MessageRole enums defined in interfaces/enums.py.

Test categories:
1. SessionStatus enum completeness and values
2. EventType enum completeness and values
3. MessageRole enum completeness and values
4. Enum behavior and compatibility
"""
import pytest
from enum import Enum

from agent_framework.interfaces.enums import SessionStatus, EventType, MessageRole


# ============================================================================
# 1. SessionStatus Enum
# ============================================================================


class TestSessionStatusEnum:
    """Independent tests for SessionStatus enum."""

    def test_session_status_is_enum(self):
        """SessionStatus must be an Enum subclass."""
        assert issubclass(SessionStatus, Enum)

    def test_session_status_is_str_enum(self):
        """SessionStatus must also inherit from str."""
        assert issubclass(SessionStatus, str)

    def test_has_active(self):
        """SessionStatus must define ACTIVE member."""
        assert hasattr(SessionStatus, "ACTIVE")
        assert SessionStatus.ACTIVE.value == "active"

    def test_has_paused(self):
        """SessionStatus must define PAUSED member."""
        assert hasattr(SessionStatus, "PAUSED")
        assert SessionStatus.PAUSED.value == "paused"

    def test_has_closed(self):
        """SessionStatus must define CLOSED member."""
        assert hasattr(SessionStatus, "CLOSED")
        assert SessionStatus.CLOSED.value == "closed"

    def test_member_count(self):
        """SessionStatus must have exactly 3 members."""
        assert len(list(SessionStatus)) == 3

    def test_all_values_lowercase(self):
        """All values must be lowercase strings."""
        for status in SessionStatus:
            assert isinstance(status.value, str)
            assert status.value == status.value.lower()

    def test_string_comparison(self):
        """Members should be comparable with string values."""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.PAUSED == "paused"
        assert SessionStatus.CLOSED == "closed"

    def test_from_string(self):
        """Can construct from string value."""
        assert SessionStatus("active") == SessionStatus.ACTIVE
        assert SessionStatus("paused") == SessionStatus.PAUSED
        assert SessionStatus("closed") == SessionStatus.CLOSED

    def test_invalid_value_raises(self):
        """Raises ValueError for unknown string."""
        with pytest.raises(ValueError):
            SessionStatus("unknown")

    def test_is_hashable(self):
        """Members must be hashable."""
        s = {SessionStatus.ACTIVE, SessionStatus.CLOSED}
        assert len(s) == 2


# ============================================================================
# 2. EventType Enum
# ============================================================================


class TestEventTypeEnum:
    """Independent tests for EventType enum."""

    def test_event_type_is_enum(self):
        """EventType must be an Enum subclass."""
        assert issubclass(EventType, Enum)

    def test_event_type_is_str_enum(self):
        """EventType must also inherit from str."""
        assert issubclass(EventType, str)

    def test_has_thought(self):
        """EventType must define THOUGHT."""
        assert hasattr(EventType, "THOUGHT")
        assert EventType.THOUGHT.value == "thought"

    def test_has_action(self):
        """EventType must define ACTION."""
        assert hasattr(EventType, "ACTION")
        assert EventType.ACTION.value == "action"

    def test_has_observation(self):
        """EventType must define OBSERVATION."""
        assert hasattr(EventType, "OBSERVATION")
        assert EventType.OBSERVATION.value == "observation"

    def test_has_text_token(self):
        """EventType must define TEXT_TOKEN."""
        assert hasattr(EventType, "TEXT_TOKEN")
        assert EventType.TEXT_TOKEN.value == "text_token"

    def test_has_final_answer(self):
        """EventType must define FINAL_ANSWER."""
        assert hasattr(EventType, "FINAL_ANSWER")
        assert EventType.FINAL_ANSWER.value == "final_answer"

    def test_has_error(self):
        """EventType must define ERROR."""
        assert hasattr(EventType, "ERROR")
        assert EventType.ERROR.value == "error"

    def test_member_count(self):
        """EventType must have exactly 16 members."""
        assert len(list(EventType)) == 16

    def test_all_values_lowercase(self):
        """All values must be lowercase strings."""
        for event_type in EventType:
            assert isinstance(event_type.value, str)
            assert event_type.value == event_type.value.lower()

    def test_string_comparison(self):
        """Members should be comparable with string values."""
        assert EventType.THOUGHT == "thought"
        assert EventType.ACTION == "action"
        assert EventType.OBSERVATION == "observation"
        assert EventType.TEXT_TOKEN == "text_token"
        assert EventType.FINAL_ANSWER == "final_answer"
        assert EventType.ERROR == "error"

    def test_from_string(self):
        """Can construct from string value."""
        assert EventType("thought") == EventType.THOUGHT
        assert EventType("final_answer") == EventType.FINAL_ANSWER

    def test_invalid_value_raises(self):
        """Raises ValueError for unknown string."""
        with pytest.raises(ValueError):
            EventType("unknown_event")

    def test_is_hashable(self):
        """Members must be hashable."""
        s = {EventType.THOUGHT, EventType.ACTION}
        assert len(s) == 2


# ============================================================================
# 3. MessageRole Enum
# ============================================================================


class TestMessageRoleEnum:
    """Independent tests for MessageRole enum."""

    def test_message_role_is_enum(self):
        """MessageRole must be an Enum subclass."""
        assert issubclass(MessageRole, Enum)

    def test_message_role_is_str_enum(self):
        """MessageRole must also inherit from str."""
        assert issubclass(MessageRole, str)

    def test_has_user(self):
        """MessageRole must define USER."""
        assert hasattr(MessageRole, "USER")
        assert MessageRole.USER.value == "user"

    def test_has_assistant(self):
        """MessageRole must define ASSISTANT."""
        assert hasattr(MessageRole, "ASSISTANT")
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_has_system(self):
        """MessageRole must define SYSTEM."""
        assert hasattr(MessageRole, "SYSTEM")
        assert MessageRole.SYSTEM.value == "system"

    def test_member_count(self):
        """MessageRole must have exactly 3 members."""
        assert len(list(MessageRole)) == 3

    def test_all_values_lowercase(self):
        """All values must be lowercase strings."""
        for role in MessageRole:
            assert isinstance(role.value, str)
            assert role.value == role.value.lower()

    def test_string_comparison(self):
        """Members should be comparable with string values."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"

    def test_from_string(self):
        """Can construct from string value."""
        assert MessageRole("user") == MessageRole.USER
        assert MessageRole("assistant") == MessageRole.ASSISTANT
        assert MessageRole("system") == MessageRole.SYSTEM

    def test_invalid_value_raises(self):
        """Raises ValueError for unknown string."""
        with pytest.raises(ValueError):
            MessageRole("unknown_role")

    def test_is_hashable(self):
        """Members must be hashable."""
        s = {MessageRole.USER, MessageRole.ASSISTANT}
        assert len(s) == 2


# ============================================================================
# 4. Enum Compatibility
# ============================================================================


class TestEnumCompatibility:
    """Test enum compatibility with framework usage."""

    def test_session_status_in_dict(self):
        """SessionStatus can be used as dict value."""
        d = {"status": SessionStatus.ACTIVE}
        assert d["status"] == "active"

    def test_event_type_in_dict(self):
        """EventType can be used as dict value."""
        d = {"type": EventType.FINAL_ANSWER}
        assert d["type"] == "final_answer"

    def test_message_role_in_dict(self):
        """MessageRole can be used as dict value."""
        d = {"role": MessageRole.USER}
        assert d["role"] == "user"

    def test_enums_json_serializable(self):
        """Enum values should be JSON-serializable strings."""
        import json
        data = {
            "status": SessionStatus.ACTIVE.value,
            "event_type": EventType.THOUGHT.value,
            "role": MessageRole.ASSISTANT.value,
        }
        serialized = json.dumps(data)
        assert '"active"' in serialized
        assert '"thought"' in serialized
        assert '"assistant"' in serialized
