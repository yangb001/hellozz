"""Tests for interfaces/enums.py enumeration types."""
import pytest
from agent_framework.interfaces.enums import SessionStatus, EventType, MessageRole


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_has_expected_values(self):
        """Test that all expected values exist."""
        assert hasattr(SessionStatus, "ACTIVE")
        assert hasattr(SessionStatus, "PAUSED")
        assert hasattr(SessionStatus, "CLOSED")

    def test_string_values(self):
        """Test that enum values are strings."""
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.CLOSED.value == "closed"

    def test_is_str_enum(self):
        """Test that enum inherits from str."""
        assert isinstance(SessionStatus.ACTIVE, str)

    def test_comparison_with_strings(self):
        """Test that enum can be compared with strings."""
        assert SessionStatus.ACTIVE == "active"
        assert SessionStatus.PAUSED == "paused"
        assert SessionStatus.CLOSED == "closed"


class TestEventType:
    """Tests for EventType enum."""

    def test_has_expected_values(self):
        """Test that all expected values exist."""
        assert hasattr(EventType, "THOUGHT")
        assert hasattr(EventType, "ACTION")
        assert hasattr(EventType, "OBSERVATION")
        assert hasattr(EventType, "TEXT_TOKEN")
        assert hasattr(EventType, "FINAL_ANSWER")
        assert hasattr(EventType, "ERROR")

    def test_string_values(self):
        """Test that enum values are strings."""
        assert EventType.THOUGHT.value == "thought"
        assert EventType.ACTION.value == "action"
        assert EventType.OBSERVATION.value == "observation"
        assert EventType.TEXT_TOKEN.value == "text_token"
        assert EventType.FINAL_ANSWER.value == "final_answer"
        assert EventType.ERROR.value == "error"

    def test_is_str_enum(self):
        """Test that enum inherits from str."""
        assert isinstance(EventType.THOUGHT, str)

    def test_comparison_with_strings(self):
        """Test that enum can be compared with strings."""
        assert EventType.FINAL_ANSWER == "final_answer"
        assert EventType.TEXT_TOKEN == "text_token"


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_has_expected_values(self):
        """Test that all expected values exist."""
        assert hasattr(MessageRole, "USER")
        assert hasattr(MessageRole, "ASSISTANT")
        assert hasattr(MessageRole, "SYSTEM")

    def test_string_values(self):
        """Test that enum values are strings."""
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_is_str_enum(self):
        """Test that enum inherits from str."""
        assert isinstance(MessageRole.USER, str)

    def test_comparison_with_strings(self):
        """Test that enum can be compared with strings."""
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.SYSTEM == "system"