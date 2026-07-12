"""Independent tests for interfaces/enums.py - Based on 详细设计.md specification."""
import pytest
from agent_framework.interfaces.enums import SessionStatus, EventType, MessageRole


class TestSessionStatusFromSpec:
    """Test SessionStatus enum according to spec."""

    def test_active_value_is_string(self):
        """Spec: status: str = 'active'"""
        assert SessionStatus.ACTIVE.value == "active"
        assert isinstance(SessionStatus.ACTIVE, str)

    def test_paused_value_is_string(self):
        """Spec: status: str = 'paused'"""
        assert SessionStatus.PAUSED.value == "paused"
        assert isinstance(SessionStatus.PAUSED, str)

    def test_closed_value_is_string(self):
        """Spec: status: str = 'closed'"""
        assert SessionStatus.CLOSED.value == "closed"
        assert isinstance(SessionStatus.CLOSED, str)


class TestEventTypeFromSpec:
    """Test EventType enum according to spec."""

    def test_thought_value(self):
        """Spec: thought - agent thought process"""
        assert EventType.THOUGHT.value == "thought"

    def test_action_value(self):
        """Spec: action - agent action execution"""
        assert EventType.ACTION.value == "action"

    def test_observation_value(self):
        """Spec: observation - observation result"""
        assert EventType.OBSERVATION.value == "observation"

    def test_text_token_value(self):
        """Spec: text_token - streaming token"""
        assert EventType.TEXT_TOKEN.value == "text_token"

    def test_final_answer_value(self):
        """Spec: final_answer - final response"""
        assert EventType.FINAL_ANSWER.value == "final_answer"

    def test_error_value(self):
        """Spec: error - error event"""
        assert EventType.ERROR.value == "error"

    def test_all_are_strings(self):
        """Spec: All enum values must be strings"""
        for et in EventType:
            assert isinstance(et, str)


class TestMessageRoleFromSpec:
    """Test MessageRole enum according to spec."""

    def test_user_value(self):
        """Spec: role: str = 'user'"""
        assert MessageRole.USER.value == "user"

    def test_assistant_value(self):
        """Spec: role: str = 'assistant'"""
        assert MessageRole.ASSISTANT.value == "assistant"

    def test_system_value(self):
        """Spec: role: str = 'system'"""
        assert MessageRole.SYSTEM.value == "system"

    def test_all_are_strings(self):
        """Spec: All enum values must be strings"""
        for mr in MessageRole:
            assert isinstance(mr, str)


class TestEnumEdgeCases:
    """Boundary and edge case tests."""

    def test_session_status_can_be_used_in_dict_keys(self):
        """Enums should work as dict keys"""
        mapping = {SessionStatus.ACTIVE: "ok", SessionStatus.CLOSED: "ended"}
        assert mapping[SessionStatus.ACTIVE] == "ok"

    def test_event_type_can_be_compared_directly(self):
        """Enums should support direct comparison"""
        assert EventType.THOUGHT == "thought"
        assert EventType.ACTION == "action"

    def test_str_enum_inheritance(self):
        """Spec: Enums inherit from str"""
        assert issubclass(SessionStatus, str)
        assert issubclass(EventType, str)
        assert issubclass(MessageRole, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])