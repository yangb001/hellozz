"""Independent tests for interfaces/events.py - Based on 详细设计.md specification."""
import pytest
from datetime import datetime, timezone
from agent_framework.interfaces.events import Event


class TestEventStructureFromSpec:
    """Test Event data class according to spec."""

    def test_required_field_type(self):
        """Spec: type: str - event type is required"""
        event = Event(type="thought")
        assert event.type == "thought"

    def test_default_content_is_empty_string(self):
        """Spec: content: str = ''"""
        event = Event(type="thought")
        assert event.content == ""

    def test_default_metadata_is_empty_dict(self):
        """Spec: metadata: dict = {}"""
        event = Event(type="action")
        assert event.metadata == {}

    def test_timestamp_exists(self):
        """Spec: timestamp: datetime = datetime.utcnow()"""
        event = Event(type="observation")
        assert hasattr(event, 'timestamp')
        assert isinstance(event.timestamp, datetime)


class TestEventValidTypesFromSpec:
    """Test that spec-defined event types are accepted."""

    def test_thought_event(self):
        """Spec: thought - agent thought process"""
        event = Event(type="thought")
        assert event.type == "thought"

    def test_action_event(self):
        """Spec: action - agent action"""
        event = Event(type="action", content="calling tool")
        assert event.type == "action"

    def test_observation_event(self):
        """Spec: observation - action result"""
        event = Event(type="observation", content="got result")
        assert event.type == "observation"

    def test_text_token_event(self):
        """Spec: text_token - streaming token"""
        event = Event(type="text_token", content="token123")
        assert event.type == "text_token"

    def test_final_answer_event(self):
        """Spec: final_answer - final response"""
        event = Event(type="final_answer", content="answer text")
        assert event.type == "final_answer"

    def test_error_event(self):
        """Spec: error - error occurred"""
        event = Event(type="error", content="error message")
        assert event.type == "error"


class TestEventWithMetadata:
    """Test metadata field usage."""

    def test_metadata_can_store_arbitrary_dict(self):
        """Spec: metadata can hold any key-value pairs"""
        meta = {"tool": "calculator", "result": 42}
        event = Event(type="action", metadata=meta)
        assert event.metadata == meta

    def test_metadata_default_is_mutable_empty_dict(self):
        """Default metadata should be a new dict each time"""
        e1 = Event(type="a")
        e2 = Event(type="b")
        e1.metadata["key"] = "value"
        assert e2.metadata == {}


class TestEventTimestampBehavior:
    """Test timestamp field according to spec."""

    def test_timestamp_is_timezone_aware(self):
        """Implementation uses timezone.utc, verify awareness"""
        event = Event(type="test")
        assert event.timestamp.tzinfo is not None

    def test_custom_timestamp_can_be_set(self):
        """Custom timestamp should be accepted"""
        custom_time = datetime(2026, 7, 11, 12, 0, 0, tzinfo=timezone.utc)
        event = Event(type="test", timestamp=custom_time)
        assert event.timestamp == custom_time


class TestEventPydanticIntegration:
    """Test Event as Pydantic BaseModel."""

    def test_event_is_pydantic_model(self):
        """Event should be a Pydantic model"""
        from pydantic import BaseModel
        assert issubclass(Event, BaseModel)

    def test_model_dump_includes_all_fields(self):
        """model_dump should serialize all fields"""
        event = Event(type="test", content="hello")
        data = event.model_dump()
        assert "type" in data
        assert "content" in data
        assert "metadata" in data
        assert "timestamp" in data

    def test_model_validation_requires_type(self):
        """Type field is required"""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Event(content="no type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])