"""Independent test cases for Event data class.

This module contains independent verification tests for the Event
data class defined in interfaces/events.py.

Test categories:
1. Event field definitions and types
2. Event creation with defaults
3. Event creation with custom values
4. Event immutability and validation
5. Boundary conditions
"""
import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from agent_framework.interfaces.events import Event


# ============================================================================
# 1. Event Field Definitions
# ============================================================================


class TestEventFields:
    """Test Event data class field definitions."""

    def test_has_type_field(self):
        """Event must have type field."""
        fields = Event.model_fields
        assert "type" in fields

    def test_has_content_field(self):
        """Event must have content field."""
        fields = Event.model_fields
        assert "content" in fields

    def test_has_metadata_field(self):
        """Event must have metadata field."""
        fields = Event.model_fields
        assert "metadata" in fields

    def test_has_timestamp_field(self):
        """Event must have timestamp field."""
        fields = Event.model_fields
        assert "timestamp" in fields

    def test_type_is_required(self):
        """type field must be required (no default)."""
        fields = Event.model_fields
        assert fields["type"].is_required()

    def test_content_has_default(self):
        """content field must have default value."""
        fields = Event.model_fields
        assert not fields["content"].is_required()

    def test_metadata_has_default(self):
        """metadata field must have default value."""
        fields = Event.model_fields
        assert not fields["metadata"].is_required()


# ============================================================================
# 2. Event Creation with Defaults
# ============================================================================


class TestEventDefaults:
    """Test Event creation with minimal fields."""

    def test_create_with_type_only(self):
        """Event can be created with only type field."""
        event = Event(type="thought")
        assert event.type == "thought"

    def test_default_content_empty_string(self):
        """Default content should be empty string."""
        event = Event(type="thought")
        assert event.content == ""

    def test_default_metadata_empty_dict(self):
        """Default metadata should be empty dict."""
        event = Event(type="thought")
        assert event.metadata == {}

    def test_default_timestamp_is_utc(self):
        """Default timestamp should be UTC timezone-aware."""
        event = Event(type="thought")
        assert event.timestamp.tzinfo is not None
        assert event.timestamp.tzinfo == timezone.utc

    def test_default_timestamp_is_recent(self):
        """Default timestamp should be close to current time."""
        before = datetime.now(timezone.utc)
        event = Event(type="thought")
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after


# ============================================================================
# 3. Event Creation with Custom Values
# ============================================================================


class TestEventCustomValues:
    """Test Event creation with custom values."""

    def test_create_with_all_fields(self):
        """Event can be created with all fields specified."""
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event = Event(
            type="action",
            content="calling tool",
            metadata={"tool": "search"},
            timestamp=ts
        )
        assert event.type == "action"
        assert event.content == "calling tool"
        assert event.metadata == {"tool": "search"}
        assert event.timestamp == ts

    def test_create_with_content(self):
        """Event accepts custom content."""
        event = Event(type="final_answer", content="the answer")
        assert event.content == "the answer"

    def test_create_with_metadata(self):
        """Event accepts custom metadata."""
        meta = {"key": "value", "nested": {"a": 1}}
        event = Event(type="observation", metadata=meta)
        assert event.metadata == meta

    def test_create_with_empty_content(self):
        """Event accepts empty string content."""
        event = Event(type="error", content="")
        assert event.content == ""

    def test_create_with_empty_metadata(self):
        """Event accepts empty dict metadata."""
        event = Event(type="thought", metadata={})
        assert event.metadata == {}


# ============================================================================
# 4. Event Validation
# ============================================================================


class TestEventValidation:
    """Test Event validation rules."""

    def test_missing_type_raises(self):
        """Event raises ValidationError when type is missing."""
        with pytest.raises(ValidationError):
            Event()

    def test_accepts_known_event_types(self):
        """Event accepts all known event type strings."""
        types = ["thought", "action", "observation", "text_token", "final_answer", "error"]
        for t in types:
            event = Event(type=t)
            assert event.type == t

    def test_accepts_unknown_event_type(self):
        """Event accepts unknown type strings (no enum constraint)."""
        event = Event(type="custom_type")
        assert event.type == "custom_type"

    def test_type_must_be_string(self):
        """Event type must be a string."""
        with pytest.raises(ValidationError):
            Event(type=123)


# ============================================================================
# 5. Event Behavior
# ============================================================================


class TestEventBehavior:
    """Test Event class behavior."""

    def test_event_is_pydantic_model(self):
        """Event must be a Pydantic BaseModel."""
        from pydantic import BaseModel
        assert issubclass(Event, BaseModel)

    def test_event_serialization(self):
        """Event can be serialized to dict."""
        event = Event(type="thought", content="thinking")
        data = event.model_dump()
        assert data["type"] == "thought"
        assert data["content"] == "thinking"

    def test_event_json_serialization(self):
        """Event can be serialized to JSON."""
        import json
        event = Event(type="action", content="do something")
        json_str = event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["type"] == "action"
        assert parsed["content"] == "do something"

    def test_event_equality(self):
        """Two Events with same values should be equal."""
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
        e1 = Event(type="thought", content="x", timestamp=ts)
        e2 = Event(type="thought", content="x", timestamp=ts)
        assert e1 == e2

    def test_event_inequality(self):
        """Two Events with different values should not be equal."""
        e1 = Event(type="thought", content="a")
        e2 = Event(type="thought", content="b")
        assert e1 != e2

    def test_metadata_independent_per_instance(self):
        """Each Event instance should have independent metadata."""
        e1 = Event(type="thought")
        e2 = Event(type="thought")
        e1.metadata["key"] = "value"
        assert "key" not in e2.metadata
