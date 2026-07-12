"""Tests for Event data class in interfaces module."""
import pytest
from datetime import datetime
from agent_framework.interfaces.events import Event


class TestEvent:
    """Test suite for Event data class."""

    def test_event_creation_with_required_fields(self):
        """Test creating Event with only required type field."""
        event = Event(type="thought")
        assert event.type == "thought"
        assert event.content == ""
        assert event.metadata == {}
        assert isinstance(event.timestamp, datetime)

    def test_event_creation_with_all_fields(self):
        """Test creating Event with all fields specified."""
        timestamp = datetime(2026, 7, 11, 10, 0, 0)
        event = Event(
            type="action",
            content="searching web",
            metadata={"source": "tool"},
            timestamp=timestamp
        )
        assert event.type == "action"
        assert event.content == "searching web"
        assert event.metadata == {"source": "tool"}
        assert event.timestamp == timestamp

    def test_event_default_timestamp_is_utcnow(self):
        """Test that default timestamp is generated via datetime.now(timezone.utc)."""
        from datetime import timezone
        before = datetime.now(timezone.utc)
        event = Event(type="observation")
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_event_default_metadata_is_empty_dict(self):
        """Test that default metadata is an empty dictionary."""
        event = Event(type="text_token")
        assert event.metadata == {}

    def test_event_default_content_is_empty_string(self):
        """Test that default content is an empty string."""
        event = Event(type="final_answer")
        assert event.content == ""

    def test_event_with_various_event_types(self):
        """Test Event with various valid type values."""
        valid_types = [
            "thought",
            "action",
            "observation",
            "text_token",
            "final_answer",
            "error"
        ]
        for event_type in valid_types:
            event = Event(type=event_type)
            assert event.type == event_type

    def test_event_metadata_can_holdarbitrary_data(self):
        """Test that metadata can hold arbitrary key-value pairs."""
        metadata = {
            "tool_name": "web_search",
            "confidence": 0.95,
            "tags": ["search", "web"]
        }
        event = Event(type="action", metadata=metadata)
        assert event.metadata == metadata

    def test_event_is_pydantic_basemodel(self):
        """Test Event inherits from Pydantic BaseModel."""
        from pydantic import BaseModel
        assert issubclass(Event, BaseModel)

    def test_event_model_validation(self):
        """Test Pydantic validates the model correctly."""
        event = Event(type="error", content="Something went wrong")
        assert event.type == "error"
        assert event.content == "Something went wrong"

    def test_event_to_dict(self):
        """Test Event can be converted to dictionary."""
        event = Event(type="thought", content="thinking...")
        data = event.model_dump()
        assert data["type"] == "thought"
        assert data["content"] == "thinking..."
        assert "timestamp" in data
        assert "metadata" in data