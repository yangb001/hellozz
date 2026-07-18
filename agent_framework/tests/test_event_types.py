"""Tests for Event and EventType extensions."""
import pytest
from datetime import datetime, timezone

from agent_framework.interfaces.enums import EventType
from agent_framework.interfaces.events import Event, ThinkingData


class TestEventType:
    """Test suite for EventType enum."""

    def test_existing_types(self):
        """Test existing event types are preserved."""
        assert EventType.THOUGHT == "thought"
        assert EventType.ACTION == "action"
        assert EventType.OBSERVATION == "observation"
        assert EventType.TEXT_TOKEN == "text_token"
        assert EventType.FINAL_ANSWER == "final_answer"
        assert EventType.ERROR == "error"

    def test_new_thinking_types(self):
        """Test new thinking event types."""
        assert EventType.CONTENT_TOKEN == "content_token"
        assert EventType.THINKING_START == "thinking_start"
        assert EventType.THINKING_CONTENT == "thinking_content"
        assert EventType.THINKING_END == "thinking_end"

    def test_existing_streaming_types(self):
        """Test existing streaming event types are preserved."""
        assert EventType.REASONING_CONTENT == "reasoning_content"
        assert EventType.TOOL_CALL_START == "tool_call_start"
        assert EventType.TOOL_CALL_ARGUMENT == "tool_call_argument"
        assert EventType.TOOL_CALL_END == "tool_call_end"
        assert EventType.STREAMING_START == "streaming_start"
        assert EventType.STREAMING_END == "streaming_end"

    def test_event_type_is_string_enum(self):
        """Test EventType is a string enum."""
        assert isinstance(EventType.THOUGHT, str)
        assert isinstance(EventType.THINKING_START, str)

    def test_event_type_values_are_unique(self):
        """Test all EventType values are unique."""
        values = [e.value for e in EventType]
        assert len(values) == len(set(values))


class TestThinkingData:
    """Test suite for ThinkingData class."""

    def test_init_default(self):
        """Test initialization with defaults."""
        data = ThinkingData()
        assert data.step == 0
        assert data.label == ""
        assert data.content == ""
        assert data.duration_ms is None
        assert data.token_count is None

    def test_init_with_values(self):
        """Test initialization with values."""
        data = ThinkingData(
            step=1,
            label="test",
            content="content",
            duration_ms=100,
            token_count=50
        )
        assert data.step == 1
        assert data.label == "test"
        assert data.content == "content"
        assert data.duration_ms == 100
        assert data.token_count == 50

    def test_is_pydantic_model(self):
        """Test ThinkingData is a Pydantic model."""
        from pydantic import BaseModel
        assert issubclass(ThinkingData, BaseModel)

    def test_model_dump(self):
        """Test ThinkingData can be converted to dict."""
        data = ThinkingData(step=1, label="test", content="content")
        dumped = data.model_dump()
        assert dumped["step"] == 1
        assert dumped["label"] == "test"
        assert dumped["content"] == "content"
        assert dumped["duration_ms"] is None
        assert dumped["token_count"] is None


class TestEventExtended:
    """Test suite for extended Event class."""

    def test_backward_compatibility(self):
        """Test backward compatibility with existing Event usage."""
        event = Event(type="thought", content="thinking")
        assert event.type == "thought"
        assert event.content == "thinking"
        assert event.metadata == {}
        assert isinstance(event.timestamp, datetime)
        assert event.thinking is None
        assert event.sequence == 0

    def test_event_with_thinking(self):
        """Test Event with thinking data."""
        thinking = ThinkingData(step=1, label="test", content="thinking...")
        event = Event(
            type=EventType.THINKING_CONTENT,
            content="thinking...",
            thinking=thinking
        )
        assert event.type == "thinking_content"
        assert event.thinking is not None
        assert event.thinking.step == 1
        assert event.thinking.label == "test"
        assert event.thinking.content == "thinking..."

    def test_event_with_sequence(self):
        """Test Event with sequence number."""
        event = Event(type=EventType.CONTENT_TOKEN, content="hello", sequence=42)
        assert event.sequence == 42

    def test_event_with_all_new_fields(self):
        """Test Event with all new fields."""
        thinking = ThinkingData(step=1, label="analysis", content="analyzing...")
        event = Event(
            type=EventType.THINKING_END,
            content="analysis complete",
            metadata={"duration_ms": 150},
            thinking=thinking,
            sequence=5
        )
        assert event.type == "thinking_end"
        assert event.content == "analysis complete"
        assert event.metadata == {"duration_ms": 150}
        assert event.thinking.step == 1
        assert event.sequence == 5

    def test_event_model_dump_with_thinking(self):
        """Test Event model_dump includes thinking data."""
        thinking = ThinkingData(step=1, label="test", content="content")
        event = Event(
            type=EventType.THINKING_CONTENT,
            content="content",
            thinking=thinking,
            sequence=1
        )
        dumped = event.model_dump()
        assert dumped["type"] == "thinking_content"
        assert dumped["thinking"]["step"] == 1
        assert dumped["thinking"]["label"] == "test"
        assert dumped["thinking"]["content"] == "content"
        assert dumped["sequence"] == 1

    def test_event_metadata_default_factory(self):
        """Test that metadata uses default_factory correctly."""
        event1 = Event(type="thought")
        event2 = Event(type="thought")
        # Should be different objects
        assert event1.metadata is not event2.metadata

    def test_event_timestamp_default_factory(self):
        """Test that timestamp uses default_factory correctly."""
        before = datetime.now(timezone.utc)
        event = Event(type="thought")
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_content_token_event(self):
        """Test CONTENT_TOKEN event type."""
        event = Event(type=EventType.CONTENT_TOKEN, content="Hello world")
        assert event.type == "content_token"
        assert event.content == "Hello world"

    def test_thinking_start_event(self):
        """Test THINKING_START event type."""
        thinking = ThinkingData(step=1, label="analyze")
        event = Event(
            type=EventType.THINKING_START,
            thinking=thinking
        )
        assert event.type == "thinking_start"
        assert event.thinking.step == 1

    def test_thinking_end_event(self):
        """Test THINKING_END event type."""
        thinking = ThinkingData(step=1, duration_ms=150)
        event = Event(
            type=EventType.THINKING_END,
            thinking=thinking
        )
        assert event.type == "thinking_end"
        assert event.thinking.duration_ms == 150
