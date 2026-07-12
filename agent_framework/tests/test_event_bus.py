"""Tests for EventBus pub/sub implementation."""
import pytest
import asyncio
from agent_framework.interfaces.events import Event
from agent_framework.core.event_bus import EventBus


class TestEventBus:
    """Test suite for EventBus pub/sub pattern."""

    def test_event_bus_can_be_instantiated(self):
        """Test that EventBus can be created."""
        bus = EventBus()
        assert bus is not None

    def test_subscribe_returns_subscription_id(self):
        """Test that subscribe returns an ID for later unsubscription."""
        bus = EventBus()
        callback = lambda e: None
        sub_id = bus.subscribe("test_event", callback)
        assert sub_id is not None
        assert isinstance(sub_id, str)

    def test_multiple_subscribers_get_different_ids(self):
        """Test that each subscription gets a unique ID."""
        bus = EventBus()
        callback = lambda e: None
        id1 = bus.subscribe("test_event", callback)
        id2 = bus.subscribe("test_event", callback)
        assert id1 != id2

    def test_unsubscribe_removes_subscription(self):
        """Test that unsubscribing removes the callback."""
        bus = EventBus()
        callback = lambda e: None
        sub_id = bus.subscribe("test_event", callback)
        bus.unsubscribe("test_event", sub_id)
        # After unsubscribe, callback should not be called
        assert len(bus._subscriptions.get("test_event", {})) == 0

    @pytest.mark.anyio
    async def test_publish_triggers_callback(self):
        """Test that publishing an event triggers the subscriber callback."""
        bus = EventBus()
        received_events = []

        def callback(event):
            received_events.append(event)

        bus.subscribe("test_event", callback)
        event = Event(type="test_event", content="hello")
        await bus.publish("test_event", event)

        assert len(received_events) == 1
        assert received_events[0].content == "hello"

    @pytest.mark.anyio
    async def test_publish_to_multiple_subscribers(self):
        """Test that an event is delivered to all subscribers."""
        bus = EventBus()
        received_count = [0, 0]

        def callback1(event):
            received_count[0] += 1

        def callback2(event):
            received_count[1] += 1

        bus.subscribe("test_event", callback1)
        bus.subscribe("test_event", callback2)

        event = Event(type="test_event", content="broadcast")
        await bus.publish("test_event", event)

        assert received_count[0] == 1
        assert received_count[1] == 1

    @pytest.mark.anyio
    async def test_subscribe_to_different_event_types(self):
        """Test that subscribers only receive events for their type."""
        bus = EventBus()
        received = []

        bus.subscribe("event_a", lambda e: received.append(e))

        await bus.publish("event_b", Event(type="event_b", content="not for me"))
        await bus.publish("event_a", Event(type="event_a", content="for me"))

        assert len(received) == 1
        assert received[0].content == "for me"

    @pytest.mark.anyio
    async def test_async_callback(self):
        """Test that async callbacks work correctly."""
        bus = EventBus()
        received = []

        async def async_callback(event):
            received.append(event)

        bus.subscribe("async_event", async_callback)
        await bus.publish("async_event", Event(type="async_event", content="async"))

        assert len(received) == 1

    @pytest.mark.anyio
    async def test_publish_without_subscribers_does_not_error(self):
        """Test that publishing to an event with no subscribers doesn't raise."""
        bus = EventBus()
        event = Event(type="unregistered_event", content="test")
        # Should not raise
        await bus.publish("unregistered_event", event)

    def test_subscribe_with_invalid_callback_type(self):
        """Test that non-callable subscribers raise TypeError."""
        bus = EventBus()
        with pytest.raises(TypeError):
            bus.subscribe("test_event", "not a callback")

    def test_unsubscribe_invalid_id_does_not_raise(self):
        """Test that unsubscribing a non-existent ID doesn't raise."""
        bus = EventBus()
        bus.subscribe("test_event", lambda e: None)
        # Should not raise
        bus.unsubscribe("test_event", "nonexistent_id")

    def test_event_bus_clear_removes_all_subscriptions(self):
        """Test that clear removes all subscriptions."""
        bus = EventBus()
        bus.subscribe("event1", lambda e: None)
        bus.subscribe("event2", lambda e: None)
        bus.clear()
        assert len(bus._subscriptions) == 0