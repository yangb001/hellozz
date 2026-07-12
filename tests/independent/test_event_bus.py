"""Independent test cases for EventBus module."""
import pytest
import asyncio
from agent_framework.core.event_bus import EventBus


class TestEventBusSubscribe:
    """Test suite for EventBus subscribe functionality."""

    def test_subscribe_returns_subscription_id(self):
        """Test that subscribe returns a string subscription ID."""
        bus = EventBus()
        callback = lambda e: None
        sub_id = bus.subscribe("test_event", callback)
        assert isinstance(sub_id, str)
        assert len(sub_id) > 0

    def test_subscribe_non_callable_raises_type_error(self):
        """Test that subscribing with non-callable raises TypeError."""
        bus = EventBus()
        with pytest.raises(TypeError) as exc_info:
            bus.subscribe("test_event", "not_a_callback")
        assert "callback must be callable" in str(exc_info.value)

    def test_subscribe_multiple_callbacks_same_event(self):
        """Test multiple callbacks can subscribe to the same event type."""
        bus = EventBus()
        results = []

        def callback1(e):
            results.append(("cb1", e))

        def callback2(e):
            results.append(("cb2", e))

        bus.subscribe("test_event", callback1)
        bus.subscribe("test_event", callback2)

        assert len(bus._subscriptions["test_event"]) == 2


class TestEventBusUnsubscribe:
    """Test suite for EventBus unsubscribe functionality."""

    def test_unsubscribe_removes_callback(self):
        """Test that unsubscribe removes the specified callback."""
        bus = EventBus()
        callback = lambda e: None
        sub_id = bus.subscribe("test_event", callback)
        assert len(bus._subscriptions["test_event"]) == 1

        bus.unsubscribe("test_event", sub_id)
        assert len(bus._subscriptions["test_event"]) == 0

    def test_unsubscribe_nonexistent_id_does_not_raise(self):
        """Test that unsubscribing with non-existent ID does not raise."""
        bus = EventBus()
        bus.subscribe("test_event", lambda e: None)
        bus.unsubscribe("test_event", "nonexistent_id")

    def test_unsubscribe_nonexistent_event_type_does_not_raise(self):
        """Test that unsubscribing from non-existent event type does not raise."""
        bus = EventBus()
        bus.unsubscribe("nonexistent_event", "some_id")


class TestEventBusPublish:
    """Test suite for EventBus publish functionality."""

    def test_publish_calls_sync_callback(self):
        """Test that publish invokes synchronous callbacks."""
        bus = EventBus()
        received = []

        def callback(event):
            received.append(event)

        bus.subscribe("test_event", callback)
        asyncio.run(bus.publish("test_event", {"data": "test"}))
        assert received == [{"data": "test"}]

    def test_publish_calls_async_callback(self):
        """Test that publish invokes asynchronous callbacks."""
        bus = EventBus()
        received = []

        async def callback(event):
            received.append(event)

        bus.subscribe("test_event", callback)
        asyncio.run(bus.publish("test_event", {"data": "async_test"}))
        assert received == [{"data": "async_test"}]

    def test_publish_calls_multiple_subscribers(self):
        """Test that publish calls all subscribed callbacks."""
        bus = EventBus()
        results = []

        def callback1(e):
            results.append(("cb1", e))

        def callback2(e):
            results.append(("cb2", e))

        bus.subscribe("test_event", callback1)
        bus.subscribe("test_event", callback2)
        asyncio.run(bus.publish("test_event", "shared_data"))

        assert len(results) == 2
        assert ("cb1", "shared_data") in results
        assert ("cb2", "shared_data") in results

    def test_publish_no_subscribers_does_not_raise(self):
        """Test that publishing to event with no subscribers does not raise."""
        bus = EventBus()
        asyncio.run(bus.publish("nonexistent_event", "data"))

    def test_publish_mixed_sync_async_callbacks(self):
        """Test publish handles both sync and async callbacks correctly."""
        bus = EventBus()
        sync_results = []
        async_results = []

        def sync_callback(e):
            sync_results.append(("sync", e))

        async def async_callback(e):
            async_results.append(("async", e))

        bus.subscribe("mixed_event", sync_callback)
        bus.subscribe("mixed_event", async_callback)
        asyncio.run(bus.publish("mixed_event", "mixed_data"))

        assert sync_results == [("sync", "mixed_data")]
        assert async_results == [("async", "mixed_data")]


class TestEventBusClear:
    """Test suite for EventBus clear functionality."""

    def test_clear_removes_all_subscriptions(self):
        """Test that clear removes all subscriptions."""
        bus = EventBus()
        bus.subscribe("event1", lambda e: None)
        bus.subscribe("event2", lambda e: None)
        bus.subscribe("event3", lambda e: None)

        assert len(bus._subscriptions) == 3
        bus.clear()
        assert len(bus._subscriptions) == 0

    def test_clear_on_empty_bus_does_not_raise(self):
        """Test that clear on empty bus does not raise."""
        bus = EventBus()
        bus.clear()


class TestEventBusEdgeCases:
    """Test suite for EventBus edge cases."""

    def test_publish_with_none_event(self):
        """Test publishing None as event data."""
        bus = EventBus()
        received = []

        def callback(e):
            received.append(e)

        bus.subscribe("nil_event", callback)
        asyncio.run(bus.publish("nil_event", None))
        assert received == [None]

    def test_publish_with_empty_string_event(self):
        """Test publishing empty string as event data."""
        bus = EventBus()
        received = []

        def callback(e):
            received.append(e)

        bus.subscribe("empty_event", callback)
        asyncio.run(bus.publish("empty_event", ""))
        assert received == [""]

    def test_publish_with_complex_object_event(self):
        """Test publishing complex object as event data."""
        bus = EventBus()
        received = []

        def callback(e):
            received.append(e)

        complex_obj = {"nested": {"data": [1, 2, 3]}, "string": "value"}
        bus.subscribe("complex_event", callback)
        asyncio.run(bus.publish("complex_event", complex_obj))
        assert received == [complex_obj]

    def test_multiple_subscribe_unsubscribe_cycles(self):
        """Test multiple subscribe/unsubscribe cycles.

        Each subscription is independent - same callback function
        registered twice with different IDs results in TWO invocations.
        """
        bus = EventBus()
        results = []

        def callback(e):
            results.append(e)

        id1 = bus.subscribe("cycle_event", callback)
        id2 = bus.subscribe("cycle_event", callback)

        # Both subscriptions fire (same function, but two subscriptions)
        asyncio.run(bus.publish("cycle_event", "first"))
        assert results == ["first", "first"]

        # After unsubscribing id1, only id2 remains
        bus.unsubscribe("cycle_event", id1)
        asyncio.run(bus.publish("cycle_event", "second"))
        assert results == ["first", "first", "second"]

        # After unsubscribing id2, no subscriptions remain
        bus.unsubscribe("cycle_event", id2)
        asyncio.run(bus.publish("cycle_event", "third"))
        # No new results since no subscribers
        assert results == ["first", "first", "second"]