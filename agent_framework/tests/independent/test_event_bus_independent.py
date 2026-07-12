"""Independent test cases for EventBus pub/sub implementation.

This module contains independent verification tests for the EventBus
class defined in core/event_bus.py.

Test categories:
1. EventBus initialization
2. subscribe and unsubscribe
3. publish with sync and async callbacks
4. clear operations
5. Multiple subscribers and event types
6. Boundary conditions
"""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

from agent_framework.core.event_bus import EventBus


# ============================================================================
# 1. Initialization
# ============================================================================


class TestEventBusInit:
    """Test EventBus initialization."""

    def test_event_bus_can_instantiate(self):
        """EventBus can be instantiated."""
        bus = EventBus()
        assert bus is not None

    def test_subscriptions_is_dict(self):
        """_subscriptions must be a dict-like structure."""
        bus = EventBus()
        assert hasattr(bus, "_subscriptions")

    def test_initially_empty(self):
        """Subscriptions should be empty initially."""
        bus = EventBus()
        assert len(bus._subscriptions) == 0


# ============================================================================
# 2. Subscribe and Unsubscribe
# ============================================================================


class TestSubscribeUnsubscribe:
    """Test subscribe and unsubscribe operations."""

    def test_subscribe_returns_id(self):
        """subscribe must return a subscription ID string."""
        bus = EventBus()
        sub_id = bus.subscribe("test_event", lambda e: None)
        assert isinstance(sub_id, str)
        assert len(sub_id) > 0

    def test_subscribe_different_ids(self):
        """Each subscribe call returns a unique ID."""
        bus = EventBus()
        id1 = bus.subscribe("test_event", lambda e: None)
        id2 = bus.subscribe("test_event", lambda e: None)
        assert id1 != id2

    def test_subscribe_non_callable_raises(self):
        """subscribe raises TypeError for non-callable callback."""
        bus = EventBus()
        with pytest.raises(TypeError):
            bus.subscribe("test_event", "not_callable")

    def test_subscribe_registers_callback(self):
        """subscribe registers the callback for the event type."""
        bus = EventBus()
        bus.subscribe("my_event", lambda e: None)
        assert "my_event" in bus._subscriptions
        assert len(bus._subscriptions["my_event"]) == 1

    def test_subscribe_multiple_same_event(self):
        """Multiple callbacks can subscribe to the same event type."""
        bus = EventBus()
        bus.subscribe("my_event", lambda e: None)
        bus.subscribe("my_event", lambda e: None)
        assert len(bus._subscriptions["my_event"]) == 2

    def test_subscribe_different_events(self):
        """Callbacks can subscribe to different event types."""
        bus = EventBus()
        bus.subscribe("event_a", lambda e: None)
        bus.subscribe("event_b", lambda e: None)
        assert "event_a" in bus._subscriptions
        assert "event_b" in bus._subscriptions

    def test_unsubscribe_removes_callback(self):
        """unsubscribe removes the specific callback."""
        bus = EventBus()
        sub_id = bus.subscribe("my_event", lambda e: None)
        bus.unsubscribe("my_event", sub_id)
        assert len(bus._subscriptions.get("my_event", {})) == 0

    def test_unsubscribe_nonexistent_id_no_error(self):
        """unsubscribe with unknown ID does not raise."""
        bus = EventBus()
        bus.unsubscribe("my_event", "nonexistent_id")  # Should not raise

    def test_unsubscribe_nonexistent_event_no_error(self):
        """unsubscribe for unknown event type does not raise."""
        bus = EventBus()
        bus.unsubscribe("nonexistent", "some_id")  # Should not raise

    def test_unsubscribe_one_keeps_others(self):
        """Unsubscribing one callback keeps others on same event."""
        bus = EventBus()
        id1 = bus.subscribe("my_event", lambda e: None)
        id2 = bus.subscribe("my_event", lambda e: None)
        bus.unsubscribe("my_event", id1)
        assert len(bus._subscriptions["my_event"]) == 1


# ============================================================================
# 3. Publish with Sync Callbacks
# ============================================================================


class TestPublishSync:
    """Test publish with synchronous callbacks."""

    @pytest.mark.asyncio
    async def test_publish_calls_sync_callback(self):
        """publish invokes sync callback with event data."""
        bus = EventBus()
        received = []
        bus.subscribe("my_event", lambda e: received.append(e))
        await bus.publish("my_event", "test_data")
        assert received == ["test_data"]

    @pytest.mark.asyncio
    async def test_publish_multiple_sync_callbacks(self):
        """publish invokes all sync callbacks."""
        bus = EventBus()
        received = []
        bus.subscribe("my_event", lambda e: received.append(("a", e)))
        bus.subscribe("my_event", lambda e: received.append(("b", e)))
        await bus.publish("my_event", "data")
        assert len(received) == 2
        assert ("a", "data") in received
        assert ("b", "data") in received

    @pytest.mark.asyncio
    async def test_publish_no_subscribers(self):
        """publish to event with no subscribers does not raise."""
        bus = EventBus()
        await bus.publish("no_such_event", "data")  # Should not raise

    @pytest.mark.asyncio
    async def test_publish_different_event_types(self):
        """publish only notifies subscribers of matching event type."""
        bus = EventBus()
        received_a = []
        received_b = []
        bus.subscribe("event_a", lambda e: received_a.append(e))
        bus.subscribe("event_b", lambda e: received_b.append(e))
        await bus.publish("event_a", "data_a")
        assert received_a == ["data_a"]
        assert received_b == []


# ============================================================================
# 4. Publish with Async Callbacks
# ============================================================================


class TestPublishAsync:
    """Test publish with asynchronous callbacks."""

    @pytest.mark.asyncio
    async def test_publish_calls_async_callback(self):
        """publish invokes async callback with event data."""
        bus = EventBus()
        received = []

        async def handler(e):
            received.append(e)

        bus.subscribe("my_event", handler)
        await bus.publish("my_event", "async_data")
        assert received == ["async_data"]

    @pytest.mark.asyncio
    async def test_publish_mixed_sync_async(self):
        """publish handles mix of sync and async callbacks."""
        bus = EventBus()
        received = []

        def sync_handler(e):
            received.append(("sync", e))

        async def async_handler(e):
            received.append(("async", e))

        bus.subscribe("my_event", sync_handler)
        bus.subscribe("my_event", async_handler)
        await bus.publish("my_event", "data")
        assert len(received) == 2
        assert ("sync", "data") in received
        assert ("async", "data") in received


# ============================================================================
# 5. Clear Operations
# ============================================================================


class TestClear:
    """Test clear operation."""

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        """clear removes all subscriptions."""
        bus = EventBus()
        bus.subscribe("event_a", lambda e: None)
        bus.subscribe("event_b", lambda e: None)
        bus.clear()
        assert len(bus._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_clear_then_publish_no_subscribers(self):
        """After clear, publish has no subscribers to notify."""
        bus = EventBus()
        received = []
        bus.subscribe("my_event", lambda e: received.append(e))
        bus.clear()
        await bus.publish("my_event", "data")
        assert received == []

    @pytest.mark.asyncio
    async def test_clear_idempotent(self):
        """clear can be called multiple times without error."""
        bus = EventBus()
        bus.subscribe("my_event", lambda e: None)
        bus.clear()
        bus.clear()
        assert len(bus._subscriptions) == 0

    @pytest.mark.asyncio
    async def test_subscribe_after_clear(self):
        """subscribe works normally after clear."""
        bus = EventBus()
        bus.subscribe("my_event", lambda e: None)
        bus.clear()
        received = []
        bus.subscribe("my_event", lambda e: received.append(e))
        await bus.publish("my_event", "data")
        assert received == ["data"]


# ============================================================================
# 6. Boundary Conditions
# ============================================================================


class TestBoundaryConditions:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_publish_none_event(self):
        """publish handles None as event data."""
        bus = EventBus()
        received = []
        bus.subscribe("my_event", lambda e: received.append(e))
        await bus.publish("my_event", None)
        assert received == [None]

    @pytest.mark.asyncio
    async def test_publish_dict_event(self):
        """publish handles dict as event data."""
        bus = EventBus()
        received = []
        bus.subscribe("my_event", lambda e: received.append(e))
        data = {"type": "thought", "content": "thinking"}
        await bus.publish("my_event", data)
        assert received == [data]

    @pytest.mark.asyncio
    async def test_many_subscribers(self):
        """EventBus handles many subscribers."""
        bus = EventBus()
        count = 100
        received = [0]

        for _ in range(count):
            bus.subscribe("my_event", lambda e: received.__setitem__(0, received[0] + 1))

        await bus.publish("my_event", "data")
        assert received[0] == count

    @pytest.mark.asyncio
    async def test_many_event_types(self):
        """EventBus handles many event types."""
        bus = EventBus()
        for i in range(100):
            bus.subscribe(f"event_{i}", lambda e: None)
        assert len(bus._subscriptions) == 100

    @pytest.mark.asyncio
    async def test_same_callback_multiple_events(self):
        """Same callback can subscribe to multiple events."""
        bus = EventBus()
        received = []
        callback = lambda e: received.append(e)
        bus.subscribe("event_a", callback)
        bus.subscribe("event_b", callback)
        await bus.publish("event_a", "a")
        await bus.publish("event_b", "b")
        assert received == ["a", "b"]

    @pytest.mark.asyncio
    async def test_callback_exception_propagates(self):
        """Exception in callback should propagate from publish."""
        bus = EventBus()

        def bad_callback(e):
            raise ValueError("test error")

        bus.subscribe("my_event", bad_callback)
        with pytest.raises(ValueError, match="test error"):
            await bus.publish("my_event", "data")

    def test_subscribe_with_lambda(self):
        """subscribe accepts lambda callbacks."""
        bus = EventBus()
        sub_id = bus.subscribe("my_event", lambda e: None)
        assert isinstance(sub_id, str)

    def test_subscribe_with_function(self):
        """subscribe accepts named functions."""
        bus = EventBus()

        def handler(e):
            pass

        sub_id = bus.subscribe("my_event", handler)
        assert isinstance(sub_id, str)

    def test_subscribe_with_bound_method(self):
        """subscribe accepts bound methods."""
        bus = EventBus()

        class Handler:
            def handle(self, e):
                pass

        handler = Handler()
        sub_id = bus.subscribe("my_event", handler.handle)
        assert isinstance(sub_id, str)
