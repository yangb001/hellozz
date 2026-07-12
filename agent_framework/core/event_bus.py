"""EventBus - Pub/Sub pattern implementation for inter-component communication."""
import asyncio
from typing import Dict, Callable, Any, Optional
from collections import defaultdict
import uuid


class EventBus:
    """Central event bus for publish/subscribe communication.

    Allows components to subscribe to specific event types and be notified
    when those events are published.
    """

    def __init__(self):
        """Initialize the EventBus with empty subscriptions."""
        self._subscriptions: Dict[str, Dict[str, Callable]] = defaultdict(dict)

    def subscribe(self, event_type: str, callback: Callable) -> str:
        """Subscribe to events of a specific type.

        Args:
            event_type: The type of event to subscribe to.
            callback: Callable that will be invoked when the event is published.

        Returns:
            A subscription ID that can be used to unsubscribe.

        Raises:
            TypeError: If callback is not callable.
        """
        if not callable(callback):
            raise TypeError("callback must be callable")

        subscription_id = str(uuid.uuid4())
        self._subscriptions[event_type][subscription_id] = callback
        return subscription_id

    def unsubscribe(self, event_type: str, subscription_id: str) -> None:
        """Unsubscribe from events of a specific type.

        Args:
            event_type: The type of event to unsubscribe from.
            subscription_id: The subscription ID returned from subscribe().
        """
        if event_type in self._subscriptions:
            self._subscriptions[event_type].pop(subscription_id, None)

    async def publish(self, event_type: str, event: Any) -> None:
        """Publish an event to all subscribers of that type.

        Args:
            event_type: The type of event being published.
            event: The event data to pass to subscribers.
        """
        if event_type not in self._subscriptions:
            return

        for callback in list(self._subscriptions[event_type].values()):
            if asyncio.iscoroutinefunction(callback):
                await callback(event)
            else:
                callback(event)

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._subscriptions.clear()