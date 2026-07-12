"""Base planner interface - Defines the contract for planning strategies."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any

from .session import SessionContext
from .base_memory import BaseMemory
from .events import Event


class BasePlanner(ABC):
    """Abstract base class for planning strategies.

    A planner is responsible for deciding the next action(s) the agent
    should take based on the current context, memory, and available tools.
    """

    name: str = ""
    """Planner identifier."""

    description: str = ""
    """Human-readable description of the planner strategy."""

    @abstractmethod
    async def plan_and_act(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any],
        llm_call: callable,
    ) -> AsyncIterator[Event]:
        """Execute planning-action loop, yielding events.

        Args:
            ctx: Current session context with messages and state.
            memory: Memory system for retrieving relevant context.
            tools: Dictionary of available tools by name.
            llm_call: Async callable that takes a prompt and yields response tokens.

        Yields:
            Event objects representing thoughts, actions, observations, and final answer.
        """
        ...