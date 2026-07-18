"""Base planner interface - Defines the contract for planning strategies."""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any

from .events import Event


class BasePlanner(ABC):
    """Abstract base class for planning strategies.

    A planner is responsible for deciding the next action(s) the agent
    should take based on the current context, memory, and available tools.

    After refactoring (Phase 2A), the planner uses PlannerContext to bundle
    all state (session_id, tools, memory, messages) into a single context object,
    reducing the number of parameters and eliminating side effects.
    """

    name: str = ""
    """Planner identifier."""

    description: str = ""
    """Human-readable description of the planner strategy."""

    @abstractmethod
    async def plan_and_act(
        self,
        ctx: Any,
        llm_call: callable,
    ) -> AsyncIterator[Event]:
        """Execute planning-action loop, yielding events.

        Args:
            ctx: PlannerContext containing all planner state (session_id, tools,
                 memory, messages, iteration tracking, etc.).
            llm_call: Async callable that takes (messages, tools) and yields events.

        Yields:
            Event objects representing thoughts, actions, observations, and final answer.
        """
        ...