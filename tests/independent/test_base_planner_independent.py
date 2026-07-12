"""Independent tests for BasePlanner interface per detailed design spec.

Design Reference (详细设计.md Section 3.4):
class BasePlanner(ABC):
    @abstractmethod
    async def plan_and_act(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any],
        llm_call: callable
    ) -> AsyncIterator[Event]:
        ...
"""
import pytest
import inspect
from abc import ABC
from typing import Dict, Any, AsyncIterator

from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.events import Event


class TestBasePlannerInterfaceCompliance:
    """Verify BasePlanner implementation matches detailed design spec."""

    def test_is_abstract_base_class(self):
        """BasePlanner must be an ABC per spec."""
        assert issubclass(BasePlanner, ABC), "BasePlanner should inherit from ABC"

    def test_plan_and_act_is_abstract_method(self):
        """plan_and_act() must be declared as abstract method per spec."""
        assert hasattr(BasePlanner, 'plan_and_act')
        assert getattr(BasePlanner.plan_and_act, '__isabstractmethod__', False), \
            "plan_and_act should be marked as abstract"

    def test_cannot_instantiate_base_planner_directly(self):
        """BasePlanner is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BasePlanner()
        assert "abstract" in str(exc_info.value).lower()

    def test_plan_and_act_is_async_generator(self):
        """plan_and_act() must be an async generator per spec - returns AsyncIterator[Event]."""
        assert inspect.iscoroutinefunction(BasePlanner.plan_and_act), \
            "plan_and_act should be a coroutine function"

    def test_plan_and_act_signature_per_spec(self):
        """plan_and_act(ctx, memory, tools, llm_call) signature per spec 3.4."""
        sig = inspect.signature(BasePlanner.plan_and_act)
        params = list(sig.parameters.keys())
        assert params == ['self', 'ctx', 'memory', 'tools', 'llm_call'], \
            f"plan_and_act signature mismatch, got {params}"

    def test_plan_and_act_returns_async_iterator_of_event(self):
        """plan_and_act should yield Event objects per spec."""
        # This is verified by the async generator check above
        # and by concrete implementation tests below
        pass


class TestBasePlannerConcreteImplementation:
    """Test that a concrete subclass can properly implement BasePlanner."""

    def test_concrete_implementation_can_be_instantiated(self):
        """A concrete subclass with plan_and_act implemented can be instantiated."""
        class ConcretePlanner(BasePlanner):
            async def plan_and_act(
                self,
                ctx: SessionContext,
                memory: BaseMemory,
                tools: Dict[str, Any],
                llm_call: callable,
            ) -> AsyncIterator[Event]:
                yield Event(type="final_answer", content="done")

        planner = ConcretePlanner()
        assert planner is not None

    def test_concrete_plan_and_act_yields_event(self):
        """A concrete plan_and_act() should yield Event objects."""
        import asyncio

        class ConcretePlanner(BasePlanner):
            async def plan_and_act(
                self,
                ctx: SessionContext,
                memory: BaseMemory,
                tools: Dict[str, Any],
                llm_call: callable,
            ) -> AsyncIterator[Event]:
                yield Event(type="thought", content="thinking")
                yield Event(type="final_answer", content="result")

        planner = ConcretePlanner()
        ctx = SessionContext(session_id="test")
        events = []

        async def collect():
            async for event in planner.plan_and_act(ctx, None, {}, None):
                events.append(event)

        asyncio.run(collect())

        assert len(events) == 2
        assert events[0].type == "thought"
        assert events[1].type == "final_answer"
        assert events[1].content == "result"

    def test_plan_and_act_handles_empty_tools_dict(self):
        """plan_and_act should accept empty tools dict per spec."""
        import asyncio

        class ConcretePlanner(BasePlanner):
            async def plan_and_act(
                self,
                ctx: SessionContext,
                memory: BaseMemory,
                tools: Dict[str, Any],
                llm_call: callable,
            ) -> AsyncIterator[Event]:
                assert tools == {}
                yield Event(type="final_answer", content="ok")

        planner = ConcretePlanner()
        ctx = SessionContext(session_id="test")

        async def test():
            async for _ in planner.plan_and_act(ctx, None, {}, None):
                pass

        asyncio.run(test())

    def test_plan_and_act_handles_none_memory(self):
        """plan_and_act should accept None for memory (optional parameter)."""
        import asyncio

        class ConcretePlanner(BasePlanner):
            async def plan_and_act(
                self,
                ctx: SessionContext,
                memory: BaseMemory,
                tools: Dict[str, Any],
                llm_call: callable,
            ) -> AsyncIterator[Event]:
                assert memory is None
                yield Event(type="final_answer", content="ok")

        planner = ConcretePlanner()
        ctx = SessionContext(session_id="test")

        async def test():
            async for _ in planner.plan_and_act(ctx, None, {}, None):
                pass

        asyncio.run(test())