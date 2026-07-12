"""Tests for interfaces/base_planner.py BasePlanner abstract class."""
import pytest
from abc import ABC
from typing import Dict, Any

from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.events import Event


class MockPlanner(BasePlanner):
    """Concrete implementation of BasePlanner for testing."""

    name = "mock_planner"
    description = "A mock planner for testing"

    async def plan_and_act(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any],
        llm_call: callable,
    ):
        """Yield a simple final answer event."""
        yield Event(type="final_answer", content="mock response")


class TestBasePlanner:
    """Tests for BasePlanner abstract class."""

    def test_is_abc(self):
        """Test that BasePlanner is an abstract base class."""
        assert issubclass(BasePlanner, ABC)

    def test_has_name_attribute(self):
        """Test that name attribute exists."""
        planner = MockPlanner()
        assert planner.name == "mock_planner"

    def test_has_description_attribute(self):
        """Test that description attribute exists."""
        planner = MockPlanner()
        assert planner.description == "A mock planner for testing"

    def test_plan_and_act_is_async_generator(self):
        """Test that plan_and_act is an async generator."""
        import inspect
        assert inspect.isasyncgenfunction(MockPlanner.plan_and_act)

    def test_plan_and_act_yields_events(self):
        """Test that plan_and_act yields Event objects using asyncio.run."""
        import asyncio
        planner = MockPlanner()
        ctx = SessionContext(session_id="test-session")
        memory = None
        tools = {}
        llm_call = None

        async def run_test():
            events = []
            async for event in planner.plan_and_act(ctx, memory, tools, llm_call):
                events.append(event)
            return events

        events = asyncio.run(run_test())
        assert len(events) == 1
        assert events[0].type == "final_answer"
        assert events[0].content == "mock response"


class TestBasePlannerSubclassRequirements:
    """Tests to verify subclass implementation requirements."""

    def test_subclass_must_implement_plan_and_act(self):
        """Test that subclasses must implement plan_and_act."""
        class IncompletePlanner(BasePlanner):
            name = "incomplete"
            description = "Missing plan_and_act"

        with pytest.raises(TypeError):
            IncompletePlanner()

    def test_subclass_can_inherit_name_attribute(self):
        """Test that subclasses can inherit name from base class."""
        class PlannerWithDefaultName(BasePlanner):
            description = "Has default name"

            async def plan_and_act(self, ctx, memory, tools, llm_call):
                yield Event(type="final_answer", content="test")

        planner = PlannerWithDefaultName()
        # Subclass inherits name from parent (empty string)
        assert planner.name == ""
        assert planner.description == "Has default name"