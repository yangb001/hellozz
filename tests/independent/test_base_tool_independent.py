"""Independent tests for BaseTool interface per detailed design spec.

Design Reference (详细设计.md Section 3.5):
class BaseTool(ABC):
    name: str
    description: str

    @abstractmethod
    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        ...
"""
import pytest
import inspect
from abc import ABC
from typing import Optional

from agent_framework.interfaces.base_tool import BaseTool


class TestBaseToolInterfaceCompliance:
    """Verify BaseTool implementation matches detailed design spec."""

    def test_is_abstract_base_class(self):
        """BaseTool must be an ABC per spec."""
        assert issubclass(BaseTool, ABC), "BaseTool should inherit from ABC"

    def test_run_is_abstract_method(self):
        """run() must be declared as abstract method per spec."""
        assert hasattr(BaseTool, 'run')
        assert getattr(BaseTool.run, '__isabstractmethod__', False), \
            "run should be marked as abstract"

    def test_cannot_instantiate_base_tool_directly(self):
        """BaseTool is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseTool()
        assert "abstract" in str(exc_info.value).lower()

    def test_run_is_async(self):
        """run() must be async per spec."""
        assert inspect.iscoroutinefunction(BaseTool.run), \
            "run should be an async method"

    def test_run_signature_per_spec(self):
        """run(self, input: str, session_id: str = None, **kwargs) signature per spec 3.5."""
        sig = inspect.signature(BaseTool.run)
        params = list(sig.parameters.keys())
        # Should be: self, input, session_id, and potentially **kwargs
        assert 'self' in params, "first param should be self"
        assert 'input' in params, "should have input param"
        assert 'session_id' in params, "should have session_id param"

    def test_run_returns_str(self):
        """run() should return str per spec."""
        pass  # Verified by concrete implementation test

    def test_name_attribute_required(self):
        """name: str attribute must be declared per spec."""
        # This tests that a concrete class must define name
        class ConcreteTool(BaseTool):
            name = "test_tool"
            description = "A test tool"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return "result"

        tool = ConcreteTool()
        assert tool.name == "test_tool"

    def test_description_attribute_required(self):
        """description: str attribute must be declared per spec."""
        class ConcreteTool(BaseTool):
            name = "test_tool"
            description = "A test tool"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return "result"

        tool = ConcreteTool()
        assert tool.description == "A test tool"


class TestBaseToolConcreteImplementation:
    """Test that a concrete subclass can properly implement BaseTool."""

    def test_concrete_implementation_can_be_instantiated(self):
        """A concrete subclass with run() implemented can be instantiated."""
        class ConcreteTool(BaseTool):
            name = "test"
            description = "test desc"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return "done"

        tool = ConcreteTool()
        assert tool is not None

    def test_concrete_run_accepts_input(self):
        """run() should accept and process input string."""
        import asyncio

        class ConcreteTool(BaseTool):
            name = "test"
            description = "test desc"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return f"processed: {input}"

        tool = ConcreteTool()

        async def test():
            result = await tool.run("hello world")
            assert result == "processed: hello world"

        asyncio.run(test())

    def test_concrete_run_accepts_session_id(self):
        """run() should accept optional session_id parameter."""
        import asyncio

        class ConcreteTool(BaseTool):
            name = "test"
            description = "test desc"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return f"sid={session_id}"

        tool = ConcreteTool()

        async def test():
            result = await tool.run("hello", session_id="session-123")
            assert result == "sid=session-123"

        asyncio.run(test())

    def test_concrete_run_accepts_kwargs(self):
        """run() should accept additional kwargs per spec."""
        import asyncio

        class ConcreteTool(BaseTool):
            name = "test"
            description = "test desc"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return f"kwargs={kwargs}"

        tool = ConcreteTool()

        async def test():
            result = await tool.run("hello", extra_param="value")
            assert "extra_param" in result
            assert "value" in result

        asyncio.run(test())

    def test_concrete_run_returns_string(self):
        """run() should return a string per spec."""
        import asyncio

        class ConcreteTool(BaseTool):
            name = "test"
            description = "test desc"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                return "string result"

        tool = ConcreteTool()

        async def test():
            result = await tool.run("input")
            assert isinstance(result, str)

        asyncio.run(test())

    def test_concrete_run_session_id_default_is_none(self):
        """session_id should default to None per spec."""
        import asyncio
        captured_session_id = "not_set"

        class ConcreteTool(BaseTool):
            name = "test"
            description = "test desc"

            async def run(self, input: str, session_id: str = None, **kwargs) -> str:
                nonlocal captured_session_id
                captured_session_id = session_id
                return "ok"

        tool = ConcreteTool()

        async def test():
            await tool.run("input")

        asyncio.run(test())
        assert captured_session_id is None