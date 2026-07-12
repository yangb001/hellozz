"""Independent tests for BaseMemory interface per detailed design spec.

Design Reference (详细设计.md Section 3.3):
- save(session_id: str, message: Message) -> None
- retrieve(session_id: str, query: str, user_ids: Optional[List[str]] = None, top_k: int = 5) -> str
- clear(session_id: str) -> None
- extract_long_term(session_id: str, force: bool = False) -> None
"""
import pytest
import inspect
from abc import ABC

from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.session import Message


class TestBaseMemoryInterfaceCompliance:
    """Verify BaseMemory implementation matches detailed design spec."""

    def test_is_abstract_base_class(self):
        """BaseMemory must be an ABC per spec."""
        assert issubclass(BaseMemory, ABC), "BaseMemory should inherit from ABC"

    def test_save_is_abstract_method(self):
        """save() must be declared as abstract method per spec."""
        assert hasattr(BaseMemory, 'save')
        assert getattr(BaseMemory.save, '__isabstractmethod__', False), \
            "save should be marked as abstract"

    def test_retrieve_is_abstract_method(self):
        """retrieve() must be declared as abstract method per spec."""
        assert hasattr(BaseMemory, 'retrieve')
        assert getattr(BaseMemory.retrieve, '__isabstractmethod__', False), \
            "retrieve should be marked as abstract"

    def test_clear_is_abstract_method(self):
        """clear() must be declared as abstract method per spec."""
        assert hasattr(BaseMemory, 'clear')
        assert getattr(BaseMemory.clear, '__isabstractmethod__', False), \
            "clear should be marked as abstract"

    def test_extract_long_term_is_abstract_method(self):
        """extract_long_term() must be declared as abstract method per spec."""
        assert hasattr(BaseMemory, 'extract_long_term')
        assert getattr(BaseMemory.extract_long_term, '__isabstractmethod__', False), \
            "extract_long_term should be marked as abstract"

    def test_cannot_instantiate_base_memory_directly(self):
        """BaseMemory is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseMemory()
        assert "abstract" in str(exc_info.value).lower()

    def test_save_method_signature_per_spec(self):
        """save(self, session_id: str, message: Message) signature per spec 3.3."""
        sig = inspect.signature(BaseMemory.save)
        params = list(sig.parameters.keys())
        assert params == ['self', 'session_id', 'message'], \
            f"save signature mismatch, got {params}"

    def test_retrieve_method_signature_per_spec(self):
        """retrieve signature per spec 3.3: retrieve(session_id, query, user_ids=None, top_k=5)."""
        sig = inspect.signature(BaseMemory.retrieve)
        params = list(sig.parameters.keys())
        defaults = {k: v.default for k, v in sig.parameters.items()}

        assert params == ['self', 'session_id', 'query', 'user_ids', 'top_k'], \
            f"retrieve signature mismatch, got {params}"
        assert defaults.get('user_ids') is None, "user_ids default should be None"
        assert defaults.get('top_k') == 5, "top_k default should be 5"

    def test_clear_method_signature_per_spec(self):
        """clear(self, session_id: str) signature per spec 3.3."""
        sig = inspect.signature(BaseMemory.clear)
        params = list(sig.parameters.keys())
        assert params == ['self', 'session_id'], \
            f"clear signature mismatch, got {params}"

    def test_extract_long_term_signature_per_spec(self):
        """extract_long_term(session_id: str, force: bool = False) per spec 3.3."""
        sig = inspect.signature(BaseMemory.extract_long_term)
        params = list(sig.parameters.keys())
        defaults = {k: v.default for k, v in sig.parameters.items()}

        assert params == ['self', 'session_id', 'force'], \
            f"extract_long_term signature mismatch, got {params}"
        assert defaults.get('force') is False, "force default should be False"

    def test_save_is_async(self):
        """save() must be async per spec."""
        assert inspect.iscoroutinefunction(BaseMemory.save), \
            "save should be an async method"

    def test_retrieve_is_async(self):
        """retrieve() must be async per spec."""
        assert inspect.iscoroutinefunction(BaseMemory.retrieve), \
            "retrieve should be an async method"

    def test_clear_is_async(self):
        """clear() must be async per spec."""
        assert inspect.iscoroutinefunction(BaseMemory.clear), \
            "clear should be an async method"

    def test_extract_long_term_is_async(self):
        """extract_long_term() must be async per spec."""
        assert inspect.iscoroutinefunction(BaseMemory.extract_long_term), \
            "extract_long_term should be an async method"


class TestBaseMemoryConcreteImplementation:
    """Test that a concrete subclass can properly implement BaseMemory."""

    def test_concrete_implementation_can_be_instantiated(self):
        """A concrete subclass with all methods implemented can be instantiated."""
        class ConcreteMemory(BaseMemory):
            async def save(self, session_id: str, message: Message) -> None:
                pass

            async def retrieve(
                self,
                session_id: str,
                query: str,
                user_ids=None,
                top_k=5
            ) -> str:
                return ""

            async def clear(self, session_id: str) -> None:
                pass

            async def extract_long_term(self, session_id: str, force=False) -> None:
                pass

        mem = ConcreteMemory()
        assert mem is not None

    def test_concrete_implementation_save_works(self):
        """A concrete save() can be called with proper arguments."""
        class ConcreteMemory(BaseMemory):
            async def save(self, session_id: str, message: Message) -> None:
                self.last_save = (session_id, message)

            async def retrieve(
                self,
                session_id: str,
                query: str,
                user_ids=None,
                top_k=5
            ) -> str:
                return ""

            async def clear(self, session_id: str) -> None:
                pass

            async def extract_long_term(self, session_id: str, force=False) -> None:
                pass

        import asyncio
        mem = ConcreteMemory()
        msg = Message(role="user", content="Hello")

        async def test():
            await mem.save("sid-123", msg)
            assert mem.last_save == ("sid-123", msg)

        asyncio.run(test())

    def test_concrete_implementation_retrieve_returns_string(self):
        """A concrete retrieve() should return a string per spec."""
        class ConcreteMemory(BaseMemory):
            async def save(self, session_id: str, message: Message) -> None:
                pass

            async def retrieve(
                self,
                session_id: str,
                query: str,
                user_ids=None,
                top_k=5
            ) -> str:
                return "relevant memories"

            async def clear(self, session_id: str) -> None:
                pass

            async def extract_long_term(self, session_id: str, force=False) -> None:
                pass

        import asyncio

        async def test():
            mem = ConcreteMemory()
            result = await mem.retrieve("sid", "query", top_k=3)
            assert isinstance(result, str)

        asyncio.run(test())