"""Tests for BaseMemory abstract class per specification."""
import pytest
from abc import ABC
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.session import Message


class TestBaseMemory:
    """Test suite for BaseMemory abstract class."""

    def test_base_memory_is_abstract_class(self):
        """Test that BaseMemory is an abstract base class."""
        assert issubclass(BaseMemory, ABC)

    def test_base_memory_has_save_method(self):
        """Test that BaseMemory declares save as abstract method."""
        assert hasattr(BaseMemory, 'save')
        assert getattr(BaseMemory.save, '__isabstractmethod__', False)

    def test_base_memory_has_retrieve_method(self):
        """Test that BaseMemory declares retrieve as abstract method."""
        assert hasattr(BaseMemory, 'retrieve')
        assert getattr(BaseMemory.retrieve, '__isabstractmethod__', False)

    def test_base_memory_has_clear_method(self):
        """Test that BaseMemory declares clear as abstract method."""
        assert hasattr(BaseMemory, 'clear')
        assert getattr(BaseMemory.clear, '__isabstractmethod__', False)

    def test_base_memory_has_extract_long_term_method(self):
        """Test that BaseMemory declares extract_long_term as abstract method."""
        assert hasattr(BaseMemory, 'extract_long_term')
        assert getattr(BaseMemory.extract_long_term, '__isabstractmethod__', False)

    def test_save_method_signature(self):
        """Test save method has correct signature: save(session_id, message)."""
        import inspect
        sig = inspect.signature(BaseMemory.save)
        params = list(sig.parameters.keys())
        assert params == ['self', 'session_id', 'message']

    def test_retrieve_method_signature(self):
        """Test retrieve method has correct signature: retrieve(session_id, query, user_ids=None, top_k=5)."""
        import inspect
        sig = inspect.signature(BaseMemory.retrieve)
        params = list(sig.parameters.keys())
        expected = ['self', 'session_id', 'query', 'user_ids', 'top_k']
        assert params == expected

    def test_clear_method_signature(self):
        """Test clear method has correct signature: clear(session_id)."""
        import inspect
        sig = inspect.signature(BaseMemory.clear)
        params = list(sig.parameters.keys())
        assert params == ['self', 'session_id']

    def test_extract_long_term_method_signature(self):
        """Test extract_long_term method has correct signature: extract_long_term(session_id, force=False)."""
        import inspect
        sig = inspect.signature(BaseMemory.extract_long_term)
        params = list(sig.parameters.keys())
        expected = ['self', 'session_id', 'force']
        assert params == expected

    def test_cannot_instantiate_base_memory_directly(self):
        """Test that BaseMemory cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseMemory()
        assert "abstract" in str(exc_info.value).lower()

    def test_message_type_can_be_used_with_save(self):
        """Test that Message type from interfaces.session works with save method."""
        from interfaces.session import Message
        msg = Message(role="user", content="Hello world")
        assert msg.role == "user"
        assert msg.content == "Hello world"
        assert msg.sender_id is None
