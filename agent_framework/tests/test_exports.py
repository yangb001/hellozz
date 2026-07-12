import pytest


class TestUnifiedExports:
    """Test suite for unified interface exports."""

    def test_import_event(self):
        """Test Event can be imported from interfaces."""
        from agent_framework.interfaces import Event
        assert Event is not None

    def test_import_session_context(self):
        """Test SessionContext can be imported from interfaces."""
        from agent_framework.interfaces import SessionContext
        assert SessionContext is not None

    def test_import_message(self):
        """Test Message can be imported from interfaces."""
        from agent_framework.interfaces import Message
        assert Message is not None

    def test_import_base_memory(self):
        """Test BaseMemory can be imported from interfaces."""
        from agent_framework.interfaces import BaseMemory
        assert BaseMemory is not None

    def test_import_base_planner(self):
        """Test BasePlanner can be imported from interfaces."""
        from agent_framework.interfaces import BasePlanner
        assert BasePlanner is not None

    def test_import_base_tool(self):
        """Test BaseTool can be imported from interfaces."""
        from agent_framework.interfaces import BaseTool
        assert BaseTool is not None

    def test_import_enums(self):
        """Test enum types can be imported from interfaces."""
        from agent_framework.interfaces import SessionStatus, EventType, MessageRole
        assert SessionStatus is not None
        assert EventType is not None
        assert MessageRole is not None

    def test_import_session_status_values(self):
        """Test SessionStatus enum values."""
        from agent_framework.interfaces import SessionStatus
        assert SessionStatus.ACTIVE.value == "active"
        assert SessionStatus.PAUSED.value == "paused"
        assert SessionStatus.CLOSED.value == "closed"

    def test_import_event_type_values(self):
        """Test EventType enum values."""
        from agent_framework.interfaces import EventType
        assert EventType.THOUGHT.value == "thought"
        assert EventType.ACTION.value == "action"
        assert EventType.OBSERVATION.value == "observation"
        assert EventType.TEXT_TOKEN.value == "text_token"
        assert EventType.FINAL_ANSWER.value == "final_answer"
        assert EventType.ERROR.value == "error"

    def test_import_message_role_values(self):
        """Test MessageRole enum values."""
        from agent_framework.interfaces import MessageRole
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.SYSTEM.value == "system"

    def test_star_import(self):
        """Test star import brings in all expected items."""
        from agent_framework.interfaces import (
            Event, SessionContext, Message,
            BaseMemory, BasePlanner, BaseTool,
            SessionStatus, EventType, MessageRole
        )
        assert Event is not None
        assert SessionContext is not None
        assert Message is not None
        assert BaseMemory is not None
        assert BasePlanner is not None
        assert BaseTool is not None


class TestTypesModule:
    """Test suite for types.py shared type aliases."""

    def test_import_types_module(self):
        """Test types module can be imported."""
        from agent_framework.interfaces import types
        assert types is not None

    def test_import_type_aliases(self):
        """Test type aliases can be imported from interfaces.types."""
        from agent_framework.interfaces.types import (
            SessionId, UserId, ToolName, Prompt, Response,
            EventStream, MemoryResult, ToolResult
        )
        assert SessionId is not None
        assert UserId is not None
        assert ToolName is not None
        assert Prompt is not None
        assert Response is not None
        assert EventStream is not None
        assert MemoryResult is not None
        assert ToolResult is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])