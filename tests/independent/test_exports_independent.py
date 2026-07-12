"""Independent tests for interfaces module exports per detailed design spec.

Design Reference (详细设计.md):
- interfaces/__init__.py should export: Event, SessionContext, Message, BaseMemory, BasePlanner, BaseTool, SessionStatus, EventType, MessageRole, and types
- interfaces/types.py should define type aliases: SessionId, UserId, ToolName, Prompt, Response, EventStream, MemoryResult, ToolResult
"""
import pytest
from typing import AsyncIterator


class TestInterfacesUnifiedExport:
    """Verify all specified interfaces are exported from agent_framework.interfaces."""

    def test_event_exported(self):
        """Event class should be exported per spec."""
        from agent_framework.interfaces import Event
        assert Event is not None
        # Event should have type, content, metadata, timestamp fields
        assert hasattr(Event, 'model_fields') or hasattr(Event, '__fields__')

    def test_session_context_exported(self):
        """SessionContext class should be exported per spec."""
        from agent_framework.interfaces import SessionContext
        assert SessionContext is not None

    def test_message_exported(self):
        """Message class should be exported per spec."""
        from agent_framework.interfaces import Message
        assert Message is not None

    def test_base_memory_exported(self):
        """BaseMemory class should be exported per spec."""
        from agent_framework.interfaces import BaseMemory
        assert BaseMemory is not None

    def test_base_planner_exported(self):
        """BasePlanner class should be exported per spec."""
        from agent_framework.interfaces import BasePlanner
        assert BasePlanner is not None

    def test_base_tool_exported(self):
        """BaseTool class should be exported per spec."""
        from agent_framework.interfaces import BaseTool
        assert BaseTool is not None

    def test_session_status_exported(self):
        """SessionStatus enum should be exported per spec."""
        from agent_framework.interfaces import SessionStatus
        assert SessionStatus is not None
        assert hasattr(SessionStatus, 'ACTIVE') or hasattr(SessionStatus, 'values')

    def test_event_type_exported(self):
        """EventType enum should be exported per spec."""
        from agent_framework.interfaces import EventType
        assert EventType is not None

    def test_message_role_exported(self):
        """MessageRole enum should be exported per spec."""
        from agent_framework.interfaces import MessageRole
        assert MessageRole is not None


class TestTypesModuleExport:
    """Verify types module exports per spec."""

    def test_session_id_exported(self):
        """SessionId type alias should be str per spec."""
        from agent_framework.interfaces import SessionId
        assert SessionId is str

    def test_user_id_exported(self):
        """UserId type alias should be str per spec."""
        from agent_framework.interfaces import UserId
        assert UserId is str

    def test_tool_name_exported(self):
        """ToolName type alias should be str per spec."""
        from agent_framework.interfaces import ToolName
        assert ToolName is str

    def test_prompt_exported(self):
        """Prompt type alias should be str per spec."""
        from agent_framework.interfaces import Prompt
        assert Prompt is str

    def test_response_exported(self):
        """Response type alias should be Union[str, AsyncIterator[str]] per spec."""
        from agent_framework.interfaces import Response

    def test_event_stream_exported(self):
        """EventStream type alias should be AsyncIterator[Event] per spec."""
        from agent_framework.interfaces import EventStream

    def test_memory_result_exported(self):
        """MemoryResult type alias should be str per spec."""
        from agent_framework.interfaces import MemoryResult
        assert MemoryResult is str

    def test_tool_result_exported(self):
        """ToolResult type alias should be str per spec."""
        from agent_framework.interfaces import ToolResult
        assert ToolResult is str


class TestTypesModuleDirectImport:
    """Verify types can be imported directly from agent_framework.interfaces.types."""

    def test_session_id_from_types(self):
        """SessionId should be importable from types module."""
        from agent_framework.interfaces.types import SessionId
        assert SessionId is str

    def test_user_id_from_types(self):
        """UserId should be importable from types module."""
        from agent_framework.interfaces.types import UserId
        assert UserId is str

    def test_tool_name_from_types(self):
        """ToolName should be importable from types module."""
        from agent_framework.interfaces.types import ToolName
        assert ToolName is str

    def test_prompt_from_types(self):
        """Prompt should be importable from types module."""
        from agent_framework.interfaces.types import Prompt
        assert Prompt is str

    def test_memory_result_from_types(self):
        """MemoryResult should be importable from types module."""
        from agent_framework.interfaces.types import MemoryResult
        assert MemoryResult is str

    def test_tool_result_from_types(self):
        """ToolResult should be importable from types module."""
        from agent_framework.interfaces.types import ToolResult
        assert ToolResult is str


class TestAllExportList:
    """Verify __all__ contains all expected exports."""

    def test_all_list_exists(self):
        """__all__ should be defined in interfaces package."""
        from agent_framework.interfaces import __all__
        assert __all__ is not None

    def test_event_in_all(self):
        """Event should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'Event' in __all__

    def test_session_context_in_all(self):
        """SessionContext should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'SessionContext' in __all__

    def test_message_in_all(self):
        """Message should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'Message' in __all__

    def test_base_memory_in_all(self):
        """BaseMemory should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'BaseMemory' in __all__

    def test_base_planner_in_all(self):
        """BasePlanner should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'BasePlanner' in __all__

    def test_base_tool_in_all(self):
        """BaseTool should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'BaseTool' in __all__

    def test_session_status_in_all(self):
        """SessionStatus should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'SessionStatus' in __all__

    def test_event_type_in_all(self):
        """EventType should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'EventType' in __all__

    def test_message_role_in_all(self):
        """MessageRole should be in __all__."""
        from agent_framework.interfaces import __all__
        assert 'MessageRole' in __all__

    def test_type_aliases_in_all(self):
        """Type aliases should be in __all__ per spec."""
        from agent_framework.interfaces import __all__
        expected_types = [
            'SessionId', 'UserId', 'ToolName', 'Prompt',
            'Response', 'EventStream', 'MemoryResult', 'ToolResult'
        ]
        for t in expected_types:
            assert t in __all__, f"{t} should be in __all__"


class TestBackwardCompatibility:
    """Verify imports from submodules still work."""

    def test_event_from_events_submodule(self):
        """Event should still be importable from agent_framework.interfaces.events."""
        from agent_framework.interfaces.events import Event
        assert Event is not None

    def test_session_context_from_session_submodule(self):
        """SessionContext should still be importable from agent_framework.interfaces.session."""
        from agent_framework.interfaces.session import SessionContext
        assert SessionContext is not None

    def test_message_from_session_submodule(self):
        """Message should still be importable from agent_framework.interfaces.session."""
        from agent_framework.interfaces.session import Message
        assert Message is not None

    def test_base_memory_from_base_memory_submodule(self):
        """BaseMemory should still be importable from agent_framework.interfaces.base_memory."""
        from agent_framework.interfaces.base_memory import BaseMemory
        assert BaseMemory is not None

    def test_base_planner_from_base_planner_submodule(self):
        """BasePlanner should still be importable from agent_framework.interfaces.base_planner."""
        from agent_framework.interfaces.base_planner import BasePlanner
        assert BasePlanner is not None

    def test_base_tool_from_base_tool_submodule(self):
        """BaseTool should still be importable from agent_framework.interfaces.base_tool."""
        from agent_framework.interfaces.base_tool import BaseTool
        assert BaseTool is not None