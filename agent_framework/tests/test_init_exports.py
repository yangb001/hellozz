"""Tests for unified exports from interfaces module."""
import pytest
from agent_framework.interfaces.events import Event
from agent_framework.interfaces.session import SessionContext
from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.interfaces.enums import SessionStatus, EventType, MessageRole


class TestInterfacesUnifiedExports:
    """Test suite verifying all interfaces are properly exported."""

    def test_event_is_exported(self):
        """Test Event class is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import Event as ExportedEvent
        assert ExportedEvent is Event

    def test_session_status_is_exported(self):
        """Test SessionStatus enum is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import SessionStatus as ExportedSessionStatus
        assert ExportedSessionStatus is SessionStatus

    def test_event_type_is_exported(self):
        """Test EventType enum is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import EventType as ExportedEventType
        assert ExportedEventType is EventType

    def test_message_role_is_exported(self):
        """Test MessageRole enum is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import MessageRole as ExportedMessageRole
        assert ExportedMessageRole is MessageRole

    def test_message_is_exported(self):
        """Test Message class is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import Message
        assert Message is not None

    def test_session_context_is_exported(self):
        """Test SessionContext class is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import SessionContext as ExportedSessionContext
        assert ExportedSessionContext is SessionContext

    def test_base_planner_is_exported(self):
        """Test BasePlanner class is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import BasePlanner as ExportedBasePlanner
        assert ExportedBasePlanner is BasePlanner

    def test_base_memory_is_exported(self):
        """Test BaseMemory class is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import BaseMemory
        assert BaseMemory is not None

    def test_base_tool_is_exported(self):
        """Test BaseTool class is exported from agent_framework.interfaces."""
        from agent_framework.interfaces import BaseTool
        assert BaseTool is not None

    def test_all_exports_in_whitelist(self):
        """Test that all items in __all__ are properly exported."""
        from agent_framework.interfaces import __all__
        expected_exports = {
            "Event",
            "SessionStatus",
            "EventType",
            "MessageRole",
            "Message",
            "SessionContext",
            "BasePlanner",
            "BaseMemory",
            "BaseTool",
        }
        for item in expected_exports:
            assert item in __all__, f"{item} should be in __all__"

    def test_exported_event_is_same_as_source(self):
        """Test exported Event is identical to the source class."""
        from agent_framework.interfaces import Event
        from agent_framework.interfaces.events import Event as SourceEvent
        assert Event is SourceEvent

    def test_exported_session_context_is_same_as_source(self):
        """Test exported SessionContext is identical to the source class."""
        from agent_framework.interfaces import SessionContext
        from agent_framework.interfaces.session import SessionContext as SourceSessionContext
        assert SessionContext is SourceSessionContext

    def test_exported_base_planner_is_same_as_source(self):
        """Test exported BasePlanner is identical to the source class."""
        from agent_framework.interfaces import BasePlanner
        from agent_framework.interfaces.base_planner import BasePlanner as SourceBasePlanner
        assert BasePlanner is SourceBasePlanner


class TestTypesModuleExports:
    """Test suite for types.py module exports."""

    def test_types_module_exists(self):
        """Test that types.py module exists and can be imported."""
        from agent_framework.interfaces import types
        assert types is not None

    def test_session_id_type_alias_exists(self):
        """Test SessionId type alias is exported."""
        from agent_framework.interfaces.types import SessionId
        assert SessionId is str

    def test_user_id_type_alias_exists(self):
        """Test UserId type alias is exported."""
        from agent_framework.interfaces.types import UserId
        assert UserId is str

    def test_tool_name_type_alias_exists(self):
        """Test ToolName type alias is exported."""
        from agent_framework.interfaces.types import ToolName
        assert ToolName is str

    def test_prompt_type_alias_exists(self):
        """Test Prompt type alias is exported."""
        from agent_framework.interfaces.types import Prompt
        assert Prompt is str

    def test_response_type_alias_exists(self):
        """Test Response type alias is exported."""
        from agent_framework.interfaces.types import Response

    def test_event_stream_type_alias_exists(self):
        """Test EventStream type alias is exported."""
        from agent_framework.interfaces.types import EventStream

    def test_memory_result_type_alias_exists(self):
        """Test MemoryResult type alias is exported."""
        from agent_framework.interfaces.types import MemoryResult
        assert MemoryResult is str

    def test_tool_result_type_alias_exists(self):
        """Test ToolResult type alias is exported."""
        from agent_framework.interfaces.types import ToolResult
        assert ToolResult is str

    def test_types_exported_from_interfaces_package(self):
        """Test type aliases can be imported from interfaces package."""
        from agent_framework.interfaces import SessionId, UserId, ToolName, Prompt
        assert SessionId is str
        assert UserId is str
        assert ToolName is str
        assert Prompt is str


class TestInterfacesImportCompatibility:
    """Test suite verifying imports work correctly."""

    def test_can_import_from_interfaces_submodule(self):
        """Test that imports from submodules still work."""
        from agent_framework.interfaces.events import Event
        from agent_framework.interfaces.session import SessionContext
        from agent_framework.interfaces.base_planner import BasePlanner
        assert Event is not None
        assert SessionContext is not None
        assert BasePlanner is not None

    def test_can_import_from_top_level_interfaces(self):
        """Test that imports from top-level interfaces package work."""
        from agent_framework.interfaces import Event, SessionContext, BasePlanner
        assert Event is not None
        assert SessionContext is not None
        assert BasePlanner is not None