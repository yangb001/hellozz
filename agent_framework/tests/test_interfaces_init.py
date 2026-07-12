"""Tests for interfaces/__init__.py unified exports."""
import pytest


class TestInterfacesUnifiedExports:
    """Test suite verifying all interfaces are properly exported."""

    def test_event_is_exported(self):
        """Test Event class is exported from interfaces."""
        from interfaces import Event
        assert Event is not None

    def test_session_status_is_exported(self):
        """Test SessionStatus enum is exported from interfaces."""
        from interfaces import SessionStatus
        assert SessionStatus is not None

    def test_event_type_is_exported(self):
        """Test EventType enum is exported from interfaces."""
        from interfaces import EventType
        assert EventType is not None

    def test_message_role_is_exported(self):
        """Test MessageRole enum is exported from interfaces."""
        from interfaces import MessageRole
        assert MessageRole is not None

    def test_message_is_exported(self):
        """Test Message class is exported from interfaces."""
        from interfaces import Message
        assert Message is not None

    def test_session_context_is_exported(self):
        """Test SessionContext class is exported from interfaces."""
        from interfaces import SessionContext
        assert SessionContext is not None

    def test_base_planner_is_exported(self):
        """Test BasePlanner class is exported from interfaces."""
        from interfaces import BasePlanner
        assert BasePlanner is not None

    def test_base_memory_is_exported(self):
        """Test BaseMemory class is exported from interfaces."""
        from interfaces import BaseMemory
        assert BaseMemory is not None

    def test_base_tool_is_exported(self):
        """Test BaseTool class is exported from interfaces."""
        from interfaces import BaseTool
        assert BaseTool is not None

    def test_all_exports_in_all(self):
        """Test that all items are in __all__."""
        from interfaces import __all__
        expected = [
            "Event",
            "SessionContext",
            "Message",
            "BaseMemory",
            "BasePlanner",
            "BaseTool",
            "SessionStatus",
            "EventType",
            "MessageRole",
        ]
        for item in expected:
            assert item in __all__, f"{item} should be in __all__"


class TestTypesModuleExports:
    """Test suite for interfaces/types.py module."""

    def test_types_module_exists(self):
        """Test types.py module can be imported."""
        from interfaces import types
        assert types is not None

    def test_session_id_exported(self):
        """Test SessionId type alias is exported."""
        from interfaces import SessionId
        assert SessionId is str

    def test_user_id_exported(self):
        """Test UserId type alias is exported."""
        from interfaces import UserId
        assert UserId is str

    def test_tool_name_exported(self):
        """Test ToolName type alias is exported."""
        from interfaces import ToolName
        assert ToolName is str

    def test_prompt_exported(self):
        """Test Prompt type alias is exported."""
        from interfaces import Prompt
        assert Prompt is str

    def test_response_exported(self):
        """Test Response type alias is exported."""
        from interfaces import Response

    def test_event_stream_exported(self):
        """Test EventStream type alias is exported."""
        from interfaces import EventStream

    def test_memory_result_exported(self):
        """Test MemoryResult type alias is exported."""
        from interfaces import MemoryResult
        assert MemoryResult is str

    def test_tool_result_exported(self):
        """Test ToolResult type alias is exported."""
        from interfaces import ToolResult
        assert ToolResult is str

    def test_types_in_all(self):
        """Test type aliases are in __all__."""
        from interfaces import __all__
        expected_types = [
            "SessionId",
            "UserId",
            "ToolName",
            "Prompt",
            "Response",
            "EventStream",
            "MemoryResult",
            "ToolResult",
        ]
        for item in expected_types:
            assert item in __all__, f"{item} should be in __all__"


class TestImportCompatibility:
    """Test suite for backward compatibility."""

    def test_import_from_submodule(self):
        """Test imports from submodules still work."""
        from interfaces.events import Event
        from interfaces.session import SessionContext
        assert Event is not None
        assert SessionContext is not None

    def test_import_from_package(self):
        """Test imports from interfaces package work."""
        from interfaces import Event, SessionContext, BasePlanner
        assert Event is not None
        assert SessionContext is not None
        assert BasePlanner is not None