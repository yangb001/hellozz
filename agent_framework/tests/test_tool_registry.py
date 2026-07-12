"""Tests for ToolRegistry - tool registration center."""

import pytest
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.tools.registry import ToolRegistry


class MockTool(BaseTool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", description: str = "A mock tool"):
        self.name = name
        self.description = description

    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        return f"mock result for: {input}"


class AnotherMockTool(BaseTool):
    """Another mock tool for testing."""

    def __init__(self):
        self.name = "another_tool"
        self.description = "Another mock tool"

    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        return "another result"


class TestToolRegistry:
    """Test cases for ToolRegistry."""

    def test_init_creates_empty_registry(self):
        """Registry should be empty on initialization."""
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_register_single_tool(self):
        """Should register a single tool successfully."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        assert "mock_tool" in registry.list_tools()

    def test_register_multiple_tools(self):
        """Should register multiple tools."""
        registry = ToolRegistry()
        tool1 = MockTool("tool1", "First tool")
        tool2 = MockTool("tool2", "Second tool")
        registry.register(tool1)
        registry.register(tool2)
        tools = registry.list_tools()
        assert "tool1" in tools
        assert "tool2" in tools
        assert len(tools) == 2

    def test_register_duplicate_name_raises_error(self):
        """Should raise ValueError when registering tool with duplicate name."""
        registry = ToolRegistry()
        tool1 = MockTool("same_name", "First tool")
        tool2 = MockTool("same_name", "Second tool")
        registry.register(tool1)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_unregister_existing_tool(self):
        """Should unregister an existing tool."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        registry.unregister("mock_tool")
        assert "mock_tool" not in registry.list_tools()

    def test_unregister_nonexistent_tool_raises_error(self):
        """Should raise KeyError when unregistering non-existent tool."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.unregister("nonexistent")

    def test_get_existing_tool(self):
        """Should return the tool instance for a registered name."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        retrieved = registry.get("mock_tool")
        assert retrieved is tool

    def test_get_nonexistent_tool_raises_error(self):
        """Should raise KeyError when getting non-existent tool."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_tools_returns_names(self):
        """Should return list of tool names."""
        registry = ToolRegistry()
        tool1 = MockTool("alpha", "Alpha tool")
        tool2 = MockTool("beta", "Beta tool")
        registry.register(tool1)
        registry.register(tool2)
        names = registry.list_tools()
        assert isinstance(names, list)
        assert set(names) == {"alpha", "beta"}

    def test_to_dict_returns_name_to_tool_mapping(self):
        """Should return dictionary mapping names to tool instances."""
        registry = ToolRegistry()
        tool1 = MockTool("tool_a", "Tool A")
        tool2 = MockTool("tool_b", "Tool B")
        registry.register(tool1)
        registry.register(tool2)
        d = registry.to_dict()
        assert isinstance(d, dict)
        assert d["tool_a"] is tool1
        assert d["tool_b"] is tool2

    def test_to_dict_empty_registry(self):
        """Should return empty dict for empty registry."""
        registry = ToolRegistry()
        assert registry.to_dict() == {}

    def test_unregister_then_register_same_name(self):
        """Should allow re-registering a name after unregistering."""
        registry = ToolRegistry()
        tool1 = MockTool("my_tool", "First version")
        registry.register(tool1)
        registry.unregister("my_tool")
        tool2 = MockTool("my_tool", "Second version")
        registry.register(tool2)
        assert registry.get("my_tool") is tool2

    def test_register_preserves_tool_instance(self):
        """Should store the exact tool instance, not a copy."""
        registry = ToolRegistry()
        tool = MockTool()
        registry.register(tool)
        assert registry.get("mock_tool") is tool
