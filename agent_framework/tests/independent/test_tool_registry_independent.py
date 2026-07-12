"""Independent tests for ToolRegistry implementation.

验证内容（基于详细设计.md 第3.5节）：
- register/unregister 方法
- get/list_tools 方法
- 工具名称唯一性
- 边界条件

本测试文件完全独立编写，不使用开发者编写的测试用例。
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_framework.tools.registry import ToolRegistry
from agent_framework.interfaces.base_tool import BaseTool


# ─────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────

def _make_tool(name: str, description: str = "test tool") -> BaseTool:
    """创建模拟工具。"""
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = description
    tool.run = AsyncMock(return_value="result")
    return tool


# ─────────────────────────────────────────────────────────
# 1. 初始化
# ─────────────────────────────────────────────────────────

class TestToolRegistryInit:
    """验证 ToolRegistry 初始化。"""

    def test_initial_tools_empty(self):
        """初始化后工具列表应为空。"""
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_initial_dict_empty(self):
        """初始化后 to_dict 应返回空字典。"""
        registry = ToolRegistry()
        assert registry.to_dict() == {}


# ─────────────────────────────────────────────────────────
# 2. register 方法
# ─────────────────────────────────────────────────────────

class TestRegisterMethod:
    """验证 register 方法行为。"""

    def test_register_adds_tool(self):
        """register 应成功添加工具。"""
        registry = ToolRegistry()
        tool = _make_tool("search")

        registry.register(tool)

        assert "search" in registry.list_tools()

    def test_register_stores_tool_reference(self):
        """register 应保存工具实例引用。"""
        registry = ToolRegistry()
        tool = _make_tool("search")

        registry.register(tool)

        assert registry.get("search") is tool

    def test_register_multiple_tools(self):
        """应能注册多个不同名称的工具。"""
        registry = ToolRegistry()
        tool1 = _make_tool("search")
        tool2 = _make_tool("calculator")

        registry.register(tool1)
        registry.register(tool2)

        assert len(registry.list_tools()) == 2
        assert registry.get("search") is tool1
        assert registry.get("calculator") is tool2

    def test_register_duplicate_name_raises_value_error(self):
        """注册同名工具应抛出 ValueError。"""
        registry = ToolRegistry()
        tool1 = _make_tool("search")
        tool2 = _make_tool("search")

        registry.register(tool1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(tool2)

    def test_register_duplicate_error_message_contains_name(self):
        """重复注册错误消息应包含工具名称。"""
        registry = ToolRegistry()
        tool1 = _make_tool("my_tool")
        tool2 = _make_tool("my_tool")

        registry.register(tool1)

        with pytest.raises(ValueError, match="my_tool"):
            registry.register(tool2)


# ─────────────────────────────────────────────────────────
# 3. unregister 方法
# ─────────────────────────────────────────────────────────

class TestUnregisterMethod:
    """验证 unregister 方法行为。"""

    def test_unregister_removes_tool(self):
        """unregister 应成功移除工具。"""
        registry = ToolRegistry()
        tool = _make_tool("search")
        registry.register(tool)

        registry.unregister("search")

        assert "search" not in registry.list_tools()

    def test_unregister_nonexistent_raises_key_error(self):
        """注销不存在的工具应抛出 KeyError。"""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="nonexistent"):
            registry.unregister("nonexistent")

    def test_unregister_error_message_contains_name(self):
        """注销不存在工具的错误消息应包含工具名称。"""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="my_tool"):
            registry.unregister("my_tool")

    def test_unregister_allows_re_registration(self):
        """注销后应能重新注册同名工具。"""
        registry = ToolRegistry()
        tool1 = _make_tool("search")
        tool2 = _make_tool("search")

        registry.register(tool1)
        registry.unregister("search")
        registry.register(tool2)

        assert registry.get("search") is tool2

    def test_unregister_only_removes_specified_tool(self):
        """unregister 应只移除指定工具，不影响其他工具。"""
        registry = ToolRegistry()
        tool1 = _make_tool("search")
        tool2 = _make_tool("calculator")
        registry.register(tool1)
        registry.register(tool2)

        registry.unregister("search")

        assert "search" not in registry.list_tools()
        assert "calculator" in registry.list_tools()
        assert registry.get("calculator") is tool2


# ─────────────────────────────────────────────────────────
# 4. get 方法
# ─────────────────────────────────────────────────────────

class TestGetMethod:
    """验证 get 方法行为。"""

    def test_get_returns_registered_tool(self):
        """get 应返回已注册的工具。"""
        registry = ToolRegistry()
        tool = _make_tool("search")
        registry.register(tool)

        result = registry.get("search")

        assert result is tool

    def test_get_nonexistent_raises_key_error(self):
        """获取不存在的工具应抛出 KeyError。"""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="nonexistent"):
            registry.get("nonexistent")

    def test_get_error_message_contains_name(self):
        """获取不存在工具的错误消息应包含工具名称。"""
        registry = ToolRegistry()

        with pytest.raises(KeyError, match="my_tool"):
            registry.get("my_tool")

    def test_get_returns_correct_tool_among_many(self):
        """多个工具时应返回正确的工具。"""
        registry = ToolRegistry()
        tools = {name: _make_tool(name) for name in ["a", "b", "c"]}
        for tool in tools.values():
            registry.register(tool)

        for name, expected_tool in tools.items():
            assert registry.get(name) is expected_tool


# ─────────────────────────────────────────────────────────
# 5. list_tools 方法
# ─────────────────────────────────────────────────────────

class TestListToolsMethod:
    """验证 list_tools 方法行为。"""

    def test_list_tools_returns_empty_list(self):
        """无注册工具时应返回空列表。"""
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_list_tools_returns_all_names(self):
        """应返回所有已注册工具的名称。"""
        registry = ToolRegistry()
        registry.register(_make_tool("search"))
        registry.register(_make_tool("calculator"))

        names = registry.list_tools()

        assert len(names) == 2
        assert "search" in names
        assert "calculator" in names

    def test_list_tools_returns_list_type(self):
        """list_tools 应返回列表类型。"""
        registry = ToolRegistry()
        registry.register(_make_tool("search"))

        result = registry.list_tools()

        assert isinstance(result, list)

    def test_list_tools_reflects_current_state(self):
        """list_tools 应反映当前注册状态。"""
        registry = ToolRegistry()
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))

        assert len(registry.list_tools()) == 2

        registry.unregister("a")

        assert len(registry.list_tools()) == 1
        assert "b" in registry.list_tools()


# ─────────────────────────────────────────────────────────
# 6. to_dict 方法
# ─────────────────────────────────────────────────────────

class TestToDictMethod:
    """验证 to_dict 方法行为。"""

    def test_to_dict_returns_empty_dict(self):
        """无注册工具时应返回空字典。"""
        registry = ToolRegistry()
        assert registry.to_dict() == {}

    def test_to_dict_contains_all_tools(self):
        """应包含所有已注册工具。"""
        registry = ToolRegistry()
        tool1 = _make_tool("search")
        tool2 = _make_tool("calculator")
        registry.register(tool1)
        registry.register(tool2)

        result = registry.to_dict()

        assert len(result) == 2
        assert result["search"] is tool1
        assert result["calculator"] is tool2

    def test_to_dict_returns_dict_type(self):
        """to_dict 应返回字典类型。"""
        registry = ToolRegistry()
        registry.register(_make_tool("search"))

        result = registry.to_dict()

        assert isinstance(result, dict)

    def test_to_dict_is_copy(self):
        """to_dict 应返回副本，修改不影响内部状态。"""
        registry = ToolRegistry()
        registry.register(_make_tool("search"))

        result = registry.to_dict()
        result["new_key"] = _make_tool("new")

        assert "new_key" not in registry.list_tools()
        assert len(registry.list_tools()) == 1


# ─────────────────────────────────────────────────────────
# 7. 边界条件
# ─────────────────────────────────────────────────────────

class TestBoundaryConditions:
    """验证边界条件处理。"""

    def test_tool_with_empty_name(self):
        """应能注册名称为空字符串的工具。"""
        registry = ToolRegistry()
        tool = _make_tool("")

        registry.register(tool)

        assert registry.get("") is tool

    def test_tool_with_special_characters_name(self):
        """应能注册名称含特殊字符的工具。"""
        registry = ToolRegistry()
        tool = _make_tool("my-tool_v2.0")

        registry.register(tool)

        assert registry.get("my-tool_v2.0") is tool

    def test_tool_with_unicode_name(self):
        """应能注册 Unicode 名称的工具。"""
        registry = ToolRegistry()
        tool = _make_tool("搜索工具")

        registry.register(tool)

        assert registry.get("搜索工具") is tool

    def test_register_then_unregister_cycle(self):
        """应支持多次注册/注销循环。"""
        registry = ToolRegistry()

        for i in range(5):
            tool = _make_tool("tool")
            registry.register(tool)
            assert registry.get("tool") is tool
            registry.unregister("tool")
            assert "tool" not in registry.list_tools()

    def test_single_tool_operations(self):
        """单个工具的完整操作流程。"""
        registry = ToolRegistry()
        tool = _make_tool("search", "Search the web")

        # 注册
        registry.register(tool)
        assert "search" in registry.list_tools()
        assert registry.get("search") is tool
        assert registry.to_dict()["search"] is tool

        # 注销
        registry.unregister("search")
        assert "search" not in registry.list_tools()
        assert registry.to_dict() == {}

    def test_many_tools(self):
        """应能注册大量工具。"""
        registry = ToolRegistry()
        tools = [_make_tool(f"tool_{i}") for i in range(100)]

        for tool in tools:
            registry.register(tool)

        assert len(registry.list_tools()) == 100
        for i in range(100):
            assert registry.get(f"tool_{i}") is tools[i]
