"""Tests for ToolExecutor"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_framework.core.tool_executor import ToolExecutor


class TestToolExecutor:
    """测试 ToolExecutor 类"""

    def test_init(self):
        """测试初始化"""
        tools = {"calc": MagicMock()}
        executor = ToolExecutor(tools)
        assert executor.tools == tools

    def test_has_tool(self):
        """测试检查工具"""
        tools = {"calc": MagicMock()}
        executor = ToolExecutor(tools)

        assert executor.has_tool("calc") == True
        assert executor.has_tool("unknown") == False

    def test_get_tool_names(self):
        """测试获取工具名称"""
        tools = {"calc": MagicMock(), "search": MagicMock()}
        executor = ToolExecutor(tools)

        names = executor.get_tool_names()
        assert "calc" in names
        assert "search" in names

    def test_parse_arguments_none(self):
        """测试解析 None 参数"""
        executor = ToolExecutor({})
        assert executor._parse_arguments(None) == ""

    def test_parse_arguments_dict_with_input(self):
        """测试解析带 input 键的字典"""
        executor = ToolExecutor({})
        assert executor._parse_arguments({"input": "test"}) == "test"

    def test_parse_arguments_dict_without_input(self):
        """测试解析不带 input 键的字典"""
        executor = ToolExecutor({})
        result = executor._parse_arguments({"expression": "1+1"})
        assert result == "1+1"

    def test_parse_arguments_json_string(self):
        """测试解析 JSON 字符串"""
        executor = ToolExecutor({})
        assert executor._parse_arguments('{"input": "test"}') == "test"

    def test_parse_arguments_plain_string(self):
        """测试解析普通字符串"""
        executor = ToolExecutor({})
        assert executor._parse_arguments("test") == "test"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """测试执行异步工具"""
        tool = AsyncMock()
        tool.run = AsyncMock(return_value="result")
        executor = ToolExecutor({"tool": tool})

        result = await executor.execute("tool", {"input": "test"})
        assert result == "result"
        tool.run.assert_called_once_with("test", session_id=None)

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        """测试执行同步工具"""
        tool = MagicMock()
        tool.run = MagicMock(return_value="result")
        executor = ToolExecutor({"tool": tool})

        result = await executor.execute("tool", {"input": "test"})
        assert result == "result"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """测试执行未知工具"""
        executor = ToolExecutor({})

        result = await executor.execute("unknown", {"input": "test"})
        assert "Error" in result
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_execute_tool_error(self):
        """测试工具执行错误"""
        tool = AsyncMock()
        tool.run = AsyncMock(side_effect=Exception("test error"))
        executor = ToolExecutor({"tool": tool})

        result = await executor.execute("tool", {"input": "test"})
        assert "Error" in result
        assert "test error" in result
