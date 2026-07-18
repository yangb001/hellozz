"""Tests for PlannerContext"""
import pytest
from unittest.mock import AsyncMock, MagicMock
from agent_framework.core.planner_context import PlannerContext


class TestPlannerContext:
    """测试 PlannerContext 类"""

    def test_init(self):
        """测试初始化"""
        ctx = PlannerContext(session_id="test")
        assert ctx.session_id == "test"
        assert ctx.iteration == 0
        assert ctx.max_iterations == 10
        assert ctx.tool_executor is not None

    def test_increment_iteration(self):
        """测试增加迭代次数"""
        ctx = PlannerContext(session_id="test")

        assert ctx.increment_iteration() == 1
        assert ctx.increment_iteration() == 2
        assert ctx.iteration == 2

    def test_is_max_iterations_reached(self):
        """测试检查最大迭代次数"""
        ctx = PlannerContext(session_id="test", max_iterations=3)

        assert ctx.is_max_iterations_reached() == False

        ctx.increment_iteration()
        ctx.increment_iteration()
        ctx.increment_iteration()

        assert ctx.is_max_iterations_reached() == True

    def test_start_tool_call(self):
        """测试开始工具调用"""
        ctx = PlannerContext(session_id="test")

        ctx.start_tool_call({"tool_name": "calc", "arguments": ""})
        assert ctx.has_pending_tool_call() == True
        assert ctx.pending_tool_call["tool_name"] == "calc"

    def test_update_tool_args(self):
        """测试更新工具参数"""
        ctx = PlannerContext(session_id="test")

        ctx.start_tool_call({"tool_name": "calc", "arguments": ""})
        ctx.update_tool_args('{"input":')
        ctx.update_tool_args('"1+1"}')

        assert ctx.accumulated_args == '{"input":"1+1"}'
        assert ctx.pending_tool_call["arguments"] == '{"input":"1+1"}'

    def test_complete_tool_call(self):
        """测试完成工具调用"""
        ctx = PlannerContext(session_id="test")

        ctx.start_tool_call({"tool_name": "calc", "arguments": '{"input": "1+1"}'})
        completed = ctx.complete_tool_call()

        assert completed["tool_name"] == "calc"
        assert ctx.has_pending_tool_call() == False
        assert ctx.has_completed_tool_calls() == True

    def test_add_tool_result(self):
        """测试添加工具结果"""
        ctx = PlannerContext(session_id="test")

        ctx.add_tool_result("call_1", "result")
        assert len(ctx.completed_tool_results) == 1
        assert ctx.completed_tool_results[0]["tool_call_id"] == "call_1"

    def test_clear_completed_calls(self):
        """测试清除已完成的工具调用"""
        ctx = PlannerContext(session_id="test")

        ctx.start_tool_call({"tool_name": "calc"})
        ctx.complete_tool_call()
        ctx.add_tool_result("call_1", "result")

        ctx.clear_completed_calls()
        assert ctx.has_completed_tool_calls() == False
        assert len(ctx.completed_tool_results) == 0

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """测试执行工具"""
        tool = AsyncMock()
        tool.run = AsyncMock(return_value="result")
        ctx = PlannerContext(
            session_id="test",
            tools={"calc": tool}
        )

        result = await ctx.execute_tool("calc", {"input": "1+1"})
        assert result == "result"
