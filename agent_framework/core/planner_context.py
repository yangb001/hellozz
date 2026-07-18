"""Planner Context - 规划器上下文"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.llm_types import ChatMessage
from agent_framework.core.tool_executor import ToolExecutor
from agent_framework.core.thinking_recorder import ThinkingRecorder


@dataclass
class PlannerContext:
    """规划器上下文

    包含规划器执行所需的所有状态，显式传递，避免副作用。

    使用方式:
        ctx = PlannerContext(
            session_id="xxx",
            tools={"calculator": calculator},
            messages=[...],
            memory=memory
        )

        # 在规划器中使用
        async for event in planner.plan_and_act(ctx, llm_call):
            ...
    """

    # 基本信息
    session_id: str
    tools: Dict[str, Any] = field(default_factory=dict)

    # 消息和记忆
    messages: List[ChatMessage] = field(default_factory=list)
    memory: Optional[BaseMemory] = None

    # 工具执行器（自动初始化）
    tool_executor: Optional[ToolExecutor] = field(default=None, repr=False)

    # 思考记录器
    thinking_recorder: Optional[ThinkingRecorder] = field(default=None, repr=False)

    # 执行状态
    iteration: int = 0
    max_iterations: int = 10

    # 工具调用追踪
    pending_tool_call: Optional[Dict[str, Any]] = None
    accumulated_args: str = ""
    completed_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    completed_tool_results: List[Dict[str, Any]] = field(default_factory=list)

    # 文本累积（用于流式响应）
    accumulated_text: str = ""

    def __post_init__(self):
        """初始化后处理"""
        if self.tool_executor is None:
            self.tool_executor = ToolExecutor(self.tools)
        if self.thinking_recorder is None:
            self.thinking_recorder = ThinkingRecorder()

    def increment_iteration(self) -> int:
        """增加迭代次数

        Returns:
            当前迭代次数
        """
        self.iteration += 1
        return self.iteration

    def is_max_iterations_reached(self) -> bool:
        """检查是否达到最大迭代次数

        Returns:
            是否达到
        """
        return self.iteration >= self.max_iterations

    def start_tool_call(self, tool_call: Dict[str, Any]):
        """开始工具调用

        Args:
            tool_call: 工具调用信息
        """
        self.pending_tool_call = tool_call
        self.accumulated_args = ""

    def update_tool_args(self, args_chunk: str):
        """更新工具参数

        Args:
            args_chunk: 参数片段
        """
        self.accumulated_args += args_chunk
        if self.pending_tool_call:
            self.pending_tool_call["arguments"] = self.accumulated_args

    def complete_tool_call(self) -> Optional[Dict[str, Any]]:
        """完成工具调用

        Returns:
            完成的工具调用信息
        """
        if self.pending_tool_call:
            completed = self.pending_tool_call.copy()
            self.completed_tool_calls.append(completed)
            self.pending_tool_call = None
            self.accumulated_args = ""
            return completed
        return None

    def add_tool_result(self, tool_call_id: str, content: str):
        """添加工具结果

        Args:
            tool_call_id: 工具调用 ID
            content: 结果内容
        """
        self.completed_tool_results.append({
            "tool_call_id": tool_call_id,
            "content": content
        })

    def clear_completed_calls(self):
        """清除已完成的工具调用"""
        self.completed_tool_calls.clear()
        self.completed_tool_results.clear()

    def has_pending_tool_call(self) -> bool:
        """检查是否有待处理的工具调用

        Returns:
            是否有
        """
        return self.pending_tool_call is not None

    def has_completed_tool_calls(self) -> bool:
        """检查是否有已完成的工具调用

        Returns:
            是否有
        """
        return len(self.completed_tool_calls) > 0

    def add_text(self, text: str):
        """累积文本内容

        Args:
            text: 要添加的文本
        """
        self.accumulated_text += text

    def get_accumulated_text(self) -> str:
        """获取累积的文本

        Returns:
            累积的文本内容
        """
        return self.accumulated_text

    def clear_accumulated_text(self):
        """清除累积的文本"""
        self.accumulated_text = ""

    async def execute_tool(self, tool_name: str, arguments: Any) -> str:
        """执行工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数

        Returns:
            执行结果
        """
        return await self.tool_executor.execute(
            tool_name=tool_name,
            arguments=arguments,
            session_id=self.session_id
        )

    def to_session_context(self) -> Optional[SessionContext]:
        """转换为 SessionContext（如果需要）

        Returns:
            SessionContext 对象
        """
        # 这个方法用于向后兼容
        # 在新架构中，应该直接使用 PlannerContext
        return None
