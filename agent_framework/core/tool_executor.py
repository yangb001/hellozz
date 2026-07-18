"""Tool Executor - 统一工具参数解析和执行"""
import json
import inspect
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ToolExecutor:
    """工具执行器

    统一处理工具参数解析和执行，解决以下问题：
    - 参数解析假设 "input" 键
    - 同步/异步工具统一处理
    - 错误处理一致
    """

    def __init__(self, tools: Dict[str, Any]):
        """初始化工具执行器

        Args:
            tools: 工具字典，key 为工具名，value 为工具对象
        """
        self.tools = tools

    async def execute(
        self,
        tool_name: str,
        arguments: Any,
        session_id: Optional[str] = None
    ) -> str:
        """执行工具

        Args:
            tool_name: 工具名称
            arguments: 工具参数（JSON 字符串或字典）
            session_id: 会话 ID

        Returns:
            执行结果字符串
        """
        # 1. 解析参数
        input_value = self._parse_arguments(arguments)

        # 2. 查找工具
        tool = self.tools.get(tool_name)
        if not tool:
            error_msg = f"Unknown tool: {tool_name}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

        # 3. 执行工具
        try:
            is_async = inspect.iscoroutinefunction(tool.run)

            if is_async:
                result = await tool.run(input_value, session_id=session_id)
            else:
                result = tool.run(input_value, session_id=session_id)

            # 确保返回字符串
            if not isinstance(result, str):
                result = str(result)

            logger.debug(f"Tool {tool_name} result: {result[:100]}...")
            return result

        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            return f"Error: {error_msg}"

    def _parse_arguments(self, arguments: Any) -> Any:
        """解析工具参数

        支持多种参数格式：
        - JSON 字符串: '{"input": "value"}'
        - 字典: {"input": "value"}
        - 普通字符串: "value"

        优先返回 "input" 键的值，否则返回第一个值。

        Args:
            arguments: 原始参数

        Returns:
            解析后的参数值
        """
        if arguments is None:
            return ""

        # 如果是字典，提取 input 或第一个值
        if isinstance(arguments, dict):
            if "input" in arguments:
                return arguments["input"]
            # 返回第一个值
            values = list(arguments.values())
            return values[0] if values else str(arguments)

        # 如果是字符串，尝试解析 JSON
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    if "input" in parsed:
                        return parsed["input"]
                    # 返回第一个值
                    values = list(parsed.values())
                    return values[0] if values else arguments
                return parsed
            except json.JSONDecodeError:
                # 不是 JSON，直接返回
                return arguments

        # 其他类型，转为字符串
        return str(arguments)

    def has_tool(self, tool_name: str) -> bool:
        """检查工具是否存在

        Args:
            tool_name: 工具名称

        Returns:
            是否存在
        """
        return tool_name in self.tools

    def get_tool_names(self) -> list:
        """获取所有工具名称

        Returns:
            工具名称列表
        """
        return list(self.tools.keys())

    def get_tool(self, tool_name: str) -> Any:
        """获取工具对象

        Args:
            tool_name: 工具名称

        Returns:
            工具对象，不存在返回 None
        """
        return self.tools.get(tool_name)
