"""Tool-call Planner - LLM planning strategy using chat completions API.

This module implements a planning pattern where the agent uses
tool_calls to perform actions and receives observations until
it reaches a final answer.

Uses the chat completions API format with messages array
and tool_calls support.

参考：详细设计.md 第7节

重构说明 (Phase 2A):
- 使用 PlannerContext 显式传递状态，移除副作用状态
- 集成 ThinkingRecorder 记录思考过程
- 使用 EventType 枚举替代字符串比较
- 方法签名简化，只接收 PlannerContext 和 llm_call

重构说明 (Phase 3):
- 统一流式和非流式处理路径
- 添加 _process_events 方法统一处理所有事件
- 添加 _handle_thinking_event 方法处理思考事件
- 添加 _execute_tool_from_ctx 方法从上下文执行工具
- 简化 plan_and_act 和 _handle_chat_response
"""
import asyncio
import json
import logging
from typing import AsyncIterator, Dict, Any, List

from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.interfaces.events import Event
from agent_framework.interfaces.enums import EventType
from agent_framework.interfaces.llm_types import FunctionCall, ToolCall, ChatResponse, ChatMessage
from agent_framework.core.planner_context import PlannerContext
from agent_framework.core.thinking_recorder import ThinkingRecorder

logger = logging.getLogger(__name__)


class ToolCallPlanner(BasePlanner):
    """Tool-call planner using chat completions API.

    Implements a planning pattern where the agent:
    1. Uses tool_calls to perform actions
    2. Observes the result from tool execution
    3. Repeats until reaching a final answer

    This implementation uses the modern chat completions API format
    with messages array instead of prompt string manipulation.

    重构后使用 PlannerContext 显式传递所有状态，避免副作用。

    Attributes:
        name: Planner identifier, defaults to "tool_call".
        description: Human-readable description.
        max_iterations: Maximum number of reasoning iterations before forced stop.
    """

    name: str = "tool_call"
    description: str = "Tool-call planner using chat completions API with tool_calls support"

    def __init__(self, name: str = "tool_call", description: str = None, max_iterations: int = 10):
        """Initialize the ToolCallPlanner.

        Args:
            name: Planner identifier.
            description: Human-readable description.
            max_iterations: Maximum number of reasoning iterations.
        """
        self.name = name
        self.description = description or self.description
        self.max_iterations = max_iterations

    async def _process_events(
        self,
        events: AsyncIterator,
        ctx: PlannerContext
    ) -> AsyncIterator[Event]:
        """统一处理所有流式事件

        Args:
            events: 事件流
            ctx: 规划器上下文

        Yields:
            处理后的事件
        """
        async for event in events:
            # 处理思考事件
            if event.type in (
                EventType.THINKING_START.value,
                EventType.THINKING_CONTENT.value,
                EventType.THINKING_END.value
            ):
                self._handle_thinking_event(event, ctx)
                yield event
                continue

            # 处理内容事件
            if event.type in (EventType.CONTENT_TOKEN.value, EventType.TEXT_TOKEN.value):
                ctx.add_text(event.content)
                yield event
                continue

            # 处理工具调用开始
            if event.type == EventType.TOOL_CALL_START.value:
                tool_name = event.metadata.get("tool_name", "") if event.metadata else ""
                tool_call_id = event.metadata.get("tool_call_id", "") if event.metadata else ""
                initial_args = event.metadata.get("arguments", "") if event.metadata else ""
                ctx.pending_tool_call = {
                    "tool_name": tool_name,
                    "tool_call_id": tool_call_id,
                    "arguments": initial_args
                }
                ctx.accumulated_args = initial_args
                yield event
                continue

            # 处理工具调用参数 (full accumulated, replace)
            if event.type == EventType.TOOL_CALL_ARGUMENT.value:
                arg_content = event.content if event.content else ""
                if arg_content:
                    ctx.accumulated_args = arg_content
                if ctx.pending_tool_call:
                    ctx.pending_tool_call["arguments"] = ctx.accumulated_args
                yield event
                continue

            # 处理工具调用结束
            if event.type == EventType.TOOL_CALL_END.value:
                if ctx.pending_tool_call:
                    # 执行工具
                    result, is_error = await self._execute_tool_from_ctx(ctx)

                    yield Event(
                        type=EventType.ACTION.value,
                        content=f"Calling {ctx.completed_tool_calls[-1]['tool_name']}..."
                    )
                    if is_error:
                        yield Event(type=EventType.ERROR.value, content=result)
                    else:
                        yield Event(type=EventType.OBSERVATION.value, content=result)
                yield event
                continue

            # 处理最终答案
            if event.type == EventType.FINAL_ANSWER.value:
                yield event
                return

            # 处理流结束
            if event.type == EventType.STREAMING_END.value:
                if not ctx.has_completed_tool_calls():
                    yield Event(type=EventType.FINAL_ANSWER.value, content=ctx.get_accumulated_text())
                return

            # 其他事件，直接转发
            yield event

    def _handle_thinking_event(self, event: Event, ctx: PlannerContext):
        """处理思考事件

        Args:
            event: 思考事件
            ctx: 规划器上下文
        """
        if event.type == EventType.THINKING_START.value:
            label = event.thinking.label if event.thinking else ""
            ctx.thinking_recorder.start_thinking(label)

        elif event.type == EventType.THINKING_CONTENT.value:
            ctx.thinking_recorder.add_content(event.content)

        elif event.type == EventType.THINKING_END.value:
            ctx.thinking_recorder.end_thinking()

    async def _execute_tool_from_ctx(self, ctx: PlannerContext):
        """从上下文执行工具

        Args:
            ctx: 规划器上下文

        Returns:
            Tuple of (result, is_error): result 是执行结果字符串，is_error 标识是否为错误
        """
        tool_call = ctx.pending_tool_call
        if not tool_call:
            return "Error: No pending tool call", True

        tool_name = tool_call.get("tool_name", "")
        tool_call_id = tool_call.get("tool_call_id", "")
        arguments = tool_call.get("arguments", "")

        # Parse tool arguments
        try:
            tool_args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            tool_args = {"input": arguments}

        # Execute tool
        if not ctx.tool_executor.has_tool(tool_name):
            error_msg = f"Unknown tool: {tool_name}"
            logger.error(error_msg)
            # Track completed call for message building
            ctx.completed_tool_calls.append({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments
            })
            ctx.add_tool_result(tool_call_id, f"Error: {error_msg}")
            ctx.pending_tool_call = None
            ctx.accumulated_args = ""
            return f"Error: {error_msg}", True

        try:
            result = await ctx.execute_tool(tool_name, tool_args)
            # Track completed call for message building
            ctx.completed_tool_calls.append({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments
            })
            ctx.add_tool_result(tool_call_id, result)
            ctx.pending_tool_call = None
            ctx.accumulated_args = ""
            return result, False
        except Exception as e:
            error_msg = f"Error executing tool {tool_name}: {e}"
            logger.error(error_msg, exc_info=True)
            ctx.completed_tool_calls.append({
                "tool_name": tool_name,
                "tool_call_id": tool_call_id,
                "arguments": arguments
            })
            ctx.add_tool_result(tool_call_id, f"Error: {error_msg}")
            ctx.pending_tool_call = None
            ctx.accumulated_args = ""
            return f"Error: {error_msg}", True

    async def plan_and_act(
        self,
        ctx: PlannerContext,
        llm_call: callable,
    ) -> AsyncIterator[Event]:
        """Execute ToolCall planning loop, yielding events.

        使用 PlannerContext 显式传递所有状态：
        - ctx.messages: 消息列表（替代原来的 messages 局部变量）
        - ctx.tools: 工具字典（替代原来的 tools 参数）
        - ctx.memory: 记忆系统（替代原来的 memory 参数）
        - ctx.session_id: 会话 ID（替代原来的 ctx.session_id）

        Args:
            ctx: PlannerContext containing all planner state.
            llm_call: Async callable that takes (messages, tools) and returns ChatResponse or async iterator.

        Yields:
            Event objects representing thoughts, actions, observations, and final answer.
        """
        # 清理思考记录器
        ctx.thinking_recorder.clear()

        # Build initial messages array
        system_message = await self._build_system_message(ctx)
        ctx.messages.insert(0, system_message)

        while not ctx.is_max_iterations_reached():
            ctx.increment_iteration()
            logger.debug(f"ToolCall iteration {ctx.iteration}/{ctx.max_iterations}")

            # Convert ChatMessage objects to dicts for JSON serialization
            try:
                messages_dicts = [self._chat_message_to_dict(m) for m in ctx.messages]
                response_or_events = llm_call(messages_dicts, ctx.tools)
                # Await coroutines (async functions that return ChatResponse)
                if asyncio.iscoroutine(response_or_events):
                    response_or_events = await response_or_events
            except Exception as e:
                logger.error(f"Error calling LLM: {e}", exc_info=True)
                yield Event(type=EventType.ERROR.value, content=f"LLM call failed: {e}")
                break

            # Check if llm_call returns an async iterator (streaming) or a ChatResponse
            if hasattr(response_or_events, '__aiter__'):
                # Streaming response - use unified event processing
                # Reset tool call tracking for this iteration
                ctx.completed_tool_calls.clear()
                ctx.completed_tool_results.clear()
                ctx.clear_accumulated_text()

                try:
                    async for event in self._process_events(response_or_events, ctx):
                        yield event
                        if event.type == EventType.FINAL_ANSWER.value:
                            return
                except Exception as e:
                    logger.error(f"Error during streaming: {e}", exc_info=True)
                    yield Event(type=EventType.ERROR.value, content=f"LLM call failed: {e}")
                    return

                # After streaming completes, check if tool calls were executed
                if not ctx.has_completed_tool_calls():
                    # No tool calls - yield final_answer so frontend knows response is complete
                    yield Event(type=EventType.FINAL_ANSWER.value, content="")
                    return

                # Tool calls were executed - add assistant message with tool_calls
                tool_call_objects = [
                    ToolCall(
                        id=tc["tool_call_id"],
                        type="function",
                        function=FunctionCall(
                            name=tc["tool_name"],
                            arguments=tc["arguments"]
                        )
                    )
                    for tc in ctx.completed_tool_calls
                ]
                ctx.messages.append(ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=tool_call_objects
                ))
                # Add tool results AFTER assistant message (correct order)
                for tr in ctx.completed_tool_results:
                    ctx.messages.append(ChatMessage(
                        role="tool",
                        content=tr["content"],
                        tool_call_id=tr["tool_call_id"]
                    ))
                # Continue to next iteration to get LLM's response to tool results
                continue
            else:
                # Non-streaming response (ChatResponse)
                response = response_or_events

            # Handle ChatResponse
            if isinstance(response, ChatResponse):
                has_tool_calls_in_response = False
                async for event in self._handle_chat_response(response, ctx):
                    yield event
                    if event.type == EventType.FINAL_ANSWER.value:
                        return
                    if event.type == EventType.ACTION.value:
                        has_tool_calls_in_response = True
                # Use tool_calls presence as sole indicator to continue
                if has_tool_calls_in_response:
                    continue
                # No tool calls and no final_answer - shouldn't happen but handle gracefully
                break
            elif isinstance(response, str):
                # Legacy string format - yield as text token and continue
                yield Event(type=EventType.TEXT_TOKEN.value, content=response)
            else:
                logger.warning(f"Unexpected response type from llm_call: {type(response)}")

        # Check if max iterations reached
        if ctx.is_max_iterations_reached():
            warning_msg = f"ToolCall loop reached maximum iterations ({ctx.max_iterations}) without final answer"
            logger.warning(warning_msg)
            yield Event(type=EventType.ERROR.value, content=warning_msg)

    async def _handle_chat_response(
        self,
        response: ChatResponse,
        ctx: PlannerContext,
    ) -> AsyncIterator[Event]:
        """Handle a ChatResponse from LLM.

        使用 PlannerContext 替代多个参数：
        - ctx.messages: 消息列表
        - ctx.tools: 工具字典
        - ctx.session_id: 会话 ID
        - ctx.iteration: 当前迭代次数

        Args:
            response: The ChatResponse from LLM.
            ctx: PlannerContext containing all planner state.

        Yields:
            Event objects based on response content and tool calls.
        """
        if response.has_tool_calls:
            # Add assistant message with tool calls to messages
            ctx.messages.append(ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # Execute each tool call and add results to messages
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_call_id = tool_call.id
                tool_args_raw = tool_call.function.arguments

                yield Event(type=EventType.ACTION.value, content=f"Calling {tool_name}...")
                logger.debug(f"Tool call: {tool_name} with args: {tool_args_raw}")

                # Execute tool via PlannerContext
                result = await ctx.execute_tool(tool_name, tool_args_raw)

                # Add tool result message
                ctx.messages.append(ChatMessage(
                    role="tool",
                    content=result,
                    tool_call_id=tool_call_id
                ))

                yield Event(type=EventType.OBSERVATION.value, content=result)

        elif response.content:
            # No tool calls - this is the final answer
            yield Event(type=EventType.FINAL_ANSWER.value, content=response.content)
            logger.debug(f"ToolCall planner completed with final answer at iteration {ctx.iteration}")

            # Add assistant message to messages
            ctx.messages.append(ChatMessage(role="assistant", content=response.content))

    async def _build_system_message(
        self,
        ctx: PlannerContext,
    ) -> ChatMessage:
        """构建 system 消息。

        返回单个 system 消息，内容包括：
        - 工具定义
        - 记忆上下文

        使用 PlannerContext 替代多个参数：
        - ctx.memory: 记忆系统
        - ctx.tools: 工具字典
        - ctx.session_id: 会话 ID

        注意：原版 _build_system_message 从 SessionContext.messages 中获取最新用户消息
        用于记忆检索。重构后 PlannerContext.messages 初始为空，记忆检索需要
        调用方确保 ctx 中有原始会话消息（通过 PlannerContext.session_messages
        或在调用 _build_system_message 前设置 ctx.messages）。

        Args:
            ctx: PlannerContext containing all planner state.

        Returns:
            ChatMessage object with role="system".
        """
        # Build system message with tools
        system_content = self._build_system_message_content(ctx.tools)

        # Retrieve memory context and inject into system message (I7)
        memory_context = ""
        try:
            if ctx.memory:
                latest_user_msg = None
                # 从 ctx.messages 中查找最新用户消息（用于记忆检索）
                # 注意：首次调用时 ctx.messages 可能为空，记忆检索会被跳过
                # 调用方应确保在 plan_and_act 调用前设置 ctx.messages
                for msg in reversed(ctx.messages):
                    if msg.role == "user":
                        latest_user_msg = msg.content
                        break

                if latest_user_msg:
                    # 注意：原版使用 ctx.participants 作为 user_ids 参数
                    # PlannerContext 暂无 participants 字段，传 None
                    # TODO: 后续需要在 PlannerContext 中添加 participants 支持
                    memory_context = await ctx.memory.retrieve(
                        ctx.session_id,
                        latest_user_msg,
                        user_ids=None  # 原版: ctx.participants
                    )
        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")

        if memory_context:
            system_content += f"\n\n## Relevant Context from Memory\n{memory_context}"

        return ChatMessage(role="system", content=system_content)

    def _build_system_message_content(self, tools: Dict[str, Any]) -> str:
        """Build the system message content with tool instructions.

        Args:
            tools: Dictionary of available tools.

        Returns:
            Formatted system message string.
        """
        parts = [
            "你是一个智能助手，既能认真解决问题，也能轻松聊天互动。",
            "",
            "## 意图识别",
            "",
            "根据用户消息判断意图：",
            "- **问题/任务**：专注分析，提供准确、有条理的解答",
            "- **闲聊/玩笑**：轻松回应，保持友好自然的对话氛围",
            "",
            "## 图片内容处理",
            "",
            "如果消息中包含【图片识别内容】标记，这是前端OCR识别的图片文字。请：",
            "- 提取其中的关键信息（问题、数据、要点）",
            "- 忽略格式标记和无关描述",
            "- 将提取内容理解为用户的实际输入",
            "",
            "格式示例：",
            "```",
            "【图片识别内容】",
            "识别出的文字内容...",
            "【/图片识别内容】",
            "```",
            "",
            "## 回复原则",
            "",
            "- 解答问题时：清晰准确，逻辑分明",
            "- 闲聊互动时：自然友好，简洁不啰嗦",
            "- 根据场景灵活切换风格",
            "",
        ]

        # Add available tools
        if tools:
            parts.append("## 可用工具")
            parts.append("")
            for tool_name, tool in tools.items():
                description = getattr(tool, 'description', 'No description available')
                parts.append(f"- **{tool_name}**: {description}")
            parts.append("")

        return "\n".join(parts)

    def _chat_message_to_dict(self, msg: ChatMessage) -> Dict[str, Any]:
        """Convert a ChatMessage to a dictionary for JSON serialization.

        Args:
            msg: ChatMessage object to convert.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        result = {"role": msg.role, "content": msg.content}
        if msg.name:
            result["name"] = msg.name
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in msg.tool_calls
            ]
        return result


# Backward compatibility alias - OLD NAME: ToolCallPlanner
ReActPlanner = ToolCallPlanner
