"""Tool-call Planner - LLM planning strategy using chat completions API.

This module implements a planning pattern where the agent uses
tool_calls to perform actions and receives observations until
it reaches a final answer.

Uses the chat completions API format with messages array
and tool_calls support.

参考：详细设计.md 第7节
"""
import json
import logging
from typing import AsyncIterator, Dict, Any, Optional, List

from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event

logger = logging.getLogger(__name__)


from agent_framework.interfaces.llm_types import FunctionCall, ToolCall, ChatResponse, ChatMessage


class ToolCallPlanner(BasePlanner):
    """Tool-call planner using chat completions API.

    Implements a planning pattern where the agent:
    1. Uses tool_calls to perform actions
    2. Observes the result from tool execution
    3. Repeats until reaching a final answer

    This implementation uses the modern chat completions API format
    with messages array instead of prompt string manipulation.

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

    async def plan_and_act(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any],
        llm_call: callable,
    ) -> AsyncIterator[Event]:
        """Execute ToolCall planning loop, yielding events.

        Args:
            ctx: Current session context with messages and state.
            memory: Memory system for retrieving relevant context.
            tools: Dictionary of available tools by name.
            llm_call: Async callable that takes (messages, tools) and returns ChatResponse.

        Yields:
            Event objects representing thoughts, actions, observations, and final answer.
        """
        # Build initial messages array
        messages = await self._build_messages(ctx, memory, tools)

        iteration = 0
        has_tool_calls = True  # Start with true to enter loop on first iteration

        while has_tool_calls and iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"ToolCall iteration {iteration}/{self.max_iterations}")
            has_tool_calls = False  # Reset, will be set true if tool calls found

            # Call LLM with messages and tools
            # Convert ChatMessage objects to dicts for JSON serialization
            try:
                # Re-convert messages to dicts each iteration to include newly added tool results
                messages_dicts = [self._chat_message_to_dict(m) for m in messages]
                response_or_events = llm_call(messages_dicts, tools)

                # Check if llm_call returns an async iterator (streaming) or a ChatResponse
                if hasattr(response_or_events, '__aiter__'):
                    # Streaming response - need to collect tool calls and execute them
                    pending_tool_call: Optional[Dict[str, Any]] = None
                    accumulated_args = ""
                    completed_tool_calls: List[Dict[str, Any]] = []
                    completed_tool_results: List[Dict[str, Any]] = []

                    async for event in response_or_events:
                        # Track tool call info during streaming
                        if event.type == "tool_call_start":
                            # New tool call starting - store the info
                            tool_name = event.metadata.get("tool_name", "") if event.metadata else ""
                            tool_call_id = event.metadata.get("tool_call_id", "") if event.metadata else ""
                            # 从 metadata 获取初始 arguments（LLM 可能一次性返回完整参数）
                            initial_args = event.metadata.get("arguments", "") if event.metadata else ""
                            pending_tool_call = {
                                "tool_name": tool_name,
                                "tool_call_id": tool_call_id,
                                "arguments": initial_args
                            }
                            accumulated_args = initial_args
                            yield event

                        elif event.type == "tool_call_argument":
                            # Accumulate argument chunks
                            arg_chunk = event.content if event.content else ""
                            accumulated_args += arg_chunk
                            if pending_tool_call:
                                pending_tool_call["arguments"] = accumulated_args
                            yield event

                        elif event.type == "tool_call_end":
                            # Tool call complete - execute the tool
                            if pending_tool_call:
                                tool_name = pending_tool_call["tool_name"]
                                tool_call_id = pending_tool_call["tool_call_id"]
                                tool_args_raw = pending_tool_call["arguments"]

                                # Parse tool arguments
                                try:
                                    tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
                                except json.JSONDecodeError:
                                    tool_args = {"input": tool_args_raw}

                                yield Event(type="action", content=f"Calling {tool_name}...")

                                # Execute tool if available
                                if tool_name in tools:
                                    tool = tools[tool_name]
                                    try:
                                        import asyncio
                                        import inspect

                                        is_async = inspect.iscoroutinefunction(tool.run)
                                        session_id = ctx.session_id

                                        if is_async:
                                            # 使用 input 字段（如果有），否则使用整个 JSON 字符串
                                            input_value = tool_args.get("input", tool_args_raw) if tool_args else tool_args_raw
                                            result = await tool.run(input_value, session_id=session_id)
                                        else:
                                            input_value = tool_args.get("input", tool_args_raw) if tool_args else tool_args_raw
                                            result = tool.run(input_value, session_id=session_id)

                                        if not isinstance(result, str):
                                            result = str(result)

                                        # Collect tool result to add after assistant message (correct order)
                                        completed_tool_results.append({
                                            "tool_call_id": tool_call_id,
                                            "content": result
                                        })

                                        yield Event(type="observation", content=result)

                                    except Exception as e:
                                        error_msg = f"Error executing tool {tool_name}: {e}"
                                        logger.error(error_msg, exc_info=True)
                                        completed_tool_results.append({
                                            "tool_call_id": tool_call_id,
                                            "content": f"Error: {error_msg}"
                                        })
                                        yield Event(type="error", content=error_msg)
                                else:
                                    error_msg = f"Unknown tool: {tool_name}"
                                    logger.error(error_msg)
                                    completed_tool_results.append({
                                        "tool_call_id": tool_call_id,
                                        "content": f"Error: {error_msg}"
                                    })
                                    yield Event(type="error", content=error_msg)

                                # Track completed tool call for assistant message
                                completed_tool_calls.append({
                                    "tool_name": tool_name,
                                    "tool_call_id": tool_call_id,
                                    "arguments": tool_args_raw
                                })

                                pending_tool_call = None
                            yield event

                        else:
                            yield event

                        # Track if we received a final answer
                        if event.type == "final_answer":
                            return
                        if event.type == "action":
                            has_tool_calls = True
                    # After streaming completes, check if we had tool calls
                    # (has_tool_calls may have been set during tool execution)
                    if not has_tool_calls and completed_tool_calls:
                        has_tool_calls = True
                    # After streaming completes, add assistant message with tool_calls
                    if completed_tool_calls:
                        tool_call_objects = [
                            ToolCall(
                                id=tc["tool_call_id"],
                                type="function",
                                function=FunctionCall(
                                    name=tc["tool_name"],
                                    arguments=tc["arguments"]
                                )
                            )
                            for tc in completed_tool_calls
                        ]
                        messages.append(ChatMessage(
                            role="assistant",
                            content="",
                            tool_calls=tool_call_objects
                        ))
                        # Add tool results AFTER assistant message (correct order)
                        for tr in completed_tool_results:
                            messages.append(ChatMessage(
                                role="tool",
                                content=tr["content"],
                                tool_call_id=tr["tool_call_id"]
                            ))
                    # After streaming completes, continue to next iteration if needed
                    if has_tool_calls:
                        continue
                    break
                else:
                    # Non-streaming response (ChatResponse)
                    response = response_or_events
            except Exception as e:
                logger.error(f"Error calling LLM: {e}", exc_info=True)
                yield Event(type="error", content=f"LLM call failed: {e}")
                break

            # Handle response based on type
            if isinstance(response, ChatResponse):
                # Modern format with ChatResponse
                async for event in self._handle_chat_response(response, messages, iteration):
                    yield event
                    if event.type == "final_answer":
                        # Final answer received, exit loop
                        return
                    if event.type == "action":
                        # Tool execution in progress
                        has_tool_calls = True
            elif isinstance(response, str):
                # Legacy string format - yield as text token and continue
                yield Event(type="text_token", content=response)
            else:
                logger.warning(f"Unexpected response type from llm_call: {type(response)}")

        # Check if max iterations reached
        if iteration >= self.max_iterations:
            warning_msg = f"ToolCall loop reached maximum iterations ({self.max_iterations}) without final answer"
            logger.warning(warning_msg)
            yield Event(type="error", content=warning_msg)

    async def _handle_chat_response(
        self,
        response: ChatResponse,
        messages: List[ChatMessage],
        iteration: int
    ) -> AsyncIterator[Event]:
        """Handle a ChatResponse from LLM.

        Args:
            response: The ChatResponse from LLM.
            messages: Current messages array (will be modified by adding tool results).
            iteration: Current iteration number.

        Yields:
            Event objects based on response content and tool calls.
        """
        if response.has_tool_calls:
            # Add assistant message with tool calls to messages
            messages.append(ChatMessage(
                role="assistant",
                content=response.content,
                tool_calls=response.tool_calls
            ))

            # Execute each tool call and add results to messages
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args_raw = tool_call.function.arguments

                yield Event(type="action", content=f"Calling {tool_name}...")
                logger.debug(f"Tool call: {tool_name} with args: {tool_args_raw}")

                # Parse tool arguments
                try:
                    tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
                except json.JSONDecodeError:
                    tool_args = {"input": tool_args_raw}

                # Get tools from the tools_ref set during _build_messages
                # We need to pass tools through - let's get it from self
                tools = getattr(self, '_tools_ref', {})

                # Execute tool if available
                if tool_name not in tools:
                    error_msg = f"Unknown tool: {tool_name}"
                    logger.error(error_msg, exc_info=True)
                    messages.append(ChatMessage(
                        role="tool",
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call.id
                    ))
                    yield Event(type="error", content=error_msg)
                    continue

                tool = tools[tool_name]

                # Execute the tool
                try:
                    import asyncio
                    import inspect

                    is_async = inspect.iscoroutinefunction(tool.run)
                    session_id = getattr(self, '_session_id_ref', '')

                    if is_async:
                        input_value = tool_args.get("input", tool_args_raw) if tool_args else tool_args_raw
                        result = await tool.run(input_value, session_id=session_id)
                    else:
                        input_value = tool_args.get("input", tool_args_raw) if tool_args else tool_args_raw
                        result = tool.run(input_value, session_id=session_id)

                    if not isinstance(result, str):
                        result = str(result)

                    logger.debug(f"Tool {tool_name} result: {result[:100]}...")

                    # Add tool result message to messages
                    messages.append(ChatMessage(
                        role="tool",
                        content=result,
                        tool_call_id=tool_call.id
                    ))

                    yield Event(type="observation", content=result)

                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {e}"
                    logger.error(error_msg, exc_info=True)

                    messages.append(ChatMessage(
                        role="tool",
                        content=f"Error: {error_msg}",
                        tool_call_id=tool_call.id
                    ))
                    yield Event(type="error", content=error_msg)

        elif response.content:
            # No tool calls - this is the final answer
            yield Event(type="final_answer", content=response.content)
            logger.debug(f"ToolCall planner completed with final answer at iteration {iteration}")

            # Add assistant message to messages
            messages.append(ChatMessage(role="assistant", content=response.content))

    async def _build_messages(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any]
    ) -> List[ChatMessage]:
        """Build the messages array for the chat completions API.

        Constructs a messages array that includes:
        - System message with tool definitions
        - Memory context as a preceding user message
        - Conversation history from ctx.messages

        Args:
            ctx: Current session context.
            memory: Memory system for retrieving relevant context.
            tools: Dictionary of available tools.

        Returns:
            List of ChatMessage objects ready for API call.
        """
        messages: List[ChatMessage] = []

        # Store tools and session_id reference for use in _handle_chat_response
        self._tools_ref = tools
        self._session_id_ref = ctx.session_id

        # Build system message with tools
        system_content = self._build_system_message(tools)
        messages.append(ChatMessage(role="system", content=system_content))

        # Retrieve memory context and add as user message
        try:
            if ctx.messages:
                latest_user_msg = None
                for msg in reversed(ctx.messages):
                    if msg.role == "user":
                        latest_user_msg = msg.content
                        break

                if latest_user_msg:
                    memory_context = await memory.retrieve(
                        ctx.session_id,
                        latest_user_msg,
                        user_ids=ctx.participants
                    )
                    if memory_context:
                        context_msg = ChatMessage(
                            role="user",
                            content=f"Relevant context:\n{memory_context}"
                        )
                        messages.append(context_msg)
        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")

        # Add conversation history (skip system, keep user/assistant/tool)
        if ctx.messages:
            for msg in ctx.messages[-10:]:
                if msg.role == "system":
                    continue
                elif msg.role == "tool":
                    messages.append(ChatMessage(
                        role="tool",
                        content=msg.content,
                        tool_call_id=getattr(msg, 'tool_call_id', None)
                    ))
                elif msg.role in ("user", "assistant"):
                    messages.append(ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        name=getattr(msg, 'sender_id', None)
                    ))

        return messages

    def _build_system_message(self, tools: Dict[str, Any]) -> str:
        """Build the system message content with tool instructions.

        Args:
            tools: Dictionary of available tools.

        Returns:
            Formatted system message string.
        """
        parts = [
            "You are a helpful assistant that can use tools to answer questions.",
            "",
        ]

        # Add available tools
        if tools:
            parts.append("## Available Tools")
            for tool_name, tool in tools.items():
                description = getattr(tool, 'description', 'No description available')
                parts.append(f"- **{tool_name}**: {description}")
            parts.append("")

        return "\n".join(parts)

    async def _build_prompt(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any]
    ) -> str:
        """Build legacy prompt string (for backward compatibility).

        Deprecated: Use _build_messages instead for modern chat completions API.

        Args:
            ctx: Current session context.
            memory: Memory system for retrieving relevant context.
            tools: Dictionary of available tools.

        Returns:
            Formatted prompt string.
        """
        messages = await self._build_messages(ctx, memory, tools)
        prompt_parts = []

        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
            elif msg.role == "tool":
                prompt_parts.append(f"Tool Result: {msg.content}")

        return "\n\n".join(prompt_parts)

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

    def _parse_action(self, text: str):
        """Parse legacy action format (for backward compatibility).

        Deprecated: With modern ChatResponse format, parsing is handled
        by extracting content/tool_calls from the response object.

        Args:
            text: Text to parse.

        Returns:
            Action object (for backward compatibility only).
        """
        from dataclasses import dataclass

        @dataclass
        class LegacyAction:
            type: str
            tool: Optional[str] = None
            input: Optional[str] = None
            content: Optional[str] = None

        import re
        text = text.strip()

        if not text:
            return LegacyAction(type="final_answer", content="")

        # Check for final answer
        final_answer_match = re.search(r'Final Answer:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            return LegacyAction(type="final_answer", content=final_answer_match.group(1).strip())

        # Check for action
        action_match = re.search(r'Action:\s*(\S+)', text, re.IGNORECASE)
        if action_match:
            tool_name = action_match.group(1).strip()
            input_match = re.search(r'Action Input:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
            input_str = input_match.group(1).strip() if input_match else ""
            return LegacyAction(type="tool_call", tool=tool_name, input=input_str)

        # Default to thought
        return LegacyAction(type="thought", content=text)


# Backward compatibility alias - OLD NAME: ToolCallPlanner
ReActPlanner = ToolCallPlanner