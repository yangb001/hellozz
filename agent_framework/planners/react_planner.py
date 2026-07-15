"""ReAct Planner - Modern ReAct planning strategy using chat completions API.

This module implements the ReAct (Reasoning + Acting) planning pattern,
where the agent alternates between reasoning about the problem and
taking actions using tools until it reaches a final answer.

ModernReAct uses the chat completions API format with messages array
and tool_calls support instead of parsing plain text.

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


class ReActPlanner(BasePlanner):
    """Modern ReAct planner using chat completions API.

    Implements the ReAct planning pattern where the agent:
    1. Reasons about the current situation (Thought)
    2. Decides on an action (Action) via tool_calls
    3. Observes the result (Observation) from tool execution
    4. Repeats until reaching a Final Answer

    This implementation uses the modern chat completions API format
    with messages array instead of prompt string manipulation.

    Attributes:
        name: Planner identifier, defaults to "react".
        description: Human-readable description.
        max_iterations: Maximum number of reasoning iterations before forced stop.
    """

    name: str = "react"
    description: str = "Modern ReAct planner using chat completions API with tool_calls support"

    def __init__(self, name: str = "react", description: str = None, max_iterations: int = 10):
        """Initialize the ReActPlanner.

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
        """Execute Modern ReAct planning loop, yielding events.

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
            logger.debug(f"Modern ReAct iteration {iteration}/{self.max_iterations}")
            has_tool_calls = False  # Reset, will be set true if tool calls found

            # Call LLM with messages and tools
            # Convert ChatMessage objects to dicts for JSON serialization
            try:
                messages_dicts = [self._chat_message_to_dict(m) for m in messages]
                response_or_events = llm_call(messages_dicts, tools)

                # Check if llm_call returns an async iterator (streaming) or a ChatResponse
                if hasattr(response_or_events, '__aiter__'):
                    # Streaming response - yield each event directly without waiting
                    async for event in response_or_events:
                        yield event
                        # Track if we received a final answer or tool call
                        if event.type == "final_answer":
                            return
                        if event.type == "action":
                            has_tool_calls = True
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
            warning_msg = f"ReAct loop reached maximum iterations ({self.max_iterations}) without final answer"
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
                content=response.content
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
                        result = await tool.run(tool_args.get("input", ""), session_id=session_id)
                    else:
                        result = tool.run(tool_args.get("input", ""), session_id=session_id)

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
            # Check if content contains final answer indicators
            content_lower = response.content.lower()

            if any(indicator in content_lower for indicator in ['final answer:', 'finalanswer', 'the answer is']):
                # Extract and yield final answer
                final_answer = self._extract_final_answer(response.content)
                yield Event(type="final_answer", content=final_answer)
                logger.debug(f"Modern ReAct completed with final answer at iteration {iteration}")

                # Add assistant message to messages
                messages.append(ChatMessage(role="assistant", content=response.content))
            else:
                # Regular content - yield as text token
                yield Event(type="text_token", content=response.content)
                messages.append(ChatMessage(role="assistant", content=response.content))

    def _extract_final_answer(self, content: str) -> str:
        """Extract final answer text from content.

        Args:
            content: Raw content from LLM.

        Returns:
            Extracted final answer text.
        """
        import re
        # Try to find "Final Answer:" pattern
        match = re.search(r'Final Answer:\s*(.*)', content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: return content as-is
        return content.strip()

    async def _build_messages(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any]
    ) -> List[ChatMessage]:
        """Build the messages array for the chat completions API.

        Constructs a messages array that includes:
        - System message with ReAct instructions and tool definitions
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
        """Build the system message content with ReAct instructions.

        Args:
            tools: Dictionary of available tools.

        Returns:
            Formatted system message string.
        """
        parts = [
            "You are a helpful assistant that uses the ReAct (Reasoning + Acting) pattern.",
            "",
            # "## Response Format",
            # "For each response, you MUST produce one of:",
            # "",
            # "### Option 1: Tool Call (for complex tasks requiring external data)",
            # "When you need to use a tool, respond with:",
            # '{"content": "", "tool_calls": [{"id": "call_1", "name": "tool_name", "arguments": "{\\"input\\": \\"query\\"}"}]}',
            # "",
            # "### Option 2: Final Answer (for direct responses)",
            # "When you can answer directly, respond with:",
            # '{"content": "Your answer here...", "tool_calls": []}',
            # "",
            # "## Important Rules",
            # "- Use tool_calls array to request tool execution",
            # "- Tools are executed and results returned to you",
            # "- Continue reasoning until you can provide a complete answer",
            # "- The content field should be empty when using tools",
            # "- When providing final answer, put the answer in content field and empty tool_calls",
            # "",
        ]

        # Add available tools
        if tools:
            parts.append("## Available Tools")
            for tool_name, tool in tools.items():
                description = getattr(tool, 'description', 'No description available')
                parts.append(f"- **{tool_name}**: {description}")
            parts.append("")

        parts.append("Remember: Use tool_calls for external actions, content for final answers.")
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