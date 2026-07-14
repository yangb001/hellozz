"""ReAct Planner - Reasoning and Acting planning strategy.

This module implements the ReAct (Reasoning + Acting) planning pattern,
where the agent alternates between reasoning about the problem and
taking actions using tools until it reaches a final answer.

参考：详细设计.md 第7节
"""
import re
import logging
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Any, Optional, List

from agent_framework.interfaces.base_planner import BasePlanner
from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.base_memory import BaseMemory
from agent_framework.interfaces.base_tool import BaseTool
from agent_framework.interfaces.events import Event

logger = logging.getLogger(__name__)


@dataclass
class Action:
    """Represents a parsed action from LLM output.

    Attributes:
        type: Type of action (tool_call, final_answer, thought).
        tool: Name of the tool to call (for tool_call actions).
        input: Input to pass to the tool (for tool_call actions).
        content: Content text (for final_answer or thought actions).
    """
    type: str
    tool: Optional[str] = None
    input: Optional[str] = None
    content: Optional[str] = None


class ReActPlanner(BasePlanner):
    """ReAct (Reasoning + Acting) planner implementation.

    Implements the ReAct planning pattern where the agent:
    1. Reasons about the current situation (Thought)
    2. Decides on an action (Action)
    3. Observes the result (Observation)
    4. Repeats until reaching a Final Answer

    This approach combines chain-of-thought reasoning with tool usage,
    allowing the agent to iteratively solve complex problems.

    Attributes:
        name: Planner identifier, defaults to "react".
        description: Human-readable description.
        max_iterations: Maximum number of reasoning iterations before forced stop.
    """

    name: str = "react"
    description: str = "ReAct planner that alternates between reasoning and acting"

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
        """Execute ReAct planning loop, yielding events.

        Args:
            ctx: Current session context with messages and state.
            memory: Memory system for retrieving relevant context.
            tools: Dictionary of available tools by name.
            llm_call: Async callable that takes a prompt and yields response tokens.

        Yields:
            Event objects representing thoughts, actions, observations, and final answer.
        """
        # Build initial prompt with context
        prompt = await self._build_prompt(ctx, memory, tools)

        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"ReAct iteration {iteration}/{self.max_iterations}")

            # Collect LLM response
            full_text = ""
            async for token in llm_call(prompt):
                full_text += token
                yield Event(type="text_token", content=token)

            # Parse the action from LLM response
            action = self._parse_action(full_text)

            if action.type == "final_answer":
                # Yield final answer event and break
                yield Event(type="final_answer", content=action.content)
                logger.debug(f"ReAct completed with final answer after {iteration} iterations")
                break

            elif action.type == "tool_call":
                # Check if tool exists
                if action.tool not in tools:
                    error_msg = f"Unknown tool: {action.tool}. Available tools: {list(tools.keys())}"
                    logger.error(error_msg, exc_info=True)
                    yield Event(type="error", content=error_msg)
                    break

                # Yield action event
                yield Event(type="action", content=f"Calling {action.tool}...")

                # Execute the tool
                try:
                    tool = tools[action.tool]
                    # Handle both async and sync tool.run methods
                    import asyncio
                    import inspect

                    # Check if tool.run is a coroutine function
                    is_async = inspect.iscoroutinefunction(tool.run)

                    if is_async:
                        result = await tool.run(action.input, session_id=ctx.session_id)
                    else:
                        # For sync functions, call directly
                        result = tool.run(action.input, session_id=ctx.session_id)

                    # Ensure result is a string
                    if not isinstance(result, str):
                        result = str(result)

                    # Append observation to prompt for next iteration
                    prompt += f"\nObservation: {result}"

                    # Yield observation event
                    yield Event(type="observation", content=result)
                    logger.debug(f"Tool {action.tool} executed successfully")

                except Exception as e:
                    error_msg = f"Error executing tool {action.tool}: {str(e)}"
                    logger.error(error_msg, exc_info=True)
                    yield Event(type="error", content=error_msg)
                    break

            else:
                # For thought or unknown action types, continue the loop
                logger.debug(f"Action type: {action.type}, continuing loop")

        else:
            # Max iterations reached without final answer
            warning_msg = f"ReAct loop reached maximum iterations ({self.max_iterations}) without final answer"
            logger.warning(warning_msg)
            yield Event(type="error", content=warning_msg)

    async def _build_prompt(
        self,
        ctx: SessionContext,
        memory: BaseMemory,
        tools: Dict[str, Any]
    ) -> str:
        """Build the system prompt for the LLM.

        Constructs a prompt that includes:
        - System instructions for ReAct behavior
        - Available tools and their descriptions
        - Tool call examples
        - Memory context (relevant past information)
        - Conversation history

        Args:
            ctx: Current session context.
            memory: Memory system for retrieving relevant context.
            tools: Dictionary of available tools.

        Returns:
            Formatted prompt string.
        """
        # Start with ReAct system instructions
        prompt_parts = [
            "You are a helpful assistant that uses the ReAct (Reasoning + Acting) pattern.",
            "",
            "## Response Format",
            "For each step, you MUST follow one of these formats:",
            "",
            "### Option 1: Use a tool",
            "Thought: [your reasoning about what to do next]",
            "Action: [tool_name]",
            "Action Input: [input for the tool]",
            "",
            "### Option 2: Provide final answer",
            "Thought: [your final reasoning]",
            "Final Answer: [your complete answer to the user]",
            "",
            "## Important Rules",
            "- Always start with 'Thought:' to explain your reasoning",
            "- After using a tool, you will receive an 'Observation:' with the result",
            "- Continue reasoning until you can provide a 'Final Answer:'",
            "- Use the exact format shown above for tool calls",
            "",
        ]

        # Add available tools with examples
        if tools:
            prompt_parts.append("## Available Tools")
            for tool_name, tool in tools.items():
                description = getattr(tool, 'description', 'No description available')
                prompt_parts.append(f"- **{tool_name}**: {description}")
            prompt_parts.append("")

            # Add tool call example
            first_tool_name = list(tools.keys())[0]
            prompt_parts.append("## Tool Call Example")
            prompt_parts.append("Thought: I need to search for information about Python.")
            prompt_parts.append(f"Action: {first_tool_name}")
            prompt_parts.append("Action Input: Python programming language")
            prompt_parts.append("")
            prompt_parts.append("After receiving the observation, continue with:")
            prompt_parts.append("Thought: Based on the search results, I can now answer.")
            prompt_parts.append("Final Answer: Python is a high-level programming language...")
            prompt_parts.append("")

        # Retrieve and add memory context
        try:
            if ctx.messages:
                # Get the latest user message for memory retrieval
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
                        prompt_parts.append("## Relevant Context from Memory")
                        prompt_parts.append(memory_context)
                        prompt_parts.append("")
        except Exception as e:
            logger.warning(f"Failed to retrieve memory context: {e}")

        # Add conversation history
        if ctx.messages:
            prompt_parts.append("## Conversation History")
            for msg in ctx.messages[-10:]:  # Last 10 messages
                sender = msg.sender_id or msg.role
                prompt_parts.append(f"[{msg.role}] ({sender}): {msg.content}")
            prompt_parts.append("")

        prompt_parts.append("## Current Task")
        prompt_parts.append("Now, let's continue the conversation using the ReAct pattern.")
        prompt_parts.append("Remember to use 'Thought:' first, then either 'Action:' + 'Action Input:' or 'Final Answer:'.")
        prompt_parts.append("")

        return "\n".join(prompt_parts)

    def _parse_action(self, text: str) -> Action:
        """Parse an action from LLM output text.

        Analyzes the LLM response to determine the next action.
        Supports multiple formats:

        Standard formats:
        - "Final Answer: ..."
        - "Action: tool_name\nAction Input: input"
        - "Thought: ..."

        Alternative formats:
        - "Use tool: tool_name\nInput: input"
        - "Tool: tool_name\nQuery: input"
        - "Call tool_name with query: input"
        - '{"tool": "tool_name", "input": "input"}'
        - "tool_name('input')"
        - "I will use tool_name to find: input"
        - "The answer is ..."
        - "In conclusion, ..."
        - "To summarize, ..."

        Args:
            text: The complete LLM response text.

        Returns:
            Action object representing the parsed action.
        """
        # Clean the text
        text = text.strip()

        # Empty text check
        if not text:
            return Action(type="final_answer", content="")

        # Check for Final Answer (case-insensitive)
        final_answer_match = re.search(r'Final Answer:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            content = final_answer_match.group(1).strip()
            return Action(type="final_answer", content=content)

        # Check for standard Action/Action Input format (case-insensitive)
        # Use findall to get all matches and take the last one
        action_matches = re.findall(r'Action:\s*(\S+)', text, re.IGNORECASE)
        action_input_matches = re.findall(r'Action Input:\s*(.*)', text, re.IGNORECASE)

        if action_matches:
            # Take the last Action found
            tool_name = action_matches[-1].strip()
            # Take the last Action Input found (if any)
            action_input = action_input_matches[-1].strip() if action_input_matches else ""
            return Action(type="tool_call", tool=tool_name, input=action_input)

        # Check for "Use tool:" format
        use_tool_match = re.search(r'Use tool:\s*(\S+)', text, re.IGNORECASE)
        if use_tool_match:
            tool_name = use_tool_match.group(1).strip()
            input_match = re.search(r'Input:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
            action_input = input_match.group(1).strip() if input_match else ""
            return Action(type="tool_call", tool=tool_name, input=action_input)

        # Check for "Tool:" format with "Query:" or "Input:"
        tool_colon_match = re.search(r'Tool:\s*(\S+)', text, re.IGNORECASE)
        if tool_colon_match:
            tool_name = tool_colon_match.group(1).strip()
            query_match = re.search(r'(?:Query|Input):\s*(.*)', text, re.IGNORECASE | re.DOTALL)
            action_input = query_match.group(1).strip() if query_match else ""
            return Action(type="tool_call", tool=tool_name, input=action_input)

        # Check for "Call tool_name with" format
        call_match = re.search(r'Call\s+(\w+)\s+with\s+(?:query|input):\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if call_match:
            tool_name = call_match.group(1).strip()
            action_input = call_match.group(2).strip()
            return Action(type="tool_call", tool=tool_name, input=action_input)

        # Check for JSON-like format: {"tool": "name", "input": "value"}
        json_match = re.search(r'\{["\']tool["\']:\s*["\'](\w+)["\'].*?["\']input["\']:\s*["\']([^"\']*)["\']', text, re.IGNORECASE)
        if json_match:
            tool_name = json_match.group(1).strip()
            action_input = json_match.group(2).strip()
            return Action(type="tool_call", tool=tool_name, input=action_input)

        # Check for function call format: tool_name('input') or tool_name("input")
        func_match = re.search(r'(\w+)\(["\']([^"\']*)["\']\)', text)
        if func_match:
            tool_name = func_match.group(1).strip()
            action_input = func_match.group(2).strip()
            # Avoid matching common Python functions
            if tool_name not in ['print', 'len', 'str', 'int', 'float', 'list', 'dict', 'set', 'type']:
                return Action(type="tool_call", tool=tool_name, input=action_input)

        # Check for "I will use tool_name to" format
        i_will_match = re.search(r'I will use\s+(\w+)\s+to\s+\w+:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if i_will_match:
            tool_name = i_will_match.group(1).strip()
            action_input = i_will_match.group(2).strip()
            return Action(type="tool_call", tool=tool_name, input=action_input)

        # Check for alternative final answer formats
        # "The answer is ..."
        answer_is_match = re.search(r'(?:The )?[Aa]nswer is\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if answer_is_match:
            content = answer_is_match.group(1).strip()
            if content:
                return Action(type="final_answer", content=content)

        # "In conclusion, ..."
        conclusion_match = re.search(r'In conclusion,\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if conclusion_match:
            content = conclusion_match.group(1).strip()
            if content:
                return Action(type="final_answer", content=content)

        # "To summarize, ..."
        summarize_match = re.search(r'To summarize,\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if summarize_match:
            content = summarize_match.group(1).strip()
            if content:
                return Action(type="final_answer", content=content)

        # Check for Thought (case-insensitive)
        thought_match = re.search(r'Thought:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if thought_match:
            content = thought_match.group(1).strip()
            # If there's a thought but no clear action, treat as final answer
            # This handles cases where the LLM just provides a direct response
            if content and not re.search(r'Action:|Final Answer:|Use tool:|Tool:|Call\s+\w+\s+with', text, re.IGNORECASE):
                return Action(type="final_answer", content=content)
            return Action(type="thought", content=content)

        # If no pattern matches, treat the entire text as a final answer
        # This is a fallback for when LLM doesn't follow the exact format
        return Action(type="final_answer", content=text)