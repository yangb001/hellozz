"""AgentRuntime - Stateless execution engine for agent sessions.

This module provides the AgentRuntime class that executes agent logic
by orchestrating context, memory, tools, planner, and LLM gateway.

The runtime is stateless - it borrows SessionContext at runtime and
does not maintain any internal state between invocations.
"""
from typing import AsyncIterator, Dict, Any, List, Optional

from ..interfaces.session import SessionContext, Message
from ..interfaces.events import Event, ToolCallEventData
from ..interfaces.base_memory import BaseMemory
from ..interfaces.base_planner import BasePlanner
from ..interfaces.enums import EventType
from ..interfaces.llm_types import ChatResponse as PlannerChatResponse, ToolCall as PlannerToolCall, FunctionCall
from ..infrastructure.llm_gateway import StreamChatResponse, ChatResponseType
from ..core.planner_context import PlannerContext


def _tools_to_schemas(tools: Dict[str, Any]) -> List[Dict]:
    """Convert tools dict to OpenAI tool schemas format.

    Args:
        tools: Dictionary mapping tool name to BaseTool object.

    Returns:
        List of OpenAI-compatible tool schema dictionaries.
    """
    schemas = []
    for name, tool in tools.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": getattr(tool, 'description', ''),
                "parameters": getattr(tool, 'parameters', {"type": "object", "properties": {}})
            }
        })
    return schemas


class AgentRuntime:
    """Stateless agent execution engine.

    The AgentRuntime orchestrates the execution flow:
    1. Adds user message to context
    2. Saves user message to memory
    3. Creates llm_call closure for planner
    4. Runs planner to generate events
    5. Saves final answer to context and memory
    6. Updates session last_active time

    This class is stateless - all state is held in SessionContext
    and passed in as a parameter.
    """

    async def run(
        self,
        ctx: SessionContext,
        user_input: str,
        memory: BaseMemory,
        tools: Dict[str, Any],
        planner: BasePlanner,
        llm_gateway: Any
    ) -> AsyncIterator[Event]:
        """Execute agent logic and yield events.

        Args:
            ctx: Current session context with messages and state.
            user_input: User's input message text.
            memory: Memory system for storing/retrieving context.
            tools: Dictionary of available tools by name.
            planner: Planning strategy to use.
            llm_gateway: LLM gateway for generating responses.

        Yields:
            Event objects representing thoughts, actions, observations,
            and final answer.
        """
        # Step 1: Add user message to context
        user_msg = Message(role="user", content=user_input)
        ctx.messages.append(user_msg)

        # Step 2: Save user message to memory
        await memory.save(ctx.session_id, user_msg)

        # Step 3: Define llm_call closure for planner using stream_chat - yields events
        async def llm_call(messages: List[Dict], tools: Optional[List[Dict]] = None) -> AsyncIterator[Event]:
            """Call LLM gateway using stream_chat and yield events.

            Streams events directly for true streaming support:
            - text_token events for content chunks
            - tool_call_start/argument/end events for tool calls

            Args:
                messages: List of conversation messages.
                tools: List of available tools (OpenAI schema format).

            Yields:
                Event objects for each chunk of content or tool call.
            """
            # Convert tools dict to OpenAI schemas if needed
            tools_schemas = _tools_to_schemas(tools) if tools else None

            async for chunk in llm_gateway.stream_chat(messages, tools=tools_schemas):
                # Handle thinking content FIRST (优先处理，避免与content重复)
                if hasattr(chunk, 'type') and chunk.type == ChatResponseType.THINKING_CONTENT:
                    yield Event(
                        type=EventType.THINKING_CONTENT,
                        content=chunk.content,
                        metadata={}
                    )
                # Handle content tokens only when NOT thinking content
                # (同一个chunk不会同时是THINKING_CONTENT和普通content)
                elif hasattr(chunk, 'content') and chunk.content:
                    yield Event(
                        type=EventType.TEXT_TOKEN,
                        content=chunk.content,
                        metadata={"chunk_index": 0}
                    )

                # Handle tool call events
                if hasattr(chunk, 'tool_call') and chunk.tool_call:
                    tc = chunk.tool_call
                    func = tc.function
                    func_name = func.name if hasattr(func, 'name') else str(func)
                    func_args = func.arguments if hasattr(func, 'arguments') else ""

                    # Determine the event type based on ChatResponseType
                    event_type = chunk.type if hasattr(chunk, 'type') else None

                    if event_type == ChatResponseType.TOOL_CALL_START:
                        yield Event(
                            type=EventType.TOOL_CALL_START,
                            content="",
                            metadata=ToolCallEventData(
                                tool_call_id=tc.id,
                                tool_name=func_name,
                                arguments=func_args,
                                is_complete=False
                            ).__dict__
                        )
                    elif event_type == ChatResponseType.TOOL_CALL_ARGUMENT:
                        yield Event(
                            type=EventType.TOOL_CALL_ARGUMENT,
                            content=func_args,
                            metadata=ToolCallEventData(
                                tool_call_id=tc.id,
                                tool_name=func_name,
                                arguments=func_args,
                                is_complete=False
                            ).__dict__
                        )
                    elif event_type == ChatResponseType.TOOL_CALL_END:
                        yield Event(
                            type=EventType.TOOL_CALL_END,
                            content="",
                            metadata=ToolCallEventData(
                                tool_call_id=tc.id,
                                tool_name=func_name,
                                arguments=func_args,
                                is_complete=True
                            ).__dict__
                        )
                    else:
                        # Default case for backward compatibility
                        yield Event(
                            type=EventType.TOOL_CALL_END,
                            content="",
                            metadata=ToolCallEventData(
                                tool_call_id=tc.id,
                                tool_name=func_name,
                                arguments=func_args,
                                is_complete=True
                            ).__dict__
                        )

                # Handle stream done
                if hasattr(chunk, 'finish_reason') and chunk.finish_reason:
                    yield Event(
                        type=EventType.STREAMING_END,
                        content="",
                        metadata={"finish_reason": chunk.finish_reason}
                    )

        # Step 4: Create PlannerContext and run planner
        planner_ctx = PlannerContext(
            session_id=ctx.session_id,
            tools=tools,
            messages=list(ctx.messages),
            memory=memory
        )

        async for event in planner.plan_and_act(planner_ctx, llm_call):
            # Step 5: If final answer, save to context and memory
            if event.type == "final_answer":
                assistant_msg = Message(role="assistant", content=event.content)
                ctx.messages.append(assistant_msg)
                await memory.save(ctx.session_id, assistant_msg)

            yield event

        # Step 6: Update session last_active time
        from datetime import datetime, timezone
        ctx.last_active = datetime.now(timezone.utc)
