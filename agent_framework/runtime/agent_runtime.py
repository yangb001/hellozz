"""AgentRuntime - Stateless execution engine for agent sessions.

This module provides the AgentRuntime class that executes agent logic
by orchestrating context, memory, tools, planner, and LLM gateway.

The runtime is stateless - it borrows SessionContext at runtime and
does not maintain any internal state between invocations.
"""
from typing import AsyncIterator, Dict, Any

from ..interfaces.session import SessionContext, Message
from ..interfaces.events import Event
from ..interfaces.base_memory import BaseMemory
from ..interfaces.base_planner import BasePlanner


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

        # Step 3: Define llm_call closure for planner
        async def llm_call(prompt: str, **kwargs) -> AsyncIterator[str]:
            """Call LLM gateway and yield response tokens.

            Args:
                prompt: The prompt to send to the LLM.
                **kwargs: Additional arguments for the LLM.

            Yields:
                Response tokens from the LLM.
            """
            async for token in llm_gateway.stream(prompt, **kwargs):
                yield token

        # Step 4: Run planner and yield events
        async for event in planner.plan_and_act(ctx, memory, tools, llm_call):
            # Step 5: If final answer, save to context and memory
            if event.type == "final_answer":
                assistant_msg = Message(role="assistant", content=event.content)
                ctx.messages.append(assistant_msg)
                await memory.save(ctx.session_id, assistant_msg)

            yield event

        # Step 6: Update session last_active time
        from datetime import datetime, timezone
        ctx.last_active = datetime.now(timezone.utc)
