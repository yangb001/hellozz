"""Session Manager - Core component for managing agent sessions with Actor model."""
import asyncio
import logging
import uuid
from typing import Dict, Any, Callable, Optional, AsyncIterator

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event

logger = logging.getLogger("agent_framework.core.session_manager")


def generate_id() -> str:
    """Generate a unique session ID."""
    return str(uuid.uuid4())


class SessionManager:
    """Manages agent sessions with Actor model using asyncio.Queue.

    Each session has its own queue for serial message processing,
    ensuring thread-safe, sequential handling of messages per session.

    Attributes:
        memory_factory: Callable that creates a BaseMemory instance for a session.
        runtime: AgentRuntime instance for executing agent logic.
        planner: BasePlanner instance for planning strategies.
        tools: Dictionary of available tools by name.
        event_bus: EventBus instance for publishing events.
        storage: SessionStorage instance for persistence.
        llm_gateway: LLMGateway instance for LLM calls.
    """

    def __init__(
        self,
        memory_factory: Callable[[str], Any],
        runtime: Any,
        planner: Any,
        tools: Dict[str, Any],
        event_bus: Any,
        storage: Any,
        llm_gateway: Any
    ):
        """Initialize SessionManager with dependencies.

        Args:
            memory_factory: Factory function to create memory instances.
            runtime: Runtime engine for agent execution.
            planner: Planning strategy implementation.
            tools: Dictionary of available tools.
            event_bus: Event bus for publishing events.
            storage: Storage adapter for session persistence.
            llm_gateway: LLM gateway for AI calls.
        """
        self.memory_factory = memory_factory
        self.runtime = runtime
        self.planner = planner
        self.tools = tools
        self.event_bus = event_bus
        self.storage = storage
        self.llm_gateway = llm_gateway
        self._active_sessions: Dict[str, SessionContext] = {}
        self._session_queues: Dict[str, asyncio.Queue] = {}

    async def create_session(
        self,
        user_id: str,
        session_type: str = "private",
        participants: Optional[list] = None
    ) -> SessionContext:
        """Create a new session and start its actor loop.

        Args:
            user_id: ID of the user creating the session.
            session_type: Type of session ('private' or 'group').
            participants: Optional list of additional participant IDs.

        Returns:
            SessionContext for the newly created session.
        """
        sid = generate_id()
        ctx = SessionContext(
            session_id=sid,
            session_type=session_type,
            participants=[user_id] + (participants or [])
        )

        self._active_sessions[sid] = ctx
        self._session_queues[sid] = asyncio.Queue()
        asyncio.create_task(self._session_actor(sid))

        await self.storage.save(ctx)
        return ctx

    async def _session_actor(self, sid: str):
        """Actor loop for processing session messages serially.

        Supports two modes:
        - Batch mode: (user_msg, future) - returns Future with collected list
        - Stream mode: (user_msg, event_queue) - yields events via queue, None = end signal

        Args:
            sid: Session ID to process messages for.
        """
        q = self._session_queues[sid]
        while True:
            item = await q.get()
            user_msg = item[0]

            # Detect mode: batch mode has 2 items with Future, stream mode has 2 items with Queue
            result_holder = item[1]
            is_stream_mode = isinstance(result_holder, asyncio.Queue)

            try:
                ctx = self._active_sessions[sid]
                memory = self.memory_factory(sid)

                events = self.runtime.run(
                    ctx=ctx,
                    user_input=user_msg.get("content", ""),
                    memory=memory,
                    tools=self.tools,
                    planner=self.planner,
                    llm_gateway=self.llm_gateway
                )

                if is_stream_mode:
                    # Stream mode: yield events as they come
                    async for event in events:
                        await result_holder.put(event)
                        await self.event_bus.publish(sid, event)
                    result_holder.put(None)  # End signal
                else:
                    # Batch mode: collect all events
                    collected = []
                    async for event in events:
                        collected.append(event)
                        await self.event_bus.publish(sid, event)
                    result_holder.set_result(collected)

                await self.storage.save(ctx)
                asyncio.create_task(memory.extract_long_term(sid))

            except Exception as e:
                logger.error(f"Error processing message in session {sid}: {e}", exc_info=True)
                if is_stream_mode:
                    result_holder.put(None)  # End signal on error
                else:
                    result_holder.set_exception(e)

    async def process_message(
        self,
        session_id: str,
        user_msg: dict
    ) -> asyncio.Future:
        """Submit a message for processing in the session's actor queue.

        Args:
            session_id: ID of the session to process the message.
            user_msg: Message dict containing 'content' and optional fields.

        Returns:
            asyncio.Future that resolves to list of Event objects.

        Raises:
            ValueError: If session_id does not exist.
        """
        if session_id not in self._session_queues:
            # Try to resume session from storage if not in memory
            ctx = await self.resume_session(session_id)
            if ctx is None:
                raise ValueError(f"Session '{session_id}' does not exist")
            logger.info(f"Resumed session {session_id} from storage")
        future = asyncio.Future()
        await self._session_queues[session_id].put((user_msg, future))
        return future

    async def process_message_stream(
        self,
        session_id: str,
        user_msg: dict
    ) -> AsyncIterator[Event]:
        """流式处理消息，实时yield事件。

        Args:
            session_id: ID of the session to process the message.
            user_msg: Message dict containing 'content' and optional fields.

        Yields:
            Event objects as they are generated by the runtime.
            None signals end of stream.

        Raises:
            ValueError: If session_id does not exist.
        """
        if session_id not in self._session_queues:
            ctx = await self.resume_session(session_id)
            if ctx is None:
                raise ValueError(f"Session '{session_id}' does not exist")
            logger.info(f"Resumed session {session_id} from storage")

        event_queue: asyncio.Queue = asyncio.Queue()
        await self._session_queues[session_id].put((user_msg, event_queue))

        while True:
            event = await event_queue.get()
            if event is None:  # 结束信号
                break
            yield event

    async def get_session(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve an active session by ID.

        Args:
            session_id: ID of the session to retrieve.

        Returns:
            SessionContext if found, None otherwise.
        """
        return self._active_sessions.get(session_id)

    async def close_session(self, session_id: str) -> None:
        """Close a session and clean up its resources.

        Args:
            session_id: ID of the session to close.
        """
        if session_id in self._active_sessions:
            ctx = self._active_sessions[session_id]
            ctx.status = "closed"
            await self.storage.save(ctx)
            del self._active_sessions[session_id]

        if session_id in self._session_queues:
            del self._session_queues[session_id]

    async def resume_session(self, session_id: str) -> Optional[SessionContext]:
        """Resume a crashed session from storage.

        Args:
            session_id: ID of the session to resume.

        Returns:
            SessionContext if session exists in storage, None otherwise.
        """
        ctx = await self.storage.load(session_id)
        if ctx is None:
            return None

        self._active_sessions[session_id] = ctx
        self._session_queues[session_id] = asyncio.Queue()
        asyncio.create_task(self._session_actor(session_id))

        return ctx