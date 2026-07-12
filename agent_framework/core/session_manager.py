"""Session Manager - Core component for managing agent sessions with Actor model."""
import asyncio
import uuid
from typing import Dict, Any, Callable, Optional

from agent_framework.interfaces.session import SessionContext, Message
from agent_framework.interfaces.events import Event


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

        Args:
            sid: Session ID to process messages for.
        """
        q = self._session_queues[sid]
        while True:
            user_msg, future = await q.get()
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

                collected = []
                async for event in events:
                    collected.append(event)
                    await self.event_bus.publish(sid, event)

                future.set_result(collected)
                await self.storage.save(ctx)
                asyncio.create_task(memory.extract_long_term(sid))

            except Exception as e:
                future.set_exception(e)

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
            raise ValueError(f"Session '{session_id}' does not exist")
        future = asyncio.Future()
        await self._session_queues[session_id].put((user_msg, future))
        return future

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