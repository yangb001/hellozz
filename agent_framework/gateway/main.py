"""FastAPI application main module.

This module creates and configures the FastAPI application instance,
including middleware, routes, and lifecycle events.
"""
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from .dependencies import set_session_manager, clear_session_manager
from ..core.session_manager import SessionManager
from ..core.event_bus import EventBus
from ..runtime.agent_runtime import AgentRuntime
from ..planners.react_planner import ReActPlanner
from ..tools.registry import ToolRegistry
from ..tools.builtin.calculator import Calculator
from ..tools.builtin.web_search import WebSearch
from ..memory.memory_manager import MemoryManager, MemoryConfig
from ..memory.buffer_memory import BufferMemory
from ..memory.vector_memory import VectorMemory
from ..memory.extractor import MemoryExtractor
from ..infrastructure.openai_llm import OpenAILLM, OpenAIConfig
from ..infrastructure.storage.session_storage import SessionStorage
from ..infrastructure.storage.sqlite_session_storage import SQLiteSessionStorage
from ..interfaces.session import SessionContext
from ..core.config import load_config
from ..core.logging_config import setup_logging, LoggingConfig as CoreLoggingConfig, get_logger


class InMemorySessionStorage(SessionStorage):
    """Simple in-memory session storage for development/testing."""

    def __init__(self):
        self._sessions: Dict[str, SessionContext] = {}

    async def save(self, ctx: SessionContext) -> None:
        self._sessions[ctx.session_id] = ctx

    async def load(self, session_id: str) -> SessionContext | None:
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]


def build_session_manager() -> SessionManager:
    """Build and configure SessionManager with all dependencies.

    Returns:
        Configured SessionManager instance.
    """
    logger = get_logger("agent_framework.gateway")

    # Load configuration
    config = load_config("config.json")

    # Ensure data directory exists for SQLite
    data_dir = os.path.dirname(config.sqlite.path)
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)

    # Create LLM Gateway
    llm_config = config.llm
    provider_name = llm_config.default
    provider_config = llm_config.providers.get(provider_name)

    if provider_config:
        openai_config = OpenAIConfig(
            model=provider_config.model,
            base_url=provider_config.base_url,
            api_key=provider_config.api_key,
            verify_ssl=provider_config.verify_ssl,
        )
        llm_gateway = OpenAILLM(openai_config)
    else:
        # Fallback to mock if no config
        llm_gateway = None

    # Create Memory system
    buffer_memory = BufferMemory(max_tokens=config.memory.short_term_size * 100)
    vector_memory = VectorMemory(vector_store=None)  # No vector store for now
    memory_extractor = MemoryExtractor(llm_gateway=llm_gateway)
    memory_config = MemoryConfig(
        trigger=config.memory.trigger,
        every_n=config.memory.every_n,
    )
    memory_manager = MemoryManager(
        short_term=buffer_memory,
        long_term=vector_memory,
        extractor=memory_extractor,
        config=memory_config,
    )

    # Create Tools - register based on config
    tool_registry = ToolRegistry()
    enabled_tools = config.tools.enabled  # List of enabled tool names, empty means all

    # Available tools mapping
    available_tools = {
        "calculator": Calculator,
        "web_search": WebSearch,
    }

    # Register tools based on config
    for tool_name, tool_class in available_tools.items():
        if not enabled_tools or tool_name in enabled_tools:
            tool_registry.register(tool_class())
            logger.info(f"Registered tool: {tool_name}")
        else:
            logger.debug(f"Skipped tool (disabled): {tool_name}")

    # Create Planner
    planner = ReActPlanner()

    # Create Runtime
    runtime = AgentRuntime()

    # Create EventBus
    event_bus = EventBus()

    # Create Storage - use SQLite for persistent storage
    storage = SQLiteSessionStorage(db_path=config.sqlite.path)

    # Create SessionManager
    session_manager = SessionManager(
        memory_factory=lambda sid: memory_manager,
        runtime=runtime,
        planner=planner,
        tools=tool_registry.to_dict(),
        event_bus=event_bus,
        storage=storage,
        llm_gateway=llm_gateway
    )

    return session_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager.

    Handles startup and shutdown events for the FastAPI application.

    Args:
        app: FastAPI application instance.

    Yields:
        None during application lifetime.
    """
    # Startup: Initialize dependencies
    # Initialize logging first so subsequent code can use proper loggers
    try:
        config = load_config("config.json")
        logging_cfg = config.logging
        core_logging_cfg = CoreLoggingConfig(
            level=logging_cfg.level,
            log_dir=logging_cfg.log_dir,
            max_bytes=logging_cfg.max_bytes,
            backup_count=logging_cfg.backup_count,
            console_output=logging_cfg.console_output,
            file_output=logging_cfg.file_output,
        )
        setup_logging(core_logging_cfg)
    except FileNotFoundError:
        # No config file found, use default logging
        setup_logging()
    except Exception as e:
        print(f"Warning: Failed to initialize logging from config: {e}")
        setup_logging()

    logger = get_logger("agent_framework.gateway")
    logger.info("Starting Agent Framework API...")

    try:
        session_manager = build_session_manager()
        set_session_manager(session_manager)
        logger.info("SessionManager initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SessionManager: {e}")
        logger.warning("API will start but session features will be unavailable")

    yield

    # Shutdown: Cleanup resources
    logger.info("Shutting down Agent Framework API...")
    clear_session_manager()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Agent Framework API",
        description="Multi-session AI Agent framework REST and WebSocket API",
        version="0.1.0",
        lifespan=lifespan
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    from .api.rest import router as rest_router
    app.include_router(rest_router)

    from .api.websocket import router as ws_router
    app.include_router(ws_router)

    # Serve static files
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/")
        async def root():
            """Serve the main web interface."""
            return FileResponse(os.path.join(static_dir, "index.html"))

    return app


# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn
    from ..core.config import load_config

    config = load_config("config.json")
    uvicorn.run(
        "agent_framework.gateway.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True
    )
