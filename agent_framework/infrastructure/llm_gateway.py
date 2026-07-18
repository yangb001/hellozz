"""LLM Gateway - Abstract interface for Large Language Model providers.

This module defines the abstract interface for LLM providers, supporting
multiple backends (Ollama, OpenAI, Anthropic) with a unified API.

参考：详细设计.md 第9.1节
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, Union, Any


class LLMProvider(str, Enum):
    """Supported LLM provider types."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass(frozen=True)
class LLMConfig:
    """Configuration for an LLM provider instance.

    Attributes:
        provider: The LLM provider type.
        model: The model name/identifier to use.
        api_key: Optional API key for authentication.
        base_url: Optional base URL for the API endpoint.
        max_tokens: Optional maximum tokens for responses.
        temperature: Optional temperature for response generation.
    """
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


from agent_framework.interfaces.llm_types import FunctionCall, ToolCall, ChatResponse as LLMChatResponse


class ChatResponseType(str, Enum):
    """Types of chat response events."""
    CONTENT = "content"           # Text content chunk
    REASONING_CONTENT = "reasoning_content"  # Reasoning/thinking content chunk
    TOOL_CALL_START = "tool_call_start"  # Tool call initiated
    TOOL_CALL_CHUNK = "tool_call_chunk"  # Tool call arguments chunk (legacy name)
    TOOL_CALL_ARGUMENT = "tool_call_argument"  # Tool call arguments chunk (preferred name)
    TOOL_CALL_END = "tool_call_end"      # Tool call completed
    DONE = "done"                 # Stream finished


# Re-export LLMChatResponse as ChatResponse for backward compatibility
ChatResponse = LLMChatResponse


@dataclass
class StreamChatResponse:
    """Represents a chunk in a streaming chat response.

    Attributes:
        type: The type of response (content or tool_call event).
        content: Text content (for CONTENT type).
        tool_call: Tool call data (for TOOL_CALL_* types).
        finish_reason: Why the stream ended (e.g., "stop", "tool_calls").
    """
    type: ChatResponseType = ChatResponseType.CONTENT
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None
    finish_reason: Optional[str] = None


class LLMGateway(ABC):
    """Abstract base class for LLM gateway implementations.

    The LLM Gateway provides a unified interface for interacting with
    multiple LLM providers. It supports both synchronous generation
    and streaming responses.

    Implementations should handle provider-specific initialization
    and provide consistent error handling across providers.
    """

    def __init__(self, configs: Dict[str, LLMConfig]):
        """Initialize the LLM Gateway with provider configurations.

        Args:
            configs: Dictionary mapping model aliases to LLMConfig instances.
                     The "default" key should always be present for the default model.
        """
        self._configs = configs

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str = "default",
        **kwargs
    ) -> str:
        """Generate a complete response from the LLM.

        Args:
            prompt: The input prompt to send to the LLM.
            model: The model alias to use (defaults to "default").
            **kwargs: Additional provider-specific parameters.

        Returns:
            The generated response as a string.

        Raises:
            KeyError: If the specified model alias is not configured.
            RuntimeError: If the LLM request fails.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        model: str = "default",
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            prompt: The input prompt to send to the LLM.
            model: The model alias to use (defaults to "default").
            **kwargs: Additional provider-specific parameters.

        Yields:
            Tokens from the response as they are generated.

        Raises:
            KeyError: If the specified model alias is not configured.
            RuntimeError: If the LLM request fails.
        """
        ...

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
        **kwargs
    ) -> ChatResponse:
        """Generate a chat response using messages array and optional tools.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions for function calling.
            model: Model alias (defaults to "default").
            **kwargs: Additional provider-specific parameters.

        Returns:
            ChatResponse with content and/or tool_calls.

        Raises:
            KeyError: If the specified model alias is not configured.
            RuntimeError: If the LLM request fails.
        """
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
        **kwargs
    ) -> AsyncIterator[StreamChatResponse]:
        """Generate a streaming chat response with tool call support.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions for function calling.
            model: Model alias (defaults to "default").
            **kwargs: Additional provider-specific parameters.

        Yields:
            StreamChatResponse objects representing content chunks,
            tool call events (start/chunk/end), or final DONE event.

        Raises:
            KeyError: If the specified model alias is not configured.
            RuntimeError: If the LLM request fails.
        """
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Args:
            text: The text to count tokens for.

        Returns:
            The estimated token count.
        """
        ...

    def get_config(self, model: str) -> LLMConfig:
        """Get the configuration for a model alias.

        Args:
            model: The model alias to look up.

        Returns:
            The LLMConfig for the specified model alias.

        Raises:
            KeyError: If the model alias is not configured.
        """
        if model not in self._configs:
            raise KeyError(f"Model alias '{model}' is not configured")
        return self._configs[model]

    @property
    def configured_models(self) -> list[str]:
        """List all configured model aliases.

        Returns:
            List of model alias names.
        """
        return list(self._configs.keys())