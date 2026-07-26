"""OpenAI-compatible LLM - Concrete implementation of LLMGateway for OpenAI-compatible APIs.

This module provides a concrete implementation of LLMGateway that integrates
with OpenAI-compatible APIs, including:
- OpenAI (https://api.openai.com/v1)
- Mimo (https://api.mimo.ai/v1)
- Other OpenAI-compatible endpoints

API endpoints:
- POST /chat/completions - Generate chat completion (supports streaming)

参考：详细设计.md 第9.1节
"""
import json
import logging
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, List, Any, Optional

import httpx

from agent_framework.infrastructure.llm_gateway import (
    LLMGateway, LLMConfig, LLMProvider,
    ChatResponse, StreamChatResponse, ChatResponseType,
    ToolCall, FunctionCall
)

# Module logger for debugging
_logger = logging.getLogger("agent_framework.infrastructure.llm_debug")


@dataclass
class OpenAIConfig:
    """Configuration for OpenAILLM.

    Attributes:
        model: The model name (e.g., "gpt-4", "gpt-3.5-turbo", "mimo-7b").
        base_url: API base URL.
        api_key: API key for authentication.
        temperature: Sampling temperature (0.0 to 2.0).
        max_tokens: Maximum tokens to generate.
        timeout: HTTP request timeout in seconds.
        verify_ssl: Whether to verify SSL certificates. Defaults to True.
                   Set to False for local/self-hosted endpoints (e.g., Ollama).
    """
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: float = 120.0
    verify_ssl: bool = True


class OpenAILLM(LLMGateway):
    """OpenAI-compatible implementation of LLMGateway.

    This class provides integration with OpenAI-compatible APIs.
    It supports both synchronous generation and streaming responses.

    Example:
        config = OpenAIConfig(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="your-api-key"
        )
        llm = OpenAILLM(config)

        # Non-streaming
        response = await llm.generate("What is Python?")

        # Streaming
        async for token in llm.stream("What is Python?"):
            print(token, end="", flush=True)
    """

    # API endpoint
    _completions_endpoint: str = "/chat/completions"

    def __init__(self, config: OpenAIConfig):
        """Initialize OpenAILLM with configuration.

        Args:
            config: OpenAIConfig instance with model and connection settings.
        """
        self._config = config
        # Convert OpenAIConfig to LLMConfig format for parent class
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        super().__init__({"default": llm_config})

        # HTTP client (lazy initialization)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def config(self) -> OpenAIConfig:
        """Get the OpenAI configuration."""
        return self._config

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            headers = {
                "Content-Type": "application/json",
            }
            if self._config.api_key:
                headers["Authorization"] = f"Bearer {self._config.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                headers=headers,
                timeout=httpx.Timeout(self._config.timeout),
                verify=self._config.verify_ssl,
            )
        return self._client

    def _apply_optional_params(self, payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Apply optional parameters (temperature, max_tokens, etc.) to a payload dict.

        Merges config defaults with per-request kwargs overrides.

        Args:
            payload: The base payload dict to augment.
            **kwargs: Per-request parameter overrides.

        Returns:
            The augmented payload dict (mutated in place and returned).
        """
        # Add temperature (from config or kwargs)
        temperature = kwargs.get("temperature", self._config.temperature)
        if temperature is not None:
            payload["temperature"] = temperature

        # Add max_tokens (from config or kwargs)
        max_tokens = kwargs.get("max_tokens", self._config.max_tokens)
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Add other optional parameters
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            payload["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            payload["presence_penalty"] = kwargs["presence_penalty"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        return payload

    def _build_payload(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build the request payload for OpenAI API.

        Args:
            prompt: The input prompt.
            stream: Whether to stream the response.
            **kwargs: Additional parameters.

        Returns:
            Dictionary payload for the API request.
        """
        payload = {
            "model": self._config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        return self._apply_optional_params(payload, **kwargs)

    def _build_chat_payload(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Build the request payload for chat completions API.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions.
            stream: Whether to stream the response.
            **kwargs: Additional parameters.

        Returns:
            Dictionary payload for the API request.
        """
        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": stream,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        return self._apply_optional_params(payload, **kwargs)

    async def _call_openai_api(
        self,
        prompt: str,
        **kwargs
    ) -> str:
        """Make a non-streaming call to OpenAI API.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters.

        Returns:
            The generated response text.

        Raises:
            RuntimeError: If the API call fails.
        """
        client = self._get_client()
        payload = self._build_payload(prompt, stream=False, **kwargs)

        # Debug logging for LLM request
        _logger.debug(f"LLM Request | model={self._config.model} | payload={payload}")

        try:
            response = await client.post(
                self._completions_endpoint,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Debug logging for LLM response
            _logger.debug(f"LLM Response | model={self._config.model} | result={result}")

            # Extract content from OpenAI response format
            choices = result.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
            return ""
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"OpenAI API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to OpenAI: {e}") from e

    async def _call_chat_api(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> ChatResponse:
        """Make a non-streaming chat API call.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters.

        Returns:
            ChatResponse with content and/or tool_calls.

        Raises:
            RuntimeError: If the API call fails.
        """
        client = self._get_client()
        payload = self._build_chat_payload(messages, tools, stream=False, **kwargs)

        # Debug logging for LLM request
        _logger.debug(f"LLM Chat Request | model={self._config.model} | payload={payload}")

        try:
            response = await client.post(
                self._completions_endpoint,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Debug logging for LLM response
            _logger.debug(f"LLM Chat Response | model={self._config.model} | result={result}")

            # Extract content and tool_calls from response
            choices = result.get("choices", [])
            if not choices:
                return ChatResponse(content="")

            choice = choices[0]
            message = choice.get("message", {})

            content = message.get("content", "")

            # Parse raw tool_calls dicts into ToolCall objects
            raw_tool_calls = message.get("tool_calls", None)
            tool_calls = None
            if raw_tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.get("id", ""),
                        type=tc.get("type", "function"),
                        function=FunctionCall(
                            name=tc.get("function", {}).get("name", ""),
                            arguments=tc.get("function", {}).get("arguments", "")
                        )
                    )
                    for tc in raw_tool_calls
                ]

            return ChatResponse(content=content, tool_calls=tool_calls)

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"OpenAI API error: {e.response.status_code} - {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to OpenAI: {e}") from e

    async def _stream_openai_api(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Make a streaming call to OpenAI API.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters.

        Yields:
            Tokens from the response.

        Raises:
            RuntimeError: If the API call fails.
        """
        client = self._get_client()
        payload = self._build_payload(prompt, stream=True, **kwargs)

        # Debug logging for LLM streaming request
        _logger.debug(f"LLM Stream Request | model={self._config.model} | payload={payload}")

        try:
            async with client.stream(
                "POST",
                self._completions_endpoint,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        _logger.debug(f"LLM Stream Raw | line={repr(line)}")

                        # Skip empty lines and "data: [DONE]" marker (with or without space)
                        line = line.strip()
                        if not line or line == "data: [DONE]" or line == "data:[DONE]":
                            continue

                        # Remove "data:" or "data: " prefix if present
                        if line.startswith("data:"):
                            if line.startswith("data: "):
                                line = line[6:]
                            else:
                                line = line[5:]

                        try:
                            # DEBUG: Log raw line before parsing
                            data = json.loads(line)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    # Debug logging for each streaming chunk/token
                                    _logger.debug(f"LLM Stream Chunk | model={self._config.model} | token={content}")
                                    yield content
                        except Exception as e:
                            # Log malformed JSON lines - show repr to reveal invisible chars
                            _logger.warning(f"Failed to parse streaming response line: repr={repr(line)} | Length={len(line)} | Error: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"OpenAI API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to OpenAI: {e}") from e

    async def _stream_chat_api(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> AsyncIterator[StreamChatResponse]:
        """Make a streaming chat API call with tool call support.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions.
            **kwargs: Additional parameters.

        Yields:
            StreamChatResponse objects.

        Raises:
            RuntimeError: If the API call fails.
        """
        client = self._get_client()
        payload = self._build_chat_payload(messages, tools, stream=True, **kwargs)

        # Debug logging for LLM streaming request
        _logger.debug(f"LLM Stream Chat Request | model={self._config.model} | payload={payload}")

        # Track tool calls for accumulation
        tool_call_tracker: Dict[int, Dict[str, Any]] = {}

        try:
            async with client.stream(
                "POST",
                self._completions_endpoint,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        _logger.debug(f"LLM Stream Raw | line={repr(line)}")

                        # Skip empty lines and "data: [DONE]" marker (with or without space)
                        line = line.strip()
                        if not line or line == "data: [DONE]" or line == "data:[DONE]":
                            continue

                        # Remove "data:" or "data: " prefix if present
                        if line.startswith("data:"):
                            if line.startswith("data: "):
                                line = line[6:]
                            else:
                                line = line[5:]

                        try:
                            data = json.loads(line)
                            choices = data.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            finish_reason = choices[0].get("finish_reason")

                            # Yield thinking content FIRST (兼容 MiniMax: reasoning_content, OpenAI: thinking_content)
                            # 思考内容优先于普通content，因为MiniMax同一次返回中可能两者都有
                            thinking = delta.get("reasoning_content") or delta.get("thinking_content")
                            if thinking:
                                yield StreamChatResponse(
                                    type=ChatResponseType.THINKING_CONTENT,
                                    content=thinking
                                )
                            # Yield content tokens only if no thinking content
                            # (避免同一个chunk同时发送text_token和thinking_content导致重复)
                            elif delta.get("content"):
                                yield StreamChatResponse(
                                    type=ChatResponseType.CONTENT,
                                    content=delta["content"]
                                )

                            # Process tool_calls
                            if delta.get("tool_calls"):
                                for tc_data in delta["tool_calls"]:
                                    index = tc_data.get("index", 0)
                                    tc_id = tc_data.get("id")
                                    tc_type = tc_data.get("type", "function")
                                    func_data = tc_data.get("function", {})
                                    func_name = func_data.get("name", "")
                                    func_args = func_data.get("arguments", "")

                                    if index not in tool_call_tracker:
                                        # New tool call
                                        tool_call_tracker[index] = {
                                            "id": tc_id,
                                            "type": tc_type,
                                            "name": func_name,
                                            "arguments": func_args
                                        }
                                        yield StreamChatResponse(
                                            type=ChatResponseType.TOOL_CALL_START,
                                            tool_call=ToolCall(
                                                id=tc_id or f"call_{index}",
                                                type=tc_type,
                                                function=FunctionCall(
                                                    name=func_name,
                                                    arguments=func_args
                                                )
                                            )
                                        )
                                    else:
                                        # Appending to existing tool call
                                        tool_call_tracker[index]["arguments"] += func_args
                                        current_tc = tool_call_tracker[index]
                                        yield StreamChatResponse(
                                            type=ChatResponseType.TOOL_CALL_ARGUMENT,
                                            tool_call=ToolCall(
                                                id=current_tc["id"],
                                                type=current_tc["type"],
                                                function=FunctionCall(
                                                    name=current_tc["name"],
                                                    arguments=current_tc["arguments"]
                                                )
                                            )
                                        )

                            # Yield tool_call_end when tool calls are finished, then done
                            if finish_reason == "tool_calls":
                                # Handle edge case: MiniMax LLM may return finish_reason=tool_calls
                                # but with empty delta.tool_calls, so tracker is empty
                                if not tool_call_tracker:
                                    _logger.warning(
                                        f"LLM returned finish_reason=tool_calls but no tool_calls in delta. "
                                        f"This may be a malformed response from the LLM."
                                    )
                                else:
                                    # Yield TOOL_CALL_END for each tool call
                                    for tc_data in tool_call_tracker.values():
                                        yield StreamChatResponse(
                                            type=ChatResponseType.TOOL_CALL_END,
                                            tool_call=ToolCall(
                                                id=tc_data["id"],
                                                type=tc_data["type"],
                                                function=FunctionCall(
                                                    name=tc_data["name"],
                                                    arguments=tc_data["arguments"]
                                                )
                                            )
                                        )
                                yield StreamChatResponse(
                                    type=ChatResponseType.DONE,
                                    finish_reason=finish_reason
                                )
                            elif finish_reason:
                                yield StreamChatResponse(
                                    type=ChatResponseType.DONE,
                                    finish_reason=finish_reason
                                )

                        except json.JSONDecodeError as e:
                            _logger.warning(f"Failed to parse streaming response line: repr={repr(line)} | Error: {e}")

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"OpenAI API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to OpenAI: {e}") from e

    async def generate(
        self,
        prompt: str,
        model: str = "default",
        **kwargs
    ) -> str:
        """Generate a complete response from OpenAI-compatible API.

        Args:
            prompt: The input prompt.
            model: Model alias (ignored, uses configured model).
            **kwargs: Additional parameters.

        Returns:
            The generated response text.
        """
        return await self._call_openai_api(prompt, **kwargs)

    async def stream(
        self,
        prompt: str,
        model: str = "default",
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response from OpenAI-compatible API.

        Args:
            prompt: The input prompt.
            model: Model alias (ignored, uses configured model).
            **kwargs: Additional parameters.

        Yields:
            Tokens from the response.
        """
        async for token in self._stream_openai_api(prompt, **kwargs):
            yield token

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
        return await self._call_chat_api(messages, tools, **kwargs)

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
        async for response in self._stream_chat_api(messages, tools, **kwargs):
            yield response

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Since OpenAI doesn't provide a tokenization endpoint,
        this method uses an approximation based on word count.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        # Approximation: ~1.3 tokens per word for English
        # This is a rough estimate; for precise counting, use tiktoken library
        words = len(text.split())
        return int(words * 1.3) if words > 0 else 0

    async def close(self) -> None:
        """Close the HTTP client.

        Should be called when the gateway is no longer needed.
        """
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "OpenAILLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()