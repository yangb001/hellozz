"""Ollama LLM - Concrete implementation of LLMGateway for Ollama API.

This module provides a concrete implementation of LLMGateway that integrates
with Ollama's local API (http://localhost:11434 by default).

Ollama API endpoints:
- POST /api/generate - Generate completion (supports streaming)
- POST /api/chat - Chat completion using messages array (supports tools)
- POST /api/embeddings - Get embeddings (used for tokenization fallback)

参考：详细设计.md 第9.1节
"""
import json
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, Optional, List

import httpx

from agent_framework.infrastructure.llm_gateway import (
    LLMGateway, LLMConfig, LLMProvider, ChatResponse, StreamChatResponse,
    ChatResponseType, ToolCall, FunctionCall
)


@dataclass
class OllamaConfig:
    """Configuration for OllamaLLM.

    Attributes:
        model: The Ollama model name (e.g., "llama3", "mistral").
        base_url: Ollama API base URL.
        timeout: HTTP request timeout in seconds.
    """
    model: str
    base_url: str = "http://localhost:11434"
    timeout: float = 120.0


class OllamaLLM(LLMGateway):
    """Ollama implementation of LLMGateway.

    This class provides integration with locally running Ollama models.
    It supports both synchronous generation and streaming responses.

    Example:
        config = OllamaConfig(model="llama3", base_url="http://localhost:11434")
        llm = OllamaLLM(config)

        # Non-streaming
        response = await llm.generate("What is Python?")

        # Streaming
        async for token in llm.stream("What is Python?"):
            print(token, end="", flush=True)
    """

    # API endpoints
    _generate_endpoint: str = "/api/generate"
    _chat_endpoint: str = "/api/chat"
    _tokenize_endpoint: str = "/api/embeddings"

    def __init__(self, config: OllamaConfig):
        """Initialize OllamaLLM with configuration.

        Args:
            config: OllamaConfig instance with model and connection settings.
        """
        self._config = config
        # Convert OllamaConfig to LLMConfig format for parent class
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=config.model,
            base_url=config.base_url,
        )
        super().__init__({"default": llm_config})

        # HTTP client (lazy initialization)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def config(self) -> OllamaConfig:
        """Get the Ollama configuration."""
        return self._config

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client.

        Returns:
            httpx.AsyncClient instance.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout),
            )
        return self._client

    async def _call_ollama_api(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> str:
        """Make a non-streaming call to Ollama API.

        Args:
            prompt: The input prompt.
            stream: Whether to stream the response (False for this method).
            **kwargs: Additional parameters (temperature, num_predict, etc.).

        Returns:
            The generated response text.

        Raises:
            RuntimeError: If the API call fails.
        """
        client = self._get_client()

        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": stream,
        }

        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "num_predict" in kwargs:
            payload["num_predict"] = kwargs["num_predict"]
        if "max_tokens" in kwargs:
            payload["num_predict"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]

        try:
            response = await client.post(
                self._generate_endpoint,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}") from e

    async def _stream_ollama_api(
        self,
        prompt: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Make a streaming call to Ollama API.

        Args:
            prompt: The input prompt.
            **kwargs: Additional parameters.

        Yields:
            Tokens from the response.

        Raises:
            RuntimeError: If the API call fails.
        """
        client = self._get_client()

        payload = {
            "model": self._config.model,
            "prompt": prompt,
            "stream": True,
        }

        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "num_predict" in kwargs:
            payload["num_predict"] = kwargs["num_predict"]

        try:
            async with client.stream(
                "POST",
                self._generate_endpoint,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data and data["response"]:
                                yield data["response"]
                        except Exception:
                            # Skip malformed JSON lines
                            pass
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}") from e

    async def _call_ollama_tokenize(self, text: str) -> int:
        """Call Ollama API to count tokens.

        Note: Ollama doesn't have a dedicated tokenization endpoint.
        We use embeddings API as a fallback, or estimate token count.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated token count.
        """
        client = self._get_client()

        try:
            # Try using the embeddings endpoint
            response = await client.post(
                self._tokenize_endpoint,
                json={
                    "model": self._config.model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            result = response.json()
            # Ollama embeddings response doesn't directly provide token count
            # We need to use a different approach
            raise NotImplementedError("Token count not directly available")
        except Exception:
            # Fallback: estimate based on word count
            # Typical ratio: ~1.3 tokens per word for English
            words = len(text.split())
            return int(words * 1.3) if words > 0 else 0

    async def generate(
        self,
        prompt: str,
        model: str = "default",
        **kwargs
    ) -> str:
        """Generate a complete response from Ollama.

        Args:
            prompt: The input prompt.
            model: Model alias (ignored, uses configured model).
            **kwargs: Additional parameters.

        Returns:
            The generated response text.
        """
        # We ignore model parameter since OllamaLLM uses single model
        return await self._call_ollama_api(prompt, stream=False, **kwargs)

    async def stream(
        self,
        prompt: str,
        model: str = "default",
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming response from Ollama.

        Args:
            prompt: The input prompt.
            model: Model alias (ignored, uses configured model).
            **kwargs: Additional parameters.

        Yields:
            Tokens from the response.
        """
        async for token in self._stream_ollama_api(prompt, **kwargs):
            yield token

    async def count_tokens(self, text: str) -> int:
        """Count tokens in text using Ollama.

        Since Ollama doesn't have a dedicated tokenization API,
        this method uses an approximation based on word count.

        Args:
            text: The text to count tokens for.

        Returns:
            Estimated token count.
        """
        try:
            return await self._call_ollama_tokenize(text)
        except Exception:
            # Fallback estimation
            words = len(text.split())
            return int(words * 1.3) if words > 0 else 0

    async def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
        **kwargs
    ) -> ChatResponse:
        """Generate a chat response using Ollama's /api/chat endpoint.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions for function calling.
            model: Model alias (ignored, uses configured model).
            **kwargs: Additional parameters.

        Returns:
            ChatResponse with content and/or tool_calls.
        """
        client = self._get_client()

        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "num_predict" in kwargs:
            payload["num_predict"] = kwargs["num_predict"]
        if "max_tokens" in kwargs:
            payload["num_predict"] = kwargs["max_tokens"]

        try:
            response = await client.post(
                self._chat_endpoint,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Extract content
            content = result.get("message", {}).get("content", "")

            # Parse raw tool_calls dicts into ToolCall objects
            raw_tool_calls = result.get("tool_calls", None) or result.get("message", {}).get("tool_calls", None)
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
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}") from e

    async def stream_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
        **kwargs
    ) -> AsyncIterator[StreamChatResponse]:
        """Generate a streaming chat response using Ollama's /api/chat endpoint.

        Args:
            messages: List of message dicts with role and content.
            tools: Optional list of tool definitions for function calling.
            model: Model alias (ignored, uses configured model).
            **kwargs: Additional parameters.

        Yields:
            StreamChatResponse objects representing content chunks,
            tool call events, or final DONE event.
        """
        client = self._get_client()

        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
        }

        # Add tools if provided
        if tools:
            payload["tools"] = tools

        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "num_predict" in kwargs:
            payload["num_predict"] = kwargs["num_predict"]

        try:
            async with client.stream(
                "POST",
                self._chat_endpoint,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)

                            # Check for content
                            if "message" in data:
                                msg = data["message"]
                                content = msg.get("content", "")
                                if content:
                                    yield StreamChatResponse(
                                        type=ChatResponseType.CONTENT,
                                        content=content
                                    )

                                # Check for tool call
                                if "tool_calls" in msg:
                                    for tc in msg["tool_calls"]:
                                        tc_id = tc.get("id", "")
                                        func_data = tc.get("function", {})
                                        func_name = func_data.get("name", "")
                                        func_args = func_data.get("arguments", "")

                                        # Yield TOOL_CALL_START first
                                        yield StreamChatResponse(
                                            type=ChatResponseType.TOOL_CALL_START,
                                            tool_call=ToolCall(
                                                id=tc_id,
                                                type="function",
                                                function=FunctionCall(
                                                    name=func_name,
                                                    arguments=func_args
                                                )
                                            )
                                        )

                                        # Then yield TOOL_CALL_END
                                        yield StreamChatResponse(
                                            type=ChatResponseType.TOOL_CALL_END,
                                            tool_call=ToolCall(
                                                id=tc_id,
                                                type="function",
                                                function=FunctionCall(
                                                    name=func_name,
                                                    arguments=func_args
                                                )
                                            )
                                        )

                            # Check for done/summary
                            if data.get("done"):
                                yield StreamChatResponse(
                                    type=ChatResponseType.DONE,
                                    finish_reason=data.get("done_reason")
                                )
                        except Exception:
                            # Skip malformed JSON lines
                            pass
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Ollama API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to connect to Ollama: {e}") from e

    async def close(self) -> None:
        """Close the HTTP client.

        Should be called when the gateway is no longer needed.
        """
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "OllamaLLM":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
