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
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, Optional

import httpx

from agent_framework.infrastructure.llm_gateway import LLMGateway, LLMConfig, LLMProvider


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

        try:
            response = await client.post(
                self._completions_endpoint,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

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

        try:
            async with client.stream(
                "POST",
                self._completions_endpoint,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        # Skip empty lines and "data: [DONE]" marker
                        line = line.strip()
                        if not line or line == "data: [DONE]":
                            continue

                        # Remove "data: " prefix if present
                        if line.startswith("data: "):
                            line = line[6:]

                        try:
                            import json
                            data = json.loads(line)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                        except Exception:
                            # Skip malformed JSON lines
                            pass
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