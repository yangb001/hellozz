"""Tests for OpenAI-compatible LLM Gateway - TDD implementation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator

import httpx

from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
from agent_framework.infrastructure.llm_gateway import LLMConfig, LLMProvider


class TestOpenAIConfig:
    """Test OpenAIConfig data class."""

    def test_openai_config_creation(self):
        """Test creating an OpenAIConfig instance."""
        config = OpenAIConfig(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            temperature=0.7,
            max_tokens=1000
        )
        assert config.model == "gpt-4"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000

    def test_openai_config_defaults(self):
        """Test OpenAIConfig with default values."""
        config = OpenAIConfig(model="gpt-3.5-turbo")
        assert config.model == "gpt-3.5-turbo"
        assert config.base_url == "https://api.openai.com/v1"
        assert config.api_key is None
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.timeout == 120.0

    def test_openai_config_custom_values(self):
        """Test OpenAIConfig with custom values."""
        config = OpenAIConfig(
            model="mimo-7b",
            base_url="https://api.mimo.ai/v1",
            api_key="mimo-key",
            temperature=0.5,
            max_tokens=500,
            timeout=60.0
        )
        assert config.model == "mimo-7b"
        assert config.base_url == "https://api.mimo.ai/v1"
        assert config.timeout == 60.0


class TestOpenAILLMInitialization:
    """Test OpenAILLM initialization."""

    def test_openai_llm_initialization(self):
        """Test OpenAILLM initialization with config."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        assert llm.config == config
        assert llm._client is None  # Lazy initialization

    def test_openai_llm_creates_llm_config(self):
        """Test that OpenAILLM creates proper LLMConfig for parent."""
        config = OpenAIConfig(
            model="gpt-4",
            base_url="https://api.openai.com/v1",
            api_key="test-key"
        )
        llm = OpenAILLM(config)

        # Should have a default model configured
        assert "default" in llm._configs
        default_config = llm._configs["default"]
        assert default_config.provider == LLMProvider.OPENAI
        assert default_config.model == "gpt-4"
        assert default_config.base_url == "https://api.openai.com/v1"

    def test_openai_llm_client_lazy_initialization(self):
        """Test that HTTP client is created lazily."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        # Client should be None initially
        assert llm._client is None

        # Getting client should create it
        client = llm._get_client()
        assert client is not None
        assert llm._client is client

    def test_openai_llm_client_reuse(self):
        """Test that HTTP client is reused."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        client1 = llm._get_client()
        client2 = llm._get_client()

        assert client1 is client2


class TestOpenAILLMGenerate:
    """Test OpenAILLM generate method."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock OpenAI API response."""
        mock = MagicMock()
        mock.status_code = 200
        mock.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        mock.raise_for_status = MagicMock()
        return mock

    @pytest.mark.asyncio
    async def test_generate_success(self, mock_response):
        """Test successful generation."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await llm.generate("Hello, how are you?")

            assert result == "Hello! How can I help you today?"
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, mock_response):
        """Test generation with custom parameters."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            result = await llm.generate(
                "Hello",
                temperature=0.5,
                max_tokens=100
            )

            # Verify the request was made with correct parameters
            call_args = mock_client.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            assert payload["temperature"] == 0.5
            assert payload["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_generate_api_error(self):
        """Test generation with API error."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_response.text = "Unauthorized"
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Unauthorized",
                request=MagicMock(),
                response=mock_response
            )
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate("Hello")
            assert "OpenAI API error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_connection_error(self):
        """Test generation with connection error."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
            mock_get_client.return_value = mock_client

            with pytest.raises(RuntimeError) as exc_info:
                await llm.generate("Hello")
            assert "Failed to connect to OpenAI" in str(exc_info.value)


class TestOpenAILLMStream:
    """Test OpenAILLM stream method."""

    @pytest.fixture
    def mock_stream_response(self):
        """Create a mock streaming response."""
        mock = AsyncMock()
        mock.status_code = 200
        mock.raise_for_status = MagicMock()

        # Simulate streaming chunks
        chunks = [
            '{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',
            '{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
            '{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}',
            '{"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        ]

        async def mock_aiter_lines():
            for chunk in chunks:
                yield chunk

        mock.aiter_lines = mock_aiter_lines
        return mock

    @pytest.mark.asyncio
    async def test_stream_success(self, mock_stream_response):
        """Test successful streaming."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=mock_context)
            mock_get_client.return_value = mock_client

            tokens = []
            async for token in llm.stream("Hello"):
                tokens.append(token)

            assert len(tokens) == 2
            assert tokens[0] == "Hello"
            assert tokens[1] == "!"

    @pytest.mark.asyncio
    async def test_stream_with_parameters(self, mock_stream_response):
        """Test streaming with custom parameters."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=mock_context)
            mock_get_client.return_value = mock_client

            tokens = []
            async for token in llm.stream("Hello", temperature=0.3):
                tokens.append(token)

            # Verify parameters were passed
            call_args = mock_client.stream.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][2]
            assert payload["temperature"] == 0.3

    @pytest.mark.asyncio
    async def test_stream_api_error(self):
        """Test streaming with API error."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Rate limited",
                request=MagicMock(),
                response=mock_response
            )
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=mock_context)
            mock_get_client.return_value = mock_client

            with pytest.raises(RuntimeError) as exc_info:
                async for _ in llm.stream("Hello"):
                    pass
            assert "OpenAI API error" in str(exc_info.value)


class TestOpenAILLMCountTokens:
    """Test OpenAILLM count_tokens method."""

    @pytest.mark.asyncio
    async def test_count_tokens_estimation(self):
        """Test token count estimation."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        # Test with simple text
        count = await llm.count_tokens("Hello world")
        assert count > 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_count_tokens_empty_text(self):
        """Test token count for empty text."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        count = await llm.count_tokens("")
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_tokens_long_text(self):
        """Test token count for longer text."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        text = "This is a longer text with multiple words to test the token estimation."
        count = await llm.count_tokens(text)
        assert count > 5  # Should be more than 5 tokens for this text


class TestOpenAILLMClose:
    """Test OpenAILLM close method."""

    @pytest.mark.asyncio
    async def test_close_client(self):
        """Test closing the HTTP client."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        # Create client first
        client = llm._get_client()
        assert client is not None

        # Close the client
        await llm.close()

        # Client should be closed
        assert llm._client is None or llm._client.is_closed

    @pytest.mark.asyncio
    async def test_close_no_client(self):
        """Test closing when no client exists."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        # Should not raise error
        await llm.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test using OpenAILLM as async context manager."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")

        async with OpenAILLM(config) as llm:
            assert llm is not None
            client = llm._get_client()
            assert client is not None

        # After exiting context, client should be closed


class TestOpenAILLMIntegration:
    """Integration tests for OpenAILLM."""

    @pytest.mark.asyncio
    async def test_generate_and_stream_consistency(self):
        """Test that generate and stream produce consistent results."""
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)

        # Mock responses for both methods
        mock_generate_response = MagicMock()
        mock_generate_response.status_code = 200
        mock_generate_response.json.return_value = {
            "choices": [{"message": {"content": "Hello!"}}]
        }
        mock_generate_response.raise_for_status = MagicMock()

        mock_stream_response = AsyncMock()
        mock_stream_response.status_code = 200
        mock_stream_response.raise_for_status = MagicMock()

        chunks = [
            '{"choices":[{"delta":{"content":"Hello!"},"finish_reason":null}]}',
            '{"choices":[{"delta":{},"finish_reason":"stop"}]}'
        ]

        async def mock_aiter_lines():
            for chunk in chunks:
                yield chunk

        mock_stream_response.aiter_lines = mock_aiter_lines

        with patch.object(llm, '_get_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_generate_response)

            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_stream_response)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            mock_client.stream = MagicMock(return_value=mock_context)

            mock_get_client.return_value = mock_client

            # Test generate
            generate_result = await llm.generate("Hello")

            # Test stream
            stream_tokens = []
            async for token in llm.stream("Hello"):
                stream_tokens.append(token)

            # Both should produce "Hello!"
            assert generate_result == "Hello!"
            assert "".join(stream_tokens) == "Hello!"

    @pytest.mark.asyncio
    async def test_multiple_models_support(self):
        """Test that different models can be configured."""
        config1 = OpenAIConfig(model="gpt-4", api_key="key1")
        config2 = OpenAIConfig(model="gpt-3.5-turbo", api_key="key2")

        llm1 = OpenAILLM(config1)
        llm2 = OpenAILLM(config2)

        assert llm1.config.model == "gpt-4"
        assert llm2.config.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_custom_base_url(self):
        """Test custom base URL configuration."""
        config = OpenAIConfig(
            model="mimo-7b",
            base_url="https://api.mimo.ai/v1",
            api_key="mimo-key"
        )
        llm = OpenAILLM(config)

        assert llm.config.base_url == "https://api.mimo.ai/v1"

        # Client should be configured with custom URL
        client = llm._get_client()
        assert client is not None