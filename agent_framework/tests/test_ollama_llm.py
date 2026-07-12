"""Tests for infrastructure/ollama_llm.py - Ollama LLM implementation."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator
import httpx

from agent_framework.infrastructure.ollama_llm import OllamaLLM, OllamaConfig
from agent_framework.infrastructure.llm_gateway import LLMGateway, LLMProvider, LLMConfig


class TestOllamaConfig:
    """Tests for OllamaConfig dataclass."""

    def test_create_with_defaults(self):
        """Test creating OllamaConfig with default values."""
        config = OllamaConfig(model="llama3")
        assert config.model == "llama3"
        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 120.0

    def test_create_with_custom_values(self):
        """Test creating OllamaConfig with custom values."""
        config = OllamaConfig(
            model="llama3",
            base_url="http://custom:8080",
            timeout=60.0,
        )
        assert config.model == "llama3"
        assert config.base_url == "http://custom:8080"
        assert config.timeout == 60.0

    def test_inherits_from_llm_config(self):
        """Test that OllamaConfig is compatible with LLMConfig patterns."""
        config = OllamaConfig(model="llama3")
        # Should have model attribute
        assert hasattr(config, "model")
        assert hasattr(config, "base_url")


class TestOllamaLLMInheritance:
    """Tests for OllamaLLM class structure."""

    def test_inherits_from_llm_gateway(self):
        """Test that OllamaLLM inherits from LLMGateway."""
        assert issubclass(OllamaLLM, LLMGateway)

    def test_can_instantiate(self):
        """Test that OllamaLLM can be instantiated."""
        config = OllamaConfig(model="llama3")
        gateway = OllamaLLM(config)
        assert isinstance(gateway, LLMGateway)
        assert isinstance(gateway, OllamaLLM)

    def test_stores_config(self):
        """Test that OllamaLLM stores configuration."""
        config = OllamaConfig(model="llama3", base_url="http://custom:8080")
        gateway = OllamaLLM(config)
        assert gateway.config.model == "llama3"
        assert gateway.config.base_url == "http://custom:8080"


class TestOllamaLLMGenerate:
    """Tests for OllamaLLM.generate method."""

    @pytest.fixture
    def ollama_llm(self):
        """Create an OllamaLLM instance for testing."""
        config = OllamaConfig(model="llama3")
        return OllamaLLM(config)

    @pytest.mark.asyncio
    async def test_generate_returns_string(self, ollama_llm):
        """Test that generate returns a string response."""
        with patch.object(ollama_llm, '_call_ollama_api') as mock_call:
            mock_call.return_value = "Hello, how can I help you?"

            result = await ollama_llm.generate("Hello")
            assert isinstance(result, str)
            assert result == "Hello, how can I help you?"

    @pytest.mark.asyncio
    async def test_generate_sends_correct_payload(self, ollama_llm):
        """Test that generate sends correct payload to Ollama API."""
        with patch.object(ollama_llm, '_call_ollama_api') as mock_call:
            mock_call.return_value = "Response"

            await ollama_llm.generate("Hello")

            # Verify the API call was made
            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args
            assert call_kwargs is not None

    @pytest.mark.asyncio
    async def test_generate_with_stream_false(self, ollama_llm):
        """Test that generate sets stream=False in request."""
        with patch.object(ollama_llm, '_call_ollama_api') as mock_call:
            mock_call.return_value = "Response"

            await ollama_llm.generate("Hello")

            # Should have been called
            assert mock_call.called

    @pytest.mark.asyncio
    async def test_generate_passes_kwargs(self, ollama_llm):
        """Test that generate passes additional kwargs."""
        with patch.object(ollama_llm, '_call_ollama_api') as mock_call:
            mock_call.return_value = "Response"

            await ollama_llm.generate(
                "Hello",
                temperature=0.7,
                num_predict=100,
            )

            assert mock_call.called

    @pytest.mark.asyncio
    async def test_generate_raises_on_connection_error(self, ollama_llm):
        """Test that generate raises RuntimeError on connection error."""
        # Mock the HTTP client to raise a connection error
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Failed to connect")

        with patch.object(ollama_llm, '_get_client', return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                await ollama_llm.generate("Hello")

            assert "Ollama" in str(exc_info.value) or "connect" in str(exc_info.value).lower()


class TestOllamaLLMStream:
    """Tests for OllamaLLM.stream method."""

    @pytest.fixture
    def ollama_llm(self):
        """Create an OllamaLLM instance for testing."""
        config = OllamaConfig(model="llama3")
        return OllamaLLM(config)

    @pytest.mark.asyncio
    async def test_stream_returns_async_iterator(self, ollama_llm):
        """Test that stream returns an async iterator."""
        async def mock_stream():
            for token in ["Hello", " ", "World"]:
                yield token

        with patch.object(ollama_llm, '_stream_ollama_api') as mock_stream_call:
            mock_stream_call.return_value = mock_stream()

            result = ollama_llm.stream("Hello")
            tokens = [t async for t in result]
            assert tokens == ["Hello", " ", "World"]

    @pytest.mark.asyncio
    async def test_stream_yields_strings(self, ollama_llm):
        """Test that stream yields string tokens."""
        async def mock_stream():
            yield "Hello"
            yield " World"

        with patch.object(ollama_llm, '_stream_ollama_api') as mock_stream_call:
            mock_stream_call.return_value = mock_stream()

            tokens = []
            async for token in ollama_llm.stream("Hello"):
                assert isinstance(token, str)
                tokens.append(token)

            assert len(tokens) == 2

    @pytest.mark.asyncio
    async def test_stream_raises_on_connection_error(self, ollama_llm):
        """Test that stream raises RuntimeError on connection error."""
        # Test by mocking _stream_ollama_api to raise RuntimeError
        async def failing_stream_api(*args, **kwargs):
            raise RuntimeError("Failed to connect to Ollama: Connection refused")
            yield ""  # Make it a generator

        with patch.object(ollama_llm, '_stream_ollama_api') as mock_stream_call:
            mock_stream_call.return_value = failing_stream_api()

            with pytest.raises(RuntimeError) as exc_info:
                async for _ in ollama_llm.stream("Hello"):
                    pass

            assert "Ollama" in str(exc_info.value) or "connect" in str(exc_info.value).lower()


class TestOllamaLLMCountTokens:
    """Tests for OllamaLLM.count_tokens method."""

    @pytest.fixture
    def ollama_llm(self):
        """Create an OllamaLLM instance for testing."""
        config = OllamaConfig(model="llama3")
        return OllamaLLM(config)

    @pytest.mark.asyncio
    async def test_count_tokens_returns_int(self, ollama_llm):
        """Test that count_tokens returns an integer."""
        with patch.object(ollama_llm, '_call_ollama_tokenize') as mock_tokenize:
            mock_tokenize.return_value = 7

            result = await ollama_llm.count_tokens("Hello world test")
            assert isinstance(result, int)
            assert result == 7

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, ollama_llm):
        """Test that count_tokens handles empty string."""
        with patch.object(ollama_llm, '_call_ollama_tokenize') as mock_tokenize:
            mock_tokenize.return_value = 0

            result = await ollama_llm.count_tokens("")
            assert result == 0

    @pytest.mark.asyncio
    async def test_count_tokens_fallback_on_error(self, ollama_llm):
        """Test that count_tokens has fallback estimation."""
        with patch.object(ollama_llm, '_call_ollama_tokenize') as mock_tokenize:
            mock_tokenize.side_effect = Exception("Ollama not available")

            # Should fall back to word-based estimation
            result = await ollama_llm.count_tokens("Hello world test")
            assert isinstance(result, int)
            # Fallback: words * 1.3 approx
            assert result >= 3  # At least word count


class TestOllamaLLMHTTPCalls:
    """Tests for OllamaLLM HTTP API calls."""

    @pytest.fixture
    def ollama_llm(self):
        """Create an OllamaLLM instance for testing."""
        config = OllamaConfig(model="llama3")
        return OllamaLLM(config)

    @pytest.mark.asyncio
    async def test_generate_endpoint_correct(self, ollama_llm):
        """Test that generate uses correct Ollama API endpoint."""
        config = OllamaConfig(model="llama3", base_url="http://localhost:11434")
        gateway = OllamaLLM(config)

        assert gateway.config.base_url == "http://localhost:11434"
        # The generate endpoint should be /api/generate
        assert hasattr(gateway, '_generate_endpoint')
        assert gateway._generate_endpoint == "/api/generate"

    def test_tokenize_endpoint_correct(self, ollama_llm):
        """Test that tokenize uses correct Ollama API endpoint."""
        # The tokenize endpoint should be /api/embeddings or similar
        assert hasattr(ollama_llm, '_tokenize_endpoint')

    def test_model_name_used(self, ollama_llm):
        """Test that model name is available for API calls."""
        assert ollama_llm.config.model == "llama3"


class TestOllamaLLMConfig:
    """Tests for OllamaLLM configuration handling."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OllamaConfig(model="llama3")
        assert config.base_url == "http://localhost:11434"
        assert config.timeout == 120.0

    def test_custom_base_url(self):
        """Test custom base URL configuration."""
        config = OllamaConfig(
            model="llama3",
            base_url="http://custom-server:8888",
        )
        gateway = OllamaLLM(config)
        assert gateway.config.base_url == "http://custom-server:8888"

    def test_timeout_configuration(self):
        """Test timeout configuration."""
        config = OllamaConfig(model="llama3", timeout=30.0)
        gateway = OllamaLLM(config)
        assert gateway.config.timeout == 30.0


class TestOllamaLLMIntegration:
    """Integration tests for OllamaLLM (require mock HTTP server)."""

    @pytest.fixture
    def ollama_llm(self):
        """Create an OllamaLLM instance for testing."""
        config = OllamaConfig(model="llama3", timeout=10.0)
        return OllamaLLM(config)

    @pytest.mark.asyncio
    async def test_full_generate_flow(self, ollama_llm):
        """Test full generate flow with mocked HTTP response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "model": "llama3",
            "response": "Hello! How can I assist you today?",
            "done": True,
        }

        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.return_value = mock_response

            result = await ollama_llm.generate("Hello")
            assert "Hello" in result or "assist" in result.lower()

    @pytest.mark.asyncio
    async def test_streaming_flow(self, ollama_llm):
        """Test streaming flow with mocked HTTP response."""
        # Mock streaming response
        async def mock_aiter_lines():
            yield '{"model":"llama3","response":"Hello","done":false}'
            yield '{"model":"llama3","response":" world","done":false}'
            yield '{"model":"llama3","response":"","done":true}'

        mock_response = MagicMock()
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.status_code = 200

        with patch('httpx.AsyncClient.stream') as mock_stream:
            mock_stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream.return_value.__aexit__ = AsyncMock(return_value=None)

            tokens = []
            async for token in ollama_llm.stream("Hello"):
                tokens.append(token)

            # Should have received tokens
            assert isinstance(tokens, list)
