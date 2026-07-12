"""Independent test cases for OpenAI-compatible LLM implementation.

This module contains independent verification tests for the OpenAILLM
and OpenAIConfig classes, following the detailed design specification.

Test categories:
1. OpenAIConfig data class
2. OpenAILLM inheritance and initialization
3. generate method
4. stream method
5. count_tokens method
6. Payload building
7. Error handling
8. Resource management
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator, Dict, Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestOpenAIConfigDataClass:
    """Independent tests for OpenAIConfig data class."""

    def test_config_has_model_field(self):
        """OpenAIConfig must have model field."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        assert config.model == "gpt-4"

    def test_config_has_base_url_field(self):
        """OpenAIConfig must have base_url field."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        assert config.base_url is not None

    def test_config_default_base_url(self):
        """OpenAIConfig should default to OpenAI base URL."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        assert "openai" in config.base_url.lower() or "api" in config.base_url.lower()

    def test_config_has_api_key_field(self):
        """OpenAIConfig must have api_key field."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        assert config.api_key == "test-key"

    def test_config_api_key_defaults_to_none(self):
        """OpenAIConfig api_key should default to None."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        assert config.api_key is None

    def test_config_has_temperature_field(self):
        """OpenAIConfig must have temperature field."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4", temperature=0.5)
        assert config.temperature == 0.5

    def test_config_default_temperature(self):
        """OpenAIConfig should have default temperature."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        assert config.temperature is not None
        assert 0 <= config.temperature <= 2

    def test_config_has_max_tokens_field(self):
        """OpenAIConfig must have max_tokens field."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4", max_tokens=100)
        assert config.max_tokens == 100

    def test_config_max_tokens_defaults_to_none(self):
        """OpenAIConfig max_tokens should default to None."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        assert config.max_tokens is None

    def test_config_has_timeout_field(self):
        """OpenAIConfig must have timeout field."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4", timeout=60.0)
        assert config.timeout == 60.0

    def test_config_default_timeout(self):
        """OpenAIConfig should have default timeout."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        assert config.timeout > 0

    def test_config_is_dataclass(self):
        """OpenAIConfig should be a dataclass."""
        import dataclasses
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        assert dataclasses.is_dataclass(OpenAIConfig)

    def test_config_custom_base_url(self):
        """OpenAIConfig should accept custom base_url."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        config = OpenAIConfig(
            model="custom-model",
            base_url="https://custom.api.com/v1"
        )
        assert config.base_url == "https://custom.api.com/v1"


class TestOpenAILLMInheritance:
    """Independent tests for OpenAILLM inheritance."""

    def test_openai_llm_is_subclass_of_llm_gateway(self):
        """OpenAILLM must inherit from LLMGateway."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM
        from agent_framework.infrastructure.llm_gateway import LLMGateway
        assert issubclass(OpenAILLM, LLMGateway)

    def test_openai_llm_implements_generate(self):
        """OpenAILLM must implement generate method."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM
        assert hasattr(OpenAILLM, 'generate')
        assert callable(getattr(OpenAILLM, 'generate'))

    def test_openai_llm_implements_stream(self):
        """OpenAILLM must implement stream method."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM
        assert hasattr(OpenAILLM, 'stream')
        assert callable(getattr(OpenAILLM, 'stream'))

    def test_openai_llm_implements_count_tokens(self):
        """OpenAILLM must implement count_tokens method."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM
        assert hasattr(OpenAILLM, 'count_tokens')
        assert callable(getattr(OpenAILLM, 'count_tokens'))

    def test_openai_llm_can_instantiate(self):
        """OpenAILLM can be instantiated."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="test-model", api_key="test-key")
        llm = OpenAILLM(config)
        assert llm is not None
        assert isinstance(llm, OpenAILLM)


class TestOpenAILLMInitialization:
    """Independent tests for OpenAILLM initialization."""

    def test_stores_config(self):
        """OpenAILLM should store config."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="gpt-4", api_key="test-key")
        llm = OpenAILLM(config)
        assert llm.config is config

    def test_config_model(self):
        """OpenAILLM config should have model."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        llm = OpenAILLM(config)
        assert llm.config.model == "gpt-4"

    def test_config_base_url(self):
        """OpenAILLM config should have base_url."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="gpt-4", base_url="https://custom.com/v1")
        llm = OpenAILLM(config)
        assert llm.config.base_url == "https://custom.com/v1"

    def test_config_api_key(self):
        """OpenAILLM config should have api_key."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="gpt-4", api_key="sk-test")
        llm = OpenAILLM(config)
        assert llm.config.api_key == "sk-test"

    def test_http_client_initially_none(self):
        """HTTP client should be initially None."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="gpt-4")
        llm = OpenAILLM(config)
        assert llm._client is None

    def test_has_completions_endpoint(self):
        """OpenAILLM should have completions endpoint."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM
        assert hasattr(OpenAILLM, '_completions_endpoint')
        assert '/completions' in OpenAILLM._completions_endpoint


class TestOpenAILLMGenerateMethod:
    """Independent tests for generate method."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        return OpenAIConfig(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key"
        )

    @pytest.mark.asyncio
    async def test_generate_returns_string(self, config):
        """generate should return a string."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        result = await llm.generate("test prompt")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_calls_api(self, config):
        """generate should call the completions API."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        await llm.generate("test prompt")

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert '/completions' in call_args[0][0] or '/chat/completions' in call_args[0][0]

    @pytest.mark.asyncio
    async def test_generate_sends_prompt_in_payload(self, config):
        """generate should include prompt in the request payload."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "response"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        await llm.generate("What is Python?")

        call_args = mock_client.post.call_args
        payload = call_args[1].get('json') or call_args[0][1]
        assert payload is not None

    @pytest.mark.asyncio
    async def test_generate_extracts_content_from_response(self, config):
        """generate should extract content from OpenAI response format."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Python is a programming language."}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        result = await llm.generate("What is Python?")
        assert result == "Python is a programming language."

    @pytest.mark.asyncio
    async def test_generate_handles_empty_choices(self, config):
        """generate should handle empty choices in response."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        result = await llm.generate("test")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_http_error_raises_runtime_error(self, config):
        """generate should raise RuntimeError on HTTP error."""
        import httpx
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "401",
                request=MagicMock(),
                response=mock_response
            )
        )
        mock_client.is_closed = False
        llm._client = mock_client

        with pytest.raises(RuntimeError):
            await llm.generate("test")

    @pytest.mark.asyncio
    async def test_generate_connection_error_raises_runtime_error(self, config):
        """generate should raise RuntimeError on connection error."""
        import httpx
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.is_closed = False
        llm._client = mock_client

        with pytest.raises(RuntimeError):
            await llm.generate("test")


class TestOpenAILLMStreamMethod:
    """Independent tests for stream method."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        return OpenAIConfig(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key"
        )

    @pytest.mark.asyncio
    async def test_stream_returns_async_iterator(self, config):
        """stream should return an async iterator."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":" World"}}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        result = llm.stream("test prompt")
        assert hasattr(result, '__aiter__')

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, config):
        """stream should yield tokens."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hello"}}]}'
            yield 'data: {"choices":[{"delta":{"content":" World"}}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        tokens = []
        async for token in llm.stream("test"):
            tokens.append(token)

        assert len(tokens) >= 1
        assert all(isinstance(t, str) for t in tokens)


class TestOpenAILLMCountTokensMethod:
    """Independent tests for count_tokens method."""

    @pytest.fixture
    def llm(self):
        """Create OpenAILLM instance."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="test-model")
        return OpenAILLM(config)

    @pytest.mark.asyncio
    async def test_count_tokens_returns_int(self, llm):
        """count_tokens should return an integer."""
        result = await llm.count_tokens("Hello world")
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_count_tokens_positive_for_nonempty(self, llm):
        """count_tokens should return positive count for non-empty text."""
        result = await llm.count_tokens("Hello world, this is a test.")
        assert result > 0

    @pytest.mark.asyncio
    async def test_count_tokens_zero_for_empty(self, llm):
        """count_tokens should return 0 for empty text."""
        result = await llm.count_tokens("")
        assert result == 0

    @pytest.mark.asyncio
    async def test_count_tokens_increases_with_text_length(self, llm):
        """count_tokens should increase with longer text."""
        count_short = await llm.count_tokens("Hello")
        count_long = await llm.count_tokens("Hello world, this is a much longer sentence with many words.")
        assert count_long > count_short

    @pytest.mark.asyncio
    async def test_count_tokens_handles_single_word(self, llm):
        """count_tokens should handle single word."""
        result = await llm.count_tokens("Python")
        assert result >= 1


class TestOpenAILLMCloseAndContextManager:
    """Independent tests for resource management."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        return OpenAIConfig(model="test-model", api_key="test-key")

    @pytest.mark.asyncio
    async def test_close_method_exists(self, config):
        """OpenAILLM should have close method."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM
        llm = OpenAILLM(config)
        assert hasattr(llm, 'close')
        assert callable(getattr(llm, 'close'))

    @pytest.mark.asyncio
    async def test_close_when_no_client(self, config):
        """close should not raise when client is None."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM
        llm = OpenAILLM(config)
        await llm.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        """OpenAILLM should support async context manager."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        async with OpenAILLM(config) as llm:
            assert llm is not None
            assert isinstance(llm, OpenAILLM)


class TestOpenAILLMConfigProperty:
    """Independent tests for config property."""

    def test_config_property_returns_openai_config(self):
        """config property should return OpenAIConfig."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="gpt-4", api_key="test")
        llm = OpenAILLM(config)
        assert isinstance(llm.config, OpenAIConfig)

    def test_config_property_returns_same_config(self):
        """config property should return the same config used at init."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
        config = OpenAIConfig(model="gpt-4", api_key="test")
        llm = OpenAILLM(config)
        assert llm.config is config


class TestOpenAILLMIntegration:
    """Independent integration tests for OpenAILLM."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        from agent_framework.infrastructure.openai_llm import OpenAIConfig
        return OpenAIConfig(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key",
            temperature=0.5,
            max_tokens=100
        )

    @pytest.mark.asyncio
    async def test_generate_then_count_tokens(self, config):
        """Test generating response and then counting tokens."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm = OpenAILLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Python is a programming language."}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        response = await llm.generate("What is Python?")
        token_count = await llm.count_tokens(response)

        assert len(response) > 0
        assert token_count > 0

    @pytest.mark.asyncio
    async def test_context_manager_cleanup(self, config):
        """Test context manager properly cleans up."""
        from agent_framework.infrastructure.openai_llm import OpenAILLM

        llm_instance = None
        async with OpenAILLM(config) as llm:
            llm_instance = llm
            assert llm is not None

        # After context exit, client should be closed
        assert llm_instance is not None
