"""Tests for infrastructure/llm_gateway.py - LLM Gateway interface."""
import pytest
from abc import ABC
from typing import AsyncIterator

from agent_framework.infrastructure.llm_gateway import (
    LLMGateway,
    LLMProvider,
    LLMConfig,
)


class TestLLMProvider:
    """Tests for LLMProvider enum."""

    def test_has_expected_providers(self):
        """Test that all expected provider values exist."""
        assert hasattr(LLMProvider, "OLLAMA")
        assert hasattr(LLMProvider, "OPENAI")
        assert hasattr(LLMProvider, "ANTHROPIC")

    def test_string_values(self):
        """Test that enum values are strings."""
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_is_str_enum(self):
        """Test that enum inherits from str."""
        assert isinstance(LLMProvider.OLLAMA, str)

    def test_comparison_with_strings(self):
        """Test that enum can be compared with strings."""
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_create_with_required_fields(self):
        """Test creating LLMConfig with required fields."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"

    def test_create_with_all_fields(self):
        """Test creating LLMConfig with all optional fields."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3",
            api_key="test-key",
            base_url="http://localhost:11434",
            max_tokens=4096,
            temperature=0.7,
        )
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3"
        assert config.api_key == "test-key"
        assert config.base_url == "http://localhost:11434"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_default_optional_fields(self):
        """Test that optional fields have correct defaults."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
        )
        assert config.api_key is None
        assert config.base_url is None
        assert config.max_tokens is None
        assert config.temperature is None

    def test_is_frozen(self):
        """Test that LLMConfig is immutable (frozen)."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            config.model = "gpt-3.5-turbo"


class TestLLMGateway:
    """Tests for LLMGateway abstract base class."""

    def test_is_abstract_class(self):
        """Test that LLMGateway is an abstract class."""
        assert issubclass(LLMGateway, ABC)

    def test_cannot_instantiate_directly(self):
        """Test that LLMGateway cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMGateway({})

    def test_has_generate_method(self):
        """Test that LLMGateway has generate method."""
        assert hasattr(LLMGateway, "generate")

    def test_has_stream_method(self):
        """Test that LLMGateway has stream method."""
        assert hasattr(LLMGateway, "stream")

    def test_has_count_tokens_method(self):
        """Test that LLMGateway has count_tokens method."""
        assert hasattr(LLMGateway, "count_tokens")

    def test_generate_is_abstract(self):
        """Test that generate is an abstract method."""
        # Get the generate method from the class
        from abc import abstractmethod
        generate = getattr(LLMGateway, "generate", None)
        assert generate is not None
        # abstractmethod decorated functions have __isabstractmethod__ attribute
        assert getattr(generate, "__isabstractmethod__", False) is True

    def test_stream_is_abstract(self):
        """Test that stream is an abstract method."""
        from abc import abstractmethod
        stream = getattr(LLMGateway, "stream", None)
        assert stream is not None
        assert getattr(stream, "__isabstractmethod__", False) is True

    def test_count_tokens_is_abstract(self):
        """Test that count_tokens is an abstract method."""
        from abc import abstractmethod
        count_tokens = getattr(LLMGateway, "count_tokens", None)
        assert count_tokens is not None
        assert getattr(count_tokens, "__isabstractmethod__", False) is True


class TestLLMGatewayConcreteImplementation:
    """Tests for a concrete implementation of LLMGateway."""

    @pytest.fixture
    def mock_gateway(self):
        """Create a mock concrete implementation for testing."""
        class MockLLMGateway(LLMGateway):
            """Mock LLM Gateway for testing."""

            async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
                """Mock generate implementation."""
                return f"Response to: {prompt}"

            async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
                """Mock stream implementation."""
                for token in ["Hello", " ", "World"]:
                    yield token

            async def count_tokens(self, text: str) -> int:
                """Mock count_tokens implementation."""
                return len(text.split())

        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        return MockLLMGateway({"default": config})

    @pytest.mark.asyncio
    async def test_generate_returns_string(self, mock_gateway):
        """Test that generate returns a string."""
        result = await mock_gateway.generate("Hello")
        assert isinstance(result, str)
        assert result == "Response to: Hello"

    @pytest.mark.asyncio
    async def test_generate_with_model_parameter(self, mock_gateway):
        """Test that generate accepts model parameter."""
        result = await mock_gateway.generate("Hello", model="default")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_with_kwargs(self, mock_gateway):
        """Test that generate accepts additional kwargs."""
        result = await mock_gateway.generate("Hello", temperature=0.7, max_tokens=100)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_stream_returns_async_iterator(self, mock_gateway):
        """Test that stream returns an async iterator of strings."""
        result = mock_gateway.stream("Hello")
        tokens = []
        async for token in result:
            tokens.append(token)
        assert tokens == ["Hello", " ", "World"]
        assert all(isinstance(t, str) for t in tokens)

    @pytest.mark.asyncio
    async def test_stream_with_model_parameter(self, mock_gateway):
        """Test that stream accepts model parameter."""
        result = mock_gateway.stream("Hello", model="default")
        tokens = [t async for t in result]
        assert len(tokens) == 3

    @pytest.mark.asyncio
    async def test_count_tokens_returns_int(self, mock_gateway):
        """Test that count_tokens returns an integer."""
        result = await mock_gateway.count_tokens("Hello world test")
        assert isinstance(result, int)
        assert result == 3

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, mock_gateway):
        """Test that count_tokens handles empty string."""
        result = await mock_gateway.count_tokens("")
        assert result == 0
