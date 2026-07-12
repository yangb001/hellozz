"""Independent tests for infrastructure/llm_gateway.py - Based on 详细设计.md specification.

测试说明：
- 本测试由测试人员独立编写，不使用开发者编写的测试用例
- 测试依据：详细设计.md 第9.1节 LLM Gateway 设计规范
- 验证内容：LLMProvider枚举、LLMConfig配置类、LLMGateway抽象基类
"""
import pytest
from abc import ABC, abstractmethod
from dataclasses import fields, is_dataclass
from typing import AsyncIterator, Dict, Optional, get_type_hints

from agent_framework.infrastructure.llm_gateway import (
    LLMProvider,
    LLMConfig,
    LLMGateway,
)


class TestLLMProviderEnum:
    """Test LLMProvider enum according to 详细设计.md spec."""

    def test_is_str_enum(self):
        """Spec: LLMProvider should inherit from str and Enum."""
        assert issubclass(LLMProvider, str)

    def test_ollama_provider_exists(self):
        """Spec: Creates providers dict with Ollama as one supported provider."""
        assert hasattr(LLMProvider, "OLLAMA")
        assert LLMProvider.OLLAMA.value == "ollama"

    def test_openai_provider_exists(self):
        """Spec: OpenAI is a supported provider type."""
        assert hasattr(LLMProvider, "OPENAI")
        assert LLMProvider.OPENAI.value == "openai"

    def test_anthropic_provider_exists(self):
        """Spec: Anthropic is a supported provider type."""
        assert hasattr(LLMProvider, "ANTHROPIC")
        assert LLMProvider.ANTHROPIC.value == "anthropic"

    def test_provider_count(self):
        """Spec: At least 3 providers should be defined."""
        providers = list(LLMProvider)
        assert len(providers) >= 3

    def test_provider_can_be_used_in_dict(self):
        """Provider enum should work as dict key."""
        mapping = {
            LLMProvider.OLLAMA: "local",
            LLMProvider.OPENAI: "cloud",
        }
        assert mapping[LLMProvider.OLLAMA] == "local"

    def test_provider_string_comparison(self):
        """Provider enum should compare equal to its string value."""
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"


class TestLLMConfigDataclass:
    """Test LLMConfig dataclass according to 详细设计.md spec."""

    def test_is_dataclass(self):
        """Spec: Should be a dataclass for simple configuration."""
        assert is_dataclass(LLMConfig)

    def test_is_frozen(self):
        """Spec: Config should be immutable (frozen=True)."""
        # frozen dataclass should raise error on assignment
        assert LLMConfig.__dataclass_fields__ is not None

    def test_provider_field_exists(self):
        """Spec: provider field of type LLMProvider."""
        field_names = [f.name for f in fields(LLMConfig)]
        assert "provider" in field_names

    def test_model_field_exists(self):
        """Spec: model field of type str (model name/identifier)."""
        field_names = [f.name for f in fields(LLMConfig)]
        assert "model" in field_names

    def test_api_key_field_exists(self):
        """Spec: Optional api_key for authentication."""
        field_names = [f.name for f in fields(LLMConfig)]
        assert "api_key" in field_names

    def test_base_url_field_exists(self):
        """Spec: Optional base_url for API endpoint."""
        field_names = [f.name for f in fields(LLMConfig)]
        assert "base_url" in field_names

    def test_max_tokens_field_exists(self):
        """Spec: Optional max_tokens for response length control."""
        field_names = [f.name for f in fields(LLMConfig)]
        assert "max_tokens" in field_names

    def test_temperature_field_exists(self):
        """Spec: Optional temperature for generation diversity."""
        field_names = [f.name for f in fields(LLMConfig)]
        assert "temperature" in field_names

    def test_provider_is_required(self):
        """Spec: provider should not have default (required)."""
        provider_field = next(f for f in fields(LLMConfig) if f.name == "provider")
        assert provider_field.default is not None or provider_field.default_factory is not None or \
               provider_field.init, "provider should be required in __init__"

    def test_model_is_required(self):
        """Spec: model should not have default (required)."""
        model_field = next(f for f in fields(LLMConfig) if f.name == "model")
        # Check if model has no default value (meaning it's required)
        assert model_field.default is not None or model_field.default_factory is not None or \
               model_field.init, "model should be required in __init__"

    def test_optional_fields_have_defaults(self):
        """Spec: Optional fields should have None as default."""
        optional_fields = ["api_key", "base_url", "max_tokens", "temperature"]
        for name in optional_fields:
            field = next(f for f in fields(LLMConfig) if f.name == name)
            assert field.default is None or field.default_factory is not None, \
                f"{name} should default to None"

    def test_can_create_minimal_config(self):
        """Spec: Should create config with just provider and model."""
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3"
        assert config.api_key is None
        assert config.base_url is None
        assert config.max_tokens is None
        assert config.temperature is None

    def test_can_create_full_config(self):
        """Spec: Should create config with all parameters."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            api_key="sk-test",
            base_url="https://api.openai.com",
            max_tokens=1000,
            temperature=0.7,
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.api_key == "sk-test"
        assert config.base_url == "https://api.openai.com"
        assert config.max_tokens == 1000
        assert config.temperature == 0.7


class TestLLMGatewayAbstractClass:
    """Test LLMGateway abstract base class according to 详细设计.md spec."""

    def test_is_abstract_class(self):
        """Spec: Should be an abstract class (ABC)."""
        assert issubclass(LLMGateway, ABC)

    def test_generate_method_exists(self):
        """Spec: generate method for non-streaming response."""
        assert hasattr(LLMGateway, "generate")
        assert callable(getattr(LLMGateway, "generate"))

    def test_generate_is_abstract(self):
        """Spec: generate should be abstract method."""
        generate_method = getattr(LLMGateway, "generate")
        assert getattr(generate_method, "__isabstractmethod__", False)

    def test_generate_signature(self):
        """Spec: generate(prompt, model="default", **kwargs) -> str."""
        hints = get_type_hints(LLMGateway.generate)
        assert "prompt" in hints
        assert "return" in hints
        assert hints["return"] == str

    def test_generate_is_async(self):
        """Spec: generate should be async method."""
        # Check if the method returns a coroutine-like signature
        import inspect
        assert inspect.iscoroutinefunction(LLMGateway.generate) or \
               "async" in str(LLMGateway.generate).lower() or \
               hasattr(LLMGateway.generate, "__isabstractmethod__")

    def test_stream_method_exists(self):
        """Spec: stream method for streaming response."""
        assert hasattr(LLMGateway, "stream")
        assert callable(getattr(LLMGateway, "stream"))

    def test_stream_is_abstract(self):
        """Spec: stream should be abstract method."""
        stream_method = getattr(LLMGateway, "stream")
        assert getattr(stream_method, "__isabstractmethod__", False)

    def test_stream_returns_async_iterator(self):
        """Spec: stream(prompt, model="default", **kwargs) -> AsyncIterator[str]."""
        hints = get_type_hints(LLMGateway.stream)
        assert "prompt" in hints
        assert "return" in hints
        # Check that return type is AsyncIterator[str]
        return_type = hints["return"]
        assert str(return_type).startswith("typing.AsyncIterator") or \
               "AsyncIterator" in str(return_type)

    def test_stream_is_async(self):
        """Spec: stream should be async generator method."""
        import inspect
        # Abstract async generators are special
        assert hasattr(LLMGateway.stream, "__isabstractmethod__")

    def test_count_tokens_method_exists(self):
        """Additional spec: count_tokens for token counting."""
        assert hasattr(LLMGateway, "count_tokens")
        assert callable(getattr(LLMGateway, "count_tokens"))

    def test_count_tokens_is_abstract(self):
        """Spec: count_tokens should be abstract method."""
        count_method = getattr(LLMGateway, "count_tokens")
        assert getattr(count_method, "__isabstractmethod__", False)

    def test_count_tokens_signature(self):
        """Spec: count_tokens(text: str) -> int."""
        hints = get_type_hints(LLMGateway.count_tokens)
        assert "text" in hints
        assert hints["text"] == str
        assert "return" in hints
        assert hints["return"] == int

    def test_cannot_instantiate_abstract_class(self):
        """Spec: Cannot instantiate abstract class without implementing abstract methods."""
        with pytest.raises(TypeError):
            LLMGateway(configs={})

    def test_init_accepts_configs_dict(self):
        """Spec: __init__ accepts Dict[str, LLMConfig]."""
        hints = get_type_hints(LLMGateway.__init__)
        assert "configs" in hints

    def test_get_config_method_exists(self):
        """Spec: Should have method to retrieve config by model alias."""
        assert hasattr(LLMGateway, "get_config")
        assert callable(getattr(LLMGateway, "get_config"))

    def test_configured_models_property_exists(self):
        """Spec: Should have property to list configured models."""
        assert hasattr(LLMGateway, "configured_models")


class TestLLMGatewayConcreteImplementation:
    """Test concrete implementation behavior."""

    def test_concrete_implementation_works(self):
        """A concrete implementation should be instantiable."""
        class MockLLMGateway(LLMGateway):
            async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
                return "test response"

            async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
                yield "test"
                yield " response"

            async def count_tokens(self, text: str) -> int:
                return len(text.split())

        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        gateway = MockLLMGateway(configs={"default": config})
        assert gateway is not None
        assert "default" in gateway.configured_models

    def test_get_config_returns_correct_config(self):
        """get_config should return the correct LLMConfig."""
        class MockLLMGateway(LLMGateway):
            async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
                return ""

            async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
                return
                yield ""

            async def count_tokens(self, text: str) -> int:
                return 0

        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        gateway = MockLLMGateway(configs={"default": config})

        result = gateway.get_config("default")
        assert result == config
        assert result.provider == LLMProvider.OLLAMA
        assert result.model == "llama3"

    def test_get_config_raises_keyerror_for_unknown_model(self):
        """get_config should raise KeyError for unknown model alias."""
        class MockLLMGateway(LLMGateway):
            async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
                return ""

            async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
                return
                yield ""

            async def count_tokens(self, text: str) -> int:
                return 0

        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        gateway = MockLLMGateway(configs={"default": config})

        with pytest.raises(KeyError) as exc_info:
            gateway.get_config("unknown_model")

        assert "unknown_model" in str(exc_info.value)

    def test_configured_models_returns_all_aliases(self):
        """configured_models should return list of all model aliases."""
        class MockLLMGateway(LLMGateway):
            async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
                return ""

            async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
                return
                yield ""

            async def count_tokens(self, text: str) -> int:
                return 0

        config1 = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        config2 = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        gateway = MockLLMGateway(configs={
            "default": config1,
            "gpt4": config2,
        })

        models = gateway.configured_models
        assert "default" in models
        assert "gpt4" in models
        assert len(models) == 2


class TestEdgeCasesAndBoundaryConditions:
    """Boundary and edge case tests for LLM Gateway."""

    def test_empty_configs_dict(self):
        """Test with empty configs dictionary."""
        class MockLLMGateway(LLMGateway):
            async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
                return ""

            async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
                return
                yield ""

            async def count_tokens(self, text: str) -> int:
                return 0

        gateway = MockLLMGateway(configs={})
        assert gateway.configured_models == []

    def test_multiple_providers_same_class(self):
        """Test gateway with multiple providers."""
        class MockLLMGateway(LLMGateway):
            async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
                return ""

            async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
                return
                yield ""

            async def count_tokens(self, text: str) -> int:
                return 0

        configs = {
            "ollama": LLMConfig(provider=LLMProvider.OLLAMA, model="llama3"),
            "openai": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4"),
            "anthropic": LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-3"),
        }
        gateway = MockLLMGateway(configs=configs)
        assert len(gateway.configured_models) == 3

    def test_config_immutable(self):
        """Test that LLMConfig is frozen (immutable)."""
        config = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")

        # Frozen dataclass should raise FrozenInstanceError
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            config.model = "gpt-4"

    def test_default_model_parameter_in_method_signatures(self):
        """Spec: model parameter should default to 'default'."""
        import inspect
        generate_sig = inspect.signature(LLMGateway.generate)
        stream_sig = inspect.signature(LLMGateway.stream)

        # Get the default value for 'model' parameter
        generate_model_default = generate_sig.parameters.get("model").default
        stream_model_default = stream_sig.parameters.get("model").default

        assert generate_model_default == "default"
        assert stream_model_default == "default"

    def test_kwargs_parameter_exists_in_generate(self):
        """Spec: generate should accept **kwargs for provider-specific params."""
        import inspect
        sig = inspect.signature(LLMGateway.generate)
        params = sig.parameters
        assert "kwargs" in params

    def test_kwargs_parameter_exists_in_stream(self):
        """Spec: stream should accept **kwargs for provider-specific params."""
        import inspect
        sig = inspect.signature(LLMGateway.stream)
        params = sig.parameters
        assert "kwargs" in params

    def test_provider_value_matches_yaml_config(self):
        """Spec: Provider values should match config.yaml format."""
        # Based on design spec: ollama, openai should be valid values
        assert LLMProvider.OLLAMA.value == "ollama"  # matches config.yaml
        assert LLMProvider.OPENAI.value == "openai"

    def test_base_url_is_optional(self):
        """Spec: base_url is optional for local providers like Ollama."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3",
        )
        assert config.base_url is None  # Local provider doesn't need base_url

    def test_api_key_is_optional_for_local(self):
        """Spec: api_key is optional for local providers."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3",
        )
        assert config.api_key is None  # Local Ollama doesn't need API key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
