"""Independent test cases for LLM Gateway interface.

This module contains independent verification tests for the LLMProvider enum,
LLMConfig dataclass, and LLMGateway abstract base class, following the detailed
design specification in section 9.1.

Test categories:
1. LLMProvider enum completeness and values
2. LLMConfig data class integrity and validation
3. LLMGateway abstract base class method signatures
4. Boundary conditions and error handling
5. Concrete implementation behavior verification
"""
import inspect
import pytest
from abc import ABC
from dataclasses import FrozenInstanceError
from enum import Enum
from typing import AsyncIterator, Dict, List, Optional, get_type_hints

from infrastructure.llm_gateway import (
    LLMConfig, LLMGateway, LLMProvider,
    ChatResponse as LLMChatResponse,
    StreamChatResponse, ChatResponseType,
)
from interfaces.llm_types import ToolCall, FunctionCall


# ============================================================================
# 1. LLMProvider Enum Tests
# ============================================================================


class TestLLMProviderEnum:
    """Independent tests for LLMProvider enum."""

    def test_llm_provider_is_enum(self):
        """LLMProvider must be an Enum subclass."""
        assert issubclass(LLMProvider, Enum), "LLMProvider 应继承自 Enum"

    def test_llm_provider_is_str_enum(self):
        """LLMProvider must also inherit from str for JSON serialization."""
        assert issubclass(LLMProvider, str), "LLMProvider 应继承自 str"

    def test_llm_provider_has_ollama(self):
        """LLMProvider must define OLLAMA member."""
        assert hasattr(LLMProvider, "OLLAMA"), "LLMProvider 缺少 OLLAMA 成员"
        assert LLMProvider.OLLAMA.value == "ollama", "OLLAMA 的值应为 'ollama'"

    def test_llm_provider_has_openai(self):
        """LLMProvider must define OPENAI member."""
        assert hasattr(LLMProvider, "OPENAI"), "LLMProvider 缺少 OPENAI 成员"
        assert LLMProvider.OPENAI.value == "openai", "OPENAI 的值应为 'openai'"

    def test_llm_provider_has_anthropic(self):
        """LLMProvider must define ANTHROPIC member."""
        assert hasattr(LLMProvider, "ANTHROPIC"), "LLMProvider 缺少 ANTHROPIC 成员"
        assert LLMProvider.ANTHROPIC.value == "anthropic", "ANTHROPIC 的值应为 'anthropic'"

    def test_llm_provider_member_count(self):
        """LLMProvider must have exactly 3 members per design."""
        members = list(LLMProvider)
        assert len(members) == 3, f"LLMProvider 应有 3 个成员，实际有 {len(members)}"

    def test_llm_provider_all_values_are_lowercase_strings(self):
        """All enum values must be lowercase strings."""
        for provider in LLMProvider:
            assert isinstance(provider.value, str), \
                f"{provider.name} 的值应为字符串类型"
            assert provider.value == provider.value.lower(), \
                f"{provider.name} 的值应为小写"

    def test_llm_provider_string_comparison(self):
        """LLMProvider members should be comparable with their string values."""
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"

    def test_llm_provider_from_string(self):
        """LLMProvider can be constructed from string value."""
        assert LLMProvider("ollama") == LLMProvider.OLLAMA
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC

    def test_llm_provider_invalid_value_raises(self):
        """LLMProvider raises ValueError for unknown provider string."""
        with pytest.raises(ValueError):
            LLMProvider("unknown_provider")

    def test_llm_provider_is_hashable(self):
        """LLMProvider members must be hashable for use as dict keys."""
        provider_set = {LLMProvider.OLLAMA, LLMProvider.OPENAI}
        assert len(provider_set) == 2

    def test_llm_provider_iteration_order(self):
        """LLMProvider iteration should follow definition order."""
        members = list(LLMProvider)
        names = [m.name for m in members]
        assert names == ["OLLAMA", "OPENAI", "ANTHROPIC"], \
            f"枚举成员顺序应为 OLLAMA, OPENAI, ANTHROPIC，实际为 {names}"


# ============================================================================
# 2. LLMConfig Dataclass Tests
# ============================================================================


class TestLLMConfigDataclass:
    """Independent tests for LLMConfig data class."""

    def test_llm_config_is_frozen(self):
        """LLMConfig must be immutable (frozen dataclass)."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        with pytest.raises(FrozenInstanceError):
            config.model = "gpt-3.5-turbo"

    def test_llm_config_has_provider_field(self):
        """LLMConfig must have provider field of type LLMProvider."""
        hints = get_type_hints(LLMConfig)
        assert "provider" in hints, "LLMConfig 缺少 provider 字段"
        assert hints["provider"] == LLMProvider, "provider 字段类型应为 LLMProvider"

    def test_llm_config_has_model_field(self):
        """LLMConfig must have model field of type str."""
        hints = get_type_hints(LLMConfig)
        assert "model" in hints, "LLMConfig 缺少 model 字段"
        assert hints["model"] == str, "model 字段类型应为 str"

    def test_llm_config_has_api_key_field(self):
        """LLMConfig must have api_key field as Optional[str]."""
        hints = get_type_hints(LLMConfig)
        assert "api_key" in hints, "LLMConfig 缺少 api_key 字段"

    def test_llm_config_has_base_url_field(self):
        """LLMConfig must have base_url field as Optional[str]."""
        hints = get_type_hints(LLMConfig)
        assert "base_url" in hints, "LLMConfig 缺少 base_url 字段"

    def test_llm_config_has_max_tokens_field(self):
        """LLMConfig must have max_tokens field as Optional[int]."""
        hints = get_type_hints(LLMConfig)
        assert "max_tokens" in hints, "LLMConfig 缺少 max_tokens 字段"

    def test_llm_config_has_temperature_field(self):
        """LLMConfig must have temperature field as Optional[float]."""
        hints = get_type_hints(LLMConfig)
        assert "temperature" in hints, "LLMConfig 缺少 temperature 字段"

    def test_llm_config_create_with_required_only(self):
        """LLMConfig can be created with only required fields (provider, model)."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"
        assert config.api_key is None
        assert config.base_url is None
        assert config.max_tokens is None
        assert config.temperature is None

    def test_llm_config_create_with_all_fields(self):
        """LLMConfig can be created with all fields specified."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3",
            api_key="test-key",
            base_url="http://localhost:11434",
            max_tokens=4096,
            temperature=0.7
        )
        assert config.provider == LLMProvider.OLLAMA
        assert config.model == "llama3"
        assert config.api_key == "test-key"
        assert config.base_url == "http://localhost:11434"
        assert config.max_tokens == 4096
        assert config.temperature == 0.7

    def test_llm_config_with_anthropic_provider(self):
        """LLMConfig works with ANTHROPIC provider."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-opus-20240229",
            api_key="sk-ant-..."
        )
        assert config.provider == LLMProvider.ANTHROPIC

    def test_llm_config_with_zero_max_tokens(self):
        """LLMConfig accepts zero max_tokens."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            max_tokens=0
        )
        assert config.max_tokens == 0

    def test_llm_config_with_zero_temperature(self):
        """LLMConfig accepts zero temperature (deterministic)."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=0.0
        )
        assert config.temperature == 0.0

    def test_llm_config_with_high_temperature(self):
        """LLMConfig accepts high temperature value."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=2.0
        )
        assert config.temperature == 2.0

    def test_llm_config_equality(self):
        """Two LLMConfig instances with same values should be equal."""
        config1 = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        config2 = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        assert config1 == config2

    def test_llm_config_inequality_different_model(self):
        """LLMConfig with different models should not be equal."""
        config1 = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        config2 = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-3.5-turbo")
        assert config1 != config2

    def test_llm_config_inequality_different_provider(self):
        """LLMConfig with different providers should not be equal."""
        config1 = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        config2 = LLMConfig(provider=LLMProvider.OLLAMA, model="gpt-4")
        assert config1 != config2

    def test_llm_config_is_hashable(self):
        """Frozen LLMConfig must be hashable for use in sets/dicts."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        config_set = {config}
        assert len(config_set) == 1

    def test_llm_config_with_empty_model_string(self):
        """LLMConfig accepts empty model string (validation left to implementation)."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model="")
        assert config.model == ""

    def test_llm_config_field_order(self):
        """LLMConfig fields should follow provider, model, then optional order."""
        import dataclasses
        fields = dataclasses.fields(LLMConfig)
        field_names = [f.name for f in fields]
        assert field_names[0] == "provider", "第一个字段应为 provider"
        assert field_names[1] == "model", "第二个字段应为 model"

    def test_llm_config_required_fields_have_no_default(self):
        """provider and model must not have default values."""
        import dataclasses
        fields = dataclasses.fields(LLMConfig)
        provider_field = next(f for f in fields if f.name == "provider")
        model_field = next(f for f in fields if f.name == "model")
        assert provider_field.default is dataclasses.MISSING, \
            "provider 字段不应有默认值"
        assert model_field.default is dataclasses.MISSING, \
            "model 字段不应有默认值"


# ============================================================================
# 3. LLMGateway Abstract Base Class Tests
# ============================================================================


class TestLLMGatewayAbstractClass:
    """Independent tests for LLMGateway abstract base class."""

    def test_llm_gateway_is_abstract(self):
        """LLMGateway must be an abstract base class."""
        assert issubclass(LLMGateway, ABC), "LLMGateway 应继承自 ABC"

    def test_llm_gateway_cannot_instantiate_directly(self):
        """Cannot directly instantiate abstract LLMGateway."""
        with pytest.raises(TypeError):
            LLMGateway(configs={})

    def test_llm_gateway_has_generate_method(self):
        """LLMGateway must have abstract generate method."""
        assert hasattr(LLMGateway, "generate"), "LLMGateway 缺少 generate 方法"
        method = getattr(LLMGateway, "generate")
        assert getattr(method, "__isabstractmethod__", False), \
            "generate 方法应为抽象方法"

    def test_llm_gateway_has_stream_method(self):
        """LLMGateway must have abstract stream method."""
        assert hasattr(LLMGateway, "stream"), "LLMGateway 缺少 stream 方法"
        method = getattr(LLMGateway, "stream")
        assert getattr(method, "__isabstractmethod__", False), \
            "stream 方法应为抽象方法"

    def test_llm_gateway_has_count_tokens_method(self):
        """LLMGateway must have abstract count_tokens method."""
        assert hasattr(LLMGateway, "count_tokens"), \
            "LLMGateway 缺少 count_tokens 方法"
        method = getattr(LLMGateway, "count_tokens")
        assert getattr(method, "__isabstractmethod__", False), \
            "count_tokens 方法应为抽象方法"

    def test_llm_gateway_has_get_config_method(self):
        """LLMGateway must have concrete get_config method."""
        assert hasattr(LLMGateway, "get_config"), \
            "LLMGateway 缺少 get_config 方法"
        # get_config 是具体方法，不是抽象方法
        method = getattr(LLMGateway, "get_config")
        assert not getattr(method, "__isabstractmethod__", False), \
            "get_config 方法不应为抽象方法"

    def test_llm_gateway_has_configured_models_property(self):
        """LLMGateway must have configured_models property."""
        assert hasattr(LLMGateway, "configured_models"), \
            "LLMGateway 缺少 configured_models 属性"

    def test_generate_method_signature(self):
        """generate method must have correct signature per design."""
        hints = get_type_hints(LLMGateway.generate)
        assert "prompt" in hints, "generate 方法缺少 prompt 参数"
        assert "model" in hints, "generate 方法缺少 model 参数"
        assert hints["prompt"] == str, "prompt 参数类型应为 str"
        assert hints["model"] == str, "model 参数类型应为 str"
        assert hints["return"] == str, "generate 方法返回类型应为 str"

    def test_stream_method_signature(self):
        """stream method must have correct signature per design."""
        hints = get_type_hints(LLMGateway.stream)
        assert "prompt" in hints, "stream 方法缺少 prompt 参数"
        assert "model" in hints, "stream 方法缺少 model 参数"
        assert hints["prompt"] == str, "prompt 参数类型应为 str"
        assert hints["model"] == str, "model 参数类型应为 str"
        # 返回类型应为 AsyncIterator[str]
        assert hints["return"] == AsyncIterator[str], \
            "stream 方法返回类型应为 AsyncIterator[str]"

    def test_count_tokens_method_signature(self):
        """count_tokens method must have correct signature per design."""
        hints = get_type_hints(LLMGateway.count_tokens)
        assert "text" in hints, "count_tokens 方法缺少 text 参数"
        assert hints["text"] == str, "text 参数类型应为 str"
        assert hints["return"] == int, "count_tokens 方法返回类型应为 int"

    def test_generate_method_is_async(self):
        """generate method must be async."""
        assert inspect.iscoroutinefunction(LLMGateway.generate), \
            "generate 方法应为 async"

    def test_count_tokens_method_is_async(self):
        """count_tokens method must be async."""
        assert inspect.iscoroutinefunction(LLMGateway.count_tokens), \
            "count_tokens 方法应为 async"

    def test_init_accepts_configs_dict(self):
        """__init__ must accept configs: Dict[str, LLMConfig]."""
        hints = get_type_hints(LLMGateway.__init__)
        assert "configs" in hints, "__init__ 缺少 configs 参数"

    def test_init_stores_configs(self):
        """__init__ must store configs in self._configs."""

        class MinimalGateway(LLMGateway):
            async def generate(self, prompt, model="default", **kwargs):
                return ""

            async def stream(self, prompt, model="default", **kwargs):
                yield ""

            async def chat(self, messages, tools=None, model="default", **kwargs):
                return LLMChatResponse(content="")

            async def stream_chat(self, messages, tools=None, model="default", **kwargs):
                yield StreamChatResponse(type=ChatResponseType.DONE)

            async def count_tokens(self, text):
                return 0

        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        gw = MinimalGateway(configs={"default": config})
        assert hasattr(gw, "_configs"), "LLMGateway 应存储 _configs 属性"
        assert gw._configs == {"default": config}


# ============================================================================
# 4. Concrete Implementation Tests (via Mock)
# ============================================================================


class MockLLMGateway(LLMGateway):
    """Mock implementation for testing concrete behavior."""

    def __init__(self, configs: Dict[str, LLMConfig]):
        super().__init__(configs)
        self.call_log: List[Dict] = []

    async def generate(self, prompt: str, model: str = "default", **kwargs) -> str:
        self.call_log.append({"method": "generate", "prompt": prompt, "model": model})
        config = self.get_config(model)
        return f"Mock response from {config.model}"

    async def stream(self, prompt: str, model: str = "default", **kwargs) -> AsyncIterator[str]:
        self.call_log.append({"method": "stream", "prompt": prompt, "model": model})
        config = self.get_config(model)
        for word in f"Mock streamed response from {config.model}".split():
            yield word + " "

    async def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
        **kwargs
    ) -> LLMChatResponse:
        self.call_log.append({"method": "chat", "messages": messages, "model": model})
        config = self.get_config(model)
        return LLMChatResponse(content=f"Mock chat response from {config.model}")

    async def stream_chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: str = "default",
        **kwargs
    ) -> AsyncIterator[StreamChatResponse]:
        self.call_log.append({"method": "stream_chat", "messages": messages, "model": model})
        config = self.get_config(model)
        for word in f"Mock streamed chat from {config.model}".split():
            yield StreamChatResponse(type=ChatResponseType.CONTENT, content=word + " ")
        yield StreamChatResponse(type=ChatResponseType.DONE)

    async def count_tokens(self, text: str) -> int:
        self.call_log.append({"method": "count_tokens", "text": text})
        return len(text.split())


class TestLLMGatewayConcreteImplementation:
    """Tests for LLMGateway concrete implementation behavior."""

    def _make_gateway(self) -> MockLLMGateway:
        """Helper to create a gateway with standard test configs."""
        return MockLLMGateway(configs={
            "default": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4"),
            "light": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-3.5-turbo"),
            "local": LLMConfig(provider=LLMProvider.OLLAMA, model="llama3"),
        })

    def test_concrete_implementation_can_instantiate(self):
        """Concrete implementation can be instantiated."""
        gw = self._make_gateway()
        assert isinstance(gw, LLMGateway)
        assert isinstance(gw, ABC)

    def test_concrete_implementation_with_single_config(self):
        """Concrete implementation works with single config."""
        gw = MockLLMGateway(configs={
            "default": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        })
        assert "default" in gw.configured_models

    @pytest.mark.asyncio
    async def test_generate_returns_string(self):
        """generate method must return a string."""
        gw = self._make_gateway()
        result = await gw.generate("Hello")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_default_model(self):
        """generate uses 'default' model when not specified."""
        gw = self._make_gateway()
        result = await gw.generate("test prompt")
        assert "gpt-4" in result

    @pytest.mark.asyncio
    async def test_generate_specific_model(self):
        """generate uses specified model alias."""
        gw = self._make_gateway()
        result = await gw.generate("test prompt", model="light")
        assert "gpt-3.5-turbo" in result

    @pytest.mark.asyncio
    async def test_generate_local_model(self):
        """generate works with local OLLAMA model."""
        gw = self._make_gateway()
        result = await gw.generate("test prompt", model="local")
        assert "llama3" in result

    @pytest.mark.asyncio
    async def test_generate_unconfigured_model_raises_key_error(self):
        """generate raises KeyError for unconfigured model alias."""
        gw = self._make_gateway()
        with pytest.raises(KeyError):
            await gw.generate("test", model="nonexistent")

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        """stream method must yield string tokens."""
        gw = self._make_gateway()
        tokens = []
        async for token in gw.stream("Hello"):
            tokens.append(token)
            assert isinstance(token, str)
        assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_stream_default_model(self):
        """stream uses 'default' model when not specified."""
        gw = self._make_gateway()
        tokens = []
        async for token in gw.stream("test prompt"):
            tokens.append(token)
        full_text = "".join(tokens)
        assert "gpt-4" in full_text

    @pytest.mark.asyncio
    async def test_stream_specific_model(self):
        """stream uses specified model alias."""
        gw = self._make_gateway()
        tokens = []
        async for token in gw.stream("test prompt", model="local"):
            tokens.append(token)
        full_text = "".join(tokens)
        assert "llama3" in full_text

    @pytest.mark.asyncio
    async def test_stream_unconfigured_model_raises_key_error(self):
        """stream raises KeyError for unconfigured model alias."""
        gw = self._make_gateway()
        with pytest.raises(KeyError):
            async for _ in gw.stream("test", model="nonexistent"):
                pass

    @pytest.mark.asyncio
    async def test_count_tokens_returns_int(self):
        """count_tokens method must return an integer."""
        gw = self._make_gateway()
        result = await gw.count_tokens("Hello world test")
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_count_tokens_word_based(self):
        """count_tokens returns word count in mock implementation."""
        gw = self._make_gateway()
        result = await gw.count_tokens("one two three four five")
        assert result == 5

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self):
        """count_tokens handles empty string."""
        gw = self._make_gateway()
        result = await gw.count_tokens("")
        assert result == 0

    def test_get_config_returns_llm_config(self):
        """get_config must return LLMConfig instance."""
        gw = self._make_gateway()
        config = gw.get_config("default")
        assert isinstance(config, LLMConfig)

    def test_get_config_default_model(self):
        """get_config returns correct config for 'default' alias."""
        gw = self._make_gateway()
        config = gw.get_config("default")
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4"

    def test_get_config_light_model(self):
        """get_config returns correct config for 'light' alias."""
        gw = self._make_gateway()
        config = gw.get_config("light")
        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-3.5-turbo"

    def test_get_config_unknown_alias_raises_key_error(self):
        """get_config raises KeyError for unknown model alias."""
        gw = self._make_gateway()
        with pytest.raises(KeyError, match="nonexistent"):
            gw.get_config("nonexistent")

    def test_get_config_error_message_includes_alias(self):
        """KeyError message should include the missing alias name."""
        gw = self._make_gateway()
        with pytest.raises(KeyError) as exc_info:
            gw.get_config("missing-model")
        assert "missing-model" in str(exc_info.value)

    def test_configured_models_returns_list(self):
        """configured_models must return a list."""
        gw = self._make_gateway()
        models = gw.configured_models
        assert isinstance(models, list)

    def test_configured_models_contains_all_aliases(self):
        """configured_models must contain all configured aliases."""
        gw = self._make_gateway()
        models = gw.configured_models
        assert "default" in models
        assert "light" in models
        assert "local" in models

    def test_configured_models_count(self):
        """configured_models count matches number of configs."""
        gw = self._make_gateway()
        assert len(gw.configured_models) == 3

    def test_configured_models_empty_for_empty_configs(self):
        """configured_models returns empty list for empty configs."""
        gw = MockLLMGateway(configs={})
        assert gw.configured_models == []


# ============================================================================
# 5. Boundary Conditions and Error Handling Tests
# ============================================================================


class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    def test_llm_config_with_very_long_model_name(self):
        """LLMConfig accepts very long model name."""
        long_name = "a" * 1000
        config = LLMConfig(provider=LLMProvider.OPENAI, model=long_name)
        assert config.model == long_name

    def test_llm_config_with_special_characters_in_model(self):
        """LLMConfig accepts model names with special characters."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo-preview"
        )
        assert config.model == "gpt-4-turbo-preview"

    def test_llm_config_with_url_in_base_url(self):
        """LLMConfig accepts full URL in base_url."""
        config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model="llama3",
            base_url="http://192.168.1.100:11434"
        )
        assert config.base_url == "http://192.168.1.100:11434"

    def test_llm_config_with_very_large_max_tokens(self):
        """LLMConfig accepts very large max_tokens."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            max_tokens=1000000
        )
        assert config.max_tokens == 1000000

    def test_llm_config_with_negative_max_tokens(self):
        """LLMConfig accepts negative max_tokens (validation left to impl)."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            max_tokens=-1
        )
        assert config.max_tokens == -1

    def test_llm_config_with_negative_temperature(self):
        """LLMConfig accepts negative temperature (validation left to impl)."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4",
            temperature=-0.5
        )
        assert config.temperature == -0.5

    @pytest.mark.asyncio
    async def test_generate_with_empty_prompt(self):
        """generate should accept empty prompt string."""
        gw = MockLLMGateway(configs={
            "default": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        })
        result = await gw.generate("")
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generate_with_very_long_prompt(self):
        """generate should accept very long prompt."""
        gw = MockLLMGateway(configs={
            "default": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        })
        long_prompt = "test " * 1000
        result = await gw.generate(long_prompt)
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_stream_with_empty_prompt(self):
        """stream should accept empty prompt string."""
        gw = MockLLMGateway(configs={
            "default": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        })
        tokens = []
        async for token in gw.stream(""):
            tokens.append(token)
        assert isinstance(tokens, list)

    @pytest.mark.asyncio
    async def test_count_tokens_with_whitespace_only(self):
        """count_tokens handles whitespace-only string."""
        gw = MockLLMGateway(configs={
            "default": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        })
        result = await gw.count_tokens("   ")
        assert result == 0  # str.split() with no args ignores leading/trailing whitespace

    def test_multiple_gateways_with_same_config(self):
        """Multiple gateway instances can share the same config."""
        config = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        gw1 = MockLLMGateway(configs={"default": config})
        gw2 = MockLLMGateway(configs={"default": config})
        assert gw1.get_config("default") == gw2.get_config("default")

    def test_gateway_configs_isolation(self):
        """Gateway instances do not share config dicts."""
        config1 = LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4")
        config2 = LLMConfig(provider=LLMProvider.OLLAMA, model="llama3")
        gw1 = MockLLMGateway(configs={"default": config1})
        gw2 = MockLLMGateway(configs={"default": config2})
        assert gw1.get_config("default").provider == LLMProvider.OPENAI
        assert gw2.get_config("default").provider == LLMProvider.OLLAMA


# ============================================================================
# 6. Design Compliance Tests
# ============================================================================


class TestDesignCompliance:
    """Tests verifying compliance with detailed design section 9.1."""

    def test_llm_gateway_module_exports(self):
        """Module must export LLMProvider, LLMConfig, LLMGateway."""
        from infrastructure import llm_gateway
        assert hasattr(llm_gateway, "LLMProvider"), \
            "llm_gateway 模块缺少 LLMProvider 导出"
        assert hasattr(llm_gateway, "LLMConfig"), \
            "llm_gateway 模块缺少 LLMConfig 导出"
        assert hasattr(llm_gateway, "LLMGateway"), \
            "llm_gateway 模块缺少 LLMGateway 导出"

    def test_design_requires_three_providers(self):
        """Design specifies Ollama, OpenAI, Anthropic as providers."""
        provider_values = {p.value for p in LLMProvider}
        expected = {"ollama", "openai", "anthropic"}
        assert provider_values == expected, \
            f"期望 providers={expected}，实际={provider_values}"

    def test_design_generate_signature(self):
        """Design: generate(prompt, model='default', **kwargs) -> str."""
        sig = inspect.signature(LLMGateway.generate)
        params = list(sig.parameters.keys())
        assert "prompt" in params, "generate 缺少 prompt 参数"
        assert "model" in params, "generate 缺少 model 参数"
        # model 应有默认值 "default"
        model_param = sig.parameters["model"]
        assert model_param.default == "default", \
            f"model 默认值应为 'default'，实际为 {model_param.default}"

    def test_design_stream_signature(self):
        """Design: stream(prompt, model='default', **kwargs) -> AsyncIterator[str]."""
        sig = inspect.signature(LLMGateway.stream)
        params = list(sig.parameters.keys())
        assert "prompt" in params, "stream 缺少 prompt 参数"
        assert "model" in params, "stream 缺少 model 参数"
        model_param = sig.parameters["model"]
        assert model_param.default == "default", \
            f"model 默认值应为 'default'，实际为 {model_param.default}"

    def test_design_init_accepts_config_dict(self):
        """Design: __init__(self, config) where config is dict of providers."""
        sig = inspect.signature(LLMGateway.__init__)
        params = list(sig.parameters.keys())
        assert "configs" in params, "__init__ 缺少 configs 参数"

    def test_design_config_has_provider_field(self):
        """Design: LLMConfig has provider field."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(LLMConfig)}
        assert "provider" in fields, "LLMConfig 缺少 provider 字段"

    def test_design_config_has_model_field(self):
        """Design: LLMConfig has model field."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(LLMConfig)}
        assert "model" in fields, "LLMConfig 缺少 model 字段"
