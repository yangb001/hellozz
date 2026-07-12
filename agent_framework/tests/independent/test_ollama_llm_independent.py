"""Independent tests for OllamaLLM implementation.

验证内容（基于详细设计.md 第9.1节）：
- OllamaConfig 配置完整性
- OllamaLLM 继承 LLMGateway 正确性
- 方法签名和异步特性
- 错误处理和超时
- context manager 支持

本测试文件完全独立编写，不使用开发者编写的测试用例。
"""
import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_framework.infrastructure.llm_gateway import LLMGateway, LLMConfig, LLMProvider
from agent_framework.infrastructure.ollama_llm import OllamaConfig, OllamaLLM


# ─────────────────────────────────────────────────────────
# 1. OllamaConfig 配置完整性
# ─────────────────────────────────────────────────────────

class TestOllamaConfig:
    """验证 OllamaConfig 数据类的完整性和默认值。"""

    def test_config_required_fields(self):
        """OllamaConfig 必须有 model 字段。"""
        config = OllamaConfig(model="llama3")
        assert config.model == "llama3"

    def test_config_default_base_url(self):
        """base_url 默认值应为 http://localhost:11434。"""
        config = OllamaConfig(model="llama3")
        assert config.base_url == "http://localhost:11434"

    def test_config_default_timeout(self):
        """timeout 默认值应为 120.0 秒。"""
        config = OllamaConfig(model="llama3")
        assert config.timeout == 120.0

    def test_config_custom_values(self):
        """OllamaConfig 应支持自定义所有字段。"""
        config = OllamaConfig(
            model="mistral",
            base_url="http://custom-host:9999",
            timeout=60.0,
        )
        assert config.model == "mistral"
        assert config.base_url == "http://custom-host:9999"
        assert config.timeout == 60.0

    def test_config_is_dataclass(self):
        """OllamaConfig 应该是一个 dataclass。"""
        import dataclasses
        assert dataclasses.is_dataclass(OllamaConfig)


# ─────────────────────────────────────────────────────────
# 2. OllamaLLM 继承 LLMGateway 正确性
# ─────────────────────────────────────────────────────────

class TestOllamaLLMInheritance:
    """验证 OllamaLLM 正确继承 LLMGateway。"""

    def test_is_subclass_of_llm_gateway(self):
        """OllamaLLM 必须是 LLMGateway 的子类。"""
        assert issubclass(OllamaLLM, LLMGateway)

    def test_implements_generate(self):
        """OllamaLLM 必须实现 generate 方法。"""
        assert hasattr(OllamaLLM, "generate")
        assert callable(getattr(OllamaLLM, "generate"))

    def test_implements_stream(self):
        """OllamaLLM 必须实现 stream 方法。"""
        assert hasattr(OllamaLLM, "stream")
        assert callable(getattr(OllamaLLM, "stream"))

    def test_implements_count_tokens(self):
        """OllamaLLM 必须实现 count_tokens 方法。"""
        assert hasattr(OllamaLLM, "count_tokens")
        assert callable(getattr(OllamaLLM, "count_tokens"))

    def test_implements_close(self):
        """OllamaLLM 必须实现 close 方法。"""
        assert hasattr(OllamaLLM, "close")
        assert callable(getattr(OllamaLLM, "close"))

    def test_implements_async_context_manager(self):
        """OllamaLLM 必须实现 __aenter__ 和 __aexit__。"""
        assert hasattr(OllamaLLM, "__aenter__")
        assert hasattr(OllamaLLM, "__aexit__")

    def test_inherits_get_config(self):
        """OllamaLLM 应继承 LLMGateway.get_config 方法。"""
        assert hasattr(OllamaLLM, "get_config")

    def test_inherits_configured_models(self):
        """OllamaLLM 应继承 LLMGateway.configured_models 属性。"""
        assert hasattr(OllamaLLM, "configured_models")


# ─────────────────────────────────────────────────────────
# 3. 方法签名和异步特性
# ─────────────────────────────────────────────────────────

class TestMethodSignatures:
    """验证方法签名符合设计规范。"""

    def test_generate_is_coroutine(self):
        """generate 方法必须是异步协程。"""
        assert inspect.iscoroutinefunction(OllamaLLM.generate)

    def test_stream_is_async_generator(self):
        """stream 方法必须是异步生成器函数。"""
        # 检查是否为 async generator function
        assert inspect.isasyncgenfunction(OllamaLLM.stream)

    def test_count_tokens_is_coroutine(self):
        """count_tokens 方法必须是异步协程。"""
        assert inspect.iscoroutinefunction(OllamaLLM.count_tokens)

    def test_close_is_coroutine(self):
        """close 方法必须是异步协程。"""
        assert inspect.iscoroutinefunction(OllamaLLM.close)

    def test_aenter_is_coroutine(self):
        """__aenter__ 方法必须是异步协程。"""
        assert inspect.iscoroutinefunction(OllamaLLM.__aenter__)

    def test_aexit_is_coroutine(self):
        """__aexit__ 方法必须是异步协程。"""
        assert inspect.iscoroutinefunction(OllamaLLM.__aexit__)

    def test_generate_signature(self):
        """generate 方法签名应包含 prompt 和 model 参数。"""
        sig = inspect.signature(OllamaLLM.generate)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "prompt" in params
        assert "model" in params

    def test_stream_signature(self):
        """stream 方法签名应包含 prompt 和 model 参数。"""
        sig = inspect.signature(OllamaLLM.stream)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "prompt" in params
        assert "model" in params


# ─────────────────────────────────────────────────────────
# 4. 初始化与配置传递
# ─────────────────────────────────────────────────────────

class TestInitialization:
    """验证 OllamaLLM 初始化逻辑。"""

    def test_init_stores_config(self):
        """初始化后应能通过 config 属性获取配置。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)
        assert llm.config is config
        assert llm.config.model == "llama3"

    def test_init_creates_default_config_in_parent(self):
        """初始化时应将 OllamaConfig 转换为 LLMConfig 传递给父类。"""
        config = OllamaConfig(model="mistral")
        llm = OllamaLLM(config)
        # 父类应有 "default" 配置
        assert "default" in llm._configs
        default_cfg = llm._configs["default"]
        assert default_cfg.provider == LLMProvider.OLLAMA
        assert default_cfg.model == "mistral"

    def test_init_http_client_is_none(self):
        """初始化后 HTTP 客户端应为 None（懒加载）。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)
        assert llm._client is None

    def test_configured_models_contains_default(self):
        """configured_models 应包含 'default'。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)
        assert "default" in llm.configured_models


# ─────────────────────────────────────────────────────────
# 5. 错误处理和超时
# ─────────────────────────────────────────────────────────

class TestErrorHandling:
    """验证错误处理逻辑。"""

    @pytest.mark.asyncio
    async def test_generate_raises_runtime_error_on_http_error(self):
        """当 Ollama API 返回 HTTP 错误时，应抛出 RuntimeError。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "Server Error",
                request=MagicMock(),
                response=mock_response,
            )
        )
        mock_client.is_closed = False
        llm._client = mock_client

        with pytest.raises(RuntimeError, match="Ollama API error"):
            await llm.generate("test prompt")

    @pytest.mark.asyncio
    async def test_generate_raises_runtime_error_on_connection_error(self):
        """当无法连接 Ollama 时，应抛出 RuntimeError。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.is_closed = False
        llm._client = mock_client

        with pytest.raises(RuntimeError, match="Failed to connect to Ollama"):
            await llm.generate("test prompt")

    @pytest.mark.asyncio
    async def test_count_tokens_fallback_on_error(self):
        """当 token 计数 API 失败时，应回退到基于单词的估算。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.is_closed = False
        llm._client = mock_client

        # 应该不抛异常，返回估算值
        result = await llm.count_tokens("hello world test")
        assert isinstance(result, int)
        assert result > 0

    @pytest.mark.asyncio
    async def test_count_tokens_empty_text(self):
        """空文本的 token 计数应返回 0。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )
        mock_client.is_closed = False
        llm._client = mock_client

        result = await llm.count_tokens("")
        assert result == 0


# ─────────────────────────────────────────────────────────
# 6. Context Manager 支持
# ─────────────────────────────────────────────────────────

class TestContextManager:
    """验证异步上下文管理器支持。"""

    @pytest.mark.asyncio
    async def test_context_manager_enter_returns_self(self):
        """__aenter__ 应返回自身。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)
        result = await llm.__aenter__()
        assert result is llm

    @pytest.mark.asyncio
    async def test_context_manager_exit_calls_close(self):
        """__aexit__ 应调用 close 方法。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # 模拟一个已打开的 client
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        llm._client = mock_client

        await llm.__aexit__(None, None, None)
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_context_manager_usage(self):
        """应支持 async with 语句。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # 模拟 client
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        llm._client = mock_client

        async with llm as gateway:
            assert gateway is llm

        # 退出后应调用 close
        mock_client.aclose.assert_awaited_once()


# ─────────────────────────────────────────────────────────
# 7. HTTP 客户端懒加载
# ─────────────────────────────────────────────────────────

class TestHTTPClientLazyInit:
    """验证 HTTP 客户端懒加载逻辑。"""

    def test_get_client_creates_client_on_first_call(self):
        """首次调用 _get_client 应创建新的 httpx.AsyncClient。"""
        config = OllamaConfig(model="llama3", base_url="http://localhost:11434")
        llm = OllamaLLM(config)

        assert llm._client is None
        client = llm._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert llm._client is client

    def test_get_client_reuses_existing_client(self):
        """多次调用 _get_client 应返回同一个客户端实例。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        client1 = llm._get_client()
        client2 = llm._get_client()
        assert client1 is client2

    def test_get_client_creates_new_when_closed(self):
        """当客户端已关闭时，_get_client 应创建新实例。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # 手动注入一个模拟的已关闭客户端
        mock_closed_client = MagicMock()
        mock_closed_client.is_closed = True
        llm._client = mock_closed_client

        # 调用 _get_client 应创建新实例
        client2 = llm._get_client()
        assert client2 is not mock_closed_client
        assert client2 is llm._client
        assert isinstance(client2, httpx.AsyncClient)


# ─────────────────────────────────────────────────────────
# 8. API 端点常量
# ─────────────────────────────────────────────────────────

class TestAPIEndpoints:
    """验证 API 端点常量定义。"""

    def test_generate_endpoint(self):
        """_generate_endpoint 应为 /api/generate。"""
        assert OllamaLLM._generate_endpoint == "/api/generate"

    def test_tokenize_endpoint(self):
        """_tokenize_endpoint 应为 /api/embeddings。"""
        assert OllamaLLM._tokenize_endpoint == "/api/embeddings"


# ─────────────────────────────────────────────────────────
# 9. generate 方法参数传递
# ─────────────────────────────────────────────────────────

class TestGenerateParams:
    """验证 generate 方法正确传递参数。"""

    @pytest.mark.asyncio
    async def test_generate_passes_prompt_to_api(self):
        """generate 应将 prompt 传递给 Ollama API。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello!"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        result = await llm.generate("What is Python?")

        # 验证调用参数
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["prompt"] == "What is Python?"
        assert payload["model"] == "llama3"
        assert payload["stream"] is False
        assert result == "Hello!"

    @pytest.mark.asyncio
    async def test_generate_passes_temperature(self):
        """generate 应支持 temperature 参数。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        await llm.generate("test", temperature=0.7)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_generate_ignores_model_param(self):
        """generate 的 model 参数应被忽略，使用配置中的模型。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        # 传入不同的 model 参数，应被忽略
        await llm.generate("test", model="gpt-4")

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "llama3"  # 使用配置的模型，不是 gpt-4


# ─────────────────────────────────────────────────────────
# 10. close 方法
# ─────────────────────────────────────────────────────────

class TestCloseMethod:
    """验证 close 方法行为。"""

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """close 应关闭 HTTP 客户端。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.aclose = AsyncMock()
        llm._client = mock_client

        await llm.close()
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_noop_when_no_client(self):
        """当没有客户端时，close 不应抛异常。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)
        assert llm._client is None
        # 不应抛异常
        await llm.close()

    @pytest.mark.asyncio
    async def test_close_noop_when_already_closed(self):
        """当客户端已关闭时，close 不应再次调用 aclose。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_client = AsyncMock()
        mock_client.is_closed = True
        mock_client.aclose = AsyncMock()
        llm._client = mock_client

        await llm.close()
        mock_client.aclose.assert_not_awaited()
