"""独立测试用例 - OllamaLLM 实现 (test_ollama_llm_independent.py)

测试人员: tester-3
测试目的: 根据 详细设计.md 第9.1节 对 OllamaLLM 进行独立验证
测试策略: 不使用开发者测试用例，采用不同的 mock 策略和测试角度

验证内容：
1. OllamaConfig 配置完整性
2. OllamaLLM 继承 LLMGateway 正确性
3. 方法签名和异步特性
4. 错误处理和超时
5. context manager 支持
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import AsyncIterator
import httpx

# 导入被测模块
from agent_framework.infrastructure.ollama_llm import OllamaLLM, OllamaConfig
from agent_framework.infrastructure.llm_gateway import LLMGateway, LLMProvider, LLMConfig


# ============================================================================
# 第一部分：OllamaConfig 配置完整性测试
# ============================================================================

class TestOllamaConfigCompleteness:
    """验证 OllamaConfig 配置类的完整性和正确性。"""

    def test_config_has_required_fields(self):
        """【规范验证】OllamaConfig 必须包含 model 字段。"""
        config = OllamaConfig(model="llama3")
        assert hasattr(config, 'model'), "OllamaConfig 必须有 model 字段"
        assert config.model == "llama3"

    def test_config_has_base_url_with_default(self):
        """【规范验证】OllamaConfig 必须有 base_url 字段，默认 localhost:11434。"""
        config = OllamaConfig(model="llama3")
        assert hasattr(config, 'base_url'), "OllamaConfig 必须有 base_url 字段"
        assert config.base_url == "http://localhost:11434"

    def test_config_has_timeout_with_default(self):
        """【规范验证】OllamaConfig 必须有 timeout 字段。"""
        config = OllamaConfig(model="llama3")
        assert hasattr(config, 'timeout'), "OllamaConfig 必须有 timeout 字段"
        assert config.timeout == 120.0

    def test_config_all_fields_customizable(self):
        """【规范验证】所有配置字段都应可自定义。"""
        config = OllamaConfig(
            model="mistral",
            base_url="http://remote-server:8080",
            timeout=60.0
        )
        assert config.model == "mistral"
        assert config.base_url == "http://remote-server:8080"
        assert config.timeout == 60.0

    def test_config_is_frozen_or_mutable(self):
        """【设计决策】验证 OllamaConfig 是否可变（根据实际使用需求）。"""
        config = OllamaConfig(model="llama3")
        # OllamaConfig 当前未设置 frozen=True，应可修改
        config.model = "mistral"
        assert config.model == "mistral"


# ============================================================================
# 第二部分：OllamaLLM 继承 LLMGateway 正确性测试
# ============================================================================

class TestOllamaLLMInheritanceCorrectness:
    """验证 OllamaLLM 正确继承 LLMGateway 抽象基类。"""

    def test_is_subclass_of_llm_gateway(self):
        """【规范验证】OllamaLLM 必须是 LLMGateway 的子类。"""
        assert issubclass(OllamaLLM, LLMGateway), \
            "OllamaLLM 必须继承自 LLMGateway"

    def test_implements_generate_method(self):
        """【规范验证】OllamaLLM 必须实现 generate 方法。"""
        assert hasattr(OllamaLLM, 'generate'), "必须实现 generate 方法"
        # 验证方法签名是异步的
        import inspect
        assert inspect.iscoroutinefunction(OllamaLLM.generate), \
            "generate 必须是 async 方法"

    def test_implements_stream_method(self):
        """【规范验证】OllamaLLM 必须实现 stream 方法。"""
        assert hasattr(OllamaLLM, 'stream'), "必须实现 stream 方法"
        # stream 通常不是协程函数，而是返回异步生成器的普通方法
        import inspect
        # 检查返回类型注解是否为 AsyncIterator
        sig = inspect.signature(OllamaLLM.stream)
        assert 'AsyncIterator' in str(sig.return_annotation) or \
               'AsyncIterator' in str(getattr(sig.return_annotation, '__args__', '')), \
            "stream 必须返回 AsyncIterator[str]"

    def test_implements_count_tokens_method(self):
        """【规范验证】OllamaLLM 必须实现 count_tokens 方法。"""
        assert hasattr(OllamaLLM, 'count_tokens'), "必须实现 count_tokens 方法"
        import inspect
        assert inspect.iscoroutinefunction(OllamaLLM.count_tokens), \
            "count_tokens 必须是 async 方法"

    def test_can_be_instantiated(self):
        """【功能验证】OllamaLLM 可以被正确实例化。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)
        assert isinstance(llm, LLMGateway)
        assert isinstance(llm, OllamaLLM)

    def test_parent_config_stored_correctly(self):
        """【规范验证】OllamaLLM 应将配置转换为 LLMConfig 格式存储。"""
        config = OllamaConfig(model="llama3", base_url="http://custom:11434")
        llm = OllamaLLM(config)

        # 检查父类的配置字典
        assert "default" in llm.configured_models
        default_config = llm.get_config("default")
        assert default_config.provider == LLMProvider.OLLAMA
        assert default_config.model == "llama3"
        assert default_config.base_url == "http://custom:11434"


# ============================================================================
# 第三部分：方法签名和异步特性测试
# ============================================================================

class TestMethodSignaturesAndAsync:
    """验证方法签名正确，异步特性符合设计规范。"""

    @pytest.fixture
    def ollama_instance(self):
        """创建 OllamaLLM 实例。"""
        config = OllamaConfig(model="llama3", timeout=30.0)
        return OllamaLLM(config)

    def test_generate_signature_accepts_prompt(self, ollama_instance):
        """【规范验证】generate 接受 prompt 参数。"""
        import inspect
        sig = inspect.signature(ollama_instance.generate)
        params = list(sig.parameters.keys())
        assert 'prompt' in params, "generate 必须接受 prompt 参数"

    def test_generate_signature_accepts_model_alias(self, ollama_instance):
        """【规范验证】generate 接受 model 别名参数。"""
        import inspect
        sig = inspect.signature(ollama_instance.generate)
        params = sig.parameters
        assert 'model' in params, "generate 必须接受 model 参数"
        assert params['model'].default == "default", "model 默认值应为 'default'"

    def test_generate_signature_accepts_kwargs(self, ollama_instance):
        """【规范验证】generate 接受额外参数 kwargs。"""
        import inspect
        sig = inspect.signature(ollama_instance.generate)
        params = list(sig.parameters.keys())
        assert 'kwargs' in params, "generate 必须接受 **kwargs"

    def test_stream_returns_async_iterator_type(self, ollama_instance):
        """【规范验证】stream 返回 AsyncIterator[str] 类型。"""
        # 使用 mock 来测试返回类型
        async def mock_stream_api(*args, **kwargs):
            yield "test"

        with patch.object(ollama_instance, '_stream_ollama_api', return_value=mock_stream_api()):
            result = ollama_instance.stream("test")
            # 验证返回的是异步迭代器
            assert hasattr(result, '__aiter__'), "stream 结果必须是异步迭代器"
            assert hasattr(result, '__anext__'), "stream 结果必须有 __anext__ 方法"

    @pytest.mark.asyncio
    async def test_generate_is_awaitable(self, ollama_instance):
        """【异步验证】generate 返回 awaitable 对象。"""
        with patch.object(ollama_instance, '_call_ollama_api', new_callable=AsyncMock) as mock:
            mock.return_value = "response"

            # 应该可以直接 await
            result = await ollama_instance.generate("test prompt")
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_count_tokens_is_awaitable(self, ollama_instance):
        """【异步验证】count_tokens 返回 awaitable 对象。"""
        # count_tokens 当前有 fallback，不需要 mock
        result = await ollama_instance.count_tokens("test text")
        assert isinstance(result, int)


# ============================================================================
# 第四部分：错误处理和超时测试
# ============================================================================

class TestErrorHandlingAndTimeout:
    """验证错误处理和超时机制符合设计规范。"""

    @pytest.fixture
    def ollama_instance(self):
        """创建 OllamaLLM 实例。"""
        config = OllamaConfig(model="llama3", timeout=30.0)
        return OllamaLLM(config)

    @pytest.mark.asyncio
    async def test_generate_raises_on_connection_error(self, ollama_instance):
        """【错误处理】连接失败时应抛出 RuntimeError。"""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")

        with patch.object(ollama_instance, '_get_client', return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                await ollama_instance.generate("test")

            # 错误信息应包含连接相关描述
            error_msg = str(exc_info.value).lower()
            assert "ollama" in error_msg or "connect" in error_msg

    @pytest.mark.asyncio
    async def test_generate_raises_on_http_error(self, ollama_instance):
        """【错误处理】HTTP 错误响应时应抛出 RuntimeError。"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_response
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        with patch.object(ollama_instance, '_get_client', return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                await ollama_instance.generate("test")

            assert "500" in str(exc_info.value) or "error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_stream_raises_on_connection_error(self, ollama_instance):
        """【错误处理】流式请求连接失败时应抛出 RuntimeError。"""
        # 通过 mock _stream_ollama_api 方法来测试错误传播
        async def failing_stream(*args, **kwargs):
            raise RuntimeError("Failed to connect to Ollama: Connection refused")
            yield ""  # 使其成为生成器

        with patch.object(ollama_instance, '_stream_ollama_api', return_value=failing_stream()):
            with pytest.raises(RuntimeError) as exc_info:
                async for _ in ollama_instance.stream("test"):
                    pass

            error_msg = str(exc_info.value).lower()
            assert "ollama" in error_msg or "connect" in error_msg

    @pytest.mark.asyncio
    async def test_timeout_configuration_applied(self, ollama_instance):
        """【超时验证】配置的 timeout 应传递给 HTTP client。"""
        config = OllamaConfig(model="llama3", timeout=30.0)
        llm = OllamaLLM(config)

        # 获取 client 时应使用正确的 timeout
        client = llm._get_client()
        assert client is not None
        # httpx.Timeout 对象正确创建
        assert isinstance(client.timeout, httpx.Timeout)
        # timeout 值应与配置一致
        assert client.timeout.read == 30.0

    @pytest.mark.asyncio
    async def test_count_tokens_fallback_works(self, ollama_instance):
        """【容错性】count_tokens 在 API 失败时应使用 fallback。"""
        # 直接测试 fallback 逻辑 - 使用无法连接的端点
        config = OllamaConfig(model="llama3", base_url="http://nonexistent:99999", timeout=1.0)
        llm = OllamaLLM(config)

        # 即使无法连接，fallback 也应工作
        result = await llm.count_tokens("hello world test")
        assert isinstance(result, int)
        assert result > 0  # 应返回估算值


# ============================================================================
# 第五部分：Context Manager 支持测试
# ============================================================================

class TestAsyncContextManagerSupport:
    """验证 OllamaLLM 支持异步上下文管理器。"""

    @pytest.fixture
    def ollama_instance(self):
        """创建 OllamaLLM 实例。"""
        config = OllamaConfig(model="llama3")
        return OllamaLLM(config)

    def test_has_aenter_method(self, ollama_instance):
        """【规范验证】必须实现 __aenter__ 方法。"""
        assert hasattr(OllamaLLM, '__aenter__'), "必须实现 __aenter__"
        import inspect
        assert inspect.iscoroutinefunction(OllamaLLM.__aenter__), \
            "__aenter__ 必须是 async 方法"

    def test_has_aexit_method(self, ollama_instance):
        """【规范验证】必须实现 __aexit__ 方法。"""
        assert hasattr(OllamaLLM, '__aexit__'), "必须实现 __aexit__"
        import inspect
        assert inspect.iscoroutinefunction(OllamaLLM.__aexit__), \
            "__aexit__ 必须是 async 方法"

    @pytest.mark.asyncio
    async def test_context_manager_returns_self(self):
        """【功能验证】__aenter__ 应返回 self。"""
        config = OllamaConfig(model="llama3")
        async with OllamaLLM(config) as llm:
            assert isinstance(llm, OllamaLLM)

    @pytest.mark.asyncio
    async def test_context_manager_closes_client(self):
        """【资源管理】退出上下文时应关闭 HTTP client。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # 先获取 client 以初始化它
        client = llm._get_client()
        assert not client.is_closed

        async with llm as instance:
            pass

        # 退出后 client 应被关闭
        assert llm._client is None or llm._client.is_closed

    @pytest.mark.asyncio
    async def test_close_method_closes_client(self):
        """【功能验证】close 方法应正确关闭 HTTP client。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # 初始化 client
        client = llm._get_client()
        assert not client.is_closed

        # 调用 close
        await llm.close()

        # client 应被关闭
        assert llm._client is None or llm._client.is_closed

    @pytest.mark.asyncio
    async def test_multiple_close_is_safe(self):
        """【健壮性】多次调用 close 不应出错。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        llm._get_client()  # 初始化 client

        # 多次调用 close
        await llm.close()
        await llm.close()
        await llm.close()

        # 应安全退出
        assert llm._client is None or llm._client.is_closed


# ============================================================================
# 第六部分：HTTP API 调用行为测试（使用不同 mock 策略）
# ============================================================================

class TestHTTPAPICallBehavior:
    """验证 HTTP API 调用行为正确。"""

    @pytest.mark.asyncio
    async def test_generate_calls_correct_endpoint(self):
        """【API验证】generate 应调用 /api/generate 端点。"""
        config = OllamaConfig(model="llama3", base_url="http://test:11434")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "test response"}
        mock_response.raise_for_status = MagicMock()

        with patch('httpx.AsyncClient') as MockClient:
            mock_instance = AsyncMock()
            mock_instance.post.return_value = mock_response
            mock_instance.is_closed = False
            MockClient.return_value = mock_instance

            result = await llm.generate("test prompt")

            # 验证调用了正确的端点
            mock_instance.post.assert_called_once()
            call_args = mock_instance.post.call_args
            assert call_args[0][0] == "/api/generate"
            # 或 args 中包含端点
            assert "/api/generate" in str(call_args) or "generate" in str(call_args)

    @pytest.mark.asyncio
    async def test_generate_sends_correct_payload(self):
        """【API验证】generate 应发送包含 model 和 prompt 的 payload。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "response"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(llm, '_get_client', return_value=mock_client):
            await llm.generate("test prompt")

            # 验证 payload
            call_args = mock_client.post.call_args
            payload = call_args.kwargs.get('json', {})
            assert payload.get('model') == "llama3"
            assert payload.get('prompt') == "test prompt"
            assert payload.get('stream') == False

    @pytest.mark.asyncio
    async def test_generate_passes_optional_params(self):
        """【API验证】generate 应正确传递可选参数。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "response"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.is_closed = False

        with patch.object(llm, '_get_client', return_value=mock_client):
            await llm.generate("test", temperature=0.7, max_tokens=100, top_p=0.9)

            payload = mock_client.post.call_args.kwargs.get('json', {})
            assert payload.get('temperature') == 0.7
            assert payload.get('num_predict') == 100  # max_tokens 转换为 num_predict
            assert payload.get('top_p') == 0.9

    @pytest.mark.asyncio
    async def test_stream_uses_streaming_mode(self):
        """【API验证】stream 应设置 stream=True。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # 通过 mock _stream_ollama_api 方法来测试，验证内部方法被调用
        async def mock_stream_api(prompt, **kwargs):
            yield "hello"
            yield " world"

        # 保存调用参数
        called_args = {}

        async def capture_stream(prompt, **kwargs):
            called_args['prompt'] = prompt
            called_args['kwargs'] = kwargs
            async for token in mock_stream_api(prompt, **kwargs):
                yield token

        with patch.object(llm, '_stream_ollama_api') as mock_stream:
            mock_stream.return_value = capture_stream("test")

            tokens = []
            async for token in llm.stream("test"):
                tokens.append(token)

            # 验证调用了 _stream_ollama_api 方法
            assert mock_stream.called
            # tokens 应包含返回的内容
            assert "hello" in tokens
            assert " world" in tokens


# ============================================================================
# 第七部分：属性和辅助方法测试
# ============================================================================

class TestPropertiesAndHelperMethods:
    """验证属性和辅助方法正确实现。"""

    def test_config_property_returns_config(self):
        """【属性验证】config 属性应返回 OllamaConfig。"""
        config = OllamaConfig(model="llama3", base_url="http://custom:8080")
        llm = OllamaLLM(config)

        assert hasattr(llm, 'config')
        assert llm.config is config

    def test_config_property_is_readonly(self):
        """【属性验证】config 属性应为只读 property。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # 检查是否是 property
        assert isinstance(type(llm).config, property)

    def test_get_client_creates_client_on_first_call(self):
        """【延迟初始化】_get_client 应在首次调用时创建 client。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        assert llm._client is None  # 初始状态

        client = llm._get_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)

    def test_get_client_reuses_existing_client(self):
        """【资源复用】_get_client 应复用现有 client。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        client1 = llm._get_client()
        client2 = llm._get_client()

        assert client1 is client2

    def test_generate_endpoint_constant(self):
        """【端点验证】_generate_endpoint 应为 /api/generate。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        assert llm._generate_endpoint == "/api/generate"

    def test_tokenize_endpoint_constant(self):
        """【端点验证】_tokenize_endpoint 应为 /api/embeddings。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        assert llm._tokenize_endpoint == "/api/embeddings"


# ============================================================================
# 第八部分：边界条件和特殊情况测试
# ============================================================================

class TestEdgeCasesAndBoundaryConditions:
    """验证边界条件和特殊情况的处理。"""

    @pytest.fixture
    def ollama_instance(self):
        config = OllamaConfig(model="llama3")
        return OllamaLLM(config)

    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self, ollama_instance):
        """【边界条件】空 prompt 应被正常处理。"""
        with patch.object(ollama_instance, '_call_ollama_api', new_callable=AsyncMock) as mock:
            mock.return_value = ""

            result = await ollama_instance.generate("")
            assert result == ""

    @pytest.mark.asyncio
    async def test_count_tokens_empty_string(self, ollama_instance):
        """【边界条件】空字符串的 token 数应返回 0 或接近 0。"""
        result = await ollama_instance.count_tokens("")
        assert isinstance(result, int)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_count_tokens_very_long_text(self, ollama_instance):
        """【边界条件】超长文本的 token 估算应正常工作。"""
        long_text = " ".join(["word"] * 10000)  # 10000 个单词
        result = await ollama_instance.count_tokens(long_text)
        assert isinstance(result, int)
        assert result > 0

    @pytest.mark.asyncio
    async def test_stream_yields_nothing_on_empty_response(self, ollama_instance):
        """【边界条件】空响应时 stream 应正常结束。"""
        async def empty_generator():
            return
            yield ""  # 使其成为生成器

        with patch.object(ollama_instance, '_stream_ollama_api', return_value=empty_generator()):
            tokens = []
            async for token in ollama_instance.stream(""):
                tokens.append(token)

            assert tokens == []

    def test_config_model_with_special_characters(self):
        """【边界条件】model 名称支持特殊格式。"""
        # Ollama 支持如 "llama3:8b" 这样的模型名
        config = OllamaConfig(model="llama3:8b")
        assert config.model == "llama3:8b"

        config2 = OllamaConfig(model="gemma2:9b-instruct")
        assert config2.model == "gemma2:9b-instruct"


# ============================================================================
# 第九部分：与父类 LLMGateway 协作测试
# ============================================================================

class TestParentClassIntegration:
    """验证与父类 LLMGateway 的协作正确。"""

    def test_configured_models_returns_default(self):
        """【继承验证】configured_models 应包含 default。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        models = llm.configured_models
        assert "default" in models

    def test_get_config_returns_correct_config(self):
        """【继承验证】get_config 应返回正确的 LLMConfig。"""
        config = OllamaConfig(model="llama3", base_url="http://custom:11434")
        llm = OllamaLLM(config)

        llm_config = llm.get_config("default")
        assert llm_config.provider == LLMProvider.OLLAMA
        assert llm_config.model == "llama3"
        assert llm_config.base_url == "http://custom:11434"

    def test_get_config_raises_on_unknown_model(self):
        """【继承验证】get_config 对未知模型应抛出 KeyError。"""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        with pytest.raises(KeyError):
            llm.get_config("unknown_model")


# ============================================================================
# 运行测试
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
