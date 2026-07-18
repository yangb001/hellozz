"""Independent verification tests for LLMGateway chat()/stream_chat() interfaces.

This module tests the new chat interface methods:
1. LLMGateway.chat() returns ChatResponse
2. LLMGateway.stream_chat() yields StreamChatResponse
3. OpenAILLM and OllamaLLM implementations
4. Backward compatibility with stream() method

Tests are written independently from developer tests.
"""
import pytest
import inspect
import json
import dataclasses
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncIterator, Dict, List, Any, get_type_hints

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_framework.infrastructure.llm_gateway import (
    LLMGateway, LLMConfig, LLMProvider,
    ChatResponse, StreamChatResponse, ChatResponseType,
    ToolCall, FunctionCall
)
from agent_framework.infrastructure.openai_llm import OpenAILLM, OpenAIConfig
from agent_framework.infrastructure.ollama_llm import OllamaLLM, OllamaConfig


# ============================================================================
# 1. Verify chat()/stream_chat() method signatures in abstract class
# ============================================================================

class TestLLMGatewayChatMethodSignature:
    """Verify LLMGateway has correct chat() method signature."""

    def test_llm_gateway_has_chat_method(self):
        """LLMGateway must have chat method."""
        assert hasattr(LLMGateway, "chat"), "LLMGateway 缺少 chat 方法"
        method = getattr(LLMGateway, "chat")
        assert getattr(method, "__isabstractmethod__", False), "chat 应为抽象方法"

    def test_chat_accepts_messages_parameter(self):
        """chat method must accept messages parameter."""
        sig = inspect.signature(LLMGateway.chat)
        params = list(sig.parameters.keys())
        assert "messages" in params, "chat 缺少 messages 参数"

    def test_chat_accepts_tools_parameter(self):
        """chat method must accept tools parameter."""
        sig = inspect.signature(LLMGateway.chat)
        params = list(sig.parameters.keys())
        assert "tools" in params, "chat 缺少 tools 参数"

    def test_chat_accepts_model_parameter(self):
        """chat method must accept model parameter."""
        sig = inspect.signature(LLMGateway.chat)
        params = list(sig.parameters.keys())
        assert "model" in params, "chat 缺少 model 参数"
        model_param = sig.parameters["model"]
        assert model_param.default == "default", "model 默认值应为 'default'"

    def test_chat_method_is_async(self):
        """chat method must be async."""
        assert inspect.iscoroutinefunction(LLMGateway.chat), "chat 应为 async 方法"


class TestLLMGatewayStreamChatMethodSignature:
    """Verify LLMGateway has correct stream_chat() method signature."""

    def test_llm_gateway_has_stream_chat_method(self):
        """LLMGateway must have stream_chat method."""
        assert hasattr(LLMGateway, "stream_chat"), "LLMGateway 缺少 stream_chat 方法"
        method = getattr(LLMGateway, "stream_chat")
        assert getattr(method, "__isabstractmethod__", False), "stream_chat 应为抽象方法"

    def test_stream_chat_accepts_messages_parameter(self):
        """stream_chat method must accept messages parameter."""
        sig = inspect.signature(LLMGateway.stream_chat)
        params = list(sig.parameters.keys())
        assert "messages" in params, "stream_chat 缺少 messages 参数"

    def test_stream_chat_accepts_tools_parameter(self):
        """stream_chat method must accept tools parameter."""
        sig = inspect.signature(LLMGateway.stream_chat)
        params = list(sig.parameters.keys())
        assert "tools" in params, "stream_chat 缺少 tools 参数"

    def test_stream_chat_accepts_model_parameter(self):
        """stream_chat method must accept model parameter."""
        sig = inspect.signature(LLMGateway.stream_chat)
        params = list(sig.parameters.keys())
        assert "model" in params, "stream_chat 缺少 model 参数"
        model_param = sig.parameters["model"]
        assert model_param.default == "default", "model 默认值应为 'default'"

    def test_stream_chat_method_is_async(self):
        """stream_chat method must be async."""
        assert inspect.iscoroutinefunction(LLMGateway.stream_chat), "stream_chat 应为 async 方法"


# ============================================================================
# 2. Verify ChatResponse and StreamChatResponse data classes
# ============================================================================

class TestChatResponseDataclass:
    """Verify ChatResponse dataclass structure."""

    def test_chat_response_has_content_field(self):
        """ChatResponse must have content field."""
        fields = {f.name for f in dataclasses.fields(ChatResponse)}
        assert "content" in fields, "ChatResponse 缺少 content 字段"

    def test_chat_response_has_tool_calls_field(self):
        """ChatResponse must have tool_calls field."""
        fields = {f.name for f in dataclasses.fields(ChatResponse)}
        assert "tool_calls" in fields, "ChatResponse 缺少 tool_calls 字段"

    def test_chat_response_can_be_instantiated(self):
        """ChatResponse can be created with content only."""
        response = ChatResponse(content="Hello")
        assert response.content == "Hello"
        assert response.tool_calls == []  # Default is empty list, not None

    def test_chat_response_with_tool_calls(self):
        """ChatResponse can be created with tool_calls."""
        tool_calls = [ToolCall(id="call_1", type="function", function=FunctionCall(name="test", arguments="{}"))]
        response = ChatResponse(content="", tool_calls=tool_calls)
        assert response.tool_calls == tool_calls


class TestStreamChatResponseDataclass:
    """Verify StreamChatResponse dataclass structure."""

    def test_stream_chat_response_has_type_field(self):
        """StreamChatResponse must have type field."""
        fields = {f.name for f in dataclasses.fields(StreamChatResponse)}
        assert "type" in fields, "StreamChatResponse 缺少 type 字段"

    def test_stream_chat_response_has_content_field(self):
        """StreamChatResponse must have content field."""
        fields = {f.name for f in dataclasses.fields(StreamChatResponse)}
        assert "content" in fields, "StreamChatResponse 缺少 content 字段"

    def test_stream_chat_response_has_tool_call_field(self):
        """StreamChatResponse must have tool_call field."""
        fields = {f.name for f in dataclasses.fields(StreamChatResponse)}
        assert "tool_call" in fields, "StreamChatResponse 缺少 tool_call 字段"

    def test_stream_chat_response_has_finish_reason_field(self):
        """StreamChatResponse must have finish_reason field."""
        fields = {f.name for f in dataclasses.fields(StreamChatResponse)}
        assert "finish_reason" in fields, "StreamChatResponse 缺少 finish_reason 字段"

    def test_stream_chat_response_content_type(self):
        """StreamChatResponse can be created with CONTENT type."""
        response = StreamChatResponse(
            type=ChatResponseType.CONTENT,
            content="Hello"
        )
        assert response.type == ChatResponseType.CONTENT
        assert response.content == "Hello"

    def test_stream_chat_response_tool_call_type(self):
        """StreamChatResponse can be created with tool call type."""
        tool_call = ToolCall(
            id="call_1",
            type="function",
            function=FunctionCall(name="test", arguments="{}")
        )
        response = StreamChatResponse(
            type=ChatResponseType.TOOL_CALL_START,
            tool_call=tool_call
        )
        assert response.type == ChatResponseType.TOOL_CALL_START
        assert response.tool_call is not None

    def test_stream_chat_response_done_type(self):
        """StreamChatResponse can be created with DONE type."""
        response = StreamChatResponse(
            type=ChatResponseType.DONE,
            finish_reason="stop"
        )
        assert response.type == ChatResponseType.DONE
        assert response.finish_reason == "stop"


# ============================================================================
# 3. OpenAILLM.chat() implementation tests
# ============================================================================

class TestOpenAILLMChatMethod:
    """Verify OpenAILLM.chat() implementation."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OpenAIConfig(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key"
        )

    @pytest.fixture
    def llm(self, config):
        """Create OpenAILLM instance."""
        return OpenAILLM(config)

    def test_openai_llm_has_chat_method(self, llm):
        """OpenAILLM must have chat method."""
        assert hasattr(llm, "chat"), "OpenAILLM 缺少 chat 方法"

    @pytest.mark.asyncio
    async def test_chat_returns_chat_response(self, llm):
        """chat() should return ChatResponse instance."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello from chat",
                    "tool_calls": None
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        result = await llm.chat(messages)

        assert isinstance(result, ChatResponse), f"应返回 ChatResponse，实际返回 {type(result)}"
        assert result.content == "Hello from chat"

    @pytest.mark.asyncio
    async def test_chat_sends_messages_to_api(self, llm):
        """chat() should send messages array to API."""
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

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        await llm.chat(messages)

        call_args = mock_client.post.call_args
        payload = call_args[1].get("json")
        assert payload is not None, "应该发送 JSON payload"
        assert "messages" in payload, "payload 应包含 messages"
        assert payload["messages"] == messages, "messages 应原样传递"

    @pytest.mark.asyncio
    async def test_chat_with_tools(self, llm):
        """chat() should send tools to API when provided."""
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

        messages = [{"role": "user", "content": "Use the calculator"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "A calculator tool",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        await llm.chat(messages, tools=tools)

        call_args = mock_client.post.call_args
        payload = call_args[1].get("json")
        assert "tools" in payload, "payload 应包含 tools"
        assert payload["tools"] == tools, "tools 应原样传递"

    @pytest.mark.asyncio
    async def test_chat_extracts_tool_calls_from_response(self, llm):
        """chat() should extract tool_calls from response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'}
                        }
                    ]
                }
            }]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
        result = await llm.chat(messages)

        assert result.tool_calls is not None, "应返回 tool_calls"
        assert len(result.tool_calls) == 1
        # ToolCall is a dataclass, use attribute access not subscript
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].function.name == "get_weather"

    @pytest.mark.asyncio
    async def test_chat_http_error_raises_runtime_error(self, llm):
        """chat() should raise RuntimeError on HTTP error."""
        import httpx

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
            await llm.chat([{"role": "user", "content": "test"}])


# ============================================================================
# 4. OpenAILLM.stream_chat() implementation tests
# ============================================================================

class TestOpenAILLMStreamChatMethod:
    """Verify OpenAILLM.stream_chat() implementation."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OpenAIConfig(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key"
        )

    @pytest.fixture
    def llm(self, config):
        """Create OpenAILLM instance."""
        return OpenAILLM(config)

    def test_openai_llm_has_stream_chat_method(self, llm):
        """OpenAILLM must have stream_chat method."""
        assert hasattr(llm, "stream_chat"), "OpenAILLM 缺少 stream_chat 方法"

    @pytest.mark.asyncio
    async def test_stream_chat_returns_async_iterator(self, llm):
        """stream_chat() should return an async iterator."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":" World"},"finish_reason":"stop"}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        result = llm.stream_chat([{"role": "user", "content": "test"}])
        assert hasattr(result, "__aiter__"), "应返回 async iterator"

    @pytest.mark.asyncio
    async def test_stream_chat_yields_stream_chat_responses(self, llm):
        """stream_chat() should yield StreamChatResponse objects."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":" World"},"finish_reason":"stop"}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        responses = []
        async for resp in llm.stream_chat([{"role": "user", "content": "test"}]):
            responses.append(resp)
            assert isinstance(resp, StreamChatResponse), f"应返回 StreamChatResponse，实际返回 {type(resp)}"

        assert len(responses) >= 2, f"应至少返回2个响应，实际返回 {len(responses)}"

    @pytest.mark.asyncio
    async def test_stream_chat_yields_content_tokens(self, llm):
        """stream_chat() should yield content tokens."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":" World"},"finish_reason":"stop"}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        content_tokens = []
        async for resp in llm.stream_chat([{"role": "user", "content": "test"}]):
            if resp.type == ChatResponseType.CONTENT:
                content_tokens.append(resp.content)

        assert "Hello" in content_tokens, "应包含 'Hello' token"
        assert " World" in content_tokens, "应包含 ' World' token"

    @pytest.mark.asyncio
    async def test_stream_chat_yields_done_event(self, llm):
        """stream_chat() should yield DONE event with finish_reason."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices":[{"delta":{"content":"Done"},"finish_reason":"stop"}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        done_received = False
        finish_reason_received = None
        async for resp in llm.stream_chat([{"role": "user", "content": "test"}]):
            if resp.type == ChatResponseType.DONE:
                done_received = True
                finish_reason_received = resp.finish_reason

        assert done_received, "应收到 DONE 事件"
        assert finish_reason_received == "stop", f"finish_reason 应为 'stop'，实际为 {finish_reason_received}"

    @pytest.mark.asyncio
    async def test_stream_chat_with_tool_calls(self, llm):
        """stream_chat() should handle tool call events."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            # Tool call start
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}'
            # Tool call argument (complete JSON in one chunk)
            yield 'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\\"city\\\":\\\"Beijing\\\"}"}}]},"finish_reason":null}]}'
            # Done
            yield 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}'
            yield 'data: [DONE]'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        tool_call_events = []
        async for resp in llm.stream_chat([{"role": "user", "content": "test"}]):
            if resp.type in (ChatResponseType.TOOL_CALL_START, ChatResponseType.TOOL_CALL_ARGUMENT, ChatResponseType.TOOL_CALL_END):
                tool_call_events.append(resp.type)

        assert ChatResponseType.TOOL_CALL_START in tool_call_events, "应收到 TOOL_CALL_START 事件"
        assert ChatResponseType.TOOL_CALL_ARGUMENT in tool_call_events, "应收到 TOOL_CALL_ARGUMENT 事件"


# ============================================================================
# 5. OllamaLLM.chat() implementation tests
# ============================================================================

class TestOllamaLLMChatMethod:
    """Verify OllamaLLM.chat() implementation."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OllamaConfig(
            model="llama3",
            base_url="http://localhost:11434"
        )

    @pytest.fixture
    def llm(self, config):
        """Create OllamaLLM instance."""
        return OllamaLLM(config)

    def test_ollama_llm_has_chat_method(self, llm):
        """OllamaLLM must have chat method."""
        assert hasattr(llm, "chat"), "OllamaLLM 缺少 chat 方法"

    @pytest.mark.asyncio
    async def test_chat_returns_chat_response(self, llm):
        """chat() should return ChatResponse instance."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "message": {"content": "Hello from Ollama"}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        result = await llm.chat(messages)

        assert isinstance(result, ChatResponse), f"应返回 ChatResponse，实际返回 {type(result)}"
        assert result.content == "Hello from Ollama"

    @pytest.mark.asyncio
    async def test_chat_sends_messages_to_api(self, llm):
        """chat() should send messages to /api/chat endpoint."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"message": {"content": "response"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        llm._client = mock_client

        messages = [{"role": "user", "content": "Hello"}]
        await llm.chat(messages)

        call_args = mock_client.post.call_args
        endpoint = call_args[0][0]
        assert "/api/chat" in endpoint, f"应使用 /api/chat endpoint，实际为 {endpoint}"

        payload = call_args[1].get("json")
        assert "messages" in payload, "payload 应包含 messages"


# ============================================================================
# 6. OllamaLLM.stream_chat() implementation tests
# ============================================================================

class TestOllamaLLMStreamChatMethod:
    """Verify OllamaLLM.stream_chat() implementation."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OllamaConfig(
            model="llama3",
            base_url="http://localhost:11434"
        )

    @pytest.fixture
    def llm(self, config):
        """Create OllamaLLM instance."""
        return OllamaLLM(config)

    def test_ollama_llm_has_stream_chat_method(self, llm):
        """OllamaLLM must have stream_chat method."""
        assert hasattr(llm, "stream_chat"), "OllamaLLM 缺少 stream_chat 方法"

    @pytest.mark.asyncio
    async def test_stream_chat_returns_async_iterator(self, llm):
        """stream_chat() should return an async iterator."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield '{"message":{"content":"Hello"},"done":false}'
            yield '{"message":{"content":" World"},"done":true,"done_reason":"stop"}'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        result = llm.stream_chat([{"role": "user", "content": "test"}])
        assert hasattr(result, "__aiter__"), "应返回 async iterator"

    @pytest.mark.asyncio
    async def test_stream_chat_yields_stream_chat_responses(self, llm):
        """stream_chat() should yield StreamChatResponse objects."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield '{"message":{"content":"Hello"},"done":false}'
            yield '{"message":{"content":" World"},"done":true,"done_reason":"stop"}'

        mock_response.aiter_lines = mock_aiter_lines

        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_context)
        mock_client.is_closed = False
        llm._client = mock_client

        responses = []
        async for resp in llm.stream_chat([{"role": "user", "content": "test"}]):
            responses.append(resp)
            assert isinstance(resp, StreamChatResponse), f"应返回 StreamChatResponse，实际返回 {type(resp)}"

        assert len(responses) >= 2, f"应至少返回2个响应，实际返回 {len(responses)}"


# ============================================================================
# 7. Backward compatibility with stream() method
# ============================================================================

class TestBackwardCompatibility:
    """Verify backward compatibility with legacy stream() method."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return OpenAIConfig(
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="test-key"
        )

    @pytest.fixture
    def llm(self, config):
        """Create OpenAILLM instance."""
        return OpenAILLM(config)

    def test_openai_llm_still_has_stream_method(self, llm):
        """OpenAILLM must still have stream() method for backward compatibility."""
        assert hasattr(llm, "stream"), "OpenAILLM 应保留 stream() 方法以保持向后兼容"

    def test_stream_accepts_prompt_parameter(self, llm):
        """stream() should accept prompt parameter like before."""
        sig = inspect.signature(llm.stream)
        params = list(sig.parameters.keys())
        assert "prompt" in params, "stream() 应接受 prompt 参数"

    @pytest.mark.asyncio
    async def test_stream_still_works(self, llm):
        """stream() should still yield string tokens like before."""
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
        async for token in llm.stream("test prompt"):
            tokens.append(token)
            assert isinstance(token, str), f"stream() 应返回 string，实际返回 {type(token)}"

        assert len(tokens) >= 2, f"stream() 应返回多个 token，实际返回 {len(tokens)}"

    @pytest.mark.asyncio
    async def test_stream_and_stream_chat_are_different_methods(self, llm):
        """stream() and stream_chat() should be different methods."""
        # stream() returns AsyncIterator[str]
        # stream_chat() returns AsyncIterator[StreamChatResponse]
        stream_sig = inspect.signature(llm.stream)
        stream_chat_sig = inspect.signature(llm.stream_chat)

        # They should have different return type hints
        stream_return = stream_sig.return_annotation
        stream_chat_return = stream_chat_sig.return_annotation

        # These should be different - one is AsyncIterator[str], other is AsyncIterator[StreamChatResponse]
        assert stream_return != stream_chat_return, "stream() 和 stream_chat() 应有不同的返回类型"