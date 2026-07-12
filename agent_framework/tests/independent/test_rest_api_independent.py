"""Independent tests for REST API endpoints.

验证内容（基于详细设计.md 第8节）：
- POST /api/v1/sessions 创建会话
- GET /api/v1/sessions/{session_id} 获取会话
- POST /api/v1/sessions/{session_id}/messages 发送消息
- GET /api/v1/sessions/{session_id}/messages 获取历史
- 错误处理

本测试文件完全独立编写，不使用开发者编写的测试用例。
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from agent_framework.gateway.main import app
from agent_framework.gateway.dependencies import (
    get_session_manager,
    set_session_manager,
    clear_session_manager,
)
from agent_framework.interfaces.session import SessionContext, Message


# ─────────────────────────────────────────────────────────
# 辅助工厂
# ─────────────────────────────────────────────────────────

def _make_session_context(
    session_id: str = "test-session-123",
    session_type: str = "private",
    participants: list = None,
    messages: list = None,
) -> SessionContext:
    """创建测试用 SessionContext。"""
    return SessionContext(
        session_id=session_id,
        session_type=session_type,
        participants=participants or ["user1"],
        status="active",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_active=datetime(2026, 1, 1, tzinfo=timezone.utc),
        messages=messages or [],
    )


def _make_mock_session_manager() -> AsyncMock:
    """创建模拟 SessionManager。"""
    sm = AsyncMock()
    sm.create_session = AsyncMock()
    sm.get_session = AsyncMock()
    sm.process_message = AsyncMock()
    return sm


@pytest.fixture(autouse=True)
def cleanup_dependency():
    """每个测试后清理依赖注入。"""
    yield
    clear_session_manager()


# ─────────────────────────────────────────────────────────
# 1. 健康检查和根端点
# ─────────────────────────────────────────────────────────

class TestBasicEndpoints:
    """验证基础端点。"""

    def test_health_check(self):
        """GET /api/v1/health 应返回健康状态。"""
        client = TestClient(app)
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "agent-framework"

    def test_root(self):
        """GET /api/v1/ 应返回欢迎消息。"""
        client = TestClient(app)
        response = client.get("/api/v1/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data


# ─────────────────────────────────────────────────────────
# 2. POST /api/v1/sessions 创建会话
# ─────────────────────────────────────────────────────────

class TestCreateSession:
    """验证创建会话端点。"""

    def test_create_session_success(self):
        """成功创建会话应返回 200 和会话详情。"""
        sm = _make_mock_session_manager()
        ctx = _make_session_context()
        sm.create_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.post(
            "/api/v1/sessions",
            json={"user_id": "user1", "session_type": "private"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["session_type"] == "private"
        assert data["status"] == "active"
        assert "user1" in data["participants"]

    def test_create_session_with_participants(self):
        """创建会话时应支持指定参与者。"""
        sm = _make_mock_session_manager()
        ctx = _make_session_context(participants=["user1", "user2"])
        sm.create_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.post(
            "/api/v1/sessions",
            json={
                "user_id": "user1",
                "session_type": "group",
                "participants": ["user2"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "user1" in data["participants"]
        assert "user2" in data["participants"]

    def test_create_session_calls_manager(self):
        """应调用 SessionManager.create_session。"""
        sm = _make_mock_session_manager()
        ctx = _make_session_context()
        sm.create_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        client.post(
            "/api/v1/sessions",
            json={"user_id": "user1"}
        )

        sm.create_session.assert_awaited_once_with(
            user_id="user1",
            session_type="private",
            participants=None,
        )

    def test_create_session_no_manager_returns_500(self):
        """SessionManager 未初始化时应返回 500。"""
        clear_session_manager()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/sessions",
            json={"user_id": "user1"}
        )

        assert response.status_code == 500

    def test_create_session_manager_error_returns_500(self):
        """SessionManager 抛异常时应返回 500。"""
        sm = _make_mock_session_manager()
        sm.create_session.side_effect = RuntimeError("Internal error")
        set_session_manager(sm)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/sessions",
            json={"user_id": "user1"}
        )

        assert response.status_code == 500

    def test_create_session_missing_user_id(self):
        """缺少 user_id 应返回 422。"""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/sessions",
            json={"session_type": "private"}
        )

        assert response.status_code == 422


# ─────────────────────────────────────────────────────────
# 3. GET /api/v1/sessions/{session_id} 获取会话
# ─────────────────────────────────────────────────────────

class TestGetSession:
    """验证获取会话端点。"""

    def test_get_session_success(self):
        """成功获取会话应返回 200 和会话详情。"""
        sm = _make_mock_session_manager()
        ctx = _make_session_context()
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.get("/api/v1/sessions/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert data["session_type"] == "private"
        assert data["status"] == "active"
        assert "message_count" in data

    def test_get_session_not_found(self):
        """不存在的会话应返回 404。"""
        sm = _make_mock_session_manager()
        sm.get_session.return_value = None
        set_session_manager(sm)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/sessions/nonexistent")

        assert response.status_code == 404

    def test_get_session_calls_manager(self):
        """应调用 SessionManager.get_session。"""
        sm = _make_mock_session_manager()
        ctx = _make_session_context()
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        client.get("/api/v1/sessions/test-session-123")

        sm.get_session.assert_awaited_once_with("test-session-123")

    def test_get_session_no_manager_returns_500(self):
        """SessionManager 未初始化时应返回 500。"""
        clear_session_manager()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/sessions/test-session-123")

        assert response.status_code == 500

    def test_get_session_includes_message_count(self):
        """响应应包含消息数量。"""
        sm = _make_mock_session_manager()
        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        ctx = _make_session_context(messages=messages)
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.get("/api/v1/sessions/test-session-123")

        assert response.status_code == 200
        data = response.json()
        assert data["message_count"] == 2


# ─────────────────────────────────────────────────────────
# 4. POST /api/v1/sessions/{session_id}/messages 发送消息
# ─────────────────────────────────────────────────────────

class TestSendMessage:
    """验证发送消息端点。"""

    def test_send_message_success(self):
        """成功发送消息应返回 200 和事件列表。"""
        sm = _make_mock_session_manager()

        # 创建一个可 await 的 future mock
        mock_event = MagicMock()
        mock_event.type = "final_answer"
        mock_event.content = "Hello!"
        mock_event.metadata = {}
        mock_event.timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

        # process_message 被 await 两次：future = await sm.process_message(...); events = await future
        async def mock_process(session_id, user_msg):
            async def future():
                return [mock_event]
            return future()

        sm.process_message = mock_process

        ctx = _make_session_context(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="Hello!"),
        ])
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.post(
            "/api/v1/sessions/test-session-123/messages",
            json={"content": "hi"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert "events" in data
        assert "message_count" in data

    def test_send_message_with_sender_id(self):
        """应支持指定 sender_id。"""
        sm = _make_mock_session_manager()
        called_with = {}

        async def mock_process(session_id, user_msg):
            called_with["session_id"] = session_id
            called_with["user_msg"] = user_msg
            return []

        sm.process_message = mock_process
        sm.get_session.return_value = _make_session_context()
        set_session_manager(sm)

        client = TestClient(app)
        client.post(
            "/api/v1/sessions/test-session-123/messages",
            json={"content": "hi", "sender_id": "user1"}
        )

        assert called_with["session_id"] == "test-session-123"
        assert called_with["user_msg"]["sender_id"] == "user1"

    def test_send_message_calls_process_message(self):
        """应调用 SessionManager.process_message。"""
        sm = _make_mock_session_manager()
        called = {"count": 0}

        async def mock_process(session_id, user_msg):
            called["count"] += 1
            return []

        sm.process_message = mock_process
        sm.get_session.return_value = _make_session_context()
        set_session_manager(sm)

        client = TestClient(app)
        client.post(
            "/api/v1/sessions/test-session-123/messages",
            json={"content": "hello"}
        )

        assert called["count"] == 1

    def test_send_message_no_manager_returns_500(self):
        """SessionManager 未初始化时应返回 500。"""
        clear_session_manager()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/sessions/test-session-123/messages",
            json={"content": "hello"}
        )

        assert response.status_code == 500

    def test_send_message_missing_content(self):
        """缺少 content 应返回 422。"""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/sessions/test-session-123/messages",
            json={"sender_id": "user1"}
        )

        assert response.status_code == 422


# ─────────────────────────────────────────────────────────
# 5. GET /api/v1/sessions/{session_id}/messages 获取历史
# ─────────────────────────────────────────────────────────

class TestGetMessageHistory:
    """验证获取消息历史端点。"""

    def test_get_messages_success(self):
        """成功获取消息历史应返回 200 和消息列表。"""
        sm = _make_mock_session_manager()
        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ]
        ctx = _make_session_context(messages=messages)
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.get("/api/v1/sessions/test-session-123/messages")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-123"
        assert len(data["messages"]) == 2
        assert data["total_count"] == 2

    def test_get_messages_not_found(self):
        """不存在的会话应返回 404。"""
        sm = _make_mock_session_manager()
        sm.get_session.return_value = None
        set_session_manager(sm)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/sessions/nonexistent/messages")

        assert response.status_code == 404

    def test_get_messages_with_limit(self):
        """应支持 limit 参数限制返回数量。"""
        sm = _make_mock_session_manager()
        messages = [
            Message(role="user", content="msg1"),
            Message(role="assistant", content="msg2"),
            Message(role="user", content="msg3"),
            Message(role="assistant", content="msg4"),
        ]
        ctx = _make_session_context(messages=messages)
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.get("/api/v1/sessions/test-session-123/messages?limit=2")

        assert response.status_code == 200
        data = response.json()
        assert len(data["messages"]) == 2
        assert data["total_count"] == 4  # total_count 是原始总数

    def test_get_messages_returns_recent_when_limited(self):
        """limit 应返回最近的消息。"""
        sm = _make_mock_session_manager()
        messages = [
            Message(role="user", content="old"),
            Message(role="assistant", content="old reply"),
            Message(role="user", content="new"),
            Message(role="assistant", content="new reply"),
        ]
        ctx = _make_session_context(messages=messages)
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.get("/api/v1/sessions/test-session-123/messages?limit=2")

        data = response.json()
        # 最后两条消息
        assert data["messages"][0]["content"] == "new"
        assert data["messages"][1]["content"] == "new reply"

    def test_get_messages_no_manager_returns_500(self):
        """SessionManager 未初始化时应返回 500。"""
        clear_session_manager()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/sessions/test-session-123/messages")

        assert response.status_code == 500

    def test_get_messages_empty_session(self):
        """无消息的会话应返回空列表。"""
        sm = _make_mock_session_manager()
        ctx = _make_session_context(messages=[])
        sm.get_session.return_value = ctx
        set_session_manager(sm)

        client = TestClient(app)
        response = client.get("/api/v1/sessions/test-session-123/messages")

        assert response.status_code == 200
        data = response.json()
        assert data["messages"] == []
        assert data["total_count"] == 0


# ─────────────────────────────────────────────────────────
# 6. 错误处理
# ─────────────────────────────────────────────────────────

class TestErrorHandling:
    """验证错误处理。"""

    def test_create_session_error_response_format(self):
        """错误响应应包含 error 和 detail 字段。"""
        sm = _make_mock_session_manager()
        sm.create_session.side_effect = RuntimeError("Something went wrong")
        set_session_manager(sm)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/v1/sessions",
            json={"user_id": "user1"}
        )

        assert response.status_code == 500
        # detail 字段包含错误信息
        detail = response.json().get("detail", {})
        assert "error" in detail or "detail" in detail

    def test_get_session_not_found_error_format(self):
        """404 错误应包含有意义的错误信息。"""
        sm = _make_mock_session_manager()
        sm.get_session.return_value = None
        set_session_manager(sm)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/sessions/nonexistent")

        assert response.status_code == 404
        detail = response.json().get("detail", {})
        assert "not found" in str(detail).lower() or "error" in str(detail).lower()
