# 多会话 AI Agent 框架

一个支持多会话、记忆系统、工具调用的 AI Agent 框架。

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository-url>
cd hellozz

# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖
python -m pip install -e ".[dev]" --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple
```

### 2. 配置

编辑 `config.json` 文件，配置 LLM 和其他参数：

```json
{
    "llm": {
        "default": "mimo",
        "providers": {
            "mimo": {
                "type": "openai",
                "model": "mimo-v2.5-pro",
                "base_url": "https://your-api-endpoint/v1",
                "api_key": "your-api-key",
                "verify_ssl": false
            }
        }
    }
}
```

### 3. 启动服务

```bash
# 开发环境（自动重载）
uvicorn agent_framework.gateway.main:app --reload --host 0.0.0.0 --port 8000

# 生产环境（多进程）
uvicorn agent_framework.gateway.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### 4. 访问 API

- **API 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc
- **健康检查**: http://localhost:8000/api/v1/health

---

## 📡 API 使用

### REST API

#### 创建会话

```bash
curl -X POST "http://localhost:8000/api/v1/sessions" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "session_type": "private"}'
```

#### 发送消息

```bash
curl -X POST "http://localhost:8000/api/v1/sessions/{session_id}/messages" \
  -H "Content-Type: application/json" \
  -d '{"content": "你好，请帮我写一首诗", "sender_id": "user1"}'
```

#### 获取会话信息

```bash
curl "http://localhost:8000/api/v1/sessions/{session_id}"
```

#### 获取消息历史

```bash
curl "http://localhost:8000/api/v1/sessions/{session_id}/messages?limit=10"
```

### WebSocket

```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8000/ws/chat?session_id=test&token=demo"
    async with websockets.connect(uri) as websocket:
        # 发送消息
        await websocket.send(json.dumps({
            "type": "user_message",
            "content": "你好！"
        }))
        
        # 接收事件流
        while True:
            response = await websocket.recv()
            event = json.loads(response)
            print(f"[{event['type']}] {event['content']}")

asyncio.run(chat())
```

---

## 🧪 测试

### 运行所有测试

```bash
.venv\Scripts\pytest
```

### 运行特定测试

```bash
# 单元测试
.venv\Scripts\pytest agent_framework/tests/test_*.py

# 集成测试
.venv\Scripts\pytest agent_framework/tests/integration/

# 独立测试
.venv\Scripts\pytest agent_framework/tests/independent/
```

### 运行真实 LLM 测试

```bash
.venv\Scripts\pytest agent_framework/tests/integration/test_end_to_end_real_llm.py -v
```

---

## 📁 项目结构

```
agent_framework/
├── interfaces/        # 抽象基类和数据结构
├── core/              # SessionManager, EventBus, Config
├── infrastructure/    # LLM Gateway, VectorStore
├── memory/            # 记忆系统
├── runtime/           # AgentRuntime
├── planners/          # ReActPlanner
├── tools/             # 工具注册和内置工具
├── gateway/           # FastAPI REST + WebSocket
└── tests/             # 测试文件
    ├── unit/          # 单元测试
    ├── integration/   # 集成测试
    └── independent/   # 独立测试
```

---

## 🔧 配置说明

### config.json 完整配置

```json
{
    "sqlite": {
        "path": "./data/sessions.db"
    },
    "memory": {
        "short_term_size": 20,
        "vector_db": "lancedb",
        "vector_path": "./data/vectors",
        "embedding_model": "all-MiniLM-L6-v2",
        "extraction": {
            "trigger": "smart",
            "model": "mimo"
        }
    },
    "llm": {
        "default": "mimo",
        "providers": {
            "mimo": {
                "type": "openai",
                "model": "mimo-v2.5-pro",
                "base_url": "https://token-plan-cn.xiaomimimo.com/v1",
                "api_key": "your-api-key",
                "verify_ssl": false
            }
        }
    },
    "planner": "planners.react_planner.ReActPlanner"
}
```

### 配置项说明

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `sqlite.path` | SQLite 数据库路径 | `./data/sessions.db` |
| `memory.short_term_size` | 短期记忆大小 | `20` |
| `memory.vector_db` | 向量数据库类型 | `lancedb` |
| `llm.default` | 默认 LLM 提供商 | `ollama` |
| `llm.providers.*.verify_ssl` | SSL 验证 | `true` |

---

## 🛠️ 开发指南

### 添加新工具

```python
from agent_framework.interfaces.base_tool import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "My custom tool"
    
    async def run(self, input: str, session_id: str = None, **kwargs) -> str:
        return f"Result: {input}"
```

### 添加新规划器

```python
from agent_framework.interfaces.base_planner import BasePlanner

class MyPlanner(BasePlanner):
    async def plan_and_act(self, ctx, memory, tools, llm_call):
        # 实现规划逻辑
        yield Event(type="final_answer", content="Response")
```

---

## 📊 测试覆盖

| 测试类型 | 数量 | 状态 |
|----------|------|------|
| 单元测试 | 1000+ | ✅ 通过 |
| 集成测试 | 30+ | ✅ 通过 |
| 独立测试 | 200+ | ✅ 通过 |
| 端到端测试 | 6 | ✅ 通过 |
| **总计** | **1665+** | ✅ **全部通过** |

---

## 🐛 常见问题

### 1. SSL 证书错误

```
SSL: CERTIFICATE_VERIFY_FAILED
```

**解决方案**: 在 `config.json` 中设置 `verify_ssl: false`

### 2. 端口被占用

```
Address already in use
```

**解决方案**:
```bash
# 查找占用端口的进程
netstat -ano | grep 8000

# 杀掉进程
taskkill //F //PID <进程ID>
```

### 3. 依赖安装失败

**解决方案**: 使用华为镜像源
```bash
python -m pip install <package> --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple
```

---

## 📝 开发规范

详见 `CLAUDE.md`

---

## 📄 许可证

[待添加]
