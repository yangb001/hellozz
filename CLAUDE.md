# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **multi-session AI Agent framework** with local-first architecture. Currently only design documentation exists - implementation has not begun.

Key design documents:
- `架构设计.md` - High-level architecture design
- `详细设计.md` - Detailed implementation blueprint

## Architecture Summary

```
接入层 (Gateway)     → FastAPI REST/WebSocket
会话管理器 (SessionManager) → Actor-based concurrency with SQLite persistence
Agent运行时 (AgentRuntime) → Stateless, borrows SessionContext
记忆系统 (Memory)    → BufferMemory (短期) + VectorMemory (长期, LanceDB/Chroma)
规划器 (Planners)    → ReAct / PlanExecute / Graph
基础设施             → LLM Gateway (Ollama/OpenAI) + SQLite + Embedding models
```

## Core Design Decisions

| Aspect | Decision |
|--------|----------|
| Context management | SessionManager creates/holds/persists `SessionContext` at runtime |
| Memory trigger | Smart: per-message LLM check; or Every N turns; or On session close |
| Long-term memory extraction | Uses lightweight LLM (e.g., Ollama llama3) asynchronously |
| Inter-service communication | gRPC for future microservice拆分 |
| Storage | SQLite (sessions) + LanceDB/Chroma (vectors), all local |
| Group sessions | `participants` field + per-user memory partitioning |

## Planned Directory Structure

```text
agent-framework/
├── interfaces/        # Abstract base classes (zero dependencies)
├── core/              # SessionManager, EventBus, Config
├── runtime/           # AgentRuntime (stateless)
├── memory/            # MemoryManager, BufferMemory, VectorMemory, Extractor
├── planners/          # ReActPlanner, PlanExecutePlanner, GraphPlanner
├── tools/             # ToolRegistry + builtin tools
├── gateway/           # FastAPI REST + WebSocket
├── infrastructure/    # LLM Gateway, Storage adapters
└── adapters/          # gRPC/HTTP clients (future)
```

## Status

- [ ] Implement `interfaces/` - Define all abstract protocols and data models
- [ ] Implement `core/` - SessionManager with Actor model, SQLite persistence
- [ ] Implement `infrastructure/` - LLM Gateway, SQLite storage, Vector store adapter
- [ ] Implement `memory/` - MemoryManager with Buffer + Vector memory
- [ ] Implement `runtime/` - Stateless AgentRuntime
- [ ] Implement `planners/` - ReActPlanner
- [ ] Implement `tools/` - ToolRegistry + basic builtin tools
- [ ] Implement `gateway/` - FastAPI with REST and WebSocket endpoints

## Agent Team Protocol

When using multi-agent teams for implementation:
- **【重要】Team Lead 只负责调度，禁止参与开发、测试及设计工作**
- **【重要】未经用户明确授权，禁止下发任何任务给团队成员**
- **【重要】禁止制定开发计划，所有计划必须由用户决策**
- Do NOT disband the team after tasks complete
- Agents should remain on standby, waiting for the next task
- Use `TeamCreate` to establish team structure before parallel work begins
- After completing current phase, check `TaskList` for newly available work before signaling completion
- Keep agents alive and idle rather than recreating them for subsequent phases

### 任务调度流程

1. **角色分工**
   - **开发者**：负责功能实现，编写 TDD 测试用例（tests/ 目录）
   - **测试人员**：负责独立验证，编写独立测试用例（tests/independent/ 目录）
   - **Team Lead**：仅负责任务分配和调度，禁止参与开发和测试

2. **任务分配原则**
   - 有依赖关系的任务必须串行执行
   - 无依赖关系的任务可以并行分配给多个开发者
   - 开发任务分配时同步分配对应的独立测试验证任务

3. **TDD 流程（开发方）**
   - 开发接到任务后，创建 `tests/test_xxx.py` 编写单元测试
   - 开发实现功能代码
   - 开发运行自测，确保 TDD 测试全部通过
   - 开发报告完成，附上测试结果

4. **独立测试流程（测试方）**
   - 测试人员**禁止使用开发者编写的测试用例**
   - 测试人员根据 `详细设计.md` 规范，编写独立的测试用例
   - 放置路径：`tests/independent/test_xxx_independent.py`
   - 用独立测试验证开发提交的功能
   - 报告验证结果（通过/发现缺陷）

5. **开发者空闲时的任务分配**
   - 当开发者完成任务后空闲时：
     - 预留 2 个开发者用于修复测试发现的 bug
     - 其他开发者继续承接后续阶段的新任务
   - 确保 bug 修复工作始终有人力资源

6. **验证后才算完成**
   - 一个任务只有在其独立测试用例全部通过后，才视为真正完成
   - 开发自测通过不算，必须经过测试人员独立验证

## Environment Configuration

### Virtual Environment (IMPORTANT)
- **所有开发和测试必须使用项目独立虚拟环境 `.venv`**
- 如果 `.venv` 不存在，首先创建：
  ```bash
  python -m venv .venv
  ```
- 激活虚拟环境后才能执行任何开发/测试命令

### Dependency Installation (IMPORTANT)
- **所有 pip 安装必须使用华为镜像源，使用 `python -m pip` 命令**：
  ```bash
  python -m pip install <package> --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple
  ```
- 或安装项目依赖：
  ```bash
  python -m pip install -e ".[dev]" --trusted-host mirrors.tools.huawei.com -i https://mirrors.tools.huawei.com/pypi/simple
  ```
- **如果依赖安装失败 2 次，停止尝试，报告给 Team Lead，由用户手动处理**
- 不要自动切换其他镜像源

## Testing Strategy (CRITICAL)

### 教训总结

**问题**：单元测试全部通过，但实际运行时失败
**根因**：过度使用 mock 绕过了真实组件组装，导致集成问题未被发现

### 核心测试原则

#### 1. 不要过度 mock
- ❌ 错误：所有依赖都用 mock
- ✅ 正确：关键路径使用真实组件
- **原则**：mock 只用于隔离外部依赖（网络、数据库），不用于隔离内部组件

#### 2. 必须测试组装
- 不仅测试单个组件，还要测试组件间的组装
- 验证依赖注入正确性
- **原则**：组装测试 > 单元测试

#### 3. 必须测试启动
- 验证应用能正常启动
- 验证关键组件初始化成功
- **原则**：启动测试是最后一道防线

#### 4. 必须测试配置
- 验证配置加载正确性
- 验证配置结构完整性
- **原则**：配置错误是最常见的运行时错误

### 测试金字塔

```
┌─────────────────────────────────────────────────────────────┐
│                    端到端测试 (E2E) - 少                     │
│            验证完整用户场景，使用真实组件                      │
├─────────────────────────────────────────────────────────────┤
│                    集成测试 (Integration) - 中               │
│            验证组件组装和交互，部分使用 mock                   │
├─────────────────────────────────────────────────────────────┤
│                    单元测试 (Unit) - 多                      │
│            验证单个函数/类功能，可使用 mock                    │
└─────────────────────────────────────────────────────────────┘
```

### 必须包含的测试类型

| 测试类型 | 说明 | 执行时机 |
|----------|------|----------|
| 单元测试 | 验证单个函数/类功能 | 每次提交 |
| 组装测试 | 验证组件组装和依赖注入 | 每次提交 |
| 启动测试 | 验证应用能正常启动 | 每次提交 |
| 配置验证测试 | 验证配置加载正确性 | 每次提交 |
| 端到端测试 | 验证完整用户场景 | 每次部署 |

### 开发者自检清单

完成开发后，必须检查：

- [ ] 单元测试通过
- [ ] 组装测试通过（真实组件）
- [ ] 启动测试通过
- [ ] 配置验证测试通过
- [ ] 端到端测试通过

**缺少任何一项测试，任务不能标记为完成！**

## Implementation Approach

Start with `interfaces/` (zero dependencies), then build outward. All modules should only depend on interfaces - implementation can be swapped (e.g., VectorMemory can use LanceDB or Chroma).

Key files to read before implementing:
- `详细设计.md` sections 3-4 for interface definitions
- `详细设计.md` sections 4-7 for implementation patterns
- `架构设计.md` section 4.4 for memory system details