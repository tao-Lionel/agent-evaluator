# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

Agent Evaluator 是一个可插件化的 AI Agent 评估框架，驱动 Agent ↔ Environment 交互循环，并从多个维度对 Agent 的表现进行评分。当前处于 MVP 阶段（路线图 Step 1），包含内存模拟数据库环境和 OpenAI 兼容的 Agent 适配器。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行评估（需要设置 OPENAI_API_KEY 环境变量）
python run.py                    # 使用默认 config.yaml
python run.py path/to/config.yaml

# 运行测试（不需要 API key，使用脚本化 mock agent）
python tests/test_core.py
```

## 架构

系统有四个可插拔扩展点，均通过装饰器注册到 `core/registry.py` 中的全局单例 `registry`：

```
@registry.adapter("name")      → AgentAdapter     (core/base.py)
@registry.environment("name")  → Environment      (core/base.py)
@registry.evaluator("name")    → Evaluator        (core/base.py)
@registry.user("name")         → UserSimulator    (core/base.py)
```

**Orchestrator**（`core/orchestrator.py`）是核心循环引擎，依赖上述抽象接口，不依赖具体实现。循环流程：重置 agent+env+user → 注入系统提示词 + 用户消息 → agent 生成响应 → 若为文本回复且存在 UserSimulator，则路由给 user（user 返回 None 表示满意，终止为 `USER_STOP`）→ 若为工具调用则由 env 执行 → 重复直到完成/达到最大步数 → 运行所有评估器 → 返回 `EvalResult`。无 `user` 配置时行为与之前完全一致。

**评分采用乘积组合**（非加权平均）：`overall = state_match × action_match × info_delivery`。任一维度为 0 则总分为 0。这是借鉴 τ²-Bench 的设计决策。

**状态评估**通过在全新环境副本上重放期望动作，然后比较两个 DB 状态的 MD5 哈希值来实现。这使得评估与执行顺序无关，且完全确定性。

### 当前实现

- **适配器**：`openai_fc` — 支持任何 OpenAI 兼容的函数调用 API
- **环境**：`mock_db` — 内存 CRUD 数据库（query/update/insert/delete/done 工具）
- **评估器**：`state_match`（DB 哈希比对）、`action_match`（部分参数匹配）、`info_delivery`（Agent 回复中的子串检查）
- **用户模拟器**：`scripted` — 基于关键词/默认分支匹配的脚本化用户；`llm` — LLM 角色扮演用户（需 API key）

## 新增组件

新增 adapter/environment/evaluator/user 的步骤：在对应目录下创建文件，使用 `@registry.{type}("name")` 装饰器，并在包的 `__init__.py` 中导入。`core/` 目录应无需任何修改。

## 配置

`config.yaml` 支持 `${ENV_VAR}` 环境变量展开。主要配置段：`agent`（adapter, model, api_key）、`environment`（type）、`user`（可选，type + 参数）、`evaluators`（名称列表）、`scenarios`（任务 JSON 路径）、`run`（num_trials, log_level）。

## 任务场景格式

任务定义在 JSON 数组中（`scenarios/sample_tasks.json`）。每个任务包含：`id`、`description`、`initial_message`、`initial_state`（DB 表数据）、`expected_actions`（含 `match_args` 用于部分匹配）、`expected_state`、`required_info`、`difficulty`、`max_steps`、`user_scenario`（可选，含 `persona`、`goal`、`script` 用于多轮对话模拟）。
