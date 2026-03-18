# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

Agent Evaluator 是一个可插件化的 AI Agent 评估框架，驱动 Agent ↔ Environment 交互循环，并从多个维度对 Agent 的表现进行评分。当前已完成 Step 1（核心骨架）和 Step 2（多轮对话），Step 3（高级指标+报告）部分完成。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行评估（需要设置 ZHIPU_API_KEY 环境变量）
python run.py                          # 单轮评估（config.yaml）
python run.py config_multi_turn.yaml   # 多轮对话评估

# 生成 HTML 报告（评估结束后自动生成，也可手动）
python report.py results/results_xxx.json --lang zh
python report.py results/results_xxx.json --lang en

# 运行测试（不需要 API key）
python tests/test_core.py
python tests/test_multi_turn.py
python tests/test_llm_judge.py
python tests/test_passthrough.py
python tests/test_http_bot.py
```

## 架构

系统有四个可插拔扩展点，均通过装饰器注册到 `core/registry.py` 中的全局单例 `registry`：

```
@registry.adapter("name")      → AgentAdapter     (core/base.py)
@registry.environment("name")  → Environment      (core/base.py)
@registry.evaluator("name")    → Evaluator        (core/base.py)
@registry.user("name")         → UserSimulator    (core/base.py)
```

**Orchestrator**（`core/orchestrator.py`）是核心循环引擎，依赖上述抽象接口，不依赖具体实现。循环流程：重置 agent+env+user → 注入中文系统提示词 + 用户消息 → agent 生成响应 → 若为文本回复且存在 UserSimulator，则路由给 user（user 返回 None 表示满意，终止为 `USER_STOP`）→ 若为工具调用则由 env 执行 → 重复直到完成/达到最大步数 → 运行所有评估器 → 返回 `EvalResult`。无 `user` 配置时为单轮模式。

**评分采用乘积组合**（非加权平均）：`overall = Π(所有评估器分数)`。任一维度为 0 则总分为 0。借鉴 τ²-Bench 的设计。

### 当前实现

- **适配器**：`openai_fc` — OpenAI 兼容函数调用 API；`http_bot` — HTTP 接口适配
- **环境**：`mock_db` — 内存 CRUD 数据库（query/update/insert/delete/done）；`passthrough` — 透传环境
- **评估器**（5 个）：
  - `state_match` — DB 状态比对（MD5 精确匹配 + 子集匹配 fallback）
  - `action_match` — 工具调用部分参数匹配
  - `info_delivery` — Agent 回复子串检查
  - `llm_judge` — LLM 回复质量评分（1-5 分）
  - `nl_assertion` — 自然语言断言评估（LLM-as-Judge 逐条判 PASS/FAIL）
- **用户模拟器**：`scripted` — 关键词/默认分支脚本化用户；`llm` — LLM 角色扮演用户
- **报告**：`report.py` — 自包含 HTML 报告，支持中英双语

## 新增组件

在对应目录下创建文件，使用 `@registry.{type}("name")` 装饰器，并在包的 `__init__.py` 中导入。`core/` 目录应无需任何修改。

## 配置

`config.yaml` 支持 `${ENV_VAR}` 环境变量展开（不支持 `:-default` 语法）。主要配置段：

- `agent` — adapter, model, api_key, base_url, temperature
- `environment` — type
- `user`（可选）— type + 参数（scripted 或 llm）
- `evaluators` — 评估器名称列表
- `scenarios` — 任务 JSON 路径
- `run` — num_trials, log_level

## 任务场景格式

任务定义在 JSON 数组中。每个任务包含：`id`、`description`、`initial_message`、`initial_state`（DB 表数据）、`expected_actions`（含 `match_args` 用于部分匹配）、`expected_state`（用于子集匹配 fallback）、`required_info`、`difficulty`、`max_steps`、`nl_assertions`（可选，自然语言断言列表）、`user_scenario`（可选，含 `persona`、`goal`、`script` 用于多轮对话）。

当前有三套场景：
- `scenarios/sample_tasks.json` — 5 个单轮任务（easy/medium/hard）
- `scenarios/multi_turn_tasks.json` — 3 个多轮对话任务
- `scenarios/http_bot_tasks.json` — HTTP Bot 评估任务

所有场景和系统提示词均为中文。
