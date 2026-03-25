# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 项目概述

Agent Evaluator 是一个可插件化的 AI Agent 评估框架，驱动 Agent ↔ Environment 交互循环，并从多个维度对 Agent 的表现进行评分。当前已完成 Step 1（核心骨架）、Step 2（多轮对话）、Step 3（高级指标+报告+HTTP Bot 评估）和 Step 4（eval_bot 飞书 Bot + LLM 智能编排）。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行评估（需要设置 ZHIPU_API_KEY 环境变量）
python run.py                          # 单轮评估（config.yaml）
python run.py config_multi_turn.yaml   # 多轮对话评估
python run.py config_http_bot.yaml     # HTTP Bot 黑盒评估

# 生成 HTML 报告（评估结束后自动生成，也可手动）
python report.py results/results_xxx.json --lang zh
python report.py results/results_xxx.json --lang en

# 运行测试（不需要 API key）
python tests/test_core.py
python tests/test_multi_turn.py
python tests/test_llm_judge.py
python tests/test_passthrough.py
python tests/test_http_bot.py
python -m pytest tests/test_eval_bot_*.py  # eval_bot 专项测试

# 启动飞书评测 Bot（需要设置 FEISHU_* 环境变量）
uvicorn eval_bot.feishu:app --port 8102
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

- **适配器**（2 个）：
  - `openai_fc` — OpenAI 兼容函数调用 API（支持智谱等）
  - `http_bot` — HTTP 接口适配，支持三种历史模式（last/history/session）、重试机制、嵌套 JSON 路径解析
- **环境**（2 个）：
  - `mock_db` — 内存 CRUD 数据库（query/update/insert/delete/done）
  - `passthrough` — 透传环境，用于黑盒 Agent 评估（无工具、无状态变更）
- **评估器**（5 个）：
  - `state_match` — DB 状态比对（MD5 精确匹配 + 子集匹配 fallback）
  - `action_match` — 工具调用部分参数匹配
  - `info_delivery` — Agent 回复子串检查（模糊匹配）
  - `llm_judge` — LLM 回复质量评分（1-5 分，适用于黑盒 Agent，支持 INSUFFICIENT_INFO 逃生门）
  - `nl_assertion` — 自然语言断言评估（LLM-as-Judge 逐条判 PASS/FAIL）
- **指标与校准**（`core/`）：
  - `metrics.py` — Pass@k / Pass^k 一致性指标、饱和度警告
  - `calibration.py` — Grader 校准框架（LLM vs 人工评分一致性）
- **用户模拟器**（2 个）：
  - `scripted` — 关键词/默认分支脚本化用户
  - `llm` — LLM 角色扮演用户（persona + goal 驱动）
- **报告**：`report.py` — 自包含 HTML 报告，深色主题，支持中英双语，含轨迹可视化、能力雷达图、一致性指标表、饱和度警告
- **eval_bot**（`eval_bot/`）：飞书 Bot + LLM 智能编排，三个功能：
  - `quick_eval` — 对 HTTP Bot 发起快速评测（异步，结果推回飞书）
  - `query_results` — 查询历史评测结果（LLM 回答）
  - `gen_scenarios` — 为指定业务领域生成测试场景
  - `Dispatcher` — 用智谱 function calling 识别意图，路由到对应 command
  - `TaskRunner` — 线程池异步执行，支持完成回调

### 三种评估模式

1. **工具调用 Agent**（`config.yaml`）：openai_fc + mock_db → state_match, action_match, info_delivery
2. **多轮对话 Agent**（`config_multi_turn.yaml`）：openai_fc + mock_db + scripted user → 加 nl_assertion
3. **黑盒 HTTP Agent**（`config_http_bot.yaml`）：http_bot + passthrough → info_delivery, llm_judge

## 新增组件

在对应目录下创建文件，使用 `@registry.{type}("name")` 装饰器，并在包的 `__init__.py` 中导入。`core/` 目录应无需任何修改。

## 目录结构

```
core/           — 核心引擎（types, base, registry, orchestrator, metrics, calibration）
adapters/       — Agent 适配器（openai_fc, http_bot）
environments/   — 环境实现（mock_db, passthrough）
evaluators/     — 评估器（state_match, action_match, info_delivery, llm_judge, nl_assertion）
users/          — 用户模拟器（scripted, llm）
scenarios/      — 任务场景 JSON（sample_tasks, multi_turn_tasks, http_bot_tasks）
eval_bot/       — 飞书评测 Bot（dispatcher, runner, feishu webhook, commands/）
tests/          — 单元测试（test_core, test_multi_turn, test_llm_judge, test_passthrough, test_http_bot, test_eval_bot_*）
docs/           — 文档（roadmap, plans/, research/）
test_bot/       — 测试用 HTTP Bot（FastAPI 服务 + 飞书集成）
results/        — 评估结果输出（JSON + HTML 报告）
```

## 配置

`config.yaml` 支持 `${ENV_VAR}` 环境变量展开（不支持 `:-default` 语法）。主要配置段：

- `agent` — adapter, model, api_key, base_url, temperature；http_bot 额外支持 url, history_mode, max_retries, retry_delay, request_field, reply_field
- `environment` — type
- `user`（可选）— type + 参数（scripted 或 llm）
- `evaluators` — 评估器名称列表
- `scenarios` — 任务 JSON 路径
- `run` — num_trials, log_level

## 任务场景格式

任务定义在 JSON 数组中。每个任务包含：`id`、`description`、`initial_message`、`initial_state`（DB 表数据）、`expected_actions`（含 `match_args` 用于部分匹配）、`expected_state`（用于子集匹配 fallback）、`required_info`、`difficulty`、`max_steps`、`nl_assertions`（可选，自然语言断言列表）、`user_scenario`（可选，含 `persona`、`goal`、`script` 用于多轮对话）、`single_turn`（可选，用于黑盒 Agent）。

当前有四套场景：
- `scenarios/sample_tasks.json` — 5 个单轮任务（easy/medium/hard）
- `scenarios/multi_turn_tasks.json` — 3 个多轮对话任务
- `scenarios/http_bot_tasks.json` — 5 个 HTTP Bot 评估任务
- `scenarios/negative_tasks.json` — 5 个反例场景（越权、信息不足、批量删除、不存在资源、业务规则）

所有场景和系统提示词均为中文。

## 依赖

```
openai>=1.0.0        # OpenAI SDK（智谱兼容）
pyyaml>=6.0          # 配置解析
python-dotenv>=1.0.0 # .env 支持
httpx>=0.24.0        # HTTP 请求（http_bot 适配器）
fastapi>=0.100.0     # eval_bot Webhook 服务
uvicorn>=0.20.0      # eval_bot ASGI 服务器
```
