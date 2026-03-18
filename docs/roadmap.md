# Agent Evaluator 开发路线图

---

## MVP 策略

**核心原则**：先跑通一条最窄的端到端路径，再横向扩展。

```
选择标准：
  - 最容易获取的被评估 Agent → OpenAI function calling Agent
  - 最容易搭建的环境         → 内存模拟数据库（不需要 Docker）
  - 最容易验证的场景         → 工具调用类任务（客服/查询）
  - 最能体现架构价值的评估   → 多维度评判（不只是 pass/fail）
```

---

## Step 1: 核心骨架 [已完成]

**目标**：一个 Agent + 一个环境 + 一个任务 → 输出评估结果

**已实现**：
- [x] 核心类型系统 (`core/types.py`)
- [x] 三个抽象接口 (`core/base.py`: AgentAdapter, Environment, Evaluator)
- [x] 插件注册表 (`core/registry.py`)
- [x] Orchestrator 交互循环引擎 (`core/orchestrator.py`)
- [x] OpenAI Function Calling 适配器 (`adapters/openai_fc.py`)
- [x] 内存模拟数据库环境，支持 CRUD + done (`environments/mock_db.py`)
- [x] 三个评估器：状态比对、动作匹配、信息传达 (`evaluators/`)
- [x] 5 个测试场景（easy/medium/hard）(`scenarios/sample_tasks.json`)
- [x] 4 个单元测试，不依赖 API (`tests/test_core.py`)
- [x] 主入口 + 配置文件 (`run.py` + `config.yaml`)
- [x] 难度分层统计 + 各评估器分项报告

**验证点**：单元测试全部通过，能输出正确的评估分数。

---

## Step 2: 多轮对话能力 [已完成]

**目标**：加入 LLM-as-User，让评估从"Agent 直接调工具"变为"Agent 和用户对话 + 调工具"

**已实现**：
- [x] `users/llm_user.py` — LLM 模拟用户（借鉴 τ²-Bench 的角色翻转技巧）
- [x] `users/scripted_user.py` — 脚本化用户（用于确定性测试）
- [x] 修改 Orchestrator 支持 Agent ↔ User ↔ Env 三方路由
- [x] Task 结构增加 `user_scenario` 字段（persona、goal、script）
- [x] 终止条件扩展：USER_STOP
- [x] Registry 新增 `@registry.user("name")` 注册装饰器
- [x] `run.py` 集成 user 配置
- [x] 3 个多轮对话任务场景 (`scenarios/multi_turn_tasks.json`)
- [x] 多轮对话专用配置 (`config_multi_turn.yaml`)
- [x] 11 个单元测试全部通过 (`tests/test_multi_turn.py`)

**验证点**：能完成多轮对话评估，用户模拟器行为自然。

---

## Step 3: 多维报告 + 高级指标 [进行中]

**目标**：输出有诊断价值的评估报告，而非只有一个数字

**已实现**：
- [x] StateEvaluator 子集匹配 fallback（MD5 精确匹配失败时，按字段子集比较得 0.0~1.0）
- [x] NL Assertion 评估器（`nl_assertion`，LLM-as-Judge 逐条判断自然语言断言 PASS/FAIL）
- [x] HTML 评估报告生成（`report.py`，自包含深色主题，中英双语）
- [x] 轨迹可视化（分色展示 SYSTEM/USER/AGENT/TOOL/ENV 消息，可折叠）
- [x] 评估后自动生成中英双语报告（`results_*.en.html` + `results_*.zh.html`）
- [x] 全面中文化（系统提示词、任务场景、用户脚本）

**待完成**：
- [ ] Pass^k 一致性指标（借鉴 τ²-Bench，多次运行衡量结果稳定性）
- [ ] 进度率 Progress Rate（借鉴 AgentBoard，追踪每步 reward 变化）
- [ ] 能力维度雷达图（借鉴 AgentBoard 的六维映射）

**验证点**：输出的报告能明确诊断"Agent 在哪里失败、为什么失败"。

---

## Step 4: 第二种 Agent 类型（验证扩展性）

**目标**：接入浏览器自动化 Agent，验证架构的可扩展性

**计划**：
- [ ] `adapters/browser.py` — 浏览器 Agent 适配器
- [ ] `environments/browser_env.py` — Playwright 浏览器环境
- [ ] `scenarios/browser/` — 浏览器任务场景
- [ ] `evaluators/visual_accuracy.py` — 视觉准确度评估器（可选）
- [ ] 能力声明 + 自动场景匹配机制

**关键验证**：`core/` 目录下零文件修改。如果需要改，说明抽象接口设计有问题，回头修正。

---

## Step 5+: 生产化

**按需推进**：
- [ ] 并发调度（借鉴 AgentBench 的 Max-Flow 或 ThreadPoolExecutor）
- [ ] 断点续传（runs.jsonl 记录进度）
- [ ] Docker 环境隔离
- [ ] CLI 工具 (`agent-eval run / play / view / report`)
- [ ] Gymnasium RL 接口
- [ ] Web 排行榜
- [ ] 动态场景生成（评估 Agent 自适应生成测试用例）

---

## 架构参考

详见：
- [benchmark-analysis.md](./research/benchmark-analysis.md) — 三个项目的完整源码级分析
- [design-decisions.md](./research/design-decisions.md) — 每个设计决策的来源和理由
- [extensibility-analysis.md](./research/extensibility-analysis.md) — 可扩展性分析和示例
