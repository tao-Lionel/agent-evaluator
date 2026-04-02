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

## Step 3: 多维报告 + 高级指标 + HTTP Bot 评估 [已完成]

**目标**：输出有诊断价值的评估报告；支持黑盒 HTTP Bot 评估模式

**已实现**：
- [x] StateEvaluator 子集匹配 fallback（MD5 精确匹配失败时，按字段子集比较得 0.0~1.0）
- [x] NL Assertion 评估器（`nl_assertion`，LLM-as-Judge 逐条判断自然语言断言 PASS/FAIL）
- [x] LLM Judge 评估器（`llm_judge`，LLM 打分 1-5 归一化，适用于黑盒 Agent）
- [x] HTML 评估报告生成（`report.py`，自包含深色主题，中英双语）
- [x] 轨迹可视化（分色展示 SYSTEM/USER/AGENT/TOOL/ENV 消息，可折叠）
- [x] 评估后自动生成中英双语报告（`results_*.en.html` + `results_*.zh.html`）
- [x] 全面中文化（系统提示词、任务场景、用户脚本）
- [x] HTTP Bot 适配器（`http_bot`，支持 last/history/session 三种历史模式、重试机制、嵌套 JSON 路径解析）
- [x] Passthrough 透传环境（`passthrough`，无工具、无状态变更，用于黑盒评估）
- [x] HTTP Bot 评估场景（`scenarios/http_bot_tasks.json`，5 个任务）
- [x] 三种评估模式：工具调用 Agent / 多轮对话 Agent / 黑盒 HTTP Agent

**已完成**：
- [x] Pass@k / Pass^k 一致性指标（`core/metrics.py`，基于 num_trials 多次运行计算，集成到 run.py 和 HTML 报告）
- [x] 进度率 Progress Rate（Orchestrator 逐步追踪 state_match 变化，计算 progress_rate）
- [x] 能力维度雷达图（`report.py` SVG 雷达图，映射评估器→能力维度，支持中英文标签）
- [x] 效率指标（EvalResult 新增 elapsed_seconds、step_durations，Orchestrator 逐步计时）
- [x] 反例场景集（`scenarios/negative_tasks.json` 5 个场景：越权访问、信息不足、批量删除、不存在资源、业务规则）
- [x] LLM Judge 逃生门机制（支持 INSUFFICIENT_INFO 返回，fallback 0.5 分）
- [x] 评估饱和度警告（`core/metrics.py` check_saturation，通过率 > 85% 时在报告和终端提示）
- [x] Grader 校准流程（`core/calibration.py` compute_calibration，计算 agreement_rate、divergence、bias）

**验证点**：输出的报告能明确诊断"Agent 在哪里失败、为什么失败"。

---

## Step 4: eval_bot 飞书评测 Bot [已完成]

**目标**：通过飞书 Bot 提供评测入口，LLM 智能编排意图识别，支持异步评测和结果查询

**已实现**：
- [x] 飞书 Webhook 处理器（`eval_bot/feishu.py`，FastAPI 服务，端口 8102）
- [x] Dispatcher 意图路由（`eval_bot/dispatcher.py`，智谱 function calling 识别意图）
- [x] TaskRunner 异步执行（`eval_bot/runner.py`，线程池异步执行，支持完成回调）
- [x] `quick_eval` 命令 — 对 HTTP Bot 发起快速评测（异步，结果推回飞书）
- [x] `query_results` 命令 — 查询历史评测结果（LLM 回答）
- [x] `gen_scenarios` 命令 — 为指定业务领域生成测试场景
- [x] 单元测试（`tests/test_eval_bot_*.py`）

**验证点**：`core/` 目录零修改，验证了框架的可扩展性。

---

## Step 5: 生产级 Agent 缺陷评估（来自 MCP Agent 实战经验）

**目标**：将 feishu-bot 等生产级 Agent 系统踩过的真实坑，转化为专项测试场景和评估维度。让评估框架能检测出 Agent 在复杂多步任务中的典型失败模式。

**来源**：`feishu-bot` MCP Agent 架构全景文档（DeepSeek-V3 + 4 个 MCP Server + 6 个 Skill 的生产系统）

### 5.1 幻觉参数检测

**问题**：LLM 在多轮工具调用中会"幻觉"参数——上一步返回了 `token=abc123`，下一步 LLM 却传了历史对话中的旧 token，导致权限设置失败。

**计划**：
- [ ] `evaluators/hallucination_param.py` — 幻觉参数评估器，检查 Agent 跨步骤传递的关键参数（ID、token、key）是否与工具实际返回值一致
- [ ] `scenarios/hallucination_tasks.json` — 专项场景：工具返回动态 ID，后续步骤必须正确引用
- [ ] 评分规则：参数与真实值一致得 1.0，使用了幻觉值得 0.0

### 5.2 工具依赖链评估

**问题**：LLM 可能在同一轮返回多个 tool_calls，但它们之间有依赖——后者需要前者的返回值。并行执行会导致失败。

**计划**：
- [ ] `evaluators/dependency_chain.py` — 工具依赖链评估器，检查 Agent 是否正确处理了工具间的执行顺序
- [ ] `scenarios/dependency_tasks.json` — 专项场景：设计需要 3-7 步工具调用的链式任务（如查询→创建→设权限→通知），要求 Agent 正确排序
- [ ] 评分维度：依赖顺序正确性、中间结果传递正确性

### 5.3 跨轮记忆评估

**问题**：用户说"用方案 B 的分镜出提示词"，但方案 B 的内容在上一轮对话里。Agent 需要正确引用跨轮信息，而不是编造或遗忘。

**计划**：
- [ ] `scenarios/cross_turn_memory_tasks.json` — 专项场景：第 1 轮给出关键信息（方案/数据/偏好），第 2-3 轮要求 Agent 引用这些信息完成任务
- [ ] 扩展 `nl_assertion` 评估器支持跨轮断言（断言可引用前几轮的上下文）
- [ ] 评分规则：正确引用得 1.0，遗忘或编造得 0.0

### 5.4 MCP Agent 适配器

**计划**：
- [ ] `adapters/mcp_agent.py` — 通过 MCP 协议直接与被测 Agent 交互（比 HTTP 更原生）
- [ ] 支持 stdio 和 SSE 两种连接模式
- [ ] 自动发现被测 Agent 的工具列表，生成能力画像

### 5.5 评测进度实时反馈

**问题**：评测任务可能跑几分钟，用户在飞书端干等没有反馈。

**计划**：
- [ ] eval_bot 评测过程中实时推送进度（"正在评测第 3/5 个场景..."）
- [ ] 借鉴 feishu-bot 的 `on_progress` 回调 + 节流机制（2s 间隔更新卡片）
- [ ] 慢场景（>15s）追加耗时提示

**关键验证**：用 feishu-bot 作为被测 Agent，验证新评估维度能检测出已知的幻觉和依赖问题。

---

## Step 6: 第二种 Agent 类型（验证扩展性）

**目标**：接入浏览器自动化 Agent，验证架构的可扩展性

**计划**：
- [ ] `adapters/browser.py` — 浏览器 Agent 适配器
- [ ] `environments/browser_env.py` — Playwright 浏览器环境
- [ ] `scenarios/browser/` — 浏览器任务场景
- [ ] `evaluators/visual_accuracy.py` — 视觉准确度评估器（可选）
- [ ] 能力声明 + 自动场景匹配机制

**关键验证**：`core/` 目录下零文件修改。如果需要改，说明抽象接口设计有问题，回头修正。

---

## Step 7: 评估质量体系

**目标**：从"能评"到"评得准、评得稳"，建立可信赖的评估质量保障

**计划**：
- [ ] 环境隔离检查清单（文件系统、环境变量、缓存、DB、网络、进程、日志、时间 mock，确保每次 Trial 独立）
- [ ] `evaluators/citation_check.py` — 引用检查评估器（验证 Agent 回复中的声称是否有出处，防幻觉）
- [ ] Outcome 优先原则 — 降低 `action_match`（Transcript）权重，强化 `state_match`（Outcome）的主导地位
- [ ] 评估类型分类 — 区分能力评估（从低爬升，每次迭代）和回归评估（接近 100%，每次提交）
- [ ] 人工标注校准集 — 100 个人工标注样本，用于训练和校准 LLM 评分器

**验证点**：LLM Judge 与人工标注的一致性 > 80%。

---

## Step 8+: 生产化

**按需推进**：
- [ ] 并发调度（借鉴 AgentBench 的 Max-Flow 或 ThreadPoolExecutor）
- [ ] 断点续传（runs.jsonl 记录进度）
- [ ] Docker 环境隔离（容器级沙箱，支持浏览器/OS Agent 的安全评估）
- [ ] CLI 工具 (`agent-eval run / play / view / report`)
- [ ] Gymnasium RL 接口
- [ ] Web 排行榜
- [ ] 动态场景生成（评估 Agent 自适应生成测试用例）
- [ ] 瑞士奶酪多层防线（自动化评估 → 生产监控 → A/B 测试 → 用户反馈 → 人工审查，各阶段配方不同）

---

## 架构参考

详见：
- [benchmark-analysis.md](./research/benchmark-analysis.md) — 三个项目的完整源码级分析
- [design-decisions.md](./research/design-decisions.md) — 每个设计决策的来源和理由
- [extensibility-analysis.md](./research/extensibility-analysis.md) — 可扩展性分析和示例
- [promptfoo-analysis.md](./research/promptfoo-analysis.md) — promptfoo 框架分析，可借鉴的技术方案（缓存、对比评估、成本追踪、轨迹断言等）
- [Agent 评估完全指南](https://github.com/adongwanai/AgentGuide/blob/main/docs/02-tech-stack/agent-evaluation-complete-guide.md) — 基于 Anthropic 研究的评估方法论（Step 3/5 改进项的主要参考来源）
