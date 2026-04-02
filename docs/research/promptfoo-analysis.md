# promptfoo 分析：可借鉴的技术方案

> 来源：[github.com/promptfoo/promptfoo](https://github.com/promptfoo/promptfoo)（19k+ stars，TypeScript，已被 OpenAI 收购）
>
> 分析日期：2026-04-02

---

## promptfoo 是什么

promptfoo 是一个开源的 LLM 评估/测试/红队框架，核心解决三个问题：

1. **系统化评估** — 消除 prompt 调优中的试错，用数据驱动对比
2. **安全检测** — 自动化红队测试，发现 LLM 应用中的安全漏洞
3. **CI/CD 集成** — 为 LLM 应用提供自动化质量门控

设计原则：隐私优先（评估 100% 本地运行）、开发者友好（热重载、缓存）、提供商无关（50+ LLM 集成）。

---

## 核心架构

```
Prompts（模板） × Providers（LLM）× Test Cases（输入+断言）
                         ↓
                   矩阵评估引擎
                         ↓
              断言验证 → 评分聚合 → 报告输出
```

**关键概念**：

| 概念 | 说明 | 对应本项目 |
|------|------|-----------|
| Provider | LLM API 集成，实现 `callApi()` | `AgentAdapter` |
| Test Case | 输入变量 + 断言规则 | `Task`（场景） |
| Assertion | 输出验证规则（60+ 种） | `Evaluator` |
| Scenario | 组合矩阵（config × tests） | 无直接对应 |
| Derived Metric | 后置计算的组合指标 | 无直接对应 |

**评估流程**：
1. 加载配置 → 解析 providers、prompts、test cases
2. 生成评估矩阵（providers × prompts × test cases 笛卡尔积）
3. 分离串行/并行用例
4. 并发执行每个 (provider, prompt, test) 元组，带速率限制
5. 对每个结果运行断言，计算逐断言评分
6. 聚合为 prompt 级指标（通过/失败计数、命名分数）
7. 运行跨输出比较断言（select-best、max-score）
8. 计算派生指标（MathJS 表达式）
9. 存入 SQLite，生成报告

---

## 可借鉴的技术方案

### 1. 评分体系：加权平均 + 自定义公式

**promptfoo 方案**：
```
Final Score = Σ(assertion_score × weight) / Σ(weight)
```
- 每个断言可配置 `weight`（默认 1.0）
- 支持 `metric: "name"` 标签，按标签聚合
- 支持 **derived metrics**（后置 MathJS 表达式或 JS 函数组合命名指标）

```yaml
# 示例：自定义组合公式
derivedMetrics:
  - name: composite
    value: 'accuracy * 0.6 + safety * 0.4'
  - name: f1_score
    value: '2 * precision * recall / (precision + recall)'
```

**对比本项目**：本项目用乘积 `∏(all scores)`，更严格（任一为 0 则总分为 0），但不够灵活。

**借鉴建议**：
- 引入 named metrics + derived metrics 概念，让用户在配置中自定义评分公式
- 保留乘积作为默认策略，但支持切换为加权平均或自定义表达式
- 配置示例：
  ```yaml
  scoring:
    mode: product          # product | weighted_avg | custom
    custom_formula: 'state_match * 0.5 + action_match * 0.3 + info_delivery * 0.2'
  ```

**优先级**：P2（增加灵活性，但当前乘积策略已经够用）

---

### 2. Agent 轨迹断言：比 action_match 更细粒度

**promptfoo 方案**（5 种轨迹断言）：

| 断言类型 | 功能 | 本项目对应 |
|----------|------|-----------|
| `trajectory:tool-used` | 验证特定工具是否被调用（支持计数） | `action_match`（部分） |
| `trajectory:tool-sequence` | 验证工具调用**顺序**（精确/包含） | 无 |
| `trajectory:tool-args-match` | 验证参数（精确/部分匹配） | `action_match.match_args` |
| `trajectory:step-count` | 步数约束（min/max 范围） | 无 |
| `trajectory:goal-success` | LLM 判断目标是否达成 | `llm_judge`（间接） |

```yaml
# promptfoo 示例
assert:
  - type: trajectory:tool-sequence
    value:
      - search_orders      # 必须先搜索
      - update_order        # 再更新
  - type: trajectory:step-count
    value:
      min: 2
      max: 5
```

**借鉴建议**：
- 扩展 `action_match` 支持 `expected_sequence`（有序匹配）
- 新增 `step_count` 约束到 Task 定义中（`min_steps` / `max_steps` 范围，不只是上限）
- 在 `expected_actions` 中增加 `order: strict | any`（默认 `any` 保持兼容）

**优先级**：P2（对评估 Agent 策略效率有价值）

---

### 3. LLM 调用缓存 + 请求去重

**promptfoo 方案**：
- **磁盘缓存**：`~/.promptfoo/cache`，基于 KeyvFile，TTL 14 天
- **缓存键**：`fetch:v2:{url}:{serialized_options}`（排除 headers）
- **在途请求去重**：并发相同请求只发一次 API 调用，其他等待结果
- **错误不缓存**：失败响应永远不缓存（允许重试）
- **429 处理**：触发指数退避
- 环境变量控制：`PROMPTFOO_CACHE_ENABLED`、`PROMPTFOO_CACHE_TTL`、`PROMPTFOO_CACHE_PATH`

```python
# 借鉴思路（伪代码）
cache_key = hashlib.md5(f"{model}:{temperature}:{prompt}".encode()).hexdigest()
if cache_key in disk_cache:
    return disk_cache[cache_key]
result = call_llm(...)
if not result.error:
    disk_cache[cache_key] = result
```

**对比本项目**：本项目没有任何缓存机制。`llm_judge`、`nl_assertion`、`safety` 每次都重新调 LLM。`num_trials > 1` 时重复调用浪费严重。

**借鉴建议**：
- 在 `core/` 下新增 `cache.py`，实现 hash-key → JSON 文件缓存
- 对评估器的 LLM 调用（非 Agent 本身的调用）启用缓存
- 支持 `EVAL_CACHE_ENABLED=true/false` 和 `EVAL_CACHE_TTL=86400` 控制
- Agent 适配器的调用**不缓存**（评估的是 Agent 行为，需要真实响应）

**优先级**：P0（降本增效，实现简单，收益大）

---

### 4. 评估恢复（Resume）

**promptfoo 方案**：
- 将已完成的 `(testIdx, promptIdx)` 对持久化到 SQLite
- 重跑时加载已完成的对，过滤掉，只跑未完成的
- 完成后重新运行所有跨输出比较断言

**对比本项目**：本项目中断后必须全部重跑。大规模评估（几十个场景 × 多次 trial）时成本高。

**借鉴建议**：
- 评估过程中逐 task 写入中间结果到 `results/partial_<timestamp>.json`
- 重跑时检测到 partial 文件，提示用户是否续跑
- 续跑逻辑：跳过已有 `task_id + trial_index` 的组合
- 实现位置：`run.py` 或 `core/orchestrator.py`

**优先级**：P1（大规模跑评必备）

---

### 5. 矩阵测试 / 多 Agent 对比评估

**promptfoo 方案**：
```yaml
providers:
  - openai:gpt-4o
  - anthropic:claude-sonnet-4-20250514
  - ollama:llama3
prompts:
  - file://prompt_v1.txt
  - file://prompt_v2.txt
tests:
  - vars: { input: "查订单" }
  - vars: { input: "退款" }
# 自动生成 3 providers × 2 prompts × 2 tests = 12 个评估
```

Web UI 提供并排对比视图，一目了然。

**对比本项目**：本项目每次只能评估一个 Agent 配置，对比需要手动跑多次再人眼看报告。

**借鉴建议**：
- 支持配置多个 agent（或多个 config 文件路径）
- 运行后生成**对比报告**：同一场景下不同 Agent 的分数并排展示
- 实现路径：
  ```yaml
  # 方案 A：单配置多 agent
  agents:
    - name: "GPT-4o"
      adapter: openai_fc
      model: gpt-4o
    - name: "GLM-4"
      adapter: openai_fc
      model: glm-4
  ```
  ```bash
  # 方案 B：多配置对比
  python run.py --compare config_agent_a.yaml config_agent_b.yaml
  ```

**优先级**：P0（选型刚需，用户价值最大）

---

### 6. 断言分组 + 阈值（Assertion Set）

**promptfoo 方案**：
```yaml
assert:
  - type: assert-set
    threshold: 0.5    # 组内 50% 通过即可
    assert:
      - type: contains
        value: "订单号"
      - type: contains
        value: "金额"
      - type: contains
        value: "收货地址"
```

支持嵌套分组，区分"必须全通过"和"部分通过即可"。

**对比本项目**：`nl_assertion` 所有断言等权计算 `passed / total`，无法区分优先级。

**借鉴建议**：
- 在 `nl_assertions` 中支持分组和权重：
  ```json
  {
    "nl_assertions": [
      {"text": "Agent 拒绝了越权请求", "required": true},
      {"text": "Agent 给出了替代方案", "weight": 0.5},
      {"text": "Agent 语气礼貌", "weight": 0.3}
    ]
  }
  ```
- `required: true` 的断言失败则整组为 0
- 其余按权重计算

**优先级**：P2（精细化评估）

---

### 7. 成本追踪

**promptfoo 方案**：
- Provider 返回 `tokenUsage`（prompt/completion/total）和 `cost`（美元）
- 支持 `cost` 断言（如 `cost < 0.001`）
- 在报告和 Web UI 中展示每次调用的 token 消耗和费用
- 聚合总成本

**对比本项目**：完全没有成本核算。每次评估中的 LLM 调用（Agent + 评估器）的 token 消耗不可见。

**借鉴建议**：
- 在 `EvalResult` 中新增 `token_usage` 和 `estimated_cost` 字段
- Agent 适配器和评估器的 LLM 调用都记录 usage
- 在报告中新增"成本概览"部分
- 可选：在评估器中支持 `max_cost` 限制

**优先级**：P1（生产环境必需，对预算管理重要）

---

### 8. 输出格式扩展

**promptfoo 方案**：JSON、HTML、CSV、JSONL、XML、YAML、**JUnit XML**。

**对比本项目**：只有 JSON + HTML。

**借鉴建议**：
- **JUnit XML**（P3）：对接 CI/CD（GitHub Actions test-reporter、GitLab CI）
  ```xml
  <testsuites>
    <testsuite name="agent-eval" tests="5" failures="1">
      <testcase name="task_order_query" time="3.2"/>
      <testcase name="task_refund">
        <failure>state_match: 0.0 — 退款状态未更新</failure>
      </testcase>
    </testsuite>
  </testsuites>
  ```
- **CSV**（P3）：方便 Excel 做进一步分析
- **Markdown**（P3）：可直接粘贴到 PR 评论或飞书文档

**优先级**：P3（锦上添花，按需实现）

---

### 9. 红队测试框架

**promptfoo 方案**（本项目暂不直接借鉴，但值得了解）：
- 54+ 漏洞插件：prompt injection、SQL/shell injection、SSRF、数据泄露、BOLA/BFLA、PII 暴露、幻觉、毒性、偏见
- 20+ 攻击策略：Base64/Hex/ROT13 编码绕过、同形字替换、渐进式攻击、多语言攻击
- 标准数据集：HarmBench、BeaverTails、DoNotAnswer

**对比本项目**：`safety` 评估器做基础的关键词检测 + LLM 辅助判断，覆盖面有限。

**借鉴建议**：
- 当前 `safety` 评估器够用，但长期可以参考 promptfoo 的插件化攻击策略
- 特别是**编码绕过**和**渐进式攻击**，是实际红队中最常见的绕过手段

**优先级**：P3+（安全评估专项，按需推进）

---

### 10. 其他值得注意的设计

| 特性 | promptfoo 实现 | 借鉴价值 |
|------|---------------|---------|
| **中止信号** | 三级中止：用户 SIGINT、全局超时、目标错误信号 | 本项目只有 max_steps，可加全局超时 |
| **生命周期钩子** | `beforeAll`/`beforeEach`/`afterEach`/`afterAll` | 可用于评估前后的环境清理/数据准备 |
| **文件引用** | `file://` 协议 + glob 模式批量加载 | 场景文件管理更灵活 |
| **YAML 引用** | `$ref: '#/templates/name'` 复用断言模板 | 减少配置重复 |
| **Nunjucks 模板** | Prompt 和变量中支持完整模板引擎 | 场景参数化（如生成变体） |
| **串行自动降级** | 检测到 `_conversation` 变量时自动切换为串行执行 | 多轮对话场景自动适配 |

---

## 优先级总览

| 优先级 | 特性 | 投入 | 收益 | 说明 |
|--------|------|------|------|------|
| **P0** | LLM 调用缓存 | 低 | 高 | hash → 文件缓存，降本加速 |
| **P0** | 多 Agent 对比评估 | 中 | 高 | 选型刚需，差异化能力 |
| **P1** | 评估恢复（Resume） | 中 | 高 | 大规模跑评必备 |
| **P1** | 成本追踪 | 低 | 中 | Token 消耗和费用可视化 |
| **P2** | 轨迹顺序断言 | 低 | 中 | action_match 增强 |
| **P2** | 自定义评分公式 | 中 | 中 | 灵活性提升 |
| **P2** | 断言分组+阈值 | 低 | 中 | 精细化评估 |
| **P3** | JUnit XML 输出 | 低 | 低 | CI/CD 集成 |
| **P3** | CSV / Markdown 输出 | 低 | 低 | 分析/分享 |
| **P3+** | 红队攻击策略 | 高 | 中 | 安全评估专项 |

---

## 与本项目的定位差异

| 维度 | promptfoo | agent-evaluator |
|------|-----------|-----------------|
| **核心定位** | Prompt/LLM 评估 | Agent 行为评估 |
| **交互模型** | 单次 prompt → response | 多步 Agent ↔ Environment 循环 |
| **评估对象** | LLM 输出质量 | Agent 完成任务的能力 |
| **环境模拟** | 无（纯文本出入） | 有（mock_db、passthrough） |
| **工具调用** | 作为断言验证 | 作为核心交互机制 |
| **评分策略** | 加权平均 | 乘积组合 |
| **红队能力** | 强（54+ 插件） | 基础（safety 评估器） |
| **矩阵测试** | 强（多维笛卡尔积） | 无 |
| **CI/CD** | 内建支持 | 无 |
| **语言** | TypeScript | Python |

两者**互补大于竞争**：promptfoo 擅长评估"LLM 说了什么"，agent-evaluator 擅长评估"Agent 做了什么"。借鉴 promptfoo 的工程实践（缓存、恢复、对比、成本追踪），可以让 agent-evaluator 更生产就绪。
