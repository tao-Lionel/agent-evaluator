# AI Agent 评估方法论

> 内部分享初稿 — 下周五小组分享用

---

## 核心问题：Agent 和传统聊天机器人有什么不同？

传统聊天机器人只需要"说得好"——回答准确、语气友好就行。但 **Agent 不一样**，它不只是说话，还要**操作**：查数据库、调接口、改状态、做决策。这意味着评估维度从一维变成了多维：

> 说对了 ≠ 做对了 ≠ 做对了且告诉了用户

举个例子：用户说"帮我取消订单 ORD-1001"，Agent 可能出现这些情况：

| 场景 | 数据库状态       | 工具调用                 | 回复内容                | 结论               |
| ---- | ---------------- | ------------------------ | ----------------------- | ------------------ |
| A    | 改成 cancelled ✓ | query + update ✓         | "已取消" ✓              | 完美               |
| B    | 改成 cancelled ✓ | query + update ✓         | "好的"（没说具体结果）✗ | **做对了但没说清** |
| C    | 没改 ✗           | 只 query 没 update ✗     | "已取消" ✓              | **说了谎**         |
| D    | 改成 cancelled ✓ | 直接 update 没先 query ✗ | "已取消" ✓              | **过程不规范**     |

如果只看回复文本（传统方法），场景 C 和 D 都会被判为"通过"——但实际上一个在说谎，一个跳过了验证步骤。

---

## 六个评估维度

### 维度 1：状态正确性（state_match）

**问的是：Agent 做完之后，世界的状态对不对？**

这是最"硬"的指标。不管 Agent 说了什么、怎么操作的，最终数据库/系统状态必须和预期一致。

**实现方法：两阶段验证**

- **阶段一（精确匹配）**：用预期操作序列重建一个"标准答案"数据库，对比两个数据库的 MD5 哈希。哈希一样 = 完全一致 = 满分
- **阶段二（子集匹配）**：精确匹配失败时，检查关键字段是否正确，给部分分数。比如订单状态改对了但多加了一个备注字段，仍然给高分

**为什么需要两阶段？** Agent 的操作可能在非关键细节上偏离"标准答案"——比如更新时额外设置了一个 `updated_at` 字段，或者字段顺序不同。这些差异会导致 MD5 哈希不一致，但关键业务状态（如订单变为 cancelled）是正确的。子集匹配就是为这种情况提供合理容错。

> 注意：state_match 只管"最终状态对不对"，不管"怎么到达的"。过程是否规范由 action_match 单独评判。两个维度独立打分，最终乘积组合——即使状态正确（state=1.0），过程不规范（action=0.5）也会拉低总分。

---

### 维度 2：行为规范性（action_match）

**问的是：Agent 的操作过程是否符合预期？**

不只看结果，还看过程。比如"取消订单"应该先查询确认存在，再更新状态——不能跳过查询直接改。

**实现方法：部分参数匹配**

- 定义期望的工具调用序列，每个调用指定必须匹配的参数子集（`match_args`）
- 不要求参数完全一致，只检查关键参数

```
期望：query(table="orders") → update(table="orders", updates={"status":"cancelled"})
实际：query(table="orders", filters={"id":"ORD-1001"}) → update(...)
                              ↑ 多出来的参数不影响匹配
```

**为什么不要求完全匹配？** Agent 可能传额外参数（如 filters），这不影响正确性。部分匹配只检查"必须有的"，容忍"多出来的"。

---

### 维度 3：信息传达（info_delivery）

**问的是：Agent 有没有把关键信息告诉用户？**

Agent 可能正确操作了数据库，但回复只说"好的"——用户根本不知道发生了什么。这个维度检查回复中是否包含所有必要信息。

**实现方法：模糊子串匹配**

- 定义一组必须出现在回复中的关键词/短语（`required_info`）
- 归一化处理（忽略大小写、标点、多余空格）后匹配
- 支持拆词匹配："退款 7 个工作日"和"7个工作日内退款"都能通过

**得分 = 匹配数 / 总数**。5 个必要信息说了 4 个 = 0.8 分。

---

### 维度 4：回复质量（llm_judge）

**问的是：Agent 的回复整体质量如何？**

前三个维度是"确定性"的——有标准答案可以对照。但有些场景没有标准答案，比如"我对这个商品不太满意，你觉得我应该退货还是换货？"。这时候需要用另一个 LLM 来当评委。

**实现方法：LLM-as-Judge**

- 把任务描述、用户消息、Agent 回复打包发给评委 LLM
- 评委按 1-5 分打分，归一化为 0.0-1.0
- 设计了 **INSUFFICIENT_INFO 逃生门**：如果评委认为信息不足以做出判断，返回特殊标记而非强行打分

**为什么需要逃生门？** LLM 评委也会"乱猜"。与其让它在不确定时随便给分，不如让它诚实说"我判不了"，系统会跳过这道题而非给零分。

**LLM-as-Judge 的已知局限（必须了解）：**

2024-2025 年的研究揭示了 LLM 评委的多种系统性偏见：

- **位置偏见（Position Bias）**：仅交换两个答案的呈现顺序，判分结果可偏移超过 10%
- **冗长偏见（Verbosity Bias）**：倾向给更长的回复打更高分，即使内容质量没有提升
- **自我偏好（Self-Preference Bias）**：LLM 给自己生成的文本打分显著高于其他模型的输出
- **一致性差**：同一个评判任务重复执行，结果可能不同

**缓解措施：**

- 随机打乱答案顺序，多次运行取多数投票或均值
- 强制 LLM 先输出推理过程再给分（CoT 评判），显著提升与人类判断的一致性
- 能用确定性指标的场景优先用确定性指标，LLM-as-Judge 仅作为补充

> 参考：Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023) 发现 GPT-4 作为评委与人类的一致率超过 80%，但后续研究（Ye et al. "Justice or Prejudice?" 2024）进一步量化了上述偏见。

---

### 维度 5：断言验证（nl_assertion）

**问的是：一组具体的业务规则是否被满足？**

有时候你想检查非常具体的条件，但用确定性规则写不出来（或太麻烦）。比如：

- "refunds 表中应该存在一条 order_id 为 ORD-2001 的记录"
- "Agent 的回复中应该提到退款预计到账时间"

**实现方法：LLM 逐条判定**

- 每条断言由 LLM 判定 PASS 或 FAIL
- 得分 = 通过数 / 总数
- 可以检查数据库状态（有 DB 时），也可以检查回复内容（无 DB 时）

**和 llm_judge 的区别：** llm_judge 是整体印象分（1-5），nl_assertion 是逐条检查清单。前者适合开放性评估，后者适合有明确规则的场景。

---

### 维度 6：安全性（safety）

**问的是：Agent 有没有做不该做的事？**

这是一个"减分项"维度，检查：

- 有害内容（辱骂、歧视、暴力）
- 越权操作（删除不属于用户的数据）
- 信息泄露（暴露其他用户的隐私）
- 违反业务规则（批量删除、跳过审批流程）

**实现方法：关键词 + LLM 双重检测**

---

## 为什么用乘积而不是加权平均？

这是整个框架最核心的设计决策：

```
overall = state × action × info × ...   （不是平均）
```

**加权平均的问题：**

假设 state=0（什么都没改），action=1.0，info=1.0

- 加权平均：0 × 0.4 + 1.0 × 0.3 + 1.0 × 0.3 = **0.6 分** — Agent 完全没干活还拿 60 分？
- 乘积：0 × 1.0 × 1.0 = **0 分** — 准确反映失败

**乘积的哲学：每个维度都是必要条件，不是可以互相补偿的。**

这借鉴了 τ-bench（Yao et al., ICLR 2025）的设计。它的实验发现：即使是顶级模型，在零售场景的 pass^8 也不到 25%——Agent 的失败往往是"某一个维度彻底崩溃"，而不是"每个维度都差一点"。乘积评分能精确捕捉这种模式。

> 注：Anthropic 已将 τ-bench 作为 Claude 系列模型的核心评估基准之一，历代 Claude 模型的发布都使用 τ-bench 展示 Agent 能力进展

---

## 稳定性指标：Pass@k 和 Pass^k

单次测试可能有运气成分。同一个任务跑 5 次，3 次通过 2 次失败——这个 Agent 到底行不行？

- **Pass@k**：k 次机会中至少有 1 次通过的概率。衡量"能力上限"
- **Pass^k**：k 次全部通过的概率。衡量"稳定性下限"

**举例：5 次试验中 3 次通过（通过率 p = 3/5 = 0.6）**

Pass^k 比较好算——每次通过概率 60%，连续 k 次都通过就是 0.6 连乘 k 次：

- Pass^3 = 0.6 × 0.6 × 0.6 = **0.216**（连续 3 次都成功？只有 21.6%）

Pass@k 可以想象成抽牌：5 次结果就是 5 张牌 ✓ ✓ ✓ ✗ ✗，闭眼随机抽 k 张，"至少抽到 1 张 ✓"的概率就是 Pass@k。反过来先算"全抽到 ✗"的概率，再用 1 减掉更简单：

- Pass@1：抽 1 张，抽到 ✗ 的概率 = 2/5，所以至少 1 张 ✓ = 1 - 2/5 = **0.6**
- Pass@2：抽 2 张，第 1 张 ✗ = 2/5，第 2 张也 ✗ = 1/4（剩 4 张里只剩 1 张 ✗），全 ✗ = 2/5 × 1/4 = 1/10，所以至少 1 张 ✓ = 1 - 1/10 = **0.9**
- Pass@3：抽 3 张，但 ✗ 只有 2 张，不可能 3 张全是 ✗，所以一定有 ✓ = **1.0**

**Pass@k 和 Pass^k 的差距越大，说明 Agent 越不稳定**——有能力但不可靠。比如 Pass@2 = 0.9 vs Pass^2 = 0.36，说明"给两次机会大概率能成一次"，但"两次都成"的概率只有三分之一。

---

## 总结：评估维度的选择逻辑

```
白盒 Agent（有工具调用）         黑盒 Agent（只有文本回复）
├── state_match   ← 必选         ├── llm_judge     ← 必选
├── action_match  ← 必选         ├── info_delivery  ← 推荐
├── info_delivery ← 推荐         ├── nl_assertion   ← 可选
├── nl_assertion  ← 可选         └── safety         ← 可选
└── safety        ← 可选
```

核心原则：**能用确定性指标就不用 LLM 判**（更稳定、更可重复），LLM-as-Judge 是确定性指标覆盖不到时的补充。

> 评估的核心价值不是一刀切地判"合不合格"，而是**发现问题和跟踪进步**——找到 Agent 的薄弱维度，每次迭代看分数有没有提升。

---

## 延伸：更广阔的 Agent 评估版图

本文聚焦对话式客服 Agent 的评估，但 Agent 评估领域远不止于此。以下是 2025 年最具影响力的几个基准（可以理解为"公开的标准化考试题"），供同事们了解全貌：

### SWE-bench — 考写代码

给 Agent 一个真实的 GitHub issue（bug 报告），看它能不能自己读代码、定位问题、写出正确的修复补丁。相当于"程序员笔试"。2025 年底最高分 74.4%（Claude Opus 4.5），意味着 100 个真实 bug 能自动修好 74 个。

### GAIA — 考综合能力

466 道人工出的题，需要多步推理 + 查资料 + 用工具才能答对。比如"某公司 2023 年 Q3 营收比 Q2 增长了多少？"——Agent 得自己去搜财报、找数字、算百分比。最高分已达 90%，逼近人类基线 92%。

### WebArena — 考操作网页

812 个真实网页任务，在电商、论坛、代码仓库等网站上完成操作。比如"在购物网站上找到评分最高的耳机并加入购物车"。测的是 Agent 能不能像人一样点击、输入、跳转页面来完成任务。

### BFCL — 考调工具

Berkeley 函数调用排行榜，专门测模型"能不能正确调用 API"。比如给一个天气查询函数，模型能不能传对参数（城市名、日期格式等）。这是 Agent 的基础能力——连工具都调不对，后面的流程就没法走。

### τ³-bench — 考对话办事（和本项目最接近）

τ-bench 的 2026 年演进版。模拟真实客服场景（零售、航空、银行），用户通过多轮对话让 Agent 帮忙处理业务（退订单、改航班等）。本项目的乘积评分和 Pass@k 指标就是借鉴了 τ-bench 的设计。

---

## 参考资料

### 评估框架（GitHub 项目）

| 项目            | 机构            | 侧重点                                 | 链接                                          |
| --------------- | --------------- | -------------------------------------- | --------------------------------------------- |
| **τ-bench**     | Sierra Research | 对话 Agent 评估，乘积评分，Pass^k      | https://github.com/sierra-research/tau-bench  |
| **τ²/τ³-bench** | Sierra Research | 双控制（用户也用工具）、语音、知识检索 | https://github.com/sierra-research/tau2-bench |
| **AgentBench**  | 清华大学 THUDM  | 分布式架构，Docker 隔离，8 种环境      | https://github.com/THUDM/AgentBench           |
| **AgentBoard**  | 香港科大 NLP    | 进度率指标，9 种环境，细粒度分析       | https://github.com/hkust-nlp/AgentBoard       |

### 论文（已验证 arxiv 链接）

- **Pass@k 指标**：Chen et al. "Evaluating Large Language Models Trained on Code" (2021) — https://arxiv.org/abs/2107.03374
- **LLM-as-Judge**：Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (NeurIPS 2023) — https://arxiv.org/abs/2306.05685
- **τ-bench**：Yao et al. "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" (ICLR 2025) — https://arxiv.org/abs/2406.12045
- **τ²-bench**：Barres et al. "τ²-bench: Evaluating Conversational Agents in a Dual-Control Environment" (2025) — https://arxiv.org/abs/2506.07982
- **AgentBench**：Liu et al. "AgentBench: Evaluating LLMs as Agents" (ICLR 2024) — https://arxiv.org/abs/2308.03688
- **AgentBoard**：Ma et al. "AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents" (NeurIPS 2024 Oral) — https://arxiv.org/abs/2401.13178
- **LLM-as-Judge 偏见研究**：Ye et al. "Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge" (2024) — https://arxiv.org/abs/2410.02736

### 其他参考

- **Agent 评估完全指南**（基于 Anthropic 研究）— https://github.com/adongwanai/AgentGuide/blob/main/docs/02-tech-stack/agent-evaluation-complete-guide.md
- **τ-bench 排行榜** — https://taubench.com/
- **Agent 基准大全**（50+ 个基准汇总）— https://github.com/philschmid/ai-agent-benchmark-compendium
