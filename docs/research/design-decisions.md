# 架构设计决策：从调研到实现

> 记录了我们如何从三个开源项目中提炼设计思路，形成本项目的架构方案。

---

## 一、调研结论：三个项目各解决了什么问题

| 项目 | 核心贡献 | 我们借鉴了什么 |
|------|---------|--------------|
| AgentBoard (HKUST, NeurIPS'24) | 多维分析型评估 | 评估维度拆解、进度追踪、能力雷达图 |
| AgentBench (THU, ICLR'24) | 分布式工程架构 | Registry 注册表、统一接口、工厂模式 |
| τ²-Bench (Sierra Research) | 双控制对话评估 | DB 哈希比对、四重评判乘积、Pass^k、LLM-as-User |

详细的源码级分析见 [benchmark-analysis.md](./benchmark-analysis.md)。

---

## 二、核心设计决策

### 决策 1：插件化 Registry 模式

**来源**：三个项目都使用了类似的注册表模式

**我们的实现**：

```python
@registry.adapter("openai_fc")
class OpenAIFCAdapter(AgentAdapter): ...

@registry.environment("mock_db")
class MockDBEnvironment(Environment): ...

@registry.evaluator("state_match")
class StateEvaluator(Evaluator): ...
```

**为什么**：新增 Agent 类型/环境/评估器只需加文件 + 装饰器，不改框架核心代码。这是保证可扩展性的根基。

### 决策 2：三个抽象接口作为扩展点

**来源**：AgentBench 的 `AgentClient.inference()` + AgentBoard 的 `gym.Env` + τ²-Bench 的 `Evaluator`

```
AgentAdapter   — 被评估 Agent 的统一接入接口
Environment    — 任务环境的统一接口 (reset / step / get_state_hash)
Evaluator      — 评估维度的统一接口 (evaluate → 0.0~1.0)
```

**为什么**：这是"变化的部分"和"不变的部分"的分界线。Orchestrator 只依赖这三个接口，不关心具体实现。

### 决策 3：评估器乘积组合（而非加权平均）

**来源**：τ²-Bench 的四重评判乘积

```python
overall = state_match * action_match * info_delivery
# 任一为 0 → 总分为 0
```

**为什么**：乘积比加权平均更严格——你不能靠某个维度的高分来掩盖另一个维度的完全失败。这更接近真实场景：客服即使态度好（info_delivery=1.0），如果没有实际操作（state_match=0.0），任务就是失败的。

### 决策 4：DB 状态哈希比对作为核心评估手段

**来源**：τ²-Bench 的 Environment Evaluator

```
Gold 环境: 初始状态 → 执行期望动作 → hash_gold
Pred 环境: 初始状态 → 回放实际轨迹 → hash_pred
评估: hash_gold == hash_pred ? 1.0 : 0.0
```

**为什么**：
- 与执行顺序无关（Agent 可以用不同路径达到相同结果）
- 完全确定性（不依赖 LLM 判断）
- 易于扩展（任何有状态的环境都可以提供哈希）

### 决策 5：MVP 先不做 LLM-as-User

**来源**：τ²-Bench 的 UserSimulator（角色翻转技巧）

**决定**：Step 1 先用固定的 `initial_message` 作为用户输入，Step 2 再加入 LLM-as-User。

**为什么**：LLM-as-User 引入了不确定性（用户行为不可控），会增加调试难度。先验证核心评估链路正确，再加入对话复杂度。

### 决策 6：Orchestrator 在进程内编排（非 HTTP）

**舍弃的方案**：AgentBench 的 HTTP Client-Server 架构

**我们的选择**：进程内直接调用，类似 τ²-Bench

**为什么**：MVP 阶段不需要分布式。进程内调用更简单、更容易调试。后续如果需要分布式，可以在 Adapter 层包装 HTTP 调用，不影响 Orchestrator。

---

## 三、各组件的设计来源映射

```
本项目组件                    设计灵感来源
─────────────────────────────────────────────────────
core/registry.py            ← AgentBoard 的 Registry 单例 + 装饰器
core/types.py (Message)     ← τ²-Bench 的 5 种消息类型（简化版）
core/types.py (Task)        ← τ²-Bench 的 Task 结构（含 evaluation_criteria）
core/base.py (AgentAdapter) ← AgentBench 的 AgentClient.inference()
core/base.py (Environment)  ← AgentBoard 的 gym.Env (reset/step)
                              + τ²-Bench 的 get_state_hash()
core/base.py (Evaluator)    ← τ²-Bench 的 Evaluator 接口
core/orchestrator.py        ← AgentBoard 的交互循环 + τ²-Bench 的 Orchestrator
evaluators/state_evaluator  ← τ²-Bench 的 evaluator_env.py (DB 哈希比对)
evaluators/action_evaluator ← τ²-Bench 的 evaluator_action.py (match_args)
evaluators/info_evaluator   ← τ²-Bench 的 evaluator_communicate.py
run.py (难度分层统计)         ← AgentBoard 的 easy/hard 分层
```

---

## 四、暂未实现但已规划的设计

| 特性 | 来源 | 计划阶段 |
|------|------|---------|
| LLM-as-User 用户模拟 | τ²-Bench | Step 2 |
| 角色翻转技巧 | τ²-Bench UserSimulator | Step 2 |
| Pass^k 一致性指标 | τ²-Bench | Step 3 |
| 能力维度雷达图 | AgentBoard | Step 3 |
| 进度率 (Progress Rate) | AgentBoard | Step 3 |
| NL Assertion (LLM-as-Judge) | τ²-Bench | Step 3 |
| 并发调度 | AgentBench Max-Flow | Step 4+ |
| 断点续传 | AgentBench runs.jsonl | Step 4+ |
| Docker 环境隔离 | AgentBench | Step 4+ |
| Gymnasium RL 接口 | τ²-Bench | Step 4+ |
