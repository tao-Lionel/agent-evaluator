# LLM Agent 评估基准深度分析

> 本文档详细分析了三个主流 LLM Agent 评估项目的架构设计与运作方式：AgentBoard、AgentBench、τ²-Bench

---

## 目录

- [一、AgentBoard (HKUST)](#一agentboard-hkust)
- [二、AgentBench (THUDM)](#二agentbench-thudm)
- [三、τ²-Bench (Sierra Research)](#三τ²-bench-sierra-research)
- [四、三个项目对比总结](#四三个项目对比总结)

---

# 一、AgentBoard (HKUST)

> 项目地址：https://github.com/hkust-nlp/AgentBoard
> 会议：NeurIPS 2024 (Oral)

## 1.1 项目概述

AgentBoard 是由香港科技大学 NLP 团队开发的一个 LLM Agent 多轮交互评估平台，致力于"全面系统地评估 LLM Agent 在多种环境中的泛化能力"。

项目遵循四个关键设计原则：

1. **任务多样性**：包含 9 个不同领域的任务，涵盖具身 AI、游戏、网页和工具四大类
2. **多轮交互**：支持 Agent 与环境的多轮对话，反映智能体的适应性学习过程
3. **部分可观测环境**：Agent 需要通过在线探索获取完整信息，评估其世界建模能力
4. **分析型评估**：提供详细的可视化面板，展示细粒度进度率、接地准确度、困难样本分析等多维性能指标

## 1.2 支持的任务与模型

### 任务类型（9 大任务，4 个领域）

| 领域 | 任务 |
|------|------|
| 具身 AI | AlfWorld、ScienceWorld、BabyAI |
| 游戏 | Jericho、PDDL |
| 网页 | WebShop、WebArena |
| 工具 | Tool-Query、Tool-Operation |

### 支持的模型

- **闭源**：GPT-4、GPT-3.5-Turbo、Claude 2 等
- **开源**：Llama 2、CodeLlama、DeepSeek、Mistral 等

## 1.3 整体架构概览

AgentBoard 采用经典的 **Agent-Environment 交互循环** 架构，核心由 4 个模块组成：

```
┌─────────────────────────────────────────────────────────┐
│                     eval_main.py                        │
│              (主入口，编排整个评估流程)                     │
├──────────┬──────────┬───────────┬───────────────────────┤
│   LLM    │  Agent   │Environment│   Task (Evaluator)    │
│  后端层   │  策略层   │  环境层    │   任务编排层            │
├──────────┼──────────┼───────────┼───────────────────────┤
│ OpenAI   │ Vanilla  │ AlfWorld  │ Evalalfworld          │
│ Azure    │ ReAct    │ WebShop   │ EvalWebshop           │
│ Claude   │          │ BabyAI    │ EvalTool              │
│ vLLM     │          │ Jericho   │ ...                   │
│ HuggingFace│        │ PDDL      │                       │
│          │          │ WebArena  │                       │
│          │          │ Tool-*    │                       │
└──────────┴──────────┴───────────┴───────────────────────┘
         ↕                ↕                  ↕
    ┌──────────┐   ┌───────────┐    ┌──────────────┐
    │ Registry │   │  Prompts  │    │   Logger     │
    │ (插件注册) │   │ (提示模板) │    │ (指标+可视化) │
    └──────────┘   └───────────┘    └──────────────┘
```

## 1.4 插件注册系统 (Registry)

AgentBoard 使用一个全局 Registry 单例，通过装饰器注册所有组件：

```python
@registry.register_agent("VanillaAgent")
class VanillaAgent(BaseAgent): ...

@registry.register_environment("alfworld")
class AlfWorld(BaseEnvironment): ...

@registry.register_task("alfworld")
class Evalalfworld(BaseTask): ...

@registry.register_llm("gpt")
class OPENAI_GPT: ...
```

运行时通过 `load_agent(name)`、`load_environment(name)`、`load_task(name)`、`load_llm(name)` 工厂函数按名称查找，实现完全解耦。

## 1.5 核心交互循环

每个样本的评估遵循以下流程：

```
┌──────────┐    goal + init_obs     ┌──────────┐
│          │ ──────────────────────→ │          │
│ 环境 Env  │                        │ Agent    │
│          │ ←────────────────────── │ (LLM)   │
│ .step()  │        action           │ .run()   │
│          │ ──────────────────────→ │          │
│          │   obs, reward, done     │ .update()│
│          │ ←────────────────────── │          │
└──────────┘    (循环 N 步)          └──────────┘
```

**伪代码**：

```python
def evaluate_env(example_index):
    # 1. 初始化
    obs, info = env.reset()
    goal = env.get_goal()
    agent.reset(goal, init_obs=obs)

    # 2. 多轮交互循环
    for step in range(max_num_steps):  # 默认 30 步
        # Agent 基于历史生成动作
        success, action = agent.run()

        # 环境执行动作，返回新观察
        obs, reward, done, info = env.step(action)

        # 更新 Agent 记忆
        agent.update(action, obs)

        # 记录 grounding（动作是否合法）
        if action in env.get_action_space():
            grounding_count += 1

        # 记录进度变化
        if reward > last_reward:
            score_changes.append((step, reward))

        if done:
            break

    # 3. 返回指标
    return success_rate, progress_rate, grounding_acc
```

## 1.6 Agent 策略层

### 基类 BaseAgent

```python
class BaseAgent:
    reset(goal, init_obs)    # 设定目标和初始观察
    run() → (success, action) # 调用 LLM 生成下一个动作
    update(action, state)     # 将交互记录加入记忆
```

### VanillaAgent（简单反射型）

- **记忆机制**：维护 `[(type, content), ...]` 列表，如 `[('Observation', '...'), ('Action', '...')]`
- **滑动窗口**：`memory_size=100`，超出时丢弃最早的记忆
- **Token 截断**：如果 prompt 超过 `max_context_length - max_tokens`，从最早的记忆开始删除

**Prompt 构造**：

```
<instruction>你是一个有帮助的助手...</instruction>
<example>少样本示例...</example>
<goal>完成任务: 找到红色毛衣并购买</goal>

Observation: 你在搜索页面...
Action: search[red sweater]
Observation: 搜索结果显示...
Action: ← (等待 LLM 生成)
```

### ReactAgent（推理+行动型）

在 VanillaAgent 基础上增加了思考环节：

```
while 未产生 Action:
    response = LLM.generate(prompt)
    if response 以 "Think:" 开头:
        记录思考内容到记忆
        think_count += 1
    elif response 以 "Action:" 开头:
        提取动作，跳出循环
    else:
        强制下轮必须输出 Action
```

## 1.7 环境层

所有环境继承 `gym.Env`，统一接口：

| 方法 | 作用 |
|------|------|
| `reset()` | 重置环境，返回初始观察 |
| `step(action)` | 执行动作，返回 `(obs, reward, done, info)` |
| `get_action_space()` | 返回当前合法动作集 |
| `get_goal()` | 返回任务目标 |

**各环境的 reward 计算方式不同**：

| 环境 | Reward 计算 |
|------|------------|
| AlfWorld | `已完成子目标数 / 总子目标数`（0~1 连续值）|
| WebShop | 购买结果匹配度（最终才有值）|
| BabyAI | 网格世界任务完成度 |
| Jericho | 文字冒险游戏得分 |
| Tool-* | 动作是否返回 ERROR |

## 1.8 评估指标体系

### 三大核心指标

| 指标 | 含义 | 计算方式 |
|------|------|---------|
| **Success Rate (SR)** | 任务完成率 | `done ? 1 : 0`，对所有样本求均值 |
| **Progress Rate (PR)** | 进度率 | 环境返回的 reward（0~1），即使未完成也有部分分 |
| **Grounding Accuracy** | 接地准确度 | `合法动作数 / 总动作数` |

### 六大能力维度

系统将 9 个任务映射到 6 个能力维度（每个任务在每个维度有 1-3 的权重）：

```
Memory（记忆）        ← alfworld, webshop, ...
Planning（规划）      ← alfworld(1), webshop(2), pddl(3), ...
World Modeling（建模） ← alfworld(3), scienceworld(3), ...
Self-reflection（反思）← alfworld(3), jericho(2), ...
Grounding（接地）     ← alfworld(2), webshop(3), ...
Spatial Navigation    ← alfworld(2), babyai(3), ...
```

最终生成雷达图展示各维度得分。

### 难度分层

每个样本标注了 `easy`/`hard` 难度，分别计算 SR 和 PR，用于分析模型在不同难度上的表现差异。

## 1.9 配置系统

所有配置集中在一个 YAML 文件中，分 4 个部分：

```yaml
run:       # 运行参数 (max_steps, wandb, log_path)
agent:     # Agent 类型和参数 (VanillaAgent/ReactAgent, memory_size)
llm:       # 多个 LLM 配置 (按名称索引)
  gpt-3.5-turbo-16k:
    name: gpt
    engine: gpt-3.5-turbo-16k
    context_length: 16384
    temperature: 0.0
    ...
env:       # 各任务的环境配置
  alfworld:
    label_path: data/alfworld/test.jsonl
    init_prompt_path: prompts/VanillaAgent/alfworld_base.json
    ...
```

支持 `${PROJECT_PATH}` 环境变量展开。

## 1.10 主入口编排流程

```
命令行参数 + YAML 配置
        │
        ▼
   load_config() ──→ llm_config, agent_config, env_config, run_config
        │
        ▼
   load_llm("gpt") ──→ OPENAI_GPT 实例
        │
        ▼
   初始化 SummaryLogger + W&B
        │
        ▼
   ┌─── For each task ───────────────────────────────────┐
   │                                                      │
   │  load_task("alfworld") ──→ Evalalfworld 实例         │
   │    ├─ 内部创建 Agent 实例                              │
   │    ├─ 内部创建 Environment 实例                        │
   │    └─ 内部创建 TaskLogger                             │
   │                                                      │
   │  task.evaluate() ──→ 遍历所有样本                      │
   │    ├─ env.reset()                                    │
   │    ├─ agent.reset(goal, obs)                         │
   │    ├─ 多轮循环: agent.run() → env.step() → agent.update() │
   │    ├─ 记录 trajectory + metrics                       │
   │    └─ 返回 SR, PR, Grounding Acc                     │
   │                                                      │
   │  agentboard.log_run_result() ──→ all_results.txt     │
   └──────────────────────────────────────────────────────┘
        │
        ▼
   agentboard.log_summary()
        ├─ 生成维度得分 → dimension.txt
        ├─ 雷达图 (SR vs 任务)
        ├─ 雷达图 (能力维度)
        └─ 柱状图 (当前模型 vs 基线)
```

## 1.11 项目结构

```
AgentBoard/
├── agentboard/
│   ├── agents/                # Agent 实现
│   │   ├── base_agent.py      # 抽象基类
│   │   ├── vanilla_agent.py   # 简单反射 Agent
│   │   └── react_agent.py     # ReAct Agent
│   ├── environment/           # 环境实现
│   │   ├── base_env.py        # gym.Env 基类
│   │   ├── alfworld/          # AlfWorld 环境
│   │   ├── webshop_env.py     # WebShop 环境
│   │   ├── babyai_env.py      # BabyAI 环境
│   │   ├── jericho_env.py     # Jericho 环境
│   │   ├── pddl_env/          # PDDL 环境
│   │   ├── scienceworld_env.py
│   │   ├── academia_env.py    # Tool 环境
│   │   ├── movie_env.py
│   │   ├── todo_env.py
│   │   ├── sheet_env.py
│   │   └── weather_env.py
│   ├── tasks/                 # 任务评估编排
│   │   ├── base_task.py
│   │   ├── alfworld.py
│   │   ├── webshop.py
│   │   ├── webbrowse.py
│   │   └── tool.py
│   ├── llm/                   # LLM 后端
│   │   ├── openai_gpt.py
│   │   ├── azure_gpt.py
│   │   ├── claude.py
│   │   ├── vllm.py
│   │   └── huggingface.py
│   ├── prompts/               # 提示模板
│   │   ├── VanillaAgent/
│   │   └── ReactAgent/
│   ├── utils/logging/         # 日志与指标
│   │   ├── logger.py          # SummaryLogger + TaskLogger
│   │   └── agent_logger.py
│   ├── common/registry.py     # 注册表
│   └── eval_main.py           # 主入口
├── eval_configs/
│   └── main_results_all_tasks.yaml
└── data/                      # 评估数据集
```

## 1.12 设计亮点

1. **插件化架构**：通过 Registry 装饰器，新增模型/环境/Agent 只需注册即可
2. **统一的 gym 接口**：所有环境遵循 `reset/step/done` 标准，方便扩展
3. **细粒度进度追踪**：不只看最终成功与否，还记录每步的 reward 变化曲线
4. **Grounding 独立评估**：将"动作是否合法"与"任务是否完成"分开度量
5. **多维度能力画像**：通过加权映射将任务表现转化为 6 项能力得分
6. **Token 感知的记忆管理**：Agent 自动截断历史以适应 LLM 上下文窗口

## 1.13 安装与使用

**环境要求**：Python 3.8.13，支持本地安装和 Docker 部署

```bash
conda create -n agentboard python=3.8.13
conda activate agentboard
git clone https://github.com/hkust-nlp/AgentBoard.git
cd AgentBoard

# 下载数据集
wget https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz
tar -xzvf data.tar.gz

# 安装依赖
bash setup.sh

# 运行评估
python agentboard/eval_main.py \
  --cfg-path eval_configs/main_results_all_tasks.yaml \
  --tasks alfworld \
  --model gpt-3.5-turbo-0613 \
  --log_path ./results/gpt-3.5-turbo-0613
```

### 运行时间参考

- API 模型（GPT-3.5-Turbo）：约 3 小时
- 开源模型（V100 GPU）：12-28 小时

---

# 二、AgentBench (THUDM)

> 项目地址：https://github.com/THUDM/AgentBench
> 会议：ICLR 2024

## 2.1 项目概述

AgentBench 是清华大学 THUDM 团队开发的 LLM Agent 综合评估基准，与 AgentBoard 的核心区别在于：它采用了一套**分布式 Client-Server 架构**，通过 HTTP REST API 通信，支持真正的分布式部署和横向扩展。

当前版本引入了 **AgentBench FC（函数调用版本）**，基于 AgentRL 框架。

## 2.2 整体架构

```
┌───────────────────────────── Client 侧 ─────────────────────────────┐
│                                                                      │
│                        Assigner (主控调度器)                          │
│                    ┌──────────────────────────┐                      │
│                    │  Max-Flow 调度算法         │                      │
│                    │  并发控制 + 断点续传        │                      │
│                    └────────┬─────────────────┘                      │
│                             │ 为每个样本启动一个线程                    │
│              ┌──────────────┴──────────────┐                         │
│              ▼                             ▼                         │
│     ┌─────────────────┐          ┌──────────────────┐               │
│     │   AgentClient    │          │   TaskClient      │               │
│     │  .inference()    │          │  HTTP REST 代理    │               │
│     │  调用 LLM 生成    │          │  /start_sample    │               │
│     └────────┬────────┘          │  /interact        │               │
│              │                    │  /calculate_overall│               │
│              │                    └────────┬─────────┘               │
└──────────────┼─────────────────────────────┼─────────────────────────┘
               │ HTTP                        │ HTTP
               ▼                             ▼
      ┌──────────────┐            ┌────────────────────────┐
      │  LLM Server   │            │   Task Controller      │
      │  (OpenAI API,  │            │   :5000/api            │
      │   Claude,      │            │                        │
      │   FastChat,    │            │   ┌──────────────────┐ │
      │   vLLM)        │            │   │  Task Workers     │ │
      └──────────────┘            │   │  (Docker 容器)     │ │
                                   │   │  ├─ AlfWorld      │ │
                                   │   │  ├─ MySQL (DB)    │ │
                                   │   │  ├─ Freebase (KG) │ │
                                   │   │  ├─ OS Container  │ │
                                   │   │  └─ WebShop       │ │
                                   │   └──────────────────┘ │
                                   └────────────────────────┘
```

## 2.3 核心组件详解

### Assigner（主控调度器）

整个系统的大脑，负责：

- **读取配置**：解析 YAML 配置文件，动态实例化 Agent 和 Task
- **Max-Flow 调度**：将 `(agent, task, sample_index)` 三元组的分配建模为最大流问题，在 Agent 并发限制和 Task Worker 容量之间找到最优分配
- **线程管理**：每个样本评估在独立线程中运行
- **断点续传**：读取 `runs.jsonl` 跳过已完成的样本

```python
class Assigner:
    def worker_generator(self):
        # 使用最大流算法，产出 (agent, task, index) 三元组
        graph = Graph(agents, tasks, concurrency_limits)
        max_flow = MaxFlow(graph)
        yield (agent, task, index)

    def start_worker(self, agent, task, index):
        # 每个样本一个线程
        thread = Thread(target=task_client.run_sample, args=(index, agent))
        thread.start()

    def finish_callback(self, result):
        # 写入 runs.jsonl 或 error.jsonl
        # 所有样本完成后调用 calculate_overall()
```

### TaskClient（任务客户端代理）

不直接运行任务，而是通过 HTTP 与远程 Task Controller 通信：

```python
class TaskClient:
    def run_sample(self, index, agent):
        # 1. 启动会话
        response = POST("/start_sample", {name, index})
        session_id = response.session_id
        history = response.history  # 初始对话历史

        # 2. 交互循环
        while status == RUNNING:
            # Agent 生成回复
            agent_response = agent.inference(history)

            # 发送给 Task 服务器
            response = POST("/interact", {session_id, agent_response})

            # 更新历史和状态
            history = response.history
            status = response.status

        # 3. 返回结果
        return TaskClientOutput(result, history)
```

### AgentClient（Agent 接口）

统一的 LLM 调用抽象：

```python
class AgentClient:
    def inference(self, history: List[dict]) -> str:
        """传入对话历史，返回 Agent 的下一步回复"""
        ...

# 具体实现
class HTTPAgent(AgentClient):     # OpenAI / Claude 等 HTTP API
class FastChatClient(AgentClient) # 本地 FastChat 推理
class TestAgent(AgentClient)      # 测试用 mock
```

### Task 服务端（核心任务逻辑）

每个 Task 继承基类，实现 4 个方法：

```python
class Task:
    def get_indices(self) -> List[SampleIndex]:
        """返回所有样本索引"""

    async def start_sample(self, index, session: Session):
        """单个样本的异步评估逻辑（核心）"""

    def calculate_overall(self, results: List[TaskOutput]) -> dict:
        """汇总所有结果，计算总分"""

    def release(self):
        """清理资源"""
```

## 2.4 评估交互循环

以 DBBench（数据库任务）为例：

```
步骤 1: Assigner 调度 → 启动线程
步骤 2: TaskClient.POST("/start_sample", {task="dbbench", index=42})
步骤 3: Server 端 DBBenchTask.start_sample(42, session):
         │
         ├─ 初始化 MySQL Docker 容器，加载测试数据库
         ├─ session.inject(system_prompt)   ← "你是一个数据库助手..."
         ├─ session.inject(user_question)   ← "查询2023年销售额最高的产品"
         │
         └─ 循环 (最多 max_round 次):
              │
              ├─ response = session.action()    ← 等待 Agent 回复
              │   (TaskClient 调用 agent.inference() → POST /interact)
              │
              ├─ 解析 Agent 回复中的 function call:
              │   ├─ execute_sql("SELECT ...") → 执行 SQL，返回结果
              │   └─ commit_final_answer("产品A") → 提交最终答案
              │
              └─ 如果是 commit_final_answer:
                   比较答案与 ground truth → 返回 TaskSampleExecutionResult

步骤 4: 结果写入 runs.jsonl
步骤 5: 所有样本完成 → calculate_overall() → overall.json
```

### 时序图

```
  Assigner          TaskClient         Task Controller        AgentClient        LLM API
     │                  │                    │                     │                │
     │  start_worker    │                    │                     │                │
     │─────────────────→│                    │                     │                │
     │                  │ POST /start_sample │                     │                │
     │                  │───────────────────→│                     │                │
     │                  │  session_id + hist  │ start_sample()     │                │
     │                  │←───────────────────│                     │                │
     │                  │                    │                     │                │
     │                  │     agent.inference(history)             │                │
     │                  │────────────────────────────────────────→│                │
     │                  │                    │                     │ POST /chat     │
     │                  │                    │                     │───────────────→│
     │                  │                    │                     │  LLM response  │
     │                  │                    │                     │←───────────────│
     │                  │          agent_response                  │                │
     │                  │←────────────────────────────────────────│                │
     │                  │                    │                     │                │
     │                  │ POST /interact     │                     │                │
     │                  │───────────────────→│ 执行动作(SQL等)      │                │
     │                  │  obs + status      │                     │                │
     │                  │←───────────────────│                     │                │
     │                  │                    │                     │                │
     │                  │    ... 重复直到 done ...                   │                │
     │                  │                    │                     │                │
     │  finish_callback │                    │                     │                │
     │←─────────────────│                    │                     │                │
     │  写入 runs.jsonl  │                    │                     │                │
```

## 2.5 支持的任务

### FC 版本（函数调用范式）

| 任务 | 环境 | 评估内容 |
|------|------|---------|
| AlfWorld (AF) | 交互式家庭模拟 | 具身任务完成 |
| DBBench (DB) | MySQL Docker 容器 | SQL 查询与数据库操作 |
| KnowledgeGraph (KG) | Freebase 数据库 | 知识图谱推理 |
| OS Interaction (OS) | Linux Docker 容器 | 系统命令操作 |
| WebShop (WS) | 电商网站模拟 | 网页购物决策 |

### 资源消耗

| 任务 | 内存 | 启动时间 |
|------|------|---------|
| WebShop | ~15 GB | ~3 分钟 |
| Mind2Web | ~1 GB | ~5 分钟 |
| 其他 | < 500 MB | 5-20 秒 |

## 2.6 状态管理

系统定义了 7 种样本执行状态：

```python
class SampleStatus(Enum):
    RUNNING                  # 执行中
    COMPLETED                # 正常完成
    AGENT_CONTEXT_LIMIT      # Agent 上下文超限
    TASK_LIMIT_REACHED       # 任务步数限制
    AGENT_VALIDATION_FAILED  # Agent 输出格式错误
    TASK_ERROR               # 任务环境出错
    AGENT_ERROR              # Agent 调用失败
```

这些状态用于细粒度地分析失败原因（不只是"成功/失败"二分法）。

## 2.7 配置系统

AgentBench 的配置系统支持 YAML 组合，有三个特殊关键字：

| 关键字 | 作用 |
|--------|------|
| `import` | 从其他文件导入配置（支持深度合并）|
| `default` | 低优先级默认值 |
| `overwrite` | 高优先级覆盖值 |

**配置目录结构**：

```
configs/
├── assignments/          # 任务分配配置（哪个 Agent 评估哪个 Task）
│   └── default.yaml
├── agents/               # Agent 定义
│   └── openai-chat.yaml  # module + API key + 参数
├── tasks/                # Task 定义
│   ├── alfworld.yaml
│   ├── dbbench.yaml
│   └── ...
└── start_task.yaml       # 批量启动 Task Worker
```

**Agent 配置示例**：

```yaml
module: src.client.agents.http_agent.HTTPAgent
parameters:
  api_key: "sk-xxx"
  model: "gpt-4"
  base_url: "https://api.openai.com/v1"
  temperature: 0.0
  max_tokens: 4096
```

**Assignment 配置示例**：

```yaml
definition:
  agent: configs/agents/openai-chat.yaml
  task: configs/tasks/
concurrency:
  agent: 5      # 每个 Agent 最多 5 个并发
  task: 3       # 每个 Task 最多 3 个并发 Worker
assignments:
  - agent: gpt-4
    task: [alfworld, dbbench, webshop]
output: results/
```

## 2.8 项目结构

```
AgentBench/
├── configs/
│   ├── assignments/     # 任务分配配置
│   ├── agents/          # Agent 定义
│   ├── tasks/           # Task 定义
│   └── start_task.yaml  # 批量启动配置
├── src/
│   ├── assigner.py      # 主控调度器
│   ├── start_task.py    # Task 启动器
│   ├── configs.py       # ConfigLoader
│   ├── client/
│   │   ├── task.py      # TaskClient (HTTP 代理)
│   │   ├── agent.py     # AgentClient 基类
│   │   └── agents/      # 具体 Agent 实现
│   ├── server/
│   │   └── tasks/       # Task 服务端实现
│   │       ├── alfworld/
│   │       ├── dbbench/
│   │       ├── knowledgegraph/
│   │       ├── os_interaction/
│   │       └── webshop/
│   ├── typings/         # 类型定义
│   │   ├── general.py
│   │   ├── status.py
│   │   ├── output.py
│   │   ├── request.py
│   │   └── config.py
│   └── utils/           # 工具函数 (Max-Flow 等)
├── data/                # 任务数据和 Dockerfile
├── docs/                # 文档
├── extra/               # docker-compose 配置
└── scripts/             # 辅助脚本
```

## 2.9 安装与使用

```bash
# 环境准备
conda create -n agentbench python=3.9
conda activate agentbench
git clone https://github.com/THUDM/AgentBench.git
cd AgentBench
pip install -r requirements.txt

# 配置 Agent (设置 API Key)
vim configs/agents/openai-chat.yaml

# 启动 Task 服务器
python -m src.start_task -a

# 运行评估
python -m src.assigner
```

## 2.10 关键设计亮点

| 特性 | 说明 |
|------|------|
| **分布式架构** | Client-Server 分离，Task Worker 可独立扩缩容 |
| **Max-Flow 调度** | 将并发控制建模为图论最大流问题，最优利用资源 |
| **Docker 隔离** | 每个任务环境运行在独立容器中，互不干扰 |
| **断点续传** | 通过 `runs.jsonl` 记录进度，中断后可恢复 |
| **工厂模式** | `InstanceFactory` 根据配置动态实例化任何类 |
| **函数调用范式 (FC)** | 新版使用标准化 function calling 接口 |
| **异步任务执行** | 服务端 `start_sample` 是 async，支持高并发 |

---

# 三、τ²-Bench (Sierra Research)

> 项目地址：https://github.com/sierra-research/tau2-bench

## 3.1 项目概述

τ²-Bench 是 Sierra Research 开发的对话式 Agent 评估框架，核心创新是 **"双控制"（Dual-Control）设计**：**Agent 和 User 都可以调用工具**，模拟真实客服场景中客户和客服同时操作系统的情况。

与前两个项目的关键区别：
- AgentBoard / AgentBench → 评估 Agent 在各种通用环境中的能力
- τ²-Bench → 专注于评估**客服对话 Agent** 在策略约束下的表现

## 3.2 系统三层架构

```
┌─────────────────────────────────────────────────────┐
│                  Orchestrator (编排器)                │
│              管理三方之间的消息路由                      │
│                                                      │
│    ┌──────────┐   消息    ┌───────────┐   消息       │
│    │  Agent    │ ←──────→ │   User     │             │
│    │ (LLM客服)  │          │ (LLM模拟用户)│             │
│    └─────┬────┘          └─────┬─────┘             │
│          │ tool_call           │ tool_call          │
│          ▼                     ▼                     │
│    ┌────────────────────────────────────┐           │
│    │         Environment (环境)          │           │
│    │   ┌─────────────┐ ┌─────────────┐ │           │
│    │   │ Agent Tools  │ │ User Tools  │ │           │
│    │   │ (客服工具)    │ │ (用户工具)   │ │           │
│    │   └─────────────┘ └─────────────┘ │           │
│    │          ┌─────────────┐          │           │
│    │          │   Database   │          │           │
│    │          │  (模拟后端)   │          │           │
│    │          └─────────────┘          │           │
│    └────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

**双控制的含义**：
- **Agent** 可以调用工具（如查订单、修改航班）
- **User** 也可以调用工具（如在电信场景中查看自己的账单）
- 两者的工具调用都会修改同一个 Environment 状态

## 3.3 核心数据模型

### 消息类型

```python
SystemMessage      # role="system"，系统提示词
AssistantMessage   # role="assistant"，Agent 回复（文本 OR 工具调用，二选一）
UserMessage        # role="user"，User 回复（文本 OR 工具调用，二选一）
ToolMessage        # role="tool"，工具执行结果，带 requestor 标记谁发起的
MultiToolMessage   # 包装多个 ToolMessage（单轮多工具调用时）
```

关键约束：**每条消息要么有文本内容，要么有工具调用，不能同时存在。**

### Task 定义（最核心的数据结构）

```
Task
├── id: str
├── description: Description
│     ├── purpose: "用户要改签航班"
│     ├── relevant_policies: "改签政策条款..."
│     └── notes: 附加说明
│
├── user_scenario: UserScenario          ← 给 LLM 用户模拟器的剧本
│     ├── persona: "一个急躁的商务旅客"
│     └── instructions:
│           ├── reason_for_call: "航班被取消了"
│           ├── known_info: {booking_id: "ABC123"}
│           ├── unknown_info: ["不知道替代航班选项"]
│           └── task_instructions: "坚持要求全额退款"
│
├── initial_state: InitialState          ← 环境初始状态
│     ├── initialization_data: {agent_data: {...}, user_data: {...}}
│     ├── initialization_actions: [预执行的工具调用]
│     └── message_history: [预置的对话历史]
│
└── evaluation_criteria: EvaluationCriteria  ← 评估标准
      ├── actions: [期望的工具调用 + compare_args]
      ├── env_assertions: [环境状态断言]
      ├── communicate_info: ["必须告知用户的信息"]
      ├── nl_assertions: ["自然语言断言，由 LLM 评判"]
      └── reward_basis: [DB, ACTION, COMMUNICATE, NL_ASSERTION]
```

## 3.4 Orchestrator 交互循环（核心引擎）

### 初始化

```python
def initialize(self):
    # 1. 用 task.initial_state 设置环境（DB数据 + 预执行动作）
    environment.set_state(initialization_data, initialization_actions, message_history)

    # 2. 回放 message_history 恢复环境状态
    # 3. 初始化 Agent/User 状态
    # 4. 如果没有历史消息，发送默认问候: "Hi! How can I help you today?"
```

### 消息路由状态机

```
step() 的路由逻辑：

  ┌─────────┐   文本消息   ┌─────────┐
  │  AGENT  │ ←──────────→ │  USER   │
  └────┬────┘              └────┬────┘
       │ tool_call               │ tool_call
       ▼                         ▼
  ┌─────────┐              ┌─────────┐
  │   ENV   │──结果回 AGENT │   ENV   │──结果回 USER
  └─────────┘              └─────────┘
```

详细流程：

```python
def step(self):
    if to_role == USER:
        # Agent/Env → User: 调用 user.generate_next_message()
        message = user.generate(incoming_message, user_state)
        if "###STOP###" in message:  # 用户满意，结束
            terminate(USER_STOP)
        elif message.has_tool_calls:
            next_route = (USER, ENV)     # 用户发起工具调用
        else:
            next_route = (USER, AGENT)   # 用户发送文本给客服

    elif to_role == AGENT:
        # User/Env → Agent: 调用 agent.generate_next_message()
        message = agent.generate(incoming_message, agent_state)
        if "###STOP###" in message:
            terminate(AGENT_STOP)
        elif message.has_tool_calls:
            next_route = (AGENT, ENV)    # 客服发起工具调用
        else:
            next_route = (AGENT, USER)   # 客服回复用户

    elif to_role == ENV:
        # Agent/User → Env: 执行工具调用
        for tool_call in message.tool_calls:
            result = environment.get_response(tool_call)
            if result.error:
                num_errors += 1
        next_route = (ENV, from_role)    # 结果返回给调用者
```

### 终止条件

| 终止原因 | 触发条件 |
|---------|---------|
| `USER_STOP` | 用户发送 `###STOP###`（满意结束）|
| `AGENT_STOP` | Agent 发送 `###STOP###` |
| `MAX_STEPS` | 超过最大步数 |
| `TOO_MANY_ERRORS` | 工具调用错误次数过多 |
| `AGENT_ERROR` | Agent 产生无效输出 |
| `USER_ERROR` | User 模拟器出错 |

## 3.5 LLM-as-User 用户模拟器

τ²-Bench 的一大特色——用 LLM 模拟真实用户：

```python
class UserSimulator(BaseUser):
    def generate_next_message(self, message, user_state):
        # 1. 构建系统提示：模拟指南 + 用户场景剧本
        system_prompt = simulation_guidelines + f"<scenario>{user_scenario}</scenario>"

        # 2. 角色翻转（关键技巧！）
        #    因为 LLM 总是以 "assistant" 角色生成
        #    所以要把对话历史中的角色互换：
        flipped_history = flip_roles(history)
        #    UserMessage → AssistantMessage（LLM 视为自己的历史输出）
        #    AssistantMessage → UserMessage（LLM 视为对方的输入）

        # 3. 调用 LLM 生成用户回复
        response = llm.generate(system_prompt, flipped_history)

        # 4. 如果有工具调用，标记 requestor="user"
        return UserMessage(content=response)
```

## 3.6 Agent 架构

### 基类

```python
class BaseAgent[AgentState]:
    generate_next_message(message, state) -> (AssistantMessage, AgentState)
    get_init_state(message_history) -> AgentState
    stop(message, state) -> None
    is_stop(message) -> bool
```

### 三种具体 Agent

1. **LLMAgent** — 标准客服 Agent，系统提示含域策略文档，调用 LLM + 工具
2. **LLMGTAgent** (Ground Truth) — 测试用，注入期望的解决步骤到系统提示
3. **LLMSoloAgent** — 自主 Agent，处理工单而非对话，只能调用工具不能发文本

### Gymnasium Agent

`GymAgent` 包装外部 Agent 用于 Gymnasium 接口，通过线程同步实现 RL 训练。

## 3.7 Environment（环境）

```python
class Environment:
    tools: ToolKitBase         # Agent 可用工具
    user_tools: ToolKitBase    # User 可用工具

    def get_response(tool_call) -> ToolMessage:
        """执行工具调用，返回结果"""

    def set_state(initialization_data, initialization_actions, message_history):
        """回放整个消息历史恢复 DB 状态"""

    def run_env_assertion(assertion) -> bool:
        """执行环境断言（用于评估）"""

    def check_db(reference) -> bool:
        """比较当前 DB 哈希与参考值"""

    def sync_tools():
        """同步 Agent/User 工具间的共享状态"""
```

## 3.8 评估系统（四重评判）

评估在每次模拟结束后执行，由四个独立评估器组成，最终奖励是它们的**乘积**（任一失败则总分为 0）。

### 1. Environment Evaluator（环境状态比对）

```
Gold 路径:                          Predicted 路径:
初始状态 → 执行期望动作 → Gold DB    初始状态 → 回放实际轨迹 → Predicted DB
                  ↓                                    ↓
              gold_hash            ==?            predicted_hash
```

- 分别构建黄金环境（执行期望动作）和预测环境（回放实际轨迹）
- 比较两个环境的 DB 哈希值（包括 agent DB 和 user DB）
- 全部匹配 → 1.0，否则 → 0.0

### 2. Action Evaluator（动作匹配）

```python
# 检查所有期望动作是否出现在实际轨迹中
for gold_action in expected_actions:
    found = any(
        gold_action.compare_with_tool_call(actual_call)
        for actual_call in trajectory_tool_calls
    )
# 全部找到 → 1.0，否则 → 0.0
```

支持 `compare_args` 部分参数匹配（只比较关键参数）。

### 3. Communicate Evaluator（信息传达）

```python
# 检查 Agent 是否向用户传达了必要信息
for required_info in communicate_info:
    found = any(
        required_info.lower() in msg.content.lower()
        for msg in assistant_messages
    )
# 全部传达 → 1.0，否则 → 0.0
```

### 4. NL Assertion Evaluator（自然语言断言，LLM-as-Judge）

使用 LLM 对自然语言断言（如"Agent 没有泄露其他客户的信息"）进行判定。

### 最终分数计算

```python
final_reward = 1.0
if DB in reward_basis:          final_reward *= env_reward        # 0 or 1
if ACTION in reward_basis:      final_reward *= action_reward     # 0 or 1
if COMMUNICATE in reward_basis: final_reward *= communicate_reward # 0 or 1
if NL_ASSERTION in reward_basis: final_reward *= nl_reward        # 0 or 1
# 结果只有 0.0 或 1.0（严格的二值评估）
```

## 3.9 Pass^k 指标

τ²-Bench 使用 Pass^k 而非简单的成功率：

```python
def pass_hat_k(num_trials, success_count, k):
    # 从 n 次试验中，至少 k 次成功的概率估计
    return C(success_count, k) / C(num_trials, k)
```

- **Pass^1**：至少成功 1 次的概率（最宽松）
- **Pass^4**：4 次都成功的概率（最严格，衡量一致性）

计算流程：
1. 对每个 task 运行 `num_trials` 次（如 4 次）
2. 统计成功次数（reward >= 1.0）
3. 计算每个 k 的 pass^k
4. 对所有 task 取平均

## 3.10 支持的评估域

| 域 | 场景 | 特点 |
|----|------|------|
| **airline** | 航空公司客服 | 改签/退票/行李问题 |
| **retail** | 零售电商客服 | 退换货/订单查询 |
| **telecom** | 电信客服 | 套餐变更/账单查询，**用户也有工具** |
| **mock** | 测试用域 | 简单示例 |

可选变体：
- `telecom-workflow`：使用工作流格式的策略
- `llm_agent_solo`：无用户模式
- `llm_agent_gt`：具有预知计划的 Agent

## 3.11 完整执行流程

```
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4
│
├─ 1. 加载配置 → RunConfig
│
├─ 2. Registry 加载域任务
│     └─ 从 JSON 文件加载 Task 列表
│     └─ 按 split (train/test/base) 过滤
│
├─ 3. ThreadPoolExecutor 并发执行
│     └─ 对每个 (task, trial) 组合：
│
│         a. 构建 Environment（DB + 工具集）
│         b. 构建 Agent（LLM + 客服工具 + 策略文档）
│         c. 构建 User（LLM + 用户场景剧本 + 可选用户工具）
│         d. 构建 Orchestrator
│
│         e. orchestrator.run()
│            ├─ initialize() → 设置环境初始状态
│            └─ 循环 step():
│                 Agent ←文本→ User
│                 Agent →tool_call→ Env →result→ Agent
│                 User  →tool_call→ Env →result→ User
│                 直到终止条件
│            └─ 返回 SimulationRun（完整轨迹）
│
│         f. evaluate_simulation(task, simulation)
│            ├─ Env Evaluator: 比对 DB 哈希
│            ├─ Action Evaluator: 检查动作匹配
│            ├─ Communicate Evaluator: 检查信息传达
│            └─ NL Evaluator: LLM 判断断言
│            └─ 返回 RewardInfo
│
├─ 4. 汇总所有 SimulationRun → Results
│
├─ 5. 计算 AgentMetrics
│     ├─ avg_reward
│     ├─ pass^1, pass^2, pass^3, pass^4
│     └─ avg_agent_cost
│
└─ 6. 输出结果 + 保存到 data/tau2/simulations/
```

## 3.12 Gymnasium 接口（RL 支持）

支持两种 Gym 环境类型：

- **AgentGymEnv** — RL 策略扮演客服 Agent，LLM 扮演用户
- **UserGymEnv** — RL 策略扮演用户，LLM 扮演客服 Agent

两者通过线程同步实现：Orchestrator 在后台线程运行，Gym 的 `step()` 方法通过事件/队列交换消息。

## 3.13 项目结构

```
tau2-bench/
├── src/tau2/
│   ├── data_model/       # 核心数据模型
│   │   ├── message.py    # 消息类型定义
│   │   ├── tasks.py      # Task 定义
│   │   └── simulation.py # 模拟结果定义
│   ├── orchestrator/     # 编排器
│   │   └── orchestrator.py
│   ├── agent/            # Agent 接口与实现
│   │   ├── base.py
│   │   ├── llm_agent.py
│   │   └── gym_agent.py
│   ├── user/             # 用户模拟器
│   │   └── user_simulator.py
│   ├── environment/      # 环境
│   │   └── environment.py
│   ├── evaluator/        # 评估器
│   │   ├── evaluator.py
│   │   ├── evaluator_env.py
│   │   ├── evaluator_action.py
│   │   └── evaluator_communicate.py
│   ├── metrics/          # 指标计算
│   │   └── agent_metrics.py
│   ├── domains/          # 域定义
│   │   ├── airline/
│   │   ├── retail/
│   │   ├── telecom/
│   │   └── mock/
│   ├── gym/              # Gymnasium 接口
│   ├── registry.py       # 注册表
│   └── cli.py            # CLI 入口
├── data/tau2/
│   ├── domains/          # 域配置与任务数据
│   └── simulations/      # 评估结果输出
├── web/leaderboard/      # 排行榜 Web 应用
└── tests/
```

## 3.14 安装与使用

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 配置 API Key
cp .env.example .env
# 编辑 .env 填入 API Key

# 运行评估
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-trials 4

# 交互式试玩
tau2 play --domain airline

# 查看结果
tau2 view

# 查看域文档
tau2 domain airline
```

### CLI 命令汇总

| 命令 | 功能 |
|------|------|
| `tau2 run` | 执行 Agent 评估 |
| `tau2 play` | 交互式试玩模式 |
| `tau2 view` | 查看结果 |
| `tau2 domain <name>` | 查看域文档 |
| `tau2 check-data` | 验证数据配置 |
| `tau2 submit prepare` | 准备排行榜提交 |
| `tau2 submit validate` | 验证提交完整性 |

## 3.15 关键设计亮点

1. **双控制架构**：Agent 和 User 都能调用工具，更贴近真实客服场景
2. **LLM-as-User**：用 LLM 模拟真实用户行为（含角色翻转技巧）
3. **四重评判乘积**：DB 状态 x 动作匹配 x 信息传达 x NL 断言，严格的二值评估
4. **DB 哈希比对**：通过回放黄金动作和实际动作，比较最终状态哈希，与执行顺序无关
5. **Pass^k 指标**：不仅衡量能否成功，还衡量成功的一致性
6. **Gymnasium 接口**：原生支持 RL 训练
7. **无状态设计**：Agent/User 无内部状态，所有状态通过参数传递，支持任意点回放
8. **种子管理**：多层种子（run → trial → agent/user LLM）保证可复现性

---

# 四、三个项目对比总结

| 维度 | AgentBoard (HKUST) | AgentBench (THU) | τ²-Bench (Sierra) |
|------|-------------------|------------------|--------------------|
| **定位** | 分析型评估平台 | 分布式通用基准 | 对话客服评估 |
| **会议** | NeurIPS 2024 | ICLR 2024 | — |
| **架构** | 单进程直调 | Client-Server HTTP | Orchestrator 编排 |
| **通信** | 函数调用 | HTTP REST API | 进程内消息路由 |
| **任务领域** | 9 类通用任务 | 5-8 类通用任务 | 3 个客服域 |
| **Agent 范式** | Vanilla / ReAct | Function Calling | Function Calling |
| **用户模拟** | 无（环境直接反馈）| 无 | **LLM-as-User** |
| **双控制** | 无 | 无 | Agent + User 都能调工具 |
| **评估指标** | SR + PR + Grounding + 6维能力 | 任务特定指标 + 状态分析 | **Pass^k**（一致性度量）|
| **评估方法** | 连续奖励 (0~1) | 任务特定 | 严格二值 (0 or 1) |
| **环境隔离** | 共享进程 | Docker 容器 | 内存中 DB 对象 |
| **并发调度** | 顺序执行 | Max-Flow 最大流 | ThreadPoolExecutor |
| **断点续传** | 无 | 支持 (runs.jsonl) | 无 |
| **RL 支持** | 无 | 无 | Gymnasium 接口 |
| **可视化** | W&B 雷达图 | Leaderboard | taubench.com |

### 一句话总结

- **AgentBoard**：注重**细粒度分析型评估**——不只看成功/失败，还看进度率、接地准确度、六大能力维度雷达图
- **AgentBench**：注重**工程化分布式基础设施**——Docker 隔离、Max-Flow 调度、断点续传、HTTP 微服务架构
- **τ²-Bench**：注重**真实客服场景的双向交互评估**——LLM 扮演用户、双方都有工具、DB 哈希对比、Pass^k 一致性度量
