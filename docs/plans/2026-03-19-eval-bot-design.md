# eval_bot 设计方案 — 飞书评测 Agent（少量智能编排）

## 目标

在保留现有纯脚本评测流程的前提下，新增一个飞书 Bot 模块 `eval_bot/`，用户通过自然语言在飞书中发起评测、查询结果、生成场景。LLM 负责意图理解和参数生成，评测流程复用现有框架。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 模块位置 | 独立 `eval_bot/`，与 `test_bot/` 分离 | 职责不同，避免混淆 |
| 优先功能 | 快速评测、结果查询、场景生成 | 用户核心需求 |
| 编排层 LLM | 智谱 glm-4 + function calling | 复用现有依赖 |
| 评测执行 | 异步（线程池 + 主动推送） | 飞书 webhook 有超时限制 |
| 结果展示 | 文本摘要 + HTML 报告路径 | 复用现有 report.py |

## 架构

```
飞书用户
  | 自然语言消息
eval_bot/feishu.py (Webhook 接收)
  |
eval_bot/dispatcher.py (LLM 意图识别 + function calling)
  | 路由到具体能力
  |-- eval_bot/commands/quick_eval.py    -> 调用 orchestrator 跑评测
  |-- eval_bot/commands/query_results.py -> 查历史结果
  +-- eval_bot/commands/gen_scenarios.py -> LLM 生成测试场景
  |
  结果推回飞书（文本摘要 + HTML 报告路径）
```

## Dispatcher 意图识别

用智谱 function calling 定义三个 tool：

- `quick_eval(bot_url, domain?, eval_modes?)` — 对指定 HTTP Bot 进行快速评测
- `query_results(query)` — 查询历史评测结果
- `gen_scenarios(domain, count?, difficulty?)` — 生成测试场景

用户消息不匹配任何 tool 时，直接返回 LLM 文本回复（闲聊兜底）。

## Command 实现逻辑

### 1. quick_eval — 快速评测

1. 立即回复飞书："评测已开始，目标: xxx"
2. 异步线程中动态生成 config dict：
   - agent.adapter = "http_bot", agent.url = bot_url
   - environment.type = "passthrough"
   - evaluators = eval_modes or ["info_delivery", "llm_judge"]
   - scenarios = 根据 domain 选已有场景 or 默认 http_bot_tasks.json
3. 复用 run.py 核心逻辑：load_tasks -> 构建组件 -> orchestrator.run()
4. 生成 HTML 报告（复用 report.py）
5. 主动推送飞书：文本摘要 + 报告路径

### 2. query_results — 结果查询

1. 扫描 results/ 目录下的 JSON 文件
2. 将结果摘要 + 用户 query 发给 LLM
3. LLM 生成自然语言回答
4. 同步回复飞书

### 3. gen_scenarios — 场景生成

1. 用 LLM 生成符合项目 Task 格式的场景 JSON（现有场景作为 few-shot）
2. 保存到 scenarios/generated_xxx.json
3. 回复飞书确认，用户可接续"跑一下"触发 quick_eval

## 文件结构

```
eval_bot/
  __init__.py
  feishu.py            # 飞书 Webhook 接收 + 消息推送
  dispatcher.py        # LLM 意图识别（function calling）
  runner.py            # 异步任务执行器（线程池 + 结果回调）
  commands/
    __init__.py
    quick_eval.py      # 快速评测
    query_results.py   # 结果查询
    gen_scenarios.py   # 场景生成
  config.yaml          # eval_bot 自身配置
```

## 与现有代码的关系

- `eval_bot` import `core/`, `adapters/`, `environments/`, `evaluators/`, `users/`, `report.py` — 复用全部现有组件
- `run.py` 不改动 — 纯脚本模式保持原样
- `test_bot/` 不改动 — 继续作为被测 Bot

## 新增依赖

无。现有 requirements.txt 已包含 fastapi, httpx, openai, pyyaml。

## 启动方式

```bash
# 现有脚本评测（不变）
python run.py config.yaml

# 启动评测 Agent Bot
uvicorn eval_bot.feishu:app --port 8102
```
