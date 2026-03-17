# 可扩展性分析：如何接入新类型的 Agent

> 分析了本项目的扩展机制，以"浏览器自动化 Agent"为例走通完整流程。

---

## 一、架构分层：不变的 vs 变化的

```
不变的部分（框架核心）           变化的部分（插件扩展）
─────────────────────        ─────────────────────
Orchestrator (交互编排)       Agent Adapter (新适配器)
Registry (注册表)             Environment (新环境)
Evaluator 接口                Evaluator (新评估维度)
Metrics 计算                  Task/Scenario (新测试用例)
报告输出
```

## 二、扩展检查清单

新增一种 Agent 类型需要做的事情：

```
必须做：
  [1 文件] adapters/xxx.py        — 实现 AgentAdapter 接口，注册到 Registry

通常要做：
  [1 文件] environments/xxx.py    — 实现 Environment 接口（如果需要新环境）
  [1 文件] scenarios/xxx.json     — 编写测试用例

可选：
  [1 文件] evaluators/xxx.py      — 新的评估维度（如果有独特需求）

不需要动的：
  core/*                          — Orchestrator、Registry、类型定义
  其他已有的 adapters/environments/evaluators
  run.py
```

## 三、示例：接入浏览器自动化 Agent

### Step 1: 写 Adapter

```python
# adapters/browser_agent.py

@registry.adapter("browser_agent")
class BrowserAgentAdapter(AgentAdapter):
    def __init__(self, agent_endpoint: str, **kwargs):
        self.endpoint = agent_endpoint

    def reset(self):
        requests.post(f"{self.endpoint}/reset")

    def act(self, messages: list[Message]) -> Message:
        payload = self._to_browser_format(messages)
        response = requests.post(f"{self.endpoint}/act", json=payload)
        return self._from_browser_format(response.json())

    @property
    def capabilities(self) -> set[str]:
        return {"browse", "click", "type", "screenshot", "navigate"}
```

### Step 2: 写 Environment

```python
# environments/browser_env.py

@registry.environment("browser")
class BrowserEnvironment(Environment):
    def __init__(self, headless: bool = True):
        self.browser = playwright.chromium.launch(headless=headless)

    def reset(self, task: Task) -> str:
        self.page = self.browser.new_page()
        self.page.goto(task.initial_state["url"])
        return self._get_observation()

    def step(self, tool_call: ToolCall) -> StepResult:
        if tool_call.name == "click":
            self.page.click(tool_call.arguments["selector"])
        elif tool_call.name == "type":
            self.page.fill(tool_call.arguments["selector"], tool_call.arguments["text"])
        elif tool_call.name == "navigate":
            self.page.goto(tool_call.arguments["url"])
        return StepResult(observation=self._get_observation(), done=self._check_done())

    def get_state_hash(self) -> str:
        return hashlib.md5((self.page.url + self.page.content()).encode()).hexdigest()
```

### Step 3: 写测试场景

```json
{
  "id": "browser-001",
  "description": "Search for a product on e-commerce site",
  "initial_message": "Find the cheapest wireless mouse",
  "initial_state": {"url": "https://test-shop.example.com"},
  "expected_actions": [
    {"name": "type", "match_args": {"selector": "#search"}},
    {"name": "click", "match_args": {"selector": ".product-card"}}
  ]
}
```

### Step 4（可选）: 新评估维度

```python
# evaluators/visual_accuracy.py

@registry.evaluator("visual_accuracy")
class VisualAccuracyEvaluator(Evaluator):
    def evaluate(self, task, trajectory, env) -> float:
        correct_clicks = sum(1 for step in trajectory if self._is_correct_target(step))
        total_clicks = sum(1 for step in trajectory if step.tool_calls and step.tool_calls[0].name == "click")
        return correct_clicks / max(total_clicks, 1)
```

## 四、能力声明机制

通过 `capabilities` 属性，框架可以自动匹配适用的测试场景：

```python
# 浏览器 Agent 声明能力
capabilities = {"browse", "click", "type"}

# 框架自动过滤
compatible_tasks = [t for t in all_tasks if t.required_capabilities <= agent.capabilities]
```

## 五、潜在的扩展方向

| Agent 类型 | Adapter | Environment | 独特评估维度 |
|-----------|---------|-------------|-------------|
| 浏览器自动化 | HTTP/WebSocket | Playwright 浏览器 | 视觉准确度、页面状态 |
| 代码生成 | API 调用 | Docker Sandbox | 测试通过率、代码质量 |
| 数据分析 | Function Call | DB + DataFrame | 数据正确性、SQL 质量 |
| 客服对话 | Chat API | 模拟后端 | 信息传达、策略合规 |
| 操作系统 | SSH/Docker | Linux 容器 | 命令正确性、系统状态 |
| 多模态 | Multimodal API | 图文环境 | 视觉理解准确度 |
| 工作流编排 | Webhook | 多服务组合 | 流程完成度、API 调用序列 |

每种只需 **1 个 Adapter + 1 个 Environment + 1 组场景**，框架核心完全不用改。
