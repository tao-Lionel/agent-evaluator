"""Auto-generate evaluation config and scenarios for any HTTP Agent.

Two input modes:
  1. OpenAPI schema (auto-detected from URL)
  2. Request/response example (user-provided)

Core logic is framework-agnostic — CLI and eval_bot are just entry points.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
import yaml

logger = logging.getLogger(__name__)

# ── OpenAPI auto-detection ──────────────────────────────────────────────────

OPENAPI_PATHS = ["/openapi.json", "/swagger.json", "/api/openapi.json"]


def try_fetch_openapi(base_url: str, timeout: float = 10.0) -> dict | None:
    """Try to fetch OpenAPI schema from common endpoints."""
    # Strip trailing path to get base URL
    # e.g. "http://localhost:8000/api/generate" → "http://localhost:8000"
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    client = httpx.Client(timeout=timeout)
    for path in OPENAPI_PATHS:
        try:
            resp = client.get(base + path)
            if resp.status_code == 200:
                data = resp.json()
                if "paths" in data or "openapi" in data:
                    logger.info("Found OpenAPI schema at %s%s", base, path)
                    return data
        except Exception:
            continue
    return None


def parse_openapi_endpoint(schema: dict, target_url: str) -> dict | None:
    """Extract request/response schema for the target endpoint from OpenAPI spec.

    Returns:
        {"request_fields": {name: {type, required, default}}, "response_schema": {...}}
    """
    from urllib.parse import urlparse
    parsed = urlparse(target_url)
    target_path = parsed.path

    paths = schema.get("paths", {})

    # Try exact match and common variations
    candidates = [target_path, target_path.rstrip("/")]
    # Strip /api prefix if present in URL but not in schema
    if target_path.startswith("/api"):
        candidates.append(target_path[4:])

    path_schema = None
    for candidate in candidates:
        if candidate in paths:
            path_schema = paths[candidate]
            break

    if not path_schema:
        return None

    post_schema = path_schema.get("post", {})
    if not post_schema:
        return None

    # Extract request body schema
    request_fields = {}
    req_body = post_schema.get("requestBody", {})
    content = req_body.get("content", {})
    json_schema = content.get("application/json", {}).get("schema", {})

    # Resolve $ref if present
    json_schema = _resolve_ref(json_schema, schema)

    properties = json_schema.get("properties", {})
    required_fields = set(json_schema.get("required", []))

    for name, prop in properties.items():
        prop = _resolve_ref(prop, schema)
        field_info: dict[str, Any] = {
            "type": prop.get("type", "string"),
            "required": name in required_fields,
        }
        if "default" in prop:
            field_info["default"] = prop["default"]
        if "description" in prop:
            field_info["description"] = prop["description"]
        if "enum" in prop:
            field_info["enum"] = prop["enum"]
        request_fields[name] = field_info

    # Extract response schema
    responses = post_schema.get("responses", {})
    resp_200 = responses.get("200", {})
    resp_content = resp_200.get("content", {})
    resp_schema = resp_content.get("application/json", {}).get("schema", {})
    resp_schema = _resolve_ref(resp_schema, schema)

    return {
        "request_fields": request_fields,
        "response_schema": resp_schema,
        "description": post_schema.get("summary", "") or post_schema.get("description", ""),
    }


def _resolve_ref(schema: dict, root: dict) -> dict:
    """Resolve a $ref pointer in OpenAPI schema."""
    ref = schema.get("$ref")
    if not ref:
        return schema
    # "#/components/schemas/GenerateRequest" → ["components", "schemas", "GenerateRequest"]
    parts = ref.lstrip("#/").split("/")
    current = root
    for part in parts:
        current = current.get(part, {})
    return current if isinstance(current, dict) else schema


# ── Example-based schema inference ──────────────────────────────────────────

def infer_schema_from_example(
    request_example: dict,
    response_example: dict,
) -> dict:
    """Infer request/response schema from a request/response example pair."""
    request_fields = {}
    for name, value in request_example.items():
        field_info: dict[str, Any] = {
            "type": _infer_type(value),
            "required": True,
            "example": value,
        }
        request_fields[name] = field_info

    return {
        "request_fields": request_fields,
        "response_example": response_example,
    }


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


# ── Config generation ───────────────────────────────────────────────────────

def generate_config(
    url: str,
    schema_info: dict,
    agent_description: str = "",
) -> str:
    """Generate a YAML config string from schema info."""
    request_fields = schema_info.get("request_fields", {})

    # Build request_template
    request_template: dict[str, Any] = {}
    for name, info in request_fields.items():
        field_type = info.get("type", "string")
        has_default = "default" in info

        # Heuristic: which field should be ${initial_message}?
        if _is_primary_input_field(name, info):
            request_template[name] = "${initial_message}"
        elif _is_description_field(name, info):
            request_template[name] = "${description}"
        elif has_default:
            request_template[name] = info["default"]
        elif field_type == "integer":
            example = info.get("example", 10)
            request_template[name] = example
        elif field_type == "number":
            request_template[name] = info.get("example", 1.0)
        elif field_type == "boolean":
            request_template[name] = info.get("example", False)
        else:
            request_template[name] = info.get("example", "")

    # Determine reply_field
    reply_field = "."

    # Determine evaluators
    evaluators = ["llm_judge", "nl_assertion"]

    # Determine timeout based on agent type
    timeout = 120
    desc_lower = agent_description.lower()
    if any(kw in desc_lower for kw in ["视频", "video", "渲染", "render"]):
        timeout = 600
    elif any(kw in desc_lower for kw in ["图片", "image", "图像", "画"]):
        timeout = 300

    config = {
        "agent": {
            "adapter": "http_bot",
            "bot_url": url,
            "request_template": request_template,
            "reply_field": reply_field,
            "timeout": timeout,
            "max_retries": 2,
            "retry_delay": 3.0,
        },
        "environment": {"type": "passthrough"},
        "evaluators": evaluators,
        "scenarios": "",  # placeholder, will be set by caller
        "run": {
            "num_trials": 1,
            "log_level": "INFO",
        },
    }

    return yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)


def _is_primary_input_field(name: str, info: dict) -> bool:
    """Heuristic: is this the main user input field?"""
    primary_names = {
        "topic", "prompt", "query", "question", "text", "message",
        "input", "content", "instruction", "request",
    }
    return name.lower() in primary_names and info.get("type", "string") == "string"


def _is_description_field(name: str, info: dict) -> bool:
    """Heuristic: is this a secondary description/requirements field?"""
    desc_names = {"requirements", "description", "context", "system_prompt", "instructions"}
    return name.lower() in desc_names and info.get("type", "string") == "string"


# ── Scenario generation (LLM-powered) ──────────────────────────────────────

SCENARIO_GEN_PROMPT = """\
你是一个 AI Agent 评估专家。根据以下 Agent 接口信息，生成 6 个多样化的评估场景。

## Agent 信息
- URL: {url}
- 功能描述: {description}
- 请求字段: {request_fields}
- 响应结构: {response_info}

## 要求

生成 6 个场景，覆盖：
1. 基础功能 (easy) — 最简单的正常请求
2. 标准用例 (easy) — 典型使用场景
3. 复杂需求 (medium) — 包含详细要求
4. 边界情况 (medium) — 特殊输入（很短/很长/特殊字符）
5. 专业领域 (hard) — 专业性强的请求
6. 压力测试 (hard) — 模糊/困难的请求

每个场景的 nl_assertions 应该检查：
- 响应结构是否正确（必有字段是否存在）
- 内容是否与请求主题相关
- 质量是否达标（不为空、不是占位符）

## 输出格式

输出纯 JSON 数组，不要 markdown 代码块。每个元素格式：
{{
  "id": "场景ID",
  "description": "场景描述（作为 Agent 的额外要求传入）",
  "difficulty": "easy|medium|hard",
  "single_turn": true,
  "initial_message": "用户输入（映射到主要请求字段）",
  "initial_state": {{}},
  "max_steps": 1,
  "required_info": [],
  "nl_assertions": ["断言1", "断言2", "断言3"]
}}"""


def generate_scenarios_prompt(
    url: str,
    schema_info: dict,
    agent_description: str,
) -> str:
    """Build the LLM prompt for scenario generation."""
    request_fields = schema_info.get("request_fields", {})
    fields_desc = json.dumps(request_fields, ensure_ascii=False, indent=2)

    # Response info
    response_info = ""
    if "response_schema" in schema_info:
        resp = schema_info["response_schema"]
        props = resp.get("properties", {})
        if props:
            response_info = json.dumps(
                {k: v.get("type", "unknown") for k, v in props.items()},
                ensure_ascii=False,
            )
    if not response_info and "response_example" in schema_info:
        response_info = json.dumps(schema_info["response_example"], ensure_ascii=False)
    if not response_info:
        response_info = "(未知)"

    return SCENARIO_GEN_PROMPT.format(
        url=url,
        description=agent_description or "(用户未提供描述)",
        request_fields=fields_desc,
        response_info=response_info,
    )


def generate_scenarios_with_llm(
    url: str,
    schema_info: dict,
    agent_description: str,
    model: str = "glm-4-flash",
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[dict]:
    """Call LLM to generate evaluation scenarios."""
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or os.getenv("ZHIPU_API_KEY", ""),
        base_url=base_url or "https://open.bigmodel.cn/api/paas/v4",
    )

    prompt = generate_scenarios_prompt(url, schema_info, agent_description)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )

    text = response.choices[0].message.content or ""
    return _parse_scenarios_json(text)


def _parse_scenarios_json(text: str) -> list[dict]:
    """Parse JSON array from LLM response, handling markdown code blocks."""
    # Strip markdown code block if present
    text = text.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    # Try parsing directly
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try finding JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    logger.error("Failed to parse scenarios from LLM response:\n%s", text[:500])
    return []


# ── High-level orchestration ────────────────────────────────────────────────

def auto_generate(
    url: str,
    request_example: dict | None = None,
    response_example: dict | None = None,
    agent_description: str = "",
    output_prefix: str | None = None,
    use_llm: bool = True,
    model: str = "glm-4-flash",
) -> tuple[str, str, list[dict]]:
    """Auto-generate config and scenarios.

    Args:
        url: Agent HTTP endpoint URL
        request_example: sample request body (if no OpenAPI)
        response_example: sample response body (if no OpenAPI)
        agent_description: one-line description of what the agent does
        output_prefix: prefix for output file names (e.g. "ppt_agent")
        use_llm: whether to use LLM for scenario generation
        model: LLM model for scenario generation

    Returns:
        (config_yaml, scenarios_json, scenarios_list)
    """
    # Step 1: Get schema info
    schema_info = None

    # Try OpenAPI first
    openapi = try_fetch_openapi(url)
    if openapi:
        schema_info = parse_openapi_endpoint(openapi, url)
        if schema_info:
            logger.info("Auto-detected API schema from OpenAPI")
            if not agent_description:
                agent_description = schema_info.get("description", "")

    # Fall back to example
    if not schema_info:
        if not request_example or not response_example:
            raise ValueError(
                "Cannot auto-detect API schema. "
                "Please provide request_example and response_example."
            )
        schema_info = infer_schema_from_example(request_example, response_example)
        logger.info("Inferred schema from request/response example")

    # Step 2: Determine output prefix
    if not output_prefix:
        # Derive from URL path: /api/generate → generate
        from urllib.parse import urlparse
        path = urlparse(url).path.rstrip("/").split("/")[-1]
        output_prefix = path or "agent"

    scenarios_path = f"scenarios/{output_prefix}_tasks.json"

    # Step 3: Generate config
    config_yaml = generate_config(url, schema_info, agent_description)
    # Set the scenarios path in config
    config_yaml = config_yaml.replace("scenarios: ''", f"scenarios: {scenarios_path}")

    # Step 4: Generate scenarios
    scenarios = []
    if use_llm:
        scenarios = generate_scenarios_with_llm(
            url, schema_info, agent_description, model=model,
        )

    if not scenarios:
        # Fallback: generate minimal scenarios without LLM
        scenarios = _generate_fallback_scenarios(url, schema_info, agent_description)

    scenarios_json = json.dumps(scenarios, ensure_ascii=False, indent=2)

    return config_yaml, scenarios_json, scenarios


def _generate_fallback_scenarios(
    url: str,
    schema_info: dict,
    agent_description: str,
) -> list[dict]:
    """Generate basic scenarios without LLM (fallback)."""
    desc = agent_description or "测试"
    return [
        {
            "id": f"auto-basic-001",
            "description": f"基础功能测试",
            "difficulty": "easy",
            "single_turn": True,
            "initial_message": f"请完成一个简单的{desc}任务",
            "initial_state": {},
            "max_steps": 1,
            "required_info": [],
            "nl_assertions": [
                "响应应为有效的 JSON 格式",
                "响应状态应为成功（status 字段为 completed 或 success）",
            ],
        },
        {
            "id": f"auto-standard-002",
            "description": f"标准用例测试",
            "difficulty": "medium",
            "single_turn": True,
            "initial_message": f"请完成一个标准的{desc}任务，要求高质量输出",
            "initial_state": {},
            "max_steps": 1,
            "required_info": [],
            "nl_assertions": [
                "响应应为有效的 JSON 格式",
                "响应内容不应为空或占位符",
            ],
        },
    ]
