"""Tests for eval_generator: schema parsing, config generation, scenario generation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.eval_generator import (
    infer_schema_from_example,
    generate_config,
    generate_scenarios_prompt,
    _parse_scenarios_json,
    _is_primary_input_field,
    _is_description_field,
    parse_openapi_endpoint,
    _generate_fallback_scenarios,
    auto_generate,
)


def test_infer_schema_from_example():
    """Infer field types from a request/response example."""
    req = {"topic": "AI trends", "page_count": 10, "output_format": "web"}
    resp = {"slides": [{"page": 1}], "status": "completed"}

    schema = infer_schema_from_example(req, resp)

    fields = schema["request_fields"]
    assert fields["topic"]["type"] == "string"
    assert fields["page_count"]["type"] == "integer"
    assert fields["output_format"]["type"] == "string"
    assert fields["topic"]["example"] == "AI trends"
    assert schema["response_example"] == resp
    print("  test_infer_schema_from_example: PASSED")


def test_primary_input_detection():
    """Heuristic correctly identifies primary input fields."""
    assert _is_primary_input_field("topic", {"type": "string"})
    assert _is_primary_input_field("prompt", {"type": "string"})
    assert _is_primary_input_field("query", {"type": "string"})
    assert _is_primary_input_field("text", {"type": "string"})
    assert not _is_primary_input_field("page_count", {"type": "integer"})
    assert not _is_primary_input_field("topic", {"type": "integer"})
    assert not _is_primary_input_field("theme", {"type": "string"})
    print("  test_primary_input_detection: PASSED")


def test_description_field_detection():
    """Heuristic correctly identifies description/requirements fields."""
    assert _is_description_field("requirements", {"type": "string"})
    assert _is_description_field("description", {"type": "string"})
    assert _is_description_field("context", {"type": "string"})
    assert not _is_description_field("topic", {"type": "string"})
    assert not _is_description_field("requirements", {"type": "integer"})
    print("  test_description_field_detection: PASSED")


def test_generate_config_from_example():
    """Config generation maps fields correctly."""
    schema = infer_schema_from_example(
        {"topic": "test", "requirements": "formal", "page_count": 10},
        {"slides": [], "status": "ok"},
    )
    config_yaml = generate_config(
        "http://localhost:8000/api/generate",
        schema,
        "生成PPT",
    )

    assert "http://localhost:8000/api/generate" in config_yaml
    assert "${initial_message}" in config_yaml  # topic mapped
    assert "${description}" in config_yaml      # requirements mapped
    assert "passthrough" in config_yaml
    assert "llm_judge" in config_yaml
    assert "nl_assertion" in config_yaml
    print("  test_generate_config_from_example: PASSED")


def test_generate_config_timeout_heuristic():
    """Timeout adjusts based on agent description."""
    schema = infer_schema_from_example({"prompt": "x"}, {"url": "..."})

    config_video = generate_config("http://x/gen", schema, "生成视频")
    assert "600" in config_video

    config_image = generate_config("http://x/gen", schema, "生成图片")
    assert "300" in config_image

    config_text = generate_config("http://x/gen", schema, "文本摘要")
    assert "120" in config_text
    print("  test_generate_config_timeout_heuristic: PASSED")


def test_parse_scenarios_json_clean():
    """Parse a clean JSON array."""
    text = '[{"id": "t1", "description": "test"}]'
    result = _parse_scenarios_json(text)
    assert len(result) == 1
    assert result[0]["id"] == "t1"
    print("  test_parse_scenarios_json_clean: PASSED")


def test_parse_scenarios_json_markdown():
    """Parse JSON wrapped in markdown code block."""
    text = '```json\n[{"id": "t1"}]\n```'
    result = _parse_scenarios_json(text)
    assert len(result) == 1
    print("  test_parse_scenarios_json_markdown: PASSED")


def test_parse_scenarios_json_with_text():
    """Parse JSON embedded in surrounding text."""
    text = 'Here are the scenarios:\n[{"id": "t1"}]\nDone!'
    result = _parse_scenarios_json(text)
    assert len(result) == 1
    print("  test_parse_scenarios_json_with_text: PASSED")


def test_parse_scenarios_json_invalid():
    """Invalid JSON returns empty list."""
    result = _parse_scenarios_json("not json at all")
    assert result == []
    print("  test_parse_scenarios_json_invalid: PASSED")


def test_generate_scenarios_prompt():
    """Prompt contains all key information."""
    schema = infer_schema_from_example(
        {"topic": "test", "page_count": 10},
        {"slides": [{"page": 1}]},
    )
    prompt = generate_scenarios_prompt(
        "http://localhost:8000/api/generate",
        schema,
        "PPT生成",
    )
    assert "localhost:8000" in prompt
    assert "PPT生成" in prompt
    assert "topic" in prompt
    assert "page_count" in prompt
    assert "slides" in prompt
    assert "nl_assertions" in prompt
    print("  test_generate_scenarios_prompt: PASSED")


def test_fallback_scenarios():
    """Fallback scenarios are valid without LLM."""
    schema = infer_schema_from_example({"prompt": "x"}, {"result": "y"})
    scenarios = _generate_fallback_scenarios("http://x", schema, "测试Agent")
    assert len(scenarios) >= 2
    for s in scenarios:
        assert "id" in s
        assert "nl_assertions" in s
        assert "initial_message" in s
        assert s["single_turn"] is True
    print("  test_fallback_scenarios: PASSED")


def test_parse_openapi_endpoint():
    """Parse request fields from a minimal OpenAPI schema."""
    openapi = {
        "openapi": "3.0.0",
        "paths": {
            "/api/generate": {
                "post": {
                    "summary": "Generate slides",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/GenerateRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/GenerateResponse"}
                                }
                            }
                        }
                    },
                }
            }
        },
        "components": {
            "schemas": {
                "GenerateRequest": {
                    "type": "object",
                    "required": ["topic"],
                    "properties": {
                        "topic": {"type": "string"},
                        "requirements": {"type": "string", "default": ""},
                        "page_count": {"type": "integer", "default": 10},
                        "output_format": {"type": "string", "default": "web", "enum": ["web", "pptx"]},
                    },
                },
                "GenerateResponse": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "status": {"type": "string"},
                        "slides": {"type": "array"},
                    },
                },
            }
        },
    }

    result = parse_openapi_endpoint(openapi, "http://localhost:8000/api/generate")
    assert result is not None
    fields = result["request_fields"]
    assert "topic" in fields
    assert fields["topic"]["required"] is True
    assert fields["page_count"]["default"] == 10
    assert fields["output_format"]["enum"] == ["web", "pptx"]
    assert result["description"] == "Generate slides"
    print("  test_parse_openapi_endpoint: PASSED")


def test_auto_generate_from_example():
    """End-to-end: auto_generate with example, no LLM."""
    config_yaml, scenarios_json, scenarios = auto_generate(
        url="http://localhost:8000/api/generate",
        request_example={"topic": "AI趋势", "page_count": 10, "requirements": "正式风格"},
        response_example={"slides": [{"page": 1, "layout": "cover"}], "status": "completed"},
        agent_description="根据主题自动生成PPT",
        use_llm=False,
    )

    assert "http://localhost:8000/api/generate" in config_yaml
    assert "${initial_message}" in config_yaml
    assert "generate_tasks.json" in config_yaml

    data = json.loads(scenarios_json)
    assert len(data) >= 2
    for s in data:
        assert "id" in s
        assert "nl_assertions" in s
    print("  test_auto_generate_from_example: PASSED")


if __name__ == "__main__":
    print("\n=== Eval Generator Tests ===\n")
    test_infer_schema_from_example()
    test_primary_input_detection()
    test_description_field_detection()
    test_generate_config_from_example()
    test_generate_config_timeout_heuristic()
    test_parse_scenarios_json_clean()
    test_parse_scenarios_json_markdown()
    test_parse_scenarios_json_with_text()
    test_parse_scenarios_json_invalid()
    test_generate_scenarios_prompt()
    test_fallback_scenarios()
    test_parse_openapi_endpoint()
    test_auto_generate_from_example()
    print("\n=== All eval generator tests passed ===\n")
