# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agent Evaluator is a pluggable framework for evaluating AI agents on tool-calling tasks. It drives an Agent ↔ Environment interaction loop and scores the agent's performance across multiple dimensions. Currently at MVP stage (Step 1 of roadmap) with a mock database environment and OpenAI-compatible agent adapter.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluation (requires OPENAI_API_KEY env var)
python run.py                    # uses config.yaml
python run.py path/to/config.yaml

# Run tests (no API key needed — uses scripted mock agent)
python tests/test_core.py
```

## Architecture

The system has three pluggable extension points, all registered via decorator on the singleton `registry` in `core/registry.py`:

```
@registry.adapter("name")      → AgentAdapter   (core/base.py)
@registry.environment("name")  → Environment    (core/base.py)
@registry.evaluator("name")    → Evaluator      (core/base.py)
```

**Orchestrator** (`core/orchestrator.py`) is the central loop engine. It only depends on the three abstract interfaces above — never on concrete implementations. The loop: reset agent+env → feed system prompt + user message → agent acts → env steps tool calls → repeat until done/max_steps → run all evaluators → return `EvalResult`.

**Evaluation scoring uses multiplicative combination** (not weighted average): `overall = state_match × action_match × info_delivery`. Any single 0 fails the entire task. This is a deliberate design choice from τ²-Bench.

**State evaluation** works by replaying expected actions on a fresh environment copy, then comparing MD5 hashes of the two DB states. This makes evaluation order-independent and fully deterministic.

### Current implementations

- **Adapter**: `openai_fc` — any OpenAI-compatible API with function calling
- **Environment**: `mock_db` — in-memory CRUD database (query/update/insert/delete/done tools)
- **Evaluators**: `state_match` (DB hash comparison), `action_match` (partial arg matching), `info_delivery` (substring check in agent replies)

## Adding New Components

To add a new adapter/environment/evaluator: create a file in the corresponding directory, use the `@registry.{type}("name")` decorator, and import it in the package's `__init__.py`. The `core/` directory should require zero modifications.

## Configuration

`config.yaml` supports `${ENV_VAR}` expansion. Key sections: `agent` (adapter, model, api_key), `environment` (type), `evaluators` (list of names), `scenarios` (path to task JSON), `run` (num_trials, log_level).

## Task Scenario Format

Tasks are defined in JSON arrays (`scenarios/sample_tasks.json`). Each task has: `id`, `description`, `initial_message`, `initial_state` (DB tables), `expected_actions` (with `match_args` for partial matching), `expected_state`, `required_info`, `difficulty`, `max_steps`.
