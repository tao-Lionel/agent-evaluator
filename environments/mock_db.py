from __future__ import annotations

import copy
import hashlib
import json
import logging
from typing import Any

from core.types import Task, StepResult, ToolCall
from core.base import Environment
from core.registry import registry

logger = logging.getLogger(__name__)


@registry.environment("mock_db")
class MockDBEnvironment(Environment):
    """In-memory mock database that supports query / update / insert / delete / done."""

    def __init__(self):
        self.db: dict[str, list[dict]] = {}

    def reset(self, task: Task) -> str:
        self.db = copy.deepcopy(task.initial_state)
        tables = list(self.db.keys())
        counts = {t: len(rows) for t, rows in self.db.items()}
        return f"Database ready. Tables: {counts}"

    def step(self, tool_call: ToolCall) -> StepResult:
        name = tool_call.name
        args = tool_call.arguments
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return StepResult(
                observation=f"Unknown tool: '{name}'. Available: query, update, insert, delete, done",
            )
        try:
            return handler(args)
        except Exception as e:
            logger.error("Tool '%s' raised: %s", name, e)
            return StepResult(observation=f"Tool error: {e}")

    def get_state_hash(self) -> str:
        serialized = json.dumps(self.db, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(serialized.encode()).hexdigest()

    def get_tool_schemas(self) -> list[dict]:
        return [
            _schema("query", "Query records from a database table.", {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "filters": {
                        "type": "object",
                        "description": "Key-value pairs to filter rows (exact match). Empty means all rows.",
                        "default": {},
                    },
                },
                "required": ["table"],
            }),
            _schema("update", "Update records in a database table that match filters.", {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "filters": {"type": "object", "description": "Match conditions"},
                    "updates": {"type": "object", "description": "Fields to update"},
                },
                "required": ["table", "filters", "updates"],
            }),
            _schema("insert", "Insert a new record into a database table.", {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "record": {"type": "object", "description": "The record to insert"},
                },
                "required": ["table", "record"],
            }),
            _schema("delete", "Delete records from a database table that match filters.", {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Table name"},
                    "filters": {"type": "object", "description": "Match conditions"},
                },
                "required": ["table", "filters"],
            }),
            _schema("done", "Call this when the task is fully completed.", {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was done",
                    },
                },
                "required": [],
            }),
        ]

    # ── Tool implementations ──

    def _tool_query(self, args: dict) -> StepResult:
        table = args["table"]
        filters = args.get("filters", {})
        rows = self.db.get(table)
        if rows is None:
            return StepResult(observation=f"Table '{table}' does not exist.")
        results = [row for row in rows if _matches(row, filters)]
        return StepResult(observation=json.dumps(results, ensure_ascii=False, indent=2))

    def _tool_update(self, args: dict) -> StepResult:
        table = args["table"]
        filters = args["filters"]
        updates = args["updates"]
        rows = self.db.get(table)
        if rows is None:
            return StepResult(observation=f"Table '{table}' does not exist.")
        count = 0
        for row in rows:
            if _matches(row, filters):
                row.update(updates)
                count += 1
        return StepResult(observation=f"Updated {count} record(s) in '{table}'.")

    def _tool_insert(self, args: dict) -> StepResult:
        table = args["table"]
        record = args["record"]
        self.db.setdefault(table, []).append(record)
        return StepResult(observation=f"Inserted 1 record into '{table}'.")

    def _tool_delete(self, args: dict) -> StepResult:
        table = args["table"]
        filters = args["filters"]
        rows = self.db.get(table)
        if rows is None:
            return StepResult(observation=f"Table '{table}' does not exist.")
        before = len(rows)
        self.db[table] = [row for row in rows if not _matches(row, filters)]
        removed = before - len(self.db[table])
        return StepResult(observation=f"Deleted {removed} record(s) from '{table}'.")

    def _tool_done(self, args: dict) -> StepResult:
        summary = args.get("summary", "Task completed.")
        return StepResult(observation=summary, done=True)


def _matches(row: dict, filters: dict) -> bool:
    return all(row.get(k) == v for k, v in filters.items())


def _schema(name: str, desc: str, params: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": params,
        },
    }
