"""Tests for negative test scenarios — verify they load and validate correctly."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.types import Task


SCENARIOS_PATH = Path(__file__).resolve().parent.parent / "scenarios" / "negative_tasks.json"


def test_load_negative_scenarios():
    """All negative scenarios should load as valid Task objects."""
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 5, f"Expected 5 negative scenarios, got {len(data)}"

    tasks = [Task.from_dict(item) for item in data]
    for task in tasks:
        assert task.id.startswith("neg-"), f"Negative task id should start with 'neg-': {task.id}"
        assert len(task.nl_assertions) >= 2, f"Each negative task should have >= 2 nl_assertions: {task.id}"
        assert task.expected_state, f"Negative task should have expected_state: {task.id}"

    print(f"  Loaded {len(tasks)} negative scenarios")


def test_negative_scenarios_preserve_state():
    """Negative scenarios should expect the DB state to remain unchanged."""
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        initial = item["initial_state"]
        expected = item["expected_state"]
        for table, rows in expected.items():
            assert table in initial, f"Expected table '{table}' missing from initial_state in {item['id']}"
        # For most negative scenarios, state should be identical
        task_id = item["id"]
        if task_id in ("neg-unauthorized-001", "neg-destructive-003", "neg-insufficient-info-002"):
            assert initial == expected, f"State should be unchanged for {task_id}"

    print("  State preservation verified for negative scenarios")


if __name__ == "__main__":
    print("\n=== Negative Scenarios Tests ===\n")
    test_load_negative_scenarios()
    print("  PASSED")
    test_negative_scenarios_preserve_state()
    print("  PASSED")
    print("\n=== All negative tests passed ===\n")
