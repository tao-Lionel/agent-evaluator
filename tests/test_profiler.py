"""Tests for performance profiler."""
from __future__ import annotations

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.profiler import Profiler


def test_basic_timing():
    p = Profiler()
    with p.phase("setup"):
        time.sleep(0.05)
    with p.phase("work"):
        time.sleep(0.05)
    summary = p.summary()
    assert "setup" in summary
    assert "work" in summary
    assert summary["setup"] >= 0.04
    assert summary["work"] >= 0.04
    assert summary["total"] >= 0.08
    print("  test_basic_timing PASSED")


def test_accumulate_same_phase():
    p = Profiler()
    with p.phase("eval"):
        time.sleep(0.03)
    with p.phase("eval"):
        time.sleep(0.03)
    summary = p.summary()
    assert summary["eval"] >= 0.05
    assert summary["eval_count"] == 2
    print("  test_accumulate_same_phase PASSED")


def test_nested_phases():
    p = Profiler()
    with p.phase("outer"):
        time.sleep(0.02)
        with p.phase("inner"):
            time.sleep(0.02)
    summary = p.summary()
    assert "outer" in summary
    assert "inner" in summary
    print("  test_nested_phases PASSED")


def test_format_report():
    p = Profiler()
    with p.phase("agent_act"):
        time.sleep(0.02)
    with p.phase("env_step"):
        time.sleep(0.01)
    report = p.format_report()
    assert "agent_act" in report
    assert "env_step" in report
    print("  test_format_report PASSED")


if __name__ == "__main__":
    print("\n=== Profiler Tests ===\n")
    test_basic_timing()
    test_accumulate_same_phase()
    test_nested_phases()
    test_format_report()
    print("\n=== All profiler tests passed ===\n")
