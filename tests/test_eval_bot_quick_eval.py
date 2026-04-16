import unittest
from unittest.mock import MagicMock, patch

from eval_bot.commands.quick_eval import ANALYSIS_SYSTEM_PROMPT, build_eval_config, run_quick_eval


class FakeEnv:
    def get_tool_schemas(self):
        return []


class FakeEvaluator:
    def __init__(self, **kwargs):
        pass


class FakeAdapter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        FakeAdapter.instances.append(self)

    def close(self):
        self.closed = True


class TestBuildEvalConfig(unittest.TestCase):
    def test_default_config(self):
        config = build_eval_config("http://localhost:8000/chat")
        self.assertEqual(config["agent"]["adapter"], "http_bot")
        self.assertEqual(config["agent"]["bot_url"], "http://localhost:8000/chat")
        self.assertEqual(config["environment"]["type"], "passthrough")
        self.assertIn("info_delivery", config["evaluators"])
        self.assertIn("llm_judge", config["evaluators"])

    def test_custom_evaluators(self):
        config = build_eval_config(
            "http://localhost:8000/chat",
            eval_modes=["info_delivery"],
        )
        self.assertEqual(config["evaluators"], ["info_delivery"])


class TestRunQuickEval(unittest.TestCase):
    def setUp(self):
        FakeAdapter.instances.clear()

    def _mock_result(self):
        from core.types import EvalResult, TerminationReason

        mock_result = MagicMock(spec=EvalResult)
        mock_result.task_id = "t1"
        mock_result.overall_score = 0.8
        mock_result.scores = {"info_delivery": 0.8}
        mock_result.terminated = TerminationReason.SUCCESS
        mock_result.steps_taken = 1
        mock_result.summary.return_value = {
            "task_id": "t1",
            "overall_score": 0.8,
            "scores": {"info_delivery": 0.8},
        }
        return mock_result

    def _mock_task(self):
        mock_task = MagicMock()
        mock_task.id = "t1"
        mock_task.difficulty = "easy"
        mock_task.description = "test"
        mock_task.initial_message = "hi"
        return mock_task

    @patch("eval_bot.commands.quick_eval.generate_html", return_value="<html></html>")
    @patch("eval_bot.commands.quick_eval.load_results", return_value=[])
    @patch("eval_bot.commands.quick_eval.registry.get_evaluator", return_value=FakeEvaluator)
    @patch("eval_bot.commands.quick_eval.registry.get_adapter", return_value=FakeAdapter)
    @patch("eval_bot.commands.quick_eval.registry.get_environment", return_value=FakeEnv)
    @patch("eval_bot.commands.quick_eval.Orchestrator")
    @patch("eval_bot.commands.quick_eval.load_tasks")
    def test_returns_summary_and_closes_adapter(
        self,
        mock_load_tasks,
        mock_orch_cls,
        _mock_get_env,
        _mock_get_adapter,
        _mock_get_eval,
        _mock_load_results,
        _mock_generate_html,
    ):
        mock_load_tasks.return_value = [self._mock_task()]
        mock_orch = MagicMock()
        mock_orch.run.return_value = self._mock_result()
        mock_orch_cls.return_value = mock_orch

        result = run_quick_eval("http://localhost:8000/chat")

        self.assertIn("summary_text", result)
        self.assertIn("results_file", result)
        self.assertIn("评测完成", result["summary_text"])
        self.assertTrue(FakeAdapter.instances[0].closed)

    @patch("eval_bot.commands.quick_eval.registry.get_evaluator", return_value=FakeEvaluator)
    @patch("eval_bot.commands.quick_eval.registry.get_adapter", return_value=FakeAdapter)
    @patch("eval_bot.commands.quick_eval.registry.get_environment", return_value=FakeEnv)
    @patch("eval_bot.commands.quick_eval.Orchestrator")
    @patch("eval_bot.commands.quick_eval.load_tasks")
    def test_closes_adapter_on_orchestrator_error(
        self,
        mock_load_tasks,
        mock_orch_cls,
        _mock_get_env,
        _mock_get_adapter,
        _mock_get_eval,
    ):
        mock_load_tasks.return_value = [self._mock_task()]
        mock_orch = MagicMock()
        mock_orch.run.side_effect = RuntimeError("boom")
        mock_orch_cls.return_value = mock_orch

        with self.assertRaises(RuntimeError):
            run_quick_eval("http://localhost:8000/chat")

        self.assertTrue(FakeAdapter.instances[0].closed)

    def test_analysis_prompt_is_readable(self):
        self.assertIn("评测分析专家", ANALYSIS_SYSTEM_PROMPT)
        self.assertIn("失败任务", ANALYSIS_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
