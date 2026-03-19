import unittest
from unittest.mock import patch, MagicMock
from eval_bot.commands.quick_eval import run_quick_eval, build_eval_config


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
    @patch("eval_bot.commands.quick_eval.Orchestrator")
    @patch("eval_bot.commands.quick_eval.load_tasks")
    def test_returns_summary(self, mock_load_tasks, mock_orch_cls):
        from core.types import EvalResult, TerminationReason

        mock_task = MagicMock()
        mock_task.id = "t1"
        mock_task.difficulty = "easy"
        mock_task.description = "test"
        mock_task.initial_message = "hi"
        mock_load_tasks.return_value = [mock_task]

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

        mock_orch = MagicMock()
        mock_orch.run.return_value = mock_result
        mock_orch_cls.return_value = mock_orch

        result = run_quick_eval("http://localhost:8000/chat")
        self.assertIn("summary_text", result)
        self.assertIn("results_file", result)


if __name__ == "__main__":
    unittest.main()
