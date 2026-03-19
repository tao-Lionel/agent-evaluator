import time
import unittest
from eval_bot.runner import TaskRunner


class TestTaskRunner(unittest.TestCase):
    def test_submit_and_callback(self):
        """Submit a task, verify callback is called with result."""
        results = []

        def on_done(task_id, result):
            results.append((task_id, result))

        runner = TaskRunner(max_workers=1)

        def fake_work():
            time.sleep(0.1)
            return {"score": 0.85}

        runner.submit("test-001", fake_work, on_done)
        time.sleep(0.5)
        runner.shutdown()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "test-001")
        self.assertEqual(results[0][1], {"score": 0.85})

    def test_submit_error_callback(self):
        """If task raises, callback receives the exception."""
        results = []

        def on_done(task_id, result):
            results.append((task_id, result))

        runner = TaskRunner(max_workers=1)

        def bad_work():
            raise ValueError("boom")

        runner.submit("test-err", bad_work, on_done)
        time.sleep(0.5)
        runner.shutdown()

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0][1], Exception)


if __name__ == "__main__":
    unittest.main()
