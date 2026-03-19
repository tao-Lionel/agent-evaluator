import json
import os
import tempfile
import unittest
from eval_bot.commands.query_results import scan_results


class TestScanResults(unittest.TestCase):
    def test_scan_finds_json_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake results
            data = [{"task_id": "t1", "overall_score": 0.8, "scores": {"x": 0.8}}]
            path = os.path.join(tmpdir, "results_123.json")
            with open(path, "w") as f:
                json.dump(data, f)

            results = scan_results(tmpdir)
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["file"], path)

    def test_scan_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scan_results(tmpdir)
            self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
