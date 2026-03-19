import json
import unittest
from unittest.mock import patch, MagicMock
from eval_bot.commands.gen_scenarios import build_gen_prompt, parse_scenarios


class TestGenScenarios(unittest.TestCase):
    def test_build_prompt_contains_domain(self):
        prompt = build_gen_prompt("退款处理", count=3, difficulty="medium")
        self.assertIn("退款处理", prompt)
        self.assertIn("3", prompt)

    def test_parse_valid_json(self):
        raw = '''```json
[
  {
    "id": "gen-001",
    "description": "test",
    "initial_message": "hello",
    "initial_state": {},
    "max_steps": 1,
    "single_turn": true,
    "required_info": [],
    "difficulty": "easy"
  }
]
```'''
        result = parse_scenarios(raw)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "gen-001")

    def test_parse_no_json(self):
        result = parse_scenarios("no json here")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
