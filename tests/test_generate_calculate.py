"""Stage 5 contract tests using one real FinanceComplexQA question and evidence."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rag"))
from generate_calculate import MISSING, messages_for, process_response, read_jsonl


class GenerationCalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.row = next(read_jsonl(ROOT / "tests/fixtures/task4_legacy_cagr_context.jsonl"))

    def test_real_question_cagr_with_mixed_report_units(self):
        """Question 00001: 2024 what-if +50%, report 2022/2024 values."""
        plan = {
            "mode": "calculate", "operands": [
                {"id": "profit22", "value": "5596038041.58", "unit": "元",
                 "source_ref": "E3", "quote": "2022=5596038041.58 元"},
                {"id": "profit24", "value": "0.9383766273", "unit": "亿元",
                 "source_ref": "E3", "quote": "2024=0.9383766273 亿元"},
                {"id": "increase", "value": "50", "unit": "%",
                 "source_ref": "question", "quote": "提高50%"},
                {"id": "years", "value": "2", "unit": "decimal",
                 "source_ref": "question", "quote": "2022年到2024年三年间"},
            ],
            "expression": "(profit24 * (1 + increase) / profit22) ** (1 / years) - 1",
            "answer_unit": "decimal",
        }
        # The year count is inferred from endpoints; the literal 2 in the quote
        # appears inside 2022/2024, so the strict source validator must reject it.
        bad = process_response(json.dumps(plan), self.row, "rag")
        self.assertEqual(bad["status"], "invalid_plan")
        plan["operands"] = plan["operands"][:-1]
        plan["expression"] = "(profit24 * (1 + increase) / profit22) ** (1 / 2) - 1"
        good = process_response(json.dumps(plan), self.row, "rag")
        self.assertEqual(good["status"], "calculated", good)
        self.assertEqual(good["final"], {"answer": "-0.8414"})
        self.assertEqual(good["trace"]["normalized_base_values"]["profit24"],
                         "93837662.7300000000")

    def test_rejects_fabricated_quote_and_unsafe_expression(self):
        plan = {"mode": "calculate", "operands": [
            {"id": "x", "value": "999999", "unit": "元",
             "source_ref": "E3", "quote": "2022=999999 元"}],
            "expression": "x", "answer_unit": "元"}
        self.assertEqual(process_response(json.dumps(plan), self.row, "rag")["status"], "invalid_plan")
        plan["operands"][0].update(value="5596038041.58", quote="2022=5596038041.58 元")
        plan["expression"] = "__import__('os').system('whoami')"
        self.assertEqual(process_response(json.dumps(plan), self.row, "rag")["status"], "invalid_plan")

    def test_no_rag_cannot_claim_evidence(self):
        plan = {"mode": "direct", "answer": "测试", "citations": ["E3"]}
        self.assertEqual(process_response(json.dumps(plan), self.row, "no_rag")["status"], "invalid_plan")
        self.assertNotIn("context_text", json.dumps(messages_for(self.row, "no_rag"), ensure_ascii=False))
        self.assertNotIn("5596038041.58", json.dumps(messages_for(self.row, "no_rag"), ensure_ascii=False))

    def test_missing_and_gold_isolation(self):
        result = process_response('{"mode":"insufficient","reason":"无该年数据"}', self.row, "rag")
        self.assertEqual(result["final"], {"answer": MISSING})
        copied = {**self.row, "gold": 1}
        with self.assertRaises(ValueError):
            messages_for(copied, "rag")


if __name__ == "__main__":
    unittest.main()
