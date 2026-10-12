"""Independent checks for task 4 ranking and citation preserving context."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))
from rerank_context import assemble_context, sort_scores, validate_input


def sample_chunk(chunk_id="c1", text="2022年营业利润 123.4567 万元"):
    return {
        "chunk_id": chunk_id, "doc_id": "公司A_报告", "source_path": "公司A_报告.tex",
        "source_start_line": 10, "source_end_line": 14,
        "location_granularity": "table", "section_path": ["营业利润"],
        "content_type": "table", "table_id": "t1", "header_row_numbers": [1],
        "data_row_numbers": [2], "years": ["2022"], "unit_mentions": ["万元"],
        "text": text, "content_text": text,
    }


class RerankContextTests(unittest.TestCase):
    def test_scores_reorder_and_preserve_hybrid_rank(self):
        candidates = [
            {"rank": 1, "chunk_id": "c1", "doc_id": "d", "score": 0.03},
            {"rank": 2, "chunk_id": "c2", "doc_id": "d", "score": 0.02},
        ]
        ranked = sort_scores(candidates, [-1.0, 3.0])
        self.assertEqual([x["chunk_id"] for x in ranked], ["c2", "c1"])
        self.assertEqual([x["hybrid_rank"] for x in ranked], [2, 1])
        self.assertEqual([x["rerank_rank"] for x in ranked], [1, 2])

    def test_context_keeps_complete_numbers_and_provenance(self):
        chunk = sample_chunk()
        ranked = [{"chunk_id": "c1", "rerank_rank": 1, "rerank_logit": 4.0}]
        context, evidence, omitted = assemble_context(ranked, {"c1": chunk})
        self.assertIn("123.4567 万元", context)
        self.assertIn("公司A_报告.tex", context)
        self.assertIn("原文第10-14行", context)
        self.assertEqual(evidence[0]["data_row_numbers"], [2])
        self.assertEqual(evidence[0]["citation"], "E1")
        self.assertFalse(omitted)

    def test_budget_omits_whole_chunk_without_cutting_data(self):
        chunks = {"c1": sample_chunk("c1"),
                  "c2": sample_chunk("c2", "X" * 300)}
        ranked = [{"chunk_id": key, "rerank_rank": i,
                   "rerank_logit": 1.0 / i}
                  for i, key in enumerate(chunks, 1)]
        first, _, _ = assemble_context(ranked[:1], chunks)
        context, evidence, omitted = assemble_context(
            ranked, chunks, max_chars=len(first) + 5)
        self.assertEqual(context, first)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(omitted[0]["chunk_id"], "c2")

    def test_input_validation_rejects_gold_and_unknown_chunk(self):
        chunks = {f"c{i}": sample_chunk(f"c{i}") for i in range(20)}
        candidates = [{"rank": i + 1, "chunk_id": f"c{i}",
                       "doc_id": "公司A_报告", "score": 0.1}
                      for i in range(20)]
        row = {"finqa_id": "q1", "hybrid": candidates, "gold": 123}
        with self.assertRaises(ValueError):
            validate_input(row, chunks)
        del row["gold"]
        validate_input(row, chunks)
        row["hybrid"][0]["chunk_id"] = "missing"
        with self.assertRaises(ValueError):
            validate_input(row, chunks)


if __name__ == "__main__":
    unittest.main()
