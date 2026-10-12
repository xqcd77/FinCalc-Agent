"""Independent unit tests for task 3 retrieval behavior and input isolation."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))
from hybrid_retrieval import (build_bm25, bm25_search, dense_search, make_query,
                              normalize, rrf, tokenize)


class HybridRetrievalTests(unittest.TestCase):
    def test_numeric_atoms_and_company_survive_tokenization(self):
        terms = tokenize("万泰生物2024年营业利润603392.SH增长12.5%")
        self.assertIn("2024年", terms)
        self.assertIn("12.5%", terms)
        self.assertTrue(any("万泰" in term for term in terms))

    def test_bm25_prefers_exact_company_and_year(self):
        chunks = [
            {"text": "公司：万泰生物 2024年营业利润为3.5亿元"},
            {"text": "公司：万泰生物 2021年营业利润为2亿元"},
            {"text": "公司：别的企业 2024年资产总额为3.5亿元"},
        ]
        result = bm25_search(build_bm25(chunks), "万泰生物2024年营业利润")
        self.assertEqual(result[0][0], 0)

    def test_dense_uses_normalized_inner_product(self):
        vectors = normalize(np.array([[3.0, 0.0], [0.0, 2.0]], dtype=np.float32))
        result = dense_search(vectors, np.array([0.1, 0.9], dtype=np.float32))
        self.assertEqual(result[0][0], 1)

    def test_rrf_deduplicates_and_uses_both_routes(self):
        sparse = [(0, 100.0), (1, 80.0)]
        dense = [(1, 0.99), (2, 0.98)]
        result = rrf([sparse, dense])
        self.assertEqual([idx for idx, _ in result], [1, 0, 2])
        self.assertEqual(len({idx for idx, _ in result}), len(result))

    def test_query_does_not_include_evaluation_labels_or_output_template(self):
        row = {"finqa_id": "x", "company_with_period": "万泰生物",
               "question": "2024年利润？\n### 输出格式\n{\"answer\": ...}",
               "gold": "secret answer", "Reference_documents": "secret path",
               "external_knowledge": "secret formula"}
        query = make_query(row)
        self.assertEqual(set(query), {"finqa_id", "company_with_period",
                                     "question", "retrieval_query"})
        self.assertEqual(query["retrieval_query"], "万泰生物 2024年利润？")
        self.assertNotIn("secret", str(query))


if __name__ == "__main__":
    unittest.main()
