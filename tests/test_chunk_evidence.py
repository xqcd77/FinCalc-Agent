import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))

from chunk_evidence import chunk_document, chunk_prose, chunk_table, prose_spans, validate_coverage  # noqa: E402


def make_doc(section, table=None):
    return {
        "doc_id": "样例报告",
        "source_path": "样例报告.tex",
        "source_sha256": "a" * 64,
        "company_candidate_from_filename": "甲公司",
        "title": "甲公司财务报告",
        "sections": [section],
        "tables": [table] if table else [],
    }


def make_section(text="", table_ids=None):
    return {
        "section_id": "样例报告::s0001",
        "title": "利润分析",
        "path": ["经营分析", "利润分析"],
        "text": text,
        "table_ids": table_ids or [],
        "source_start_line": 20,
        "source_end_line": 35,
    }


def row(number, cells, raw_cells=None):
    return {"row": number, "cells": cells, "raw_cells": raw_cells or cells}


def make_table(rows, caption="利润表"):
    return {
        "table_id": "样例报告::t0001",
        "environment": "tabular",
        "caption": caption,
        "units_seen_near_header": ["万元"],
        "source_start_line": 25,
        "source_end_line": 32,
        "rows": rows,
    }


class ChunkEvidenceTests(unittest.TestCase):
    def test_long_prose_has_complete_overlapping_coverage(self):
        text = "甲公司2022年营业利润增长。" * 75
        spans = prose_spans(text, max_chars=500, overlap_chars=60)
        self.assertGreater(len(spans), 1)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], len(text))
        self.assertTrue(all(end - start <= 500 for start, end in spans))
        self.assertTrue(all(right[0] <= left[1] for left, right in zip(spans, spans[1:])))
        section = make_section(text)
        chunks = chunk_prose(make_doc(section), section, 500, 60)
        self.assertTrue(all(chunk["section_path"] == section["path"] for chunk in chunks))
        self.assertEqual(validate_coverage([make_doc(section)], chunks)["uncovered_prose_characters"], 0)

    def test_multilevel_header_keeps_year_to_value_mapping(self):
        slash = chr(92)
        rows = [
            row(1, ["股票代码", "利润"], ["股票代码", slash + "multicolumn{2}{c}{利润}"]),
            row(2, ["", "2022", "2024"]),
            row(3, ["600000", "10万元", "20万元"]),
        ]
        table = make_table(rows)
        section = make_section(table_ids=[table["table_id"]])
        chunks, _, aligned = chunk_table(make_doc(section, table), section, table, 3, 650)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["header_row_numbers"], [1, 2])
        self.assertEqual(chunks[0]["data_row_numbers"], [3])
        self.assertIn("2022=10万元", chunks[0]["text"])
        self.assertIn("2024=20万元", chunks[0]["text"])
        self.assertEqual(aligned, 1)

    def test_multirow_label_continues_across_chunk_boundary(self):
        slash = chr(92)
        rows = [row(1, ["公司", "指标", "2022", "2024"])]
        rows += [row(2, ["甲公司", "收入", "10", "20"], [slash + "multirow{4}{*}{甲公司}", "收入", "10", "20"])]
        rows += [row(3, ["", "利润", "2", "3"]), row(4, ["", "资产", "8", "9"]), row(5, ["", "负债", "4", "5"])]
        table = make_table(rows)
        section = make_section(table_ids=[table["table_id"]])
        chunks, fills, _ = chunk_table(make_doc(section, table), section, table, 2, 650)
        self.assertEqual([c["data_row_numbers"] for c in chunks], [[2, 3], [4, 5]])
        self.assertEqual(fills, 3)
        self.assertIn("甲公司", chunks[1]["rows"][1]["display_cells"][0])
        self.assertEqual(chunks[1]["rows"][1]["cells"][0], "")

    def test_missing_caption_uses_section_and_oversize_row_is_kept(self):
        rows = [row(1, ["指标", "2024"]), row(2, ["营业利润", "9" * 700 + "万元"])]
        table = make_table(rows, caption="")
        section = make_section(table_ids=[table["table_id"]])
        chunks, _, _ = chunk_table(make_doc(section, table), section, table, 3, 200)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0]["oversize_content"])
        self.assertIn("原文无表题", chunks[0]["text"])
        self.assertIn("经营分析 / 利润分析", chunks[0]["text"])
        self.assertIn("9" * 700, chunks[0]["text"])

    def test_stock_label_does_not_create_false_share_unit_hint(self):
        rows = [row(1, ["股票代码", "金额"]), row(2, ["600000", "10元"])]
        table = make_table(rows)
        table["units_seen_near_header"] = ["股", "元"]
        section = make_section(table_ids=[table["table_id"]])
        chunks, _, _ = chunk_table(make_doc(section, table), section, table, 3, 650)
        self.assertIn("表头附近出现的单位：元", chunks[0]["text"])
        self.assertNotIn("表头附近出现的单位：股", chunks[0]["text"])

    def test_real_parsed_report_covers_all_rows_and_prose(self):
        if not (Path(__file__).resolve().parents[1] / "data/processed/corpus/parsed_documents.jsonl").exists():
            self.skipTest("Optional full-corpus integration fixture is not distributed")
        root = Path(__file__).resolve().parents[1]
        input_path = root / "data/processed/corpus/parsed_documents.jsonl"
        self.assertTrue(input_path.is_file())
        with input_path.open("r", encoding="utf-8") as stream:
            doc = json.loads(stream.readline())
        chunks, _ = chunk_document(doc, 500, 60, 650, 3)
        self.assertTrue(chunks)
        self.assertEqual(len(chunks), len({chunk["chunk_id"] for chunk in chunks}))
        self.assertEqual(validate_coverage([doc], chunks), {
            "uncovered_prose_characters": 0,
            "missing_table_rows": 0,
            "duplicate_data_rows": 0,
            "missing_header_repetitions": 0,
        })
        self.assertTrue(all(chunk["source_path"] == doc["source_path"] for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
