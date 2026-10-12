import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "rag"))

from parse_latex import (  # noqa: E402
    _find_environment_spans,
    _section_records,
    _split_table_cells,
    _split_table_rows,
    latex_to_text,
    parse_document,
    strip_tex_comments,
)


class LatexParserTests(unittest.TestCase):
    def test_comments_respect_escaped_percent(self):
        self.assertEqual(strip_tex_comments("营业率 5\\% % note\n下一行"), "营业率 5\\% \n下一行")

    def test_table_row_and_cell_split_respects_nested_braces(self):
        sep = chr(92) * 2
        tex = "指标 & 2022 " + sep + " 收入 & " + chr(92) + "multicolumn{2}{c}{3 & 4} " + sep
        rows = _split_table_rows(tex)
        self.assertEqual(len(rows), 2)
        self.assertEqual(_split_table_cells(rows[1]), ["收入", r"\multicolumn{2}{c}{3 & 4}"])

    def test_common_financial_macros_keep_visible_values(self):
        slash = chr(92)
        first = slash + "textbf{营业利润} & " + slash + "multicolumn{2}{c}{" + slash + "underline{12.50}}"
        self.assertEqual(latex_to_text(first), "营业利润 & 12.50")
        second = slash + "SI{12.5}{亿元} " + slash + "%"
        self.assertEqual(latex_to_text(second), "12.5亿元%")

    def test_table_rule_arguments_do_not_become_cell_values(self):
        slash = chr(92)
        self.assertEqual(latex_to_text(slash + "cmidrule(lr){3-7}"), "")

    def test_layout_commands_and_list_options_do_not_pollute_prose(self):
        slash = chr(92)
        tex = slash + "vspace*{3cm}" + slash + "color{formblue}" + slash + "begin{enumerate}[label={x}, leftmargin=2em]" + slash + "item 现金" + slash + "end{enumerate}"
        tex += " 填写" + slash + "blankline{3cm}" + slash * 2 + "[4pt] " + slash + "rule{0.5" + slash + "textwidth}{0.4pt}"
        self.assertEqual(latex_to_text(tex), "现金 填写")

    def test_environment_scanner_finds_tabular_inside_table(self):
        slash = chr(92)
        text = slash + "begin{table}" + slash + "begin{tabular}{lc}A & B " + slash * 2 + slash + "end{tabular}" + slash + "end{table}"
        spans, warnings = _find_environment_spans(text, {"table", "tabular"})
        self.assertEqual({span["env"] for span in spans}, {"table", "tabular"})
        self.assertEqual(warnings, [])

    def test_real_document_has_sections_tables_and_source_hash(self):
        if not (Path(__file__).resolve().parents[1] / "data/raw/financecomplexqa/latex/corporate_reports").exists():
            self.skipTest("Optional full-corpus integration fixture is not distributed")
        root = Path(__file__).resolve().parents[1]
        source = root / "data/raw/financecomplexqa/latex/corporate_reports"
        files = sorted(source.glob("*.tex"))
        self.assertTrue(files, "FinanceComplexQA LaTeX data not found")
        doc, _ = parse_document(files[0], source)
        self.assertEqual(len(doc["source_sha256"]), 64)
        self.assertGreater(doc["section_count"], 0)
        self.assertGreater(doc["table_count"], 0)
        self.assertTrue(all("raw_latex" in table for table in doc["tables"]))
        self.assertTrue(all(table["source_start_line"] > 0 for table in doc["tables"]))


if __name__ == "__main__":
    unittest.main()
