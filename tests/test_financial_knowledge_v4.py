"""Focused exact-alias and source-neutral financial guidance regressions."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
from financial_knowledge_v4 import (
    METRICS, RECORDS, LEXICON, aliases, canonical, load_lexicon, metric_card,
    planning_guidance, term_pattern,
)


class FinancialKnowledgeTests(unittest.TestCase):
    def test_real_missing_safe_aliases(self):
        for phrase, name in [
            ("负债总额", "负债合计"), ("所有者权益总额", "所有者权益"),
            ("股东权益总额", "所有者权益"), ("股东权益合计", "所有者权益"),
            ("归属于母公司股东的权益", "归母权益"),
            ("归母所有者权益", "归母权益"), ("归母所有者权益合计", "归母权益"),
        ]:
            self.assertEqual(canonical(phrase), name)

    def test_exact_normalisation_without_guessing(self):
        self.assertEqual(canonical(" e p s "), "每股收益")
        self.assertEqual(canonical("归 属 于 母 公 司 股 东 的 权 益"), "归母权益")
        self.assertEqual(canonical("PB"), "市净率")
        self.assertEqual(canonical("不明确的调整净利润"), "不明确的调整净利润")
        self.assertEqual(canonical("归母净利润增长率"), "归母净利润增长率")
        self.assertEqual(canonical("EBITDA"), "EBITDA")

    def test_dangerous_scope_and_balance_mixups_remain_distinct(self):
        for left, right in [
            ("净利润", "归母净利润"), ("所有者权益合计", "归母权益"),
            ("货币资金", "现金"), ("现金及现金等价物", "货币资金"),
            ("固定资产原值", "固定资产"), ("平均固定资产", "固定资产"),
            ("平均总资产", "资产总计"), ("平均流动资产", "流动资产"),
            ("平均存货", "存货"), ("基本每股收益", "稀释每股收益"),
            ("扣非净利润", "净利润"), ("扣非归母净利润", "归母净利润"),
            ("营业总成本", "营业成本"), ("现金流量利息保障倍数", "利息保障倍数"),
        ]:
            self.assertNotEqual(canonical(left), canonical(right), (left, right))

    def test_compatibility_aliases_are_explicit_not_universal(self):
        self.assertIn("营业总收入", RECORDS["营业收入"]["compatibility_aliases"])
        self.assertIn("利息支出", RECORDS["利息费用"]["compatibility_aliases"])
        self.assertNotIn("利息支出", METRICS)
        self.assertEqual(canonical("营业总收入"), "营业收入")
        self.assertIn("本项目", RECORDS["营业收入"]["scope"])

    def test_related_terms_never_become_verified_aliases(self):
        self.assertNotIn("净利润", aliases("归母净利润"))
        self.assertNotIn("少数股东权益", aliases("所有者权益"))
        self.assertNotIn("现金及现金等价物", aliases("货币资金"))
        self.assertNotIn("营业总成本", aliases("营业成本"))

    def test_english_patterns_require_ascii_token_boundaries(self):
        for token in ["PE", "EPS", "EBIT", "PB"]:
            self.assertIsNone(re.search(term_pattern(token), "prefix" + token + "suffix", re.I))
            self.assertIsNone(re.search(term_pattern(token), "a_" + token, re.I))
            self.assertIsNotNone(re.search(term_pattern(token), "指标（" + token.lower() + "）", re.I))
        self.assertIsNone(re.search(term_pattern("EBIT"), "EBITDA", re.I))

    def test_cash_coverage_guidance_excludes_ordinary_ebit_card(self):
        guidance = planning_guidance("现金流量利息保障倍数是多少？", "interest_coverage")
        self.assertIn("经营活动现金流量净额", guidance)
        self.assertIn("不能用息税前利润", guidance)
        self.assertNotIn("普通利息保障倍数=", guidance)

    def test_valuation_guidance_contains_share_units_and_two_periods(self):
        guidance = planning_guidance("PB上调后静态市盈率PE是多少？")
        self.assertIn("每股收益", guidance)
        self.assertIn("估值交易日", guidance)
        self.assertIn("财报年度", guidance)
        self.assertIn("0.01", guidance)

    def test_day_types_and_inverse_cash_cycle_guidance(self):
        self.assertEqual(RECORDS["现金循环周期"]["unit_kind"], "day")
        guidance = planning_guidance("现金循环周期为1100天，存货周转天数是多少？")
        self.assertIn("现金循环周期−应收周转天数+应付周转天数", guidance)
        self.assertIn("360/365", guidance)

    def test_guidance_is_bounded_and_unknown_question_adds_nothing(self):
        question = "现金流量利息保障倍数、净资产收益率、PE、存货周转天数及复合增长率"
        self.assertLessEqual(len(planning_guidance(question).splitlines()), 2)
        self.assertTrue(all(len(card["text"]) <= 180 for card in LEXICON["planning_cards"]))
        self.assertEqual(planning_guidance("文本里只有SPEED而没有估值指标"), "")

    def test_metric_card_discloses_type_and_forbidden_scope(self):
        card = metric_card("归母所有者权益")
        self.assertIn("归母权益[金额]", card)
        self.assertIn("所有者权益", card)
        self.assertEqual(metric_card("未知指标"), "")

    def test_loader_rejects_cross_metric_alias_collision(self):
        payload = json.loads(json.dumps(LEXICON))
        payload["metrics"][1]["aliases"].append(payload["metrics"][0]["name"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "别名冲突"):
                load_lexicon(path)

    def test_loader_rejects_alias_exclusion_conflict(self):
        payload = json.loads(json.dumps(LEXICON))
        payload["metrics"][0]["do_not_substitute"].append(payload["metrics"][0]["name"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "矛盾"):
                load_lexicon(path)

    def test_loader_rejects_case_normalised_duplicate_alias(self):
        payload = json.loads(json.dumps(LEXICON))
        record = next(record for record in payload["metrics"] if "EPS" in record["aliases"])
        record["aliases"].append("eps")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "重复别名"):
                load_lexicon(path)


if __name__ == "__main__":
    unittest.main()
