"""Behavioral tests for the bounded V3 agent and sourced Decimal tool."""

import json
from pathlib import Path
import re
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rag"))
from agent_v3 import EvidenceState, RagAgentV3  # noqa: E402
from field_plan_v3 import calculate, validate_binding, validate_plan  # noqa: E402
from generate_calculate import read_jsonl  # noqa: E402


def make_chunk(ident, body, *, company="甲公司", year="2022"):
    return {
        "chunk_id": ident, "doc_id": f"{company}_报告", "company": company,
        "source_path": f"{company}_报告.tex", "source_start_line": 10,
        "source_end_line": 20, "section_path": ["财务指标"],
        "content_type": "prose", "table_id": None, "data_row_numbers": [],
        "years": [year], "content_text": body, "text": f"公司：{company}\n{body}",
    }


def make_row(question, chunk=None):
    if chunk is None:
        return {"finqa_id": "synthetic", "question": question,
                "company_with_period": "甲公司", "context_text": "",
                "selected_evidence": []}
    return {"finqa_id": "synthetic", "question": question,
            "company_with_period": "甲公司",
            "context_text": f"[E1] 来源：甲公司_报告.tex\n公司：甲公司\n{chunk['content_text']}",
            "selected_evidence": [{"citation": "E1", "chunk_id": chunk["chunk_id"],
                                   "doc_id": chunk["doc_id"]}]}


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def __call__(self, messages):
        self.calls += 1
        return json.dumps(next(self.responses), ensure_ascii=False), {}


class CharTokenizer:
    """Deterministic test double; production loads the exact Qwen tokenizer."""
    def encode(self, text, add_special_tokens=False):
        return list(text)


TOKENIZER = CharTokenizer()


class FakeRetriever:
    def __init__(self, candidates=None):
        self.candidates = candidates or []
        self.calls = []

    def search(self, field, row, seen, round_number):
        self.calls.append((field["id"], round_number))
        return [{"chunk": chunk, "query": field["metric"], "round": round_number}
                for chunk in self.candidates if chunk["chunk_id"] not in seen]


class AgentV3Tests(unittest.TestCase):
    def setUp(self):
        self.question = "假如甲公司2023年归母净利润为183286.12380957万元，2023年同比增长率是多少？"
        self.chunk = make_chunk("prior", "2022年归母净利润为1,880.0900669百万元。")
        self.plan = {"formula_id": "growth_rate", "answer_unit": "decimal", "fields": [
            {"id": "current", "metric": "归母净利润", "year": "2023", "source": "question",
             "value": "183286.12380957", "unit": "万元",
             "quote": "2023年归母净利润为183286.12380957万元"},
            {"id": "previous", "metric": "归母净利润", "year": "2022", "source": "report"},
        ]}
        self.binding = {"id": "previous", "value": "1880.0900669", "unit": "百万元",
                        "source_ref": "E1", "quote": "归母净利润为1,880.0900669百万元"}

    def test_visible_evidence_rechecked_before_wrong_refusal(self):
        row = make_row(self.question, self.chunk)
        model = ScriptedModel([self.plan, {"bindings": [], "missing": ["previous"]},
                               {"bindings": [self.binding], "missing": []}])
        retrieval = FakeRetriever()
        output = RagAgentV3(model, retrieval, {"prior": self.chunk}, TOKENIZER).run(row)
        self.assertEqual(output["status"], "calculated", output)
        self.assertTrue(output["trace"]["evidence_completeness"]["complete"])
        self.assertEqual(retrieval.calls, [])
        self.assertEqual(output["final"]["answer"], "-0.0251")

    def test_integer_valued_visible_field_is_rechecked_before_refusal(self):
        chunk = make_chunk("integer", "2022年归母净利润为1,880百万元。")
        binding = {"id": "previous", "value": "1880", "unit": "百万元",
                   "source_ref": "E1", "quote": "归母净利润为1,880百万元"}
        model = ScriptedModel([self.plan, {"bindings": [], "missing": ["previous"]},
                               {"bindings": [binding], "missing": []}])
        retrieval = FakeRetriever()
        output = RagAgentV3(model, retrieval, {"integer": chunk}, TOKENIZER).run(
            make_row(self.question, chunk))
        self.assertEqual(output["status"], "calculated", output)
        self.assertEqual(model.calls, 3)
        self.assertEqual(retrieval.calls, [])

    def test_missing_field_triggers_targeted_retrieval(self):
        row = make_row(self.question)
        model = ScriptedModel([self.plan, {"bindings": [], "missing": ["previous"]},
                               {"bindings": [self.binding], "missing": []}])
        retrieval = FakeRetriever([self.chunk])
        output = RagAgentV3(model, retrieval, {"prior": self.chunk}, TOKENIZER).run(row)
        self.assertEqual(output["status"], "calculated", output)
        self.assertEqual(retrieval.calls, [("previous", 1)])
        self.assertEqual(output["selected_evidence"][0]["chunk_id"], "prior")

    def test_retrieval_exhaustion_preserves_justified_refusal(self):
        row = make_row(self.question)
        model = ScriptedModel([self.plan, {"bindings": [], "missing": ["previous"]}])
        output = RagAgentV3(model, FakeRetriever(), {}, TOKENIZER).run(row)
        self.assertEqual(output["status"], "insufficient", output)
        self.assertFalse(output["trace"]["evidence_completeness"]["complete"])
        self.assertIn("2022", output["trace"]["reason"])

    def test_scenario_value_cannot_be_omitted_or_replaced_by_report(self):
        malformed = {**self.plan, "fields": [
            {**self.plan["fields"][0], "source": "report"}, self.plan["fields"][1]]}
        with self.assertRaisesRegex(ValueError, "Hypothetical"):
            validate_plan(malformed, self.question)

    def test_question_and_report_units_must_match_the_cited_text(self):
        wrong_question_unit = {**self.plan, "fields": [
            {**self.plan["fields"][0], "unit": "元"}, self.plan["fields"][1]]}
        with self.assertRaisesRegex(ValueError, "unit absent"):
            validate_plan(wrong_question_unit, self.question)
        row = make_row(self.question, self.chunk)
        wrong_report_unit = {**self.binding, "unit": "万元"}
        with self.assertRaisesRegex(ValueError, "unit absent"):
            validate_binding(wrong_report_unit, validate_plan(self.plan, self.question)["fields"][1],
                             row, {"E1": self.chunk})

    def test_peer_value_in_target_report_is_not_bound_to_target(self):
        body = "北大荒（600598.SH）2022年归母净利润为1,880.0900669百万元。"
        peer_chunk = make_chunk("peer", body)
        row = make_row(self.question, peer_chunk)
        model = ScriptedModel([self.plan, {"bindings": [self.binding], "missing": []},
                               {"bindings": [], "missing": ["previous"]}])
        output = RagAgentV3(model, FakeRetriever(), {"peer": peer_chunk}, TOKENIZER).run(row)
        self.assertEqual(output["status"], "insufficient", output)
        self.assertIn("peer company", str(output["trace"]["events"]), output)

    def test_initial_evidence_from_other_report_is_rejected(self):
        foreign = make_chunk("foreign", "甲公司2022年归母净利润为1,880.0900669百万元。",
                             company="乙公司")
        with self.assertRaisesRegex(ValueError, "another company's report"):
            RagAgentV3(ScriptedModel([]), FakeRetriever(), {"foreign": foreign},
                       TOKENIZER).run(make_row(self.question, foreign))

    def test_missing_evidence_is_not_accepted_as_unsupported_formula(self):
        refusal = {"formula_id": "unsupported", "reason": "未找到2022年报告数值", "fields": []}
        output = RagAgentV3(ScriptedModel([refusal, refusal]), FakeRetriever(), {},
                            TOKENIZER).run(make_row(self.question))
        self.assertEqual(output["status"], "invalid_plan", output)
        self.assertEqual(len(output["trace"]["events"]), 2)

    def test_wrong_company_evidence_and_unattributed_hundred_rejected(self):
        other = make_chunk("other", "2022年归母净利润为1,880.0900669百万元。", company="乙公司")
        row = make_row(self.question, other)
        model = ScriptedModel([self.plan, {"bindings": [self.binding], "missing": []}])
        with self.assertRaisesRegex(ValueError, "another company's report"):
            RagAgentV3(model, FakeRetriever(), {"other": other}, TOKENIZER).run(row)
        custom = {"formula_id": "custom", "answer_unit": "decimal",
                  "expression": "current / previous * 100 - 1", "fields": self.plan["fields"]}
        with self.assertRaisesRegex(ValueError, "unattributed"):
            validate_plan(custom, self.question)

    def test_fixed_growth_formula_cannot_double_convert_percent(self):
        plan = validate_plan(self.plan, self.question)
        row = make_row(self.question, self.chunk)
        tool = calculate(plan, {"previous": self.binding}, row)
        self.assertEqual(tool["expression"], "current / previous - 1")
        self.assertEqual(tool["answer_unit"], "decimal")
        self.assertEqual(tool["answer"], "-0.0251")

    def test_actual_00009_and_00006_regressions(self):
        if not (Path(__file__).resolve().parents[1] / "data/processed/eval/task4_v2_contexts.jsonl").exists():
            self.skipTest("Optional full-corpus integration fixture is not distributed")
        source = ROOT / "data/processed/eval/task4_v2_contexts.jsonl"
        rows = {x["finqa_id"]: x for x in read_jsonl(source)
                if x["finqa_id"] in ("finreport_00009", "finreport_00006")}
        needed = {e["chunk_id"] for row in rows.values() for e in row["selected_evidence"]}
        chunks = {x["chunk_id"]: x for x in read_jsonl(
            ROOT / "data/processed/corpus/evidence_chunks.jsonl") if x["chunk_id"] in needed}
        cases = [
            ("finreport_00009", "growth_rate", "current", "previous",
             "183286.12380957", "万元", "1880.0900669", "百万元", "-0.0251"),
            ("finreport_00006", "operating_margin", "revenue", "operating_profit",
             "309223291387.0938", "元", "31.54", "亿元", "0.0102"),
        ]
        for ident, formula, question_id, report_id, q_value, q_unit, e_value, e_unit, expected in cases:
            with self.subTest(ident=ident):
                row = rows[ident]
                q_quote = re.search(r"[^,，]*" + re.escape(q_value) + re.escape(q_unit),
                                    row["question"]).group()
                e_quote = re.search(r"[^，；;]*" + re.escape(e_value if ident.endswith("00006")
                                    else "1,880.0900669") + re.escape(e_unit),
                                    row["context_text"]).group()
                report_metric = e_quote.split("为")[0].split("而")[-1].strip() if ident.endswith("00009") else "营业利润"
                question_metric = report_metric if ident.endswith("00009") else "营业收入"
                plan = {"formula_id": formula, "answer_unit": "decimal", "fields": [
                    {"id": question_id, "metric": question_metric, "year": "2023" if ident.endswith("00009") else "2024",
                     "source": "question", "value": q_value, "unit": q_unit, "quote": q_quote},
                    {"id": report_id, "metric": report_metric, "year": "2022" if ident.endswith("00009") else "2024",
                     "source": "report"},
                ]}
                binding = {"id": report_id, "value": e_value, "unit": e_unit,
                           "source_ref": "E1", "quote": e_quote}
                model = ScriptedModel([plan, {"bindings": [binding], "missing": []}])
                output = RagAgentV3(model, FakeRetriever(), chunks, TOKENIZER).run(row)
                self.assertEqual(output["status"], "calculated", output)
                self.assertEqual(output["final"]["answer"], expected)

    def test_cagr_uses_calendar_years_and_decimal_power(self):
        question = "甲公司2022年营业利润为100元，2024年营业利润为121元，复合增长率是多少？"
        proposal = {"formula_id": "cagr", "answer_unit": "decimal", "fields": [
            {"id": "start", "metric": "营业利润", "year": "2022", "source": "question",
             "value": "100", "unit": "元", "quote": "2022年营业利润为100元"},
            {"id": "end", "metric": "营业利润", "year": "2024", "source": "question",
             "value": "121", "unit": "元", "quote": "2024年营业利润为121元"},
        ]}
        result = calculate(validate_plan(proposal, question), {}, make_row(question))
        self.assertEqual(result["answer"], "0.1")

    def test_relevant_late_block_survives_field_view_budget(self):
        irrelevant = make_chunk("early", "无关背景" * 500)
        irrelevant["source_start_line"] = 1
        row = make_row(self.question, irrelevant)
        row["selected_evidence"].append({"citation": "E2", "chunk_id": "prior",
                                         "doc_id": self.chunk["doc_id"]})
        state = EvidenceState(row, {"early": irrelevant, "prior": self.chunk}, TOKENIZER)
        _, refs, tokens = state.view_for_field(validate_plan(self.plan, self.question)["fields"][1])
        self.assertIn("E2", refs)
        self.assertLessEqual(tokens, 2600)

    def test_date_fields_and_order_statistic_are_calculated_by_tool(self):
        question = "甲公司2025年8月21日收盘价为10元，2025年8月22日收盘价为12元，次高价格是多少？"
        fields = [{"id": ident, "metric": "收盘价", "year": "2025", "date": date,
                   "source": "question", "value": value, "unit": "元", "quote": quote}
                  for ident, date, value, quote in [
                      ("a", "2025-08-21", "10", "2025年8月21日收盘价为10元"),
                      ("b", "2025-08-22", "12", "2025年8月22日收盘价为12元")]]
        plan = validate_plan({"formula_id": "custom", "answer_unit": "元",
                              "expression": "nth_largest(2,a,b)", "fields": fields}, question)
        self.assertEqual(calculate(plan, {}, make_row(question))["answer"], "10")
        fields[0]["date"] = "2025-02-30"
        with self.assertRaisesRegex(ValueError, "date"):
            validate_plan({**plan, "fields": fields}, question)

    def test_per_share_source_unit_is_normalized_without_scale_change(self):
        question = "甲公司2024年每股收益为1.25元，是多少元/股？"
        plan = validate_plan({"formula_id": "direct", "answer_unit": "元/股", "fields": [
            {"id": "value", "metric": "每股收益", "year": "2024", "source": "question",
             "value": "1.25", "unit": "元", "quote": "2024年每股收益为1.25元"}]}, question)
        self.assertEqual(plan["fields"][0]["unit"], "元/股")
        self.assertEqual(calculate(plan, {}, make_row(question))["answer"], "1.25")

    def test_failed_calculation_preserves_verified_fields(self):
        zero_chunk = make_chunk("zero", "2022年归母净利润为0百万元。")
        binding = {**self.binding, "value": "0", "quote": "归母净利润为0百万元"}
        model = ScriptedModel([self.plan, {"bindings": [binding]}, {"bindings": [binding]}])
        output = RagAgentV3(model, FakeRetriever(), {"zero": zero_chunk}, TOKENIZER).run(
            make_row(self.question, zero_chunk))
        self.assertEqual(output["status"], "calculation_error", output)
        self.assertTrue(output["trace"]["evidence_completeness"]["complete"])
        self.assertIn("previous", output["trace"]["bindings"])

    def test_oversized_planner_prompt_does_not_crash_event_logging(self):
        output = RagAgentV3(ScriptedModel([]), FakeRetriever(), {}, TOKENIZER).run(
            make_row("财报问题" * 1000))
        self.assertEqual(output["status"], "invalid_plan")
        self.assertIn("budget", output["error"])

    def test_wrong_year_prose_is_not_bound_using_chunk_wide_years(self):
        body = "2022年归母净利润为100百万元。2023年归母净利润为120百万元。"
        chunk = make_chunk("years", body)
        chunk["years"] = ["2022", "2023"]
        binding = {**self.binding, "value": "120", "quote": "归母净利润为120百万元"}
        with self.assertRaisesRegex(ValueError, "another year"):
            validate_binding(binding, validate_plan(self.plan, self.question)["fields"][1],
                             make_row(self.question, chunk), {"E1": chunk})

    def test_custom_unsafe_syntax_is_rejected_before_retrieval(self):
        with self.assertRaisesRegex(ValueError, "syntax"):
            validate_plan({"formula_id": "custom", "answer_unit": "decimal",
                           "expression": "current[0] / previous", "fields": self.plan["fields"]},
                          self.question)

    def test_api_failure_keeps_plan_and_call_attempt(self):
        calls = []
        def ask(messages):
            calls.append(messages)
            if len(calls) == 1:
                return json.dumps(self.plan), {}
            raise RuntimeError("test API unavailable")
        output = RagAgentV3(ask, FakeRetriever(), {}, TOKENIZER).run(make_row(self.question))
        self.assertEqual(output["status"], "api_error")
        self.assertIn("field_plan", output["trace"])
        self.assertTrue(output["trace"]["events"][-1]["call_attempted"])

    def test_hypothetical_percentage_change_cannot_be_silently_omitted(self):
        question = "假如甲公司2024年营业利润提高50%，2022年到2024年的复合增长率是多少？"
        fields = [{"id": "start", "metric": "营业利润", "year": "2022", "source": "report"},
                  {"id": "end", "metric": "营业利润", "year": "2024", "source": "report"}]
        with self.assertRaisesRegex(ValueError, "Hypothetical"):
            validate_plan({"formula_id": "cagr", "answer_unit": "decimal", "fields": fields}, question)
        fields.append({"id": "change", "metric": "提高幅度", "year": "2024", "source": "question",
                       "value": "50", "unit": "%", "quote": "提高50%"})
        plan = validate_plan({"formula_id": "custom", "answer_unit": "decimal", "fields": fields,
                              "expression": "(end * (1 + change) / start) ** (1 / 2) - 1"}, question)
        self.assertEqual(plan["fields"][-1]["value"], "50")

    def test_malformed_binding_value_is_rejected_then_rechecked(self):
        model = ScriptedModel([self.plan, {"bindings": [{**self.binding, "value": "invalid"}]},
                               {"bindings": [self.binding]}])
        output = RagAgentV3(model, FakeRetriever(), {"prior": self.chunk}, TOKENIZER).run(
            make_row(self.question, self.chunk))
        self.assertEqual(output["status"], "calculated", output)
        self.assertIn("not numeric", str(output["trace"]["events"]))


if __name__ == "__main__":
    unittest.main()
