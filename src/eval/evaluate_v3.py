"""Offline V3 evaluation. Gold is loaded only here, never by the RAG agent."""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import re
import unicodedata

from evaluate_rag import PROJECT, QUESTIONS, REFUSAL, four_places, keyed, numeric, rows


DEFAULT_INPUT = PROJECT / "data/processed/eval/task5_v3_rag_generation.jsonl"
DEFAULT_PREFIX = PROJECT / "reports/task6_v3"
V2_AUDIT = PROJECT / "reports/task6_v2_question_audit.jsonl"
PROBES = PROJECT / "reports/rag_debug_step1_probe_labels.json"
CHUNKS = PROJECT / "data/processed/corpus/evidence_chunks.jsonl"


def _norm(text: str) -> str:
    return re.sub(r"[\s,，]", "", unicodedata.normalize("NFKC", text)).lower()


def probe_coverage(generations: dict) -> dict:
    """Manual eight-case regression probe, not a full test-set gold operand label."""
    if not PROBES.exists():
        return {"available": False}
    cases = json.loads(PROBES.read_text(encoding="utf-8"))["cases"]
    chunks = {item["chunk_id"]: item for item in rows(CHUNKS)}
    outcomes = []
    for case in cases:
        ident = case["finqa_id"]
        if ident not in generations:
            continue
        output = generations[ident]
        visible_ids = output.get("shown_chunk_ids", output.get("source_chunk_ids", []))
        text = _norm(" ".join(chunks[cid]["content_text"] for cid in visible_ids))
        facets = []
        for facet in case["facets"]:
            markers = facet.get("markers", [facet.get("marker")])
            facets.append({"name": facet["name"], "present": any(_norm(m) in text for m in markers)})
        outcomes.append({"finqa_id": ident, "all_present": all(f["present"] for f in facets),
                         "facets": facets})
    return {"available": bool(outcomes), "cases": len(outcomes),
            "full_coverage_n": sum(x["all_present"] for x in outcomes),
            "facet_total": sum(len(x["facets"]) for x in outcomes),
            "facet_present": sum(f["present"] for x in outcomes for f in x["facets"]),
            "diagnostic_only": "These 8 cases participated in V2 debugging; text markers need human semantic review.",
            "details": outcomes}


def independently_labeled_coverage(path: Path | None, generations: dict, questions: dict) -> dict:
    if path is None:
        return {"available": False, "reason": "No independently reviewed operand labels supplied."}
    cases = json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal)["cases"]
    chunks = {chunk["chunk_id"]: chunk for chunk in rows(CHUNKS)}
    details, seen = [], set()
    for case in cases:
        ident = case["finqa_id"]
        if ident in seen or ident not in questions or case.get("reviewed") is not True:
            raise ValueError("Operand labels need unique known IDs and reviewed=true")
        seen.add(ident)
        fields = case.get("fields")
        if not isinstance(fields, list) or not fields:
            raise ValueError("Reviewed operand label must enumerate all required fields")
        present = []
        if ident not in generations:
            continue
        output = generations[ident]
        visible = set(output.get("shown_chunk_ids", []))
        for field in fields:
            quote = field.get("quote")
            if not isinstance(quote, str) or not quote or not field.get("unit") or not field.get("id"):
                raise ValueError("Operand label needs id, exact quote and unit")
            value = numeric(field.get("value"))
            if value is None:
                raise ValueError("Operand label needs a numeric value")
            if value not in {Decimal(match.group().replace(",", ""))
                             for match in re.finditer(r"[-+]?\d[\d,]*(?:\.\d+)?", quote)}:
                raise ValueError("Operand label value absent from quote")
            if field.get("source") == "question":
                if quote not in questions[ident]["question"]:
                    raise ValueError("Question operand quote absent from source")
                present.append(True)
            elif field.get("source") == "report":
                accepted = field.get("acceptable_chunk_ids", [])
                if not accepted or any(cid not in chunks or quote not in chunks[cid]["content_text"]
                                       for cid in accepted):
                    raise ValueError("Report operand labels need existing chunks and exact source quote")
                present.append(bool(visible.intersection(accepted)))
            else:
                raise ValueError("Unknown operand label source")
        details.append({"finqa_id": ident, "complete": all(present),
                        "fields": len(present), "present": sum(present)})
    complete = [item for item in details if item["complete"] and numeric(questions[item["finqa_id"]]["gold"]) is not None]
    refusals = sum(generations[item["finqa_id"]]["status"] == "insufficient" for item in complete)
    return {"available": bool(details), "annotated_questions": len(details),
            "complete_questions": sum(item["complete"] for item in details),
            "complete_coverage_rate": sum(item["complete"] for item in details) / len(details) if details else None,
            "false_refusal_given_complete_evidence": {"n": refusals, "denominator": len(complete),
                                                     "rate": refusals / len(complete) if complete else None},
            "details": details}


def evaluate(input_path: Path, prefix: Path, operand_labels_path: Path | None = None) -> dict:
    questions = keyed((row for path in sorted(QUESTIONS.glob("*.jsonl")) for row in rows(path)),
                      "source questions")
    generations = keyed(rows(input_path), "V3 outputs")
    if not generations or not generations.keys() <= questions.keys():
        raise ValueError("V3 output IDs are empty or outside the source question set")
    audits = []
    for ident, output in generations.items():
        gold = questions[ident]["gold"]
        gold_number = numeric(gold)
        answer = (output.get("final") or {}).get("answer")
        predicted = numeric(answer)
        status = output["status"]
        correct = (predicted is not None and four_places(predicted) == four_places(gold_number)) \
            if gold_number is not None else answer == REFUSAL
        trace = output.get("trace") or {}
        complete = (trace.get("evidence_completeness") or {}).get("complete") is True
        plan = trace.get("field_plan") or {}
        fields = plan.get("fields", [])
        report_fields = [f for f in fields if f.get("source") == "report"]
        valid_plan = bool(fields) and plan.get("formula_id") not in (None, "unsupported")
        audits.append({
            "finqa_id": ident, "gold": str(gold), "prediction": answer,
            "status": status, "correct": bool(correct),
            "gold_numeric": gold_number is not None,
            "plan_valid": valid_plan, "planned_fields": len(fields),
            "planned_report_fields": len(report_fields),
            "verified_report_fields": len(trace.get("bindings") or {}),
            "planned_field_evidence_complete": complete,
            "source_chunk_ids": output.get("source_chunk_ids", []),
            "model_calls": sum(bool(e.get("raw_response")) for e in trace.get("events", [])),
            "model_call_attempts": sum(bool(e.get("call_attempted", e.get("raw_response"))) for e in trace.get("events", [])),
            "retrieval_rounds": sum(e.get("step") == "targeted_retrieval" for e in trace.get("events", [])),
            "error": output.get("error"), "latency_seconds": output.get("latency_seconds"),
        })
    count = len(audits)
    valid = [x for x in audits if x["plan_valid"]]
    complete = [x for x in valid if x["planned_field_evidence_complete"]]
    answered_numeric = [x for x in audits if x["status"] in ("calculated", "direct") and x["gold_numeric"]]
    manual_probe = probe_coverage(generations)
    audit_by_id = {x["finqa_id"]: x for x in audits}
    complete_probe_numeric = [audit_by_id[x["finqa_id"]] for x in manual_probe.get("details", [])
                              if x["all_present"] and audit_by_id[x["finqa_id"]]["gold_numeric"]]
    r = lambda numerator, denominator: round(numerator / denominator, 6) if denominator else None
    summary = {
        "question_count": count,
        "subset_run": count != len(questions),
        "status_counts": dict(Counter(x["status"] for x in audits)),
        "final_accuracy": r(sum(x["correct"] for x in audits), count),
        "final_correct_n": sum(x["correct"] for x in audits),
        "field_plan_valid_rate": r(len(valid), count),
        "planned_report_field_coverage": {
            "verified": sum(x["verified_report_fields"] for x in valid),
            "planned": sum(x["planned_report_fields"] for x in valid),
            "rate": r(sum(x["verified_report_fields"] for x in valid),
                      sum(x["planned_report_fields"] for x in valid)),
        },
        "complete_planned_evidence_rate": r(len(complete), len(valid)),
        # `insufficient` is emitted only when the model's own planned fields
        # remain unbound, so conditioning on complete planned bindings would
        # make this rate identically zero. The independent manual probes below
        # are the available diagnostic for refusal despite visible evidence.
        "false_refusal_given_complete_planned_evidence": {
            "available": False,
            "reason": "Agent status and planned-field completeness are mutually exclusive; use independently labeled evidence probes.",
        },
        "calculation_plan_pass_rate_given_complete_evidence":
            r(sum(x["status"] in ("calculated", "direct") for x in complete), len(complete)),
        "numeric_accuracy_given_valid_calculation": {
            "correct": sum(x["correct"] for x in answered_numeric),
            "denominator": len(answered_numeric),
            "rate": r(sum(x["correct"] for x in answered_numeric), len(answered_numeric)),
        },
        "mean_model_calls": r(sum(x["model_calls"] for x in audits), count),
        "mean_targeted_retrieval_rounds": r(sum(x["retrieval_rounds"] for x in audits), count),
        "manual_probe_coverage": manual_probe,
        "independent_operand_coverage": independently_labeled_coverage(operand_labels_path, generations, questions),
        "manual_probe_false_refusal_given_all_markers_present": {
            "n": sum(x["status"] == "insufficient" for x in complete_probe_numeric),
            "denominator": len(complete_probe_numeric),
            "rate": r(sum(x["status"] == "insufficient" for x in complete_probe_numeric),
                      len(complete_probe_numeric)),
        },
        "limitations": [
            "Planned-field coverage is mechanically validated against model-created checklists, not full manually labeled operand recall.",
            "The 8 manual probe cases were used during V2 debugging and are only regression diagnostics.",
            "The 163 questions have been inspected during V1/V2 debugging; this is exploratory accuracy, not a clean held-out generalization estimate.",
        ],
    }
    # Only compare V2 RAG against V3 RAG; attaching that delta to a no-RAG
    # generation would label the wrong baseline comparison.
    is_rag_run = all(output.get("mode") == "rag" for output in generations.values())
    if is_rag_run and V2_AUDIT.exists():
        v2 = {x["finqa_id"]: x for x in rows(V2_AUDIT)
              if x["mode"] == "rag" and x["finqa_id"] in generations}
        if len(v2) == count:
            transitions = Counter(
                ("correct" if v2[x["finqa_id"]]["correct"] else "wrong") + "_to_" +
                ("correct" if x["correct"] else "wrong") for x in audits)
            summary["paired_v2_to_v3"] = {
                "n": count, "v2_correct": sum(bool(x["correct"]) for x in v2.values()),
                "v3_correct": sum(x["correct"] for x in audits),
                "accuracy_delta": r(sum(x["correct"] for x in audits) -
                                    sum(bool(x["correct"]) for x in v2.values()), count),
                "transitions": dict(transitions),
            }
    metrics_path = prefix.with_name(prefix.name + "_metrics.json")
    audit_path = prefix.with_name(prefix.name + "_question_audit.jsonl")
    if metrics_path.exists() or audit_path.exists():
        raise FileExistsError("V3 evaluation output exists; choose another --output-prefix")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with audit_path.open("w", encoding="utf-8") as handle:
        for audit in audits:
            # Numeric audit fields can be Decimal; serialize them losslessly as strings.
            handle.write(json.dumps(audit, ensure_ascii=False, default=str) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_PREFIX)
    parser.add_argument("--operand-labels", type=Path,
                        help="Independently reviewed all-required-operand labels; see V3 review report")
    args = parser.parse_args()
    print(json.dumps(evaluate(args.input, args.output_prefix, args.operand_labels), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
