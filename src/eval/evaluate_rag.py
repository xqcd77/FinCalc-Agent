"""Offline evaluation of the fixed FinanceComplexQA Chinese report runs.

Reads gold only here, after retrieval and generation have completed. Writes a
per-question audit file plus aggregate metrics. No model or GPU is required.
"""

from __future__ import annotations

import csv
from collections import Counter
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
from pathlib import Path
import re
from statistics import median


PROJECT = Path(__file__).resolve().parents[2]
QUESTIONS = PROJECT / "data/raw/financecomplexqa/questions/corporate_reports"
EVAL = PROJECT / "data/processed/eval"
REPORTS = PROJECT / "reports"
QUANTUM = Decimal("0.0001")
REFUSAL = "缺少相关数据，无法求出答案"
NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")


def rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line, parse_float=Decimal)
                except (json.JSONDecodeError, InvalidOperation) as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def keyed(items, label: str):
    result = {}
    for item in items:
        ident = item.get("finqa_id")
        if not isinstance(ident, str) or ident in result:
            raise ValueError(f"Missing or duplicate finqa_id in {label}: {ident}")
        result[ident] = item
    return result


def numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, Decimal)):
        number = Decimal(str(value))
    elif isinstance(value, str):
        clean = value.strip()
        if not NUMBER.fullmatch(clean):
            return None
        number = Decimal(clean)
    else:
        return None
    return number if number.is_finite() else None


def four_places(number: Decimal):
    return number.quantize(QUANTUM, rounding=ROUND_HALF_UP)


def unique_docs(candidates):
    seen = set()
    result = []
    for candidate in candidates:
        doc_id = candidate["doc_id"]
        if doc_id not in seen:
            result.append(doc_id)
            seen.add(doc_id)
    return result


def rank_metrics(questions, ranked, field):
    hits = Counter()
    reciprocal = Decimal(0)
    ranks = {}
    for ident, question in questions.items():
        target = Path(question["Reference_documents"]).stem
        docs = unique_docs(ranked[ident][field])
        rank = docs.index(target) + 1 if target in docs else None
        ranks[ident] = rank
        for k in (1, 3, 5, 10, 20):
            hits[k] += rank is not None and rank <= k
        if rank is not None and rank <= 10:
            reciprocal += Decimal(1) / rank
    total = len(questions)
    return {
        "hit_at_1": hits[1] / total,
        "hit_at_3": hits[3] / total,
        "hit_at_5": hits[5] / total,
        "hit_at_10": hits[10] / total,
        "hit_at_20": hits[20] / total,
        "mrr_at_10": round(float(reciprocal / total), 6),
        "ranks": ranks,
    }


def summarize_generation(records):
    total = len(records)
    statuses = Counter(record["status"] for record in records)
    numeric_gold = [r for r in records if r["gold_kind"] == "numeric"]
    refused_gold = [r for r in records if r["gold_kind"] == "refusal"]
    numeric_preds = [r for r in numeric_gold if r["prediction_numeric"] is not None]
    abs_errors = sorted(Decimal(r["absolute_error"]) for r in numeric_preds)
    relative_errors = sorted(
        Decimal(r["relative_error"]) for r in numeric_preds
        if r["relative_error"] is not None
    )
    exact = sum(r["correct"] for r in records)
    near_1e4 = sum(Decimal(r["absolute_error"]) <= Decimal("0.0001") for r in numeric_preds)
    near_1e3 = sum(Decimal(r["absolute_error"]) <= Decimal("0.001") for r in numeric_preds)
    latencies = sorted(float(r["latency_seconds"]) for r in records)
    return {
        "n": total,
        "gold_numeric_n": len(numeric_gold),
        "gold_refusal_n": len(refused_gold),
        "correct_n": exact,
        "answer_accuracy": round(exact / total, 6),
        "numeric_correct_n": sum(r["correct"] for r in numeric_gold),
        "numeric_accuracy": round(sum(r["correct"] for r in numeric_gold) / len(numeric_gold), 6),
        "refusal_correct_n": sum(r["correct"] for r in refused_gold),
        "numeric_prediction_n": len(numeric_preds),
        "absolute_error_le_0_0001_n": near_1e4,
        "absolute_error_le_0_001_n": near_1e3,
        "absolute_error_median_on_numeric_predictions": str(median(abs_errors)) if abs_errors else None,
        "relative_error_median_on_nonzero_numeric_gold": str(median(relative_errors)) if relative_errors else None,
        "status_counts": dict(statuses),
        "final_answer_format_valid_n": sum(r["final_answer_format_valid"] for r in records),
        "strict_json_answer_value_type_valid_n": sum(r["strict_json_answer_value_type_valid"] for r in records),
        "raw_json_object_valid_n": sum(r["raw_json_object_valid"] for r in records),
        "trace_mechanically_valid_n": sum(r["trace_mechanically_valid"] for r in records),
        "latency_p50_seconds": median(latencies),
        "latency_p95_seconds": latencies[(95 * total + 99) // 100 - 1],
    }


def main():
    question_list = [row for path in sorted(QUESTIONS.glob("*.jsonl")) for row in rows(path)]
    questions = keyed(question_list, "source questions")
    ranking = keyed(rows(EVAL / "task3_retrieval_rankings.jsonl"), "Task 3")
    context = keyed(rows(EVAL / "task4_reranked_contexts.jsonl"), "Task 4")
    generations = {
        mode: keyed(rows(EVAL / f"task5_{mode}_generation.jsonl"), f"Task 5 {mode}")
        for mode in ("rag", "no_rag")
    }
    for label, dataset in (("Task 3", ranking), ("Task 4", context), *generations.items()):
        if dataset.keys() != questions.keys():
            raise ValueError(f"ID set differs from gold: {label}")

    retrieval = {}
    document_ranks = {}
    for name, source, field in (
        ("bm25", ranking, "bm25"), ("dense", ranking, "dense"),
        ("hybrid", ranking, "hybrid"), ("reranked", context, "reranked"),
    ):
        result = rank_metrics(questions, source, field)
        document_ranks[name] = result.pop("ranks")
        retrieval[name] = result

    audited = []
    for ident, question in questions.items():
        gold = question["gold"]
        gold_number = numeric(gold)
        if gold_number is None and gold != REFUSAL:
            raise ValueError(f"Unknown gold answer type for {ident}: {gold!r}")
        target_doc = Path(question["Reference_documents"]).stem
        chosen = context[ident]["selected_evidence"]
        citation_map = {item["citation"]: item for item in chosen}
        selected_docs = unique_docs(chosen)
        for mode in ("rag", "no_rag"):
            generated = generations[mode][ident]
            if generated.get("mode") != mode:
                raise ValueError(f"Mode mismatch: {ident} {mode}")
            final = generated.get("final")
            answer = final.get("answer") if isinstance(final, dict) else None
            prediction = numeric(answer)
            status = generated["status"]
            gold_kind = "numeric" if gold_number is not None else "refusal"
            correct = (
                prediction is not None and four_places(prediction) == four_places(gold_number)
                if gold_number is not None else answer == REFUSAL
            )
            absolute = abs(prediction - gold_number) if prediction is not None and gold_number is not None else None
            relative = absolute / abs(gold_number) if absolute is not None and gold_number != 0 else None
            try:
                raw = json.loads(generated["raw_response"]) if generated.get("raw_response") is not None else None
            except json.JSONDecodeError:
                raw = None
            raw_valid = isinstance(raw, dict)
            final_valid = isinstance(final, dict) and set(final) == {"answer"} and (
                prediction is not None or answer == REFUSAL
            )
            strict_value_type = isinstance(final, dict) and set(final) == {"answer"} and (
                (isinstance(answer, (int, Decimal)) and not isinstance(answer, bool) and prediction is not None)
                or answer == REFUSAL
            )
            refs = []
            if status == "calculated":
                refs = [item.get("source_ref") for item in generated["trace"]["operands"]]
            elif status == "direct":
                refs = generated["trace"].get("citations", [])
            refs_valid = all(ref == "question" or (mode == "rag" and ref in citation_map) for ref in refs)
            mechanical = status in ("calculated", "direct") and refs_valid
            if mode == "rag" and status == "calculated":
                mechanical = mechanical and all(
                    item.get("source_ref") == "question" or item.get("source_ref") in citation_map
                    for item in generated["trace"]["operands"]
                )
            if correct:
                category = "correct"
            elif status == "api_error":
                category = "api_error"
            elif status == "invalid_plan":
                category = "invalid_plan"
            elif status == "insufficient":
                category = "wrong_refusal"
            elif gold_kind == "refusal":
                category = "false_answer"
            elif prediction is None:
                category = "non_numeric_answer"
            else:
                category = "numeric_mismatch"
            audited.append({
                "finqa_id": ident, "mode": mode, "question": question["question"].split("### 输出格式")[0].strip(),
                "reference_doc_id": target_doc, "gold": str(gold), "gold_kind": gold_kind,
                "prediction": answer, "prediction_numeric": str(prediction) if prediction is not None else None,
                "status": status, "correct": correct, "error_category": category,
                "absolute_error": str(absolute) if absolute is not None else None,
                "relative_error": str(relative) if relative is not None else None,
                "final_answer_format_valid": final_valid,
                "strict_json_answer_value_type_valid": strict_value_type,
                "raw_json_object_valid": raw_valid,
                "trace_mechanically_valid": mechanical,
                "reference_doc_in_context": target_doc in selected_docs if mode == "rag" else None,
                "reference_doc_rank_bm25": document_ranks["bm25"][ident],
                "reference_doc_rank_dense": document_ranks["dense"][ident],
                "reference_doc_rank_hybrid": document_ranks["hybrid"][ident],
                "reference_doc_rank_reranked": document_ranks["reranked"][ident],
                "selected_chunk_ids": generated["source_chunk_ids"],
                "selected_citations": refs,
                "trace": generated["trace"], "error": generated.get("error"),
                "latency_seconds": generated["latency_seconds"],
            })

    by_mode = {mode: [row for row in audited if row["mode"] == mode] for mode in generations}
    rag_by_id = {row["finqa_id"]: row for row in by_mode["rag"]}
    no_rag_by_id = {row["finqa_id"]: row for row in by_mode["no_rag"]}
    paired = Counter(
        ("correct" if no_rag_by_id[ident]["correct"] else "wrong") + "_to_" +
        ("correct" if rag_by_id[ident]["correct"] else "wrong")
        for ident in questions
    )
    rag = by_mode["rag"]
    results = {
        "evaluation_protocol": {
            "gold_question_count": len(questions),
            "numeric_match": "Decimal; both values rounded half-up to 4 decimal places",
            "refusal_match": f"exactly {REFUSAL}",
            "absolute_error_tolerances": ["0.0001", "0.001"],
            "retrieval_level": "unique document IDs; reference filename stem as positive label",
            "format_policy": "parseable numeric strings accepted for scoring; strict JSON number type reported separately",
            "manual_semantic_evidence_check": "not included in automatic score",
        },
        "retrieval": retrieval,
        "generation": {mode: summarize_generation(records) for mode, records in by_mode.items()},
        "paired": dict(paired),
        "rag_failure_categories": dict(Counter(row["error_category"] for row in rag)),
        "rag_reference_doc_in_context_n": sum(row["reference_doc_in_context"] for row in rag),
        "rag_wrong_with_reference_doc_in_context_n": sum(
            not row["correct"] and row["reference_doc_in_context"] for row in rag
        ),
    }
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / "task6_question_audit.jsonl").open("w", encoding="utf-8") as handle:
        for row in audited:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    fields = [
        "finqa_id", "mode", "status", "correct", "error_category", "gold", "prediction",
        "absolute_error", "relative_error", "reference_doc_in_context",
        "reference_doc_rank_hybrid", "reference_doc_rank_reranked", "error",
    ]
    with (REPORTS / "task6_question_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in audited)
    with (REPORTS / "task6_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
