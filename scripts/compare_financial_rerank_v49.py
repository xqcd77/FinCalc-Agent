"""Rebuild reranker scores before inspecting evaluation-only source labels.

All scoring uses question/company queries and raw corpus text only. The source
labels are loaded after scores have been written, exclusively for comparison.
Candidate unions and fusion weights are fixed; no parameter search on labels.
Existing generation/context artifacts are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/rag"))
sys.path.insert(0, str(ROOT / "src/eval"))
from rerank_context import MODEL_SHA256, PAIR_MAX_TOKENS, BATCH_SIZE


def rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def pool(row, dependency_candidates=()):
    result = {}
    for name, limit in (("hybrid", 20), ("bm25", 10), ("dense", 10)):
        for item in row[name][:limit]:
            entry = result.setdefault(item["chunk_id"],
                                      {"chunk_id": item["chunk_id"], "doc_id": item["doc_id"]})
            if entry["doc_id"] != item["doc_id"]:
                raise ValueError("Candidate document identity mismatch")
            entry[name + "_rank"] = item["rank"]
    for name in ("hybrid", "bm25", "dense"):
        full_ranks = {c["chunk_id"]: c["rank"] for c in row[name]}
        for ident, entry in result.items():
            if ident in full_ranks:
                entry[name + "_rank"] = full_ranks[ident]
    for rank, c in enumerate(dependency_candidates, 1):
        entry = result.setdefault(c["chunk_id"], {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"]})
        entry["dependency_rank"] = rank
    return list(result.values())


def dependency_search(row, chunks_list, index):
    from evidence_rerank_v49 import calculation_requirements
    from financial_semantics_v4 import aliases
    from hybrid_retrieval import bm25_search
    from select_context_v2 import target_entity
    target = target_entity(row["company_with_period"], row["question"])
    ranks, audits = {}, []
    for f in calculation_requirements(row):
        terms = list(dict.fromkeys([f["metric"], *aliases(f["metric"])]))[:4]
        query = " ".join([target, *terms, f.get("year") or "", f.get("date") or "", "数据 金额 单位"])
        found = []
        for position, score in bm25_search(index, query, top_k=100):
            c = chunks_list[position]
            if c.get("company") != row["company_with_period"]:
                continue
            found.append(c)
            if len(found) >= 5:
                break
        for rank, c in enumerate(found, 1):
            entry = ranks.setdefault(c["chunk_id"], {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "score": 0.0})
            entry["score"] += 1 / (60 + rank)
        audits.append({"metric": f["metric"], "year": f.get("year"), "date": f.get("date"),
                       "query": query, "candidate_ids": [c["chunk_id"] for c in found]})
    return sorted(ranks.values(), key=lambda c: (-c["score"], c["chunk_id"])), audits


def ordered(candidates, key):
    ranked = sorted(candidates, key=lambda c: (-key(c), c.get("hybrid_rank", 999), c["chunk_id"]))
    return [dict(c, rerank_rank=i + 1) for i, c in enumerate(ranked)]


def fixed_fusion(candidates, sources):
    """Equal-weight RRF, k=60 as in the existing retrieval implementation."""
    return ordered(candidates, lambda c: sum(1 / (60 + c[s]) for s in sources if s in c))


def score(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model_path = ROOT / "models/reranker/bge-reranker-large"
    if sha(model_path / "model.safetensors") != MODEL_SHA256:
        raise ValueError("Unexpected bge-reranker-large weights")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, dtype=torch.float32)
    model.to(args.device).eval()
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats()
    chunks_path = ROOT / "data/processed/corpus/evidence_chunks.jsonl"
    ranks_path = ROOT / "data/processed/eval/task3_bge_m3_rankings.jsonl"
    chunks_list = rows(chunks_path)
    chunks = {c["chunk_id"]: c for c in chunks_list}
    dependency_index = None
    if args.dependency_retrieval:
        from hybrid_retrieval import build_bm25
        dependency_index = build_bm25(chunks_list)
    queries = rows(ranks_path)
    if len({r["finqa_id"] for r in queries}) != len(queries):
        raise ValueError("Duplicate questions")
    results = []
    truncated = pairs = 0
    start = time.perf_counter()
    path = args.output / "large_scored_candidates.jsonl"
    if path.exists():
        raise FileExistsError(path)
    with path.with_suffix(".jsonl.partial").open("x", encoding="utf-8") as handle:
        for n, row in enumerate(queries, 1):
            if {"gold", "Reference_documents", "external_knowledge"} & row.keys():
                raise ValueError("Evaluation fields entered reranker input")
            extra, dependency_audit = dependency_search(row, chunks_list, dependency_index) if dependency_index else ([], [])
            candidates = pool(row, extra)
            texts = []
            for c in candidates:
                chunk = chunks[c["chunk_id"]]
                if chunk["doc_id"] != c["doc_id"]:
                    raise ValueError("Corpus document identity mismatch")
                texts.append(chunk["text"])
            for i in range(0, len(candidates), BATCH_SIZE):
                batch = texts[i:i + BATCH_SIZE]
                lengths = tokenizer([row["retrieval_query"]] * len(batch), batch,
                                    truncation=False, padding=False, add_special_tokens=True)
                truncated += sum(len(ids) > PAIR_MAX_TOKENS for ids in lengths["input_ids"])
                tokens = tokenizer([row["retrieval_query"]] * len(batch), batch, padding=True,
                                   truncation=True, max_length=PAIR_MAX_TOKENS,
                                   return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    logits = model(**tokens, return_dict=True).logits.reshape(-1).float().cpu().tolist()
                for c, logit in zip(candidates[i:i + BATCH_SIZE], logits):
                    c["rerank_logit"] = logit
                pairs += len(batch)
            raw_union = ordered(candidates, lambda c: c["rerank_logit"])
            large_ranks = {c["chunk_id"]: i + 1 for i, c in enumerate(raw_union)}
            for c in candidates:
                c["large_rank"] = large_ranks[c["chunk_id"]]
            original = [c for c in candidates if "hybrid_rank" in c]
            original_ranked = ordered(original, lambda c: c["rerank_logit"])
            same20_ranks = {c["chunk_id"]: i + 1 for i, c in enumerate(original_ranked)}
            for c in original:
                c["large_same20_rank"] = same20_ranks[c["chunk_id"]]
            strategies = {
                "large_raw_same20": original_ranked,
                "large_raw_union": raw_union,
                "large_hybrid_rrf_same20": fixed_fusion(original, ("large_same20_rank", "hybrid_rank")),
                "large_bm25_dense_rrf_union": fixed_fusion(candidates, ("large_rank", "bm25_rank", "dense_rank")),
            }
            if args.dependency_retrieval:
                strategies["large_dependency_rrf_union"] = fixed_fusion(
                    candidates, ("large_rank", "bm25_rank", "dense_rank", "dependency_rank"))
            record = {"finqa_id": row["finqa_id"], "question": row["question"],
                      "company_with_period": row["company_with_period"],
                      "retrieval_query": row["retrieval_query"], "strategies": strategies,
                      "dependency_retrieval_audit": dependency_audit}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            results.append(record)
            if n % 10 == 0 or n == len(queries):
                print(json.dumps({"scored": n, "total": len(queries), "pairs": pairs,
                                  "seconds": round(time.perf_counter() - start, 2)}, ensure_ascii=True), flush=True)
    path.with_suffix(".jsonl.partial").replace(path)
    meta = {"questions": len(results), "candidate_pairs": pairs, "model": "BAAI/bge-reranker-large",
            "model_sha256": MODEL_SHA256, "dtype": "float32", "device": args.device,
            "batch_size": BATCH_SIZE, "pair_max_tokens": PAIR_MAX_TOKENS,
            "truncated_pairs": truncated, "fusion_k": 60, "fusion_weights": "equal",
            "candidate_pool": "hybrid20 union bm25top10 union densetop10",
            "dependency_retrieval": args.dependency_retrieval,
            "dependency_module_sha256": sha(ROOT / "src/rag/evidence_rerank_v49.py") if args.dependency_retrieval else None,
            "chunks_sha256": sha(chunks_path), "rankings_sha256": sha(ranks_path),
            "seconds": time.perf_counter() - start,
            "evaluation_labels_used_for_scoring": False,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None}
    (args.output / "score_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()
    return results


def compare(args, scores):
    # Evaluation-only load occurs strictly after complete model scoring.
    labels = json.loads((ROOT / "data/processed/eval/evidence_labels_v48.json").read_text(encoding="utf-8"))
    cases = {c["finqa_id"]: c for c in labels["cases"]
             if c.get("certified") is True and any(f["source"] == "report" for f in c["fields"])}
    baseline = {r["finqa_id"]: r for r in rows(ROOT / "data/processed/eval/task3_bge_m3_rankings.jsonl")}
    old = {r["finqa_id"]: r for r in rows(ROOT / "data/processed/eval/task4_v4_8_contexts.jsonl")}
    methods = {name: {i: r[name] for i, r in baseline.items()} for name in ("bm25", "dense", "hybrid")}
    methods["old_cached_base_int8_rerank"] = {i: r["reranked"] for i, r in old.items()}
    for name in scores[0]["strategies"]:
        methods[name] = {r["finqa_id"]: r["strategies"][name] for r in scores}
    metrics, audits = {}, []
    for name, rankings in methods.items():
        total = hits = complete = 0
        rr = 0.0
        for ident, c in cases.items():
            fs = [f for f in c["fields"] if f["source"] == "report"]
            ids = [x["chunk_id"] for x in rankings[ident][:10]]
            covered = [bool(set(ids) & set(f["acceptable_chunk_ids"])) for f in fs]
            total += len(fs); hits += sum(covered); complete += all(covered)
            relevant = {x for f in fs for x in f["acceptable_chunk_ids"]}
            pos = next((i + 1 for i, x in enumerate(ids) if x in relevant), None)
            rr += 1 / pos if pos else 0
            audits.append({"method": name, "finqa_id": ident, "covered": sum(covered),
                           "required": len(fs), "complete": all(covered), "first_rank": pos})
        metrics[name] = {"labelled_questions": len(cases), "required_numeric_inputs": total,
                         "numeric_hits_at10": hits, "numeric_recall_at10": hits / total,
                         "complete_questions_at10": complete, "complete_evidence_rate_at10": complete / len(cases),
                         "chunk_mrr_at10": rr / len(cases)}
    report = {"metrics": metrics, "scope": "Same automatic labelled subset; development diagnostics, not full163 relevance gold",
              "human_reviewed_questions": sum(c.get("reviewed_by_human") is True for c in cases.values()),
              "generation_accuracy_measured": False, "parameters_tuned_on_labels": False}
    (args.output / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "question_audit.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in audits), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


def with_coverage_guard(args, scores):
    from evidence_rerank_v49 import guard_order
    from select_context_v4 import issuer_names
    from table_delivery_v48 import CompanionIndex
    chunks = CompanionIndex({c["chunk_id"]: c for c in rows(ROOT / "data/processed/corpus/evidence_chunks.jsonl")})
    names = issuer_names(chunks)
    guard_records = []
    for record in scores:
        ranked = record["strategies"].get("large_dependency_rrf_union",
                    record["strategies"]["large_bm25_dense_rrf_union"])
        ordered, audit = guard_order(record, ranked, chunks, names=names)
        record["strategies"]["coverage_guarded_large_rrf_union"] = ordered
        guard_records.append({"finqa_id": record["finqa_id"], **audit})
    (args.output / "coverage_guard_audit.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in guard_records), encoding="utf-8")
    (args.output / "guarded_scored_candidates.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in scores), encoding="utf-8")
    return scores


def validate_replay(args, scores):
    if not args.scores:
        return
    metadata = json.loads((args.scores.parent / "score_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("model_sha256") != MODEL_SHA256:
        raise ValueError("Replay scores came from a different reranker")
    if metadata.get("dependency_retrieval") and metadata.get("dependency_module_sha256") != sha(ROOT / "src/rag/evidence_rerank_v49.py"):
        raise ValueError("Stale dependency-query policy; regenerate the scored candidate pool")
    for key, path in (("chunks_sha256", ROOT / "data/processed/corpus/evidence_chunks.jsonl"),
                      ("rankings_sha256", ROOT / "data/processed/eval/task3_bge_m3_rankings.jsonl")):
        if metadata.get(key) != sha(path):
            raise ValueError("Stale score artifact: " + key)
    queries = {r["finqa_id"]: r for r in rows(ROOT / "data/processed/eval/task3_bge_m3_rankings.jsonl")}
    if len(scores) != len(queries) or {r["finqa_id"] for r in scores} != set(queries):
        raise ValueError("Replay question set mismatch")
    for r in scores:
        if any(r[k] != queries[r["finqa_id"]][k]
               for k in ("question", "company_with_period", "retrieval_query")):
            raise ValueError("Replay query changed")


def export_artifacts(args, scores):
    if not args.export_reranked and not args.export_contexts:
        return
    if not args.coverage_guard:
        raise ValueError("Production-compatible export requires explicit coverage guard")
    from select_context_v4 import build_context
    from tokenizer_v4 import native_tokenizer
    from table_delivery_v48 import CompanionIndex
    chunks = CompanionIndex({c["chunk_id"]: c for c in rows(ROOT / "data/processed/corpus/evidence_chunks.jsonl")})
    records = []
    score_path = args.scores or args.output / "large_scored_candidates.jsonl"
    score_digest = sha(score_path)
    for r in scores:
        record = {k: r[k] for k in ("finqa_id", "question", "company_with_period", "retrieval_query")}
        record["reranked"] = [dict(c, hybrid_rank=c.get("hybrid_rank"), rrf_score=sum(
            1 / (60 + c[s]) for s in ("large_rank", "bm25_rank", "dense_rank") if s in c))
            for c in r["strategies"]["coverage_guarded_large_rrf_union"]]
        record["ranking_provenance"] = {
            "model": "BAAI/bge-reranker-large", "model_sha256": MODEL_SHA256,
            "precision": "float32", "strategy": "coverage_guarded_large_rrf_union",
            "candidate_pool": "hybrid20 union bm25top10 union densetop10 plus optional dependency query candidates",
            "score_artifact_sha256": score_digest,
            "requires_eval_labels": False,
        }
        if {"gold", "Reference_documents", "external_knowledge"} & record.keys():
            raise ValueError("Evaluation-only fields in export")
        ids = [c["chunk_id"] for c in record["reranked"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate exported candidates")
        records.append(record)
    for path in (args.export_reranked, args.export_contexts):
        if path and path.exists():
            raise FileExistsError(path)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
    if args.export_reranked:
        args.export_reranked.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    if args.export_contexts:
        tok = native_tokenizer(args.tokenizer_path)
        with args.export_contexts.open("x", encoding="utf-8") as handle:
            for r in records:
                context = build_context(r, chunks, tok)
                handle.write(json.dumps(context, ensure_ascii=False) + "\n")
        print(json.dumps({"exported_contexts": len(records), "output": str(args.export_contexts)}, ensure_ascii=True), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--scores", type=Path, help="Already scored artifacts for evaluation-only replay")
    p.add_argument("--coverage-guard", action="store_true",
                   help="Apply question-only numeric requirement protection before evaluation")
    p.add_argument("--export-reranked", type=Path)
    p.add_argument("--export-contexts", type=Path)
    p.add_argument("--tokenizer-path", type=Path)
    p.add_argument("--dependency-retrieval", action="store_true",
                   help="Add same-issuer candidates from question-derived financial input queries")
    p.add_argument("--skip-comparison", action="store_true", help="Export without local evaluation annotations")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = rows(args.scores) if args.scores else score(args)
    validate_replay(args, results)
    if args.coverage_guard:
        results = with_coverage_guard(args, results)
    export_artifacts(args, results)
    if not args.skip_comparison:
        compare(args, results)


if __name__ == "__main__":
    main()
