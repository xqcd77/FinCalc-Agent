"""Task 4: rerank Task 3 hybrid candidates and assemble traceable context.

The model receives only the retrieval query and evidence text. Gold answers,
reference-document labels, and external knowledge never enter reranking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
from exact_cpu_cache import ExactCPUCache

from tokenizers import Tokenizer
from table_structure import table_structure
from evidence_contract_v4 import quotation_body


PROJECT = Path(__file__).resolve().parents[2]
CHUNKS = PROJECT / "data/processed/corpus/evidence_chunks.jsonl"
RANKINGS = PROJECT / "data/processed/eval/task3_bge_m3_rankings.jsonl"
OUTPUT = PROJECT / "data/processed/eval/task4_v4_8_reranked.jsonl"
METADATA = PROJECT / "data/processed/eval/task4_v4_8_rerank_metadata.json"
MODEL_PATH = PROJECT / "models/reranker/bge-reranker-large"
MODEL_NAME = "BAAI/bge-reranker-large"
MODEL_VARIANT = "BAAI/bge-reranker-large Transformers safetensors float32 CPU"
MODEL_SHA256 = "c5ae4e262c60ed2fb507ec587358285377ec36ee0a2e6da029f0534272b06d36"
PAIR_MAX_TOKENS = 512
BATCH_SIZE = 4
CANDIDATES = 20
CONTEXT_TOP_K = 5
CONTEXT_MAX_CHARS = 2500


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_input(row: dict, chunks: dict[str, dict]) -> None:
    forbidden = {"gold", "Reference_documents", "external_knowledge"}
    if forbidden.intersection(row):
        raise ValueError(f"Evaluation/training field leaked into {row['finqa_id']}")
    candidates = row["hybrid"]
    if len(candidates) != CANDIDATES:
        raise ValueError(f"Expected {CANDIDATES} candidates for {row['finqa_id']}")
    ids = [item["chunk_id"] for item in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate hybrid chunk in {row['finqa_id']}")
    for expected_rank, item in enumerate(candidates, 1):
        chunk = chunks.get(item["chunk_id"])
        if chunk is None or chunk["doc_id"] != item["doc_id"]:
            raise ValueError(f"Unknown/mismatched chunk in {row['finqa_id']}")
        if item["rank"] != expected_rank:
            raise ValueError(f"Nonconsecutive rank in {row['finqa_id']}")


def sort_scores(candidates: list[dict], scores: list[float]) -> list[dict]:
    if len(candidates) != len(scores):
        raise ValueError("Reranker returned wrong number of scores")
    enriched = [
        {"chunk_id": item["chunk_id"], "doc_id": item["doc_id"],
         "hybrid_rank": item["rank"], "rrf_score": item["score"],
         "rerank_logit": float(score)}
        for item, score in zip(candidates, scores)
    ]
    enriched.sort(key=lambda item: (-item["rerank_logit"],
                                    item["hybrid_rank"], item["chunk_id"]))
    for rank, item in enumerate(enriched, 1):
        item["rerank_rank"] = rank
    return enriched


def format_evidence(number: int, chunk: dict) -> str:
    section = " / ".join(chunk["section_path"]) or "前置内容"
    location = f"原文第{chunk['source_start_line']}-{chunk['source_end_line']}行"
    structure = table_structure(chunk) if chunk.get("content_type") == "table" else None
    rows=",".join(map(str,structure['data_row_numbers'])) or '无数据行' if structure else ''
    if chunk["table_id"]:
        headers = ",".join(map(str, structure["header_row_numbers"])) or "未识别"
        rows = ",".join(map(str, structure["data_row_numbers"])) or "无数据行"
        inferred = structure["inferred_header_row_numbers"]
        inferred_note = f"（含自动识别的续表头行{','.join(map(str, inferred))}）" if inferred else ""
        location += f"；表格表头行{headers}{inferred_note}、数据行{rows}"
    heading = (f"[E{number}] 来源：{chunk['source_path']}；章节：{section}；"
               f"{location}\n")
    if chunk.get("content_type") == "table":
        caption = chunk.get("table_caption") or "原文无表题"
        units = "、".join(chunk.get("unit_mentions", [])) or "原表未识别单位"
        header_lines = [f"第{row.get('row')}行={' | '.join(map(str, row.get('display_cells', [])))}"
                        for row in structure["header_rows"]]
        data_lines = [f"第{row.get('row')}行={' | '.join(map(str, row.get('display_cells', [])))}"
                      for row in structure["data_rows"]]
        header_map = ("表格列标题（按原单元格位置，含续表头）：" + "；".join(header_lines) + "\n"
                      if header_lines else "表格列标题：未识别\n")
        data_map = ("表格数据行编号：" + rows + "；指标及所有期间原值见下方原始正文（不重复列值）。\n"
                    if data_lines else "表格数据行：未识别\n")
        if data_lines and not quotation_body(chunk):
            data_map='结构解析单元格（缺少原始正文，不可用作quote）：'+'；'.join(data_lines)+'\n'
        heading += f"表题：{caption}；单位：{units}\n{header_map}{data_map}"
    return heading + '原始可引用正文（quote逐字复制此处）：\n' + quotation_body(chunk)


def assemble_context(reranked: list[dict], chunks: dict[str, dict],
                     max_chars: int = CONTEXT_MAX_CHARS,
                     top_k: int = CONTEXT_TOP_K) -> tuple[str, list[dict], list[dict]]:
    """Keep whole chunks; omit a top-five chunk if adding it breaks the budget."""
    parts = []
    evidence = []
    omitted = []
    for candidate in reranked[:top_k]:
        chunk = chunks[candidate["chunk_id"]]
        number = len(evidence) + 1
        block = format_evidence(number, chunk)
        proposed = "\n\n".join([*parts, block])
        if len(proposed) > max_chars:
            omitted.append({"chunk_id": chunk["chunk_id"],
                            "reason": "context_char_budget"})
            continue
        parts.append(block)
        evidence.append({
            "citation": f"E{number}", "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"], "source_path": chunk["source_path"],
            "source_start_line": chunk["source_start_line"],
            "source_end_line": chunk["source_end_line"],
            "location_granularity": chunk["location_granularity"],
            "section_path": chunk["section_path"],
            "content_type": chunk["content_type"], "table_id": chunk["table_id"],
            "header_row_numbers": chunk["header_row_numbers"],
            "data_row_numbers": chunk["data_row_numbers"],
            "years": chunk["years"], "unit_mentions": chunk["unit_mentions"],
            "rerank_rank": candidate["rerank_rank"],
            "rerank_logit": candidate["rerank_logit"],
        })
    return "\n\n".join(parts), evidence, omitted


class BGERerankerLarge:
    """Transformers cross-encoder wrapper with the previous CPU scoring contract."""

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_PATH), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(MODEL_PATH), local_files_only=True)
        self.model.to("cpu")
        self.model.eval()
        # Fixed model/tokenizer instance, float32 CPU and truncation settings.
        # Never cache individual pairs: ordered batch shape affects logits.
        self.score_cache = ExactCPUCache(128)

    def rerank(self, query: str, documents: list[str], batch_size: int = BATCH_SIZE):
        if batch_size < 1: raise ValueError('batch_size must be positive')
        key = (query, tuple(documents), batch_size, PAIR_MAX_TOKENS, MODEL_SHA256,
               MODEL_VARIANT)
        return self.score_cache.get_or_compute(key, lambda:
            self._rerank_uncached(query, documents, batch_size))

    def _rerank_uncached(self, query, documents, batch_size):
        scores = []
        for start in range(0, len(documents), batch_size):
            batch = documents[start:start + batch_size]
            tokens = self.tokenizer(
                [query] * len(batch), batch, padding=True, truncation=True,
                max_length=PAIR_MAX_TOKENS, return_tensors="pt")
            with self.torch.inference_mode():
                logits = self.model(**tokens, return_dict=True).logits.reshape(-1).float()
            scores.extend(logits.tolist())
        return scores


def reranker():
    weight = MODEL_PATH / "model.safetensors"
    if not weight.is_file() or file_sha256(weight) != MODEL_SHA256:
        raise ValueError(
            "Reranker weight missing or changed; run scripts/download_bge_reranker_large.py")
    return BGERerankerLarge()


def run(chunks_path: Path = CHUNKS, rankings_path: Path = RANKINGS,
        output: Path = OUTPUT, metadata_path: Path = METADATA,
        limit: int | None = None, model=None) -> dict:
    chunks = {item["chunk_id"]: item for item in read_jsonl(chunks_path)}
    rankings = list(read_jsonl(rankings_path))
    if limit is not None:
        rankings = rankings[:limit]
    if len({item["finqa_id"] for item in rankings}) != len(rankings):
        raise ValueError("Duplicate question ID in task 3 rankings")
    for item in rankings:
        validate_input(item, chunks)
    if model is None:
        model = reranker()
    audit_tokenizer = Tokenizer.from_file(str(MODEL_PATH / "tokenizer.json"))
    audit_tokenizer.no_truncation()
    latencies = []
    raw_lengths = []
    context_lengths = []
    selected_counts = []
    omitted_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rankings, 1):
            candidates = row["hybrid"]
            texts = [chunks[item["chunk_id"]]["text"] for item in candidates]
            lengths = [len(audit_tokenizer.encode(row["retrieval_query"], text).ids)
                       for text in texts]
            raw_lengths.extend(lengths)
            started = time.perf_counter()
            scores = list(model.rerank(row["retrieval_query"], texts,
                                       batch_size=BATCH_SIZE))
            seconds = time.perf_counter() - started
            latencies.append(seconds)
            ordered = sort_scores(candidates, scores)
            context, evidence, omitted = assemble_context(ordered, chunks)
            omitted_count += len(omitted)
            context_lengths.append(len(context))
            selected_counts.append(len(evidence))
            record = {
                "finqa_id": row["finqa_id"], "question": row["question"],
                "company_with_period": row["company_with_period"],
                "retrieval_query": row["retrieval_query"],
                "reranked": ordered, "context_text": context,
                "selected_evidence": evidence,
                "omitted_top5": omitted,
                "context_chars": len(context),
                "rerank_seconds": round(seconds, 5),
                "pair_token_lengths": lengths,
                "pair_truncated_count": sum(length > PAIR_MAX_TOKENS for length in lengths),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 10 == 0 or index == len(rankings):
                print(f"Reranked {index}/{len(rankings)}", flush=True)
    partial.replace(output)
    summary = {
        "question_count": len(rankings), "candidate_pairs": len(raw_lengths),
        "reranker_model": MODEL_NAME, "model_variant": MODEL_VARIANT,
        "model_sha256": MODEL_SHA256,
        "chunks_sha256": file_sha256(chunks_path),
        "rankings_sha256": file_sha256(rankings_path),
        "batch_size": BATCH_SIZE, "pair_max_tokens": PAIR_MAX_TOKENS,
        "context_top_k": CONTEXT_TOP_K, "context_max_chars": CONTEXT_MAX_CHARS,
        "pair_truncated_count": sum(length > PAIR_MAX_TOKENS for length in raw_lengths),
        "max_pair_tokens_before_truncation": max(raw_lengths, default=0),
        "mean_rerank_seconds": round(statistics.mean(latencies), 4),
        "p50_rerank_seconds": round(sorted(latencies)[len(latencies) // 2], 4),
        "p95_rerank_seconds": round(sorted(latencies)[min(len(latencies)-1, int(len(latencies)*0.95))], 4),
        "mean_selected_evidence": round(statistics.mean(selected_counts), 2),
        "questions_with_fewer_than_five_evidence": sum(n < CONTEXT_TOP_K for n in selected_counts),
        "omitted_top5_chunks": omitted_count,
        "max_context_chars": max(context_lengths, default=0),
        "output": str(output),
    }
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Small smoke run; use without --limit for all 163 questions")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.limit is None:
        result = run()
    else:
        smoke = PROJECT / "reports/task4_smoke_reranked_contexts.jsonl"
        result = run(output=smoke,
                     metadata_path=PROJECT / "reports/task4_smoke_metadata.json",
                     limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
