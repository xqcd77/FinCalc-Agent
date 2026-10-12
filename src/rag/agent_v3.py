"""Bounded plan/search/bind/calculate agent for Chinese financial-report QA.

Generation uses an OpenAI-compatible local Qwen endpoint. Search and reranking
use the project's frozen local indexes. This script never reads gold answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
import pickle
from pathlib import Path
import re
import time
import threading
from urllib import error, request

PROJECT = Path(__file__).resolve().parents[2]

import numpy as np

from field_plan_v3 import (
    BIND_SYSTEM, PLAN_SYSTEM, calculate, field_slot, validate_binding, validate_plan,
)
from generate_calculate import MISSING, MODEL, NUMBER, read_jsonl
from prompt_budget import (count_chat_tokens, count_text_tokens, ensure_prompt_fits,
                           load_tokenizer, resolve_tokenizer_path)
from select_context_v2 import Slot, slot_match_info, target_entity
from hybrid_retrieval import (
    CHUNKS, INDEX, MODEL_CACHE, MODEL_PATH, bm25_search, dense_search, embed_queries,
    encoder, rrf, source_hash, validate_index, retrieval_view,
)
from rerank_context import format_evidence, reranker


INPUT = PROJECT / "data/processed/eval/task4_v3_contexts.jsonl"
OUTPUT = PROJECT / "data/processed/eval/task5_v3_rag_generation.jsonl"
VERSION = "rag-agent-v3-review-20260930"
MAX_EVIDENCE_TOKENS = 6000
MAX_EVIDENCE_BLOCKS = 10
MAX_RETRIEVAL_ROUNDS = 2
MAX_BIND_INPUT_TOKENS = 2600
MAX_PLAN_INPUT_TOKENS = 2100
MODEL_CONTEXT_TOKENS = 3200
QUERY_RANK_CACHE_SIZE = 64


def call_model_v3(endpoint: str, messages: list[dict], timeout: int) -> tuple[str, dict]:
    """Use enough output budget for multi-field JSON plans and binding lists."""
    max_tokens = 1024 if messages[0]["content"] == PLAN_SYSTEM else 512
    payload = {"model": MODEL, "messages": messages, "temperature": 0,
               "max_tokens": max_tokens, "response_format": {"type": "json_object"}}
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(endpoint.rstrip("/") + "/chat/completions", body,
                          {"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            data = json.load(response)
    except error.HTTPError as exc:
        raise RuntimeError(f"Model HTTP {exc.code}: {exc.read(500).decode('utf-8', 'replace')}") from exc
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise ValueError("Model response content is not text")
    return content, data.get("usage", {})


def _parse_object(raw: str) -> dict:
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("Model did not return a JSON object")
    return obj


def _forbid_eval_fields(row: dict) -> None:
    if {"gold", "Reference_documents", "external_knowledge"} & row.keys():
        raise ValueError("Evaluation labels leaked into V3 agent")
    if not isinstance(row.get("question"), str) or not isinstance(row.get("selected_evidence"), list):
        raise ValueError("Malformed V3 input")


class EvidenceState:
    def __init__(self, row: dict, chunks: dict[str, dict], tokenizer):
        _forbid_eval_fields(row)
        self.row = {**row, "selected_evidence": list(row["selected_evidence"])}
        self.entity = row.get("target_entity") or target_entity(
            row["company_with_period"], row["question"])
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.by_ref = {}
        self.seen_ids = set()
        for item in self.row["selected_evidence"]:
            chunk = chunks[item["chunk_id"]]
            if chunk["doc_id"] != item["doc_id"]:
                raise ValueError("Initial evidence document mismatch")
            if chunk["company"] != row["company_with_period"]:
                raise ValueError("Initial evidence is from another company's report")
            self.by_ref[item["citation"]] = chunk
            self.seen_ids.add(item["chunk_id"])
        if len(self.by_ref) != len(self.row["selected_evidence"]):
            raise ValueError("Duplicate initial evidence references")
        if list(self.by_ref) != [f"E{i}" for i in range(1, len(self.by_ref) + 1)]:
            raise ValueError("Initial citations must be consecutive")
        if len(self.seen_ids) != len(self.by_ref):
            raise ValueError("Duplicate initial evidence chunks")
        # Binding provenance and displayed views must refer to the same corpus text.
        self.row["context_text"] = "\n\n".join(
            format_evidence(int(ref[1:]), chunk) for ref, chunk in self.by_ref.items())

    def add(self, candidate: dict) -> str | None:
        chunk = candidate["chunk"]
        source_text = chunk.get("content_text", "") + chunk.get("text", "")
        target_match = (chunk["company"] == self.row["company_with_period"] and
                        (self.row["company_with_period"] != "未识别机构" or
                         self.entity in source_text))
        if chunk["chunk_id"] in self.seen_ids or not target_match:
            return None
        if len(self.row["selected_evidence"]) >= MAX_EVIDENCE_BLOCKS:
            return None
        ref = f"E{len(self.row['selected_evidence']) + 1}"
        block = format_evidence(len(self.row["selected_evidence"]) + 1, chunk)
        proposed = "\n\n".join(filter(None, (self.row["context_text"], block)))
        if count_text_tokens(self.tokenizer, proposed) > MAX_EVIDENCE_TOKENS:
            return None
        self.row["context_text"] = proposed
        self.row["selected_evidence"].append({
            "citation": ref, "chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
            "source_path": chunk["source_path"], "source_start_line": chunk["source_start_line"],
            "source_end_line": chunk["source_end_line"], "content_type": chunk["content_type"],
            "retrieval_query": candidate["query"], "retrieval_round": candidate["round"],
        })
        self.by_ref[ref] = chunk
        self.seen_ids.add(chunk["chunk_id"])
        return ref

    def candidate_refs(self, field: dict) -> list[str]:
        """Weak signal for a retry, never proof that a field is actually bound."""
        metric_terms = [field["metric"], *field.get("aliases", [])]
        result = []
        for ref, chunk in self.by_ref.items():
            body = " ".join((chunk.get("table_caption", "") or "",
                             chunk.get("content_text", ""), chunk.get("text", "")))
            if (chunk.get("company") != self.row["company_with_period"] and
                    self.entity not in body):
                continue
            if field.get("year") and field["year"] not in body and field["year"] not in chunk.get("years", []):
                continue
            if any(term.lower() in body.lower() for term in metric_terms) and NUMBER.search(body):
                result.append(ref)
        return result

    def view_for_field(self, field: dict, hint: str = "") -> tuple[str, list[str], int]:
        """Build a token-bounded, field-specific view in source order."""
        aliases = [field["metric"], *field.get("aliases", [])]
        candidates: dict[str, list[tuple[int, str, dict, dict]]] = {}
        for item in self.row["selected_evidence"]:
            ref = item["citation"]
            chunk = self.by_ref[ref]
            body = " ".join((chunk.get("table_caption", "") or "",
                             chunk.get("content_text", ""), chunk.get("text", "")))
            metric_hit = any(a.lower() in body.lower() for a in aliases)
            year_hit = not field.get("year") or field["year"] in body or field["year"] in chunk.get("years", [])
            table_info = slot_match_info(field_slot(field),
                                         chunk, self.row["company_with_period"], self.entity)
            table_hit = bool(table_info.get("matched"))
            score = 6 * metric_hit + 3 * year_hit + 2 * table_hit
            score += int(bool(re.search(r"\d[\d,]*\.\d+", body)))
            if item.get("retrieval_round") and metric_hit:
                score += 2
            candidates.setdefault(chunk.get("doc_id", ""), []).append((score, ref, chunk, item))
        # Choose relevant blocks before restoring source order. Otherwise an
        # irrelevant early paragraph can crowd out a useful late table.
        ordered = sorted((entry for group in candidates.values() for entry in group),
                         key=lambda x: (-x[0], int(x[1][1:])))
        chosen = []
        parts, refs, used_tokens = [], [], 0
        for entry in ordered:
            _, ref, chunk, _ = entry
            block = format_evidence(int(ref[1:]), chunk)
            proposed_parts = [*parts, block]
            proposed = "\n\n".join(proposed_parts)
            user = (f"目标公司：{self.entity}\n问题：{self.row['question']}\n"
                    f"只绑定此字段：{json.dumps(field, ensure_ascii=False)}\n"
                    f"证据视图：\n{proposed or '无可显示的证据'}\n{hint}")
            used_tokens = count_chat_tokens(self.tokenizer, [
                {"role": "system", "content": BIND_SYSTEM},
                {"role": "user", "content": user},
            ])
            if used_tokens > MAX_BIND_INPUT_TOKENS:
                continue
            parts = proposed_parts
            refs.append(ref)
            chosen.append(entry)
        chosen.sort(key=lambda x: (x[2].get("doc_id", ""),
                                  x[2].get("source_start_line", 0),
                                  x[2].get("source_end_line", 0), int(x[1][1:])))
        view = "\n\n".join(format_evidence(int(ref[1:]), chunk)
                              for _, ref, chunk, _ in chosen)
        user = (f"目标公司：{self.entity}\n问题：{self.row['question']}\n"
                f"只绑定此字段：{json.dumps(field, ensure_ascii=False)}\n"
                f"证据视图：\n{view or '无可显示的证据'}\n{hint}")
        used_tokens = count_chat_tokens(self.tokenizer, [
            {"role": "system", "content": BIND_SYSTEM}, {"role": "user", "content": user}])
        return view, [entry[1] for entry in chosen], used_tokens


class TargetedRetriever:
    """Search the existing BM25 + BGE index and rerank within the target report."""

    def __init__(self, chunks: list[dict] | None = None, model=None, ranker=None):
        self.chunks = chunks or list(read_jsonl(CHUNKS))
        meta = json.loads((INDEX / "metadata.json").read_text(encoding="utf-8"))
        with (INDEX / "bm25.pkl").open("rb") as handle:
            self.bm25 = pickle.load(handle)
        self.vectors = np.load(INDEX / "dense_vectors.npy", mmap_mode="r")
        validate_index(meta,CHUNKS,self.vectors)
        if len(self.chunks) != self.bm25["n"] or len(self.chunks) != len(self.vectors):
            raise ValueError("Search index length mismatch")
        self.model = model or encoder(MODEL_CACHE)
        self.ranker = ranker or reranker()
        # A failed binding can trigger another search with the same exact query.
        # Cache only the query-dependent global rankings; company/seen filtering,
        # RRF, field verification, and reranking still run for each search.
        self._query_rank_cache = OrderedDict()
        self.query_rank_cache_hits = 0
        self.query_rank_cache_misses = 0
        # The rank cache, embedder and reranker are shared by concurrent question
        # workers. Keep each retrieval transaction isolated and deterministic.
        self._search_lock = threading.RLock()

    def _query_rankings(self, query: str):
        cached = self._query_rank_cache.get(query)
        if cached is not None:
            self._query_rank_cache.move_to_end(query)
            self.query_rank_cache_hits += 1
            return cached
        self.query_rank_cache_misses += 1
        sparse = tuple(bm25_search(self.bm25, query, top_k=len(self.chunks)))
        vector = embed_queries(self.model, [query])[0]
        dense = tuple(dense_search(self.vectors, vector, top_k=len(self.chunks)))
        cached = (sparse, dense)
        self._query_rank_cache[query] = cached
        if len(self._query_rank_cache) > QUERY_RANK_CACHE_SIZE:
            self._query_rank_cache.popitem(last=False)
        return cached

    def search(self, field: dict, row: dict, seen: set[str], round_number: int) -> list[dict]:
        with self._search_lock:
            return self._search_locked(field, row, seen, round_number)

    def _search_locked(self, field: dict, row: dict, seen: set[str], round_number: int) -> list[dict]:
        company = row["company_with_period"]
        entity = row.get("target_entity") or target_entity(company, row["question"])
        query = " ".join(filter(None, [entity, field.get("date") or field.get("year"), field["metric"],
                                      " ".join(field.get("aliases", [])[:2])]))
        eligible = {i for i, chunk in enumerate(self.chunks)
                    if chunk["company"] == company and
                    (company != "未识别机构" or entity in chunk.get("text", ""))
                    and chunk["chunk_id"] not in seen}
        if not eligible:
            return []
        sparse_ranked, dense_ranked = self._query_rankings(query)
        sparse = [pair for pair in sparse_ranked if pair[0] in eligible][:100]
        dense = [pair for pair in dense_ranked if pair[0] in eligible][:100]
        fused = rrf([sparse, dense], top_k=160)
        candidates = [(index, score) for index, score in fused
                      if (self.chunks[index]["company"] == company and
                          (company != "未识别机构" or
                           entity in self.chunks[index].get("text", "")))
                      and self.chunks[index]["chunk_id"] not in seen][:20]
        if not candidates:
            return []
        scores = list(self.ranker.rerank(query, [self.chunks[i]["text"] for i, _ in candidates],
                                         batch_size=4))
        ranked = sorted(zip(candidates, scores), key=lambda x: (
            -int(bool(slot_match_info(field_slot(field), self.chunks[x[0][0]], company, entity).get("matched"))),
            -float(x[1]), -x[0][1]))
        return [{"chunk": self.chunks[index], "query": query, "round": round_number,
                 "rrf_score": float(rrf_score), "rerank_score": float(score)}
                for ((index, rrf_score), score) in ranked]


class RagAgentV3:
    def __init__(self, ask, retriever, chunks: dict[str, dict], tokenizer,
                 max_retrieval_rounds: int = MAX_RETRIEVAL_ROUNDS):
        self.ask = ask
        self.retriever = retriever
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_retrieval_rounds = max_retrieval_rounds

    def _ask(self, system: str, user: str, events: list[dict], step: str) -> dict:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        is_plan = system == PLAN_SYSTEM
        prompt_budget = MAX_PLAN_INPUT_TOKENS if is_plan else MAX_BIND_INPUT_TOKENS
        output_budget = 1024 if is_plan else 512
        if prompt_budget + output_budget > MODEL_CONTEXT_TOKENS:
            raise ValueError("V3 prompt plus generation reserve exceeds model context")
        prompt_tokens = ensure_prompt_fits(self.tokenizer, messages, prompt_budget, stage=step)
        event = {"step": step, "call_attempted": True,
                 "prompt_tokens": prompt_tokens, "prompt_token_budget": prompt_budget}
        events.append(event)
        try:
            raw, usage = self.ask(messages)
        except (RuntimeError, OSError) as exc:
            event["error"] = f"{type(exc).__name__}: {exc}"
            raise
        event.update(raw_response=raw, usage=usage)
        return _parse_object(raw)

    def _plan(self, state: EvidenceState, events: list[dict]) -> dict:
        question = state.row["question"]
        user = f"目标公司：{state.entity}\n问题：{question}"
        error = None
        for attempt in range(2):
            try:
                proposal = self._ask(PLAN_SYSTEM, user + (f"\n上次规划错误：{error}。请修正。" if error else ""),
                                     events, "plan" if attempt == 0 else "plan_repair")
                valid = validate_plan(proposal, question)
                if valid["formula_id"] == "unsupported":
                    reason = valid["reason"]
                    if attempt == 0 or re.search(r"缺少|未找到|未提供|没有.*(?:数据|证据|数值)", reason):
                        error = "缺少报告数值不等于题意不明；请列出待检索字段并选择公式。"
                        events[-1]["validation_error"] = error
                        continue
                return valid
            except (ValueError, TypeError, KeyError, SyntaxError) as exc:
                error = str(exc)
                if events:
                    events[-1]["validation_error"] = error
        raise ValueError(f"Field plan invalid after repair: {error}")

    def _bind(self, state: EvidenceState, plan: dict, bindings: dict,
              events: list[dict], round_number: int, hint: str = "") -> dict:
        missing = [f for f in plan["fields"] if f["source"] == "report" and f["id"] not in bindings]
        if not missing:
            return bindings
        for field in missing:
            view, shown_refs, view_tokens = state.view_for_field(field, hint)
            user = (f"目标公司：{state.entity}\n问题：{state.row['question']}\n"
                    f"只绑定此字段：{json.dumps(field, ensure_ascii=False)}\n"
                    f"证据视图：\n{view or '无可显示的证据'}\n{hint}")
            step = f"bind_round_{round_number}_{field['id']}"
            try:
                proposal = self._ask(BIND_SYSTEM, user, events, step)
            except json.JSONDecodeError as exc:
                events[-1]["validation_error"] = str(exc)
                proposal = self._ask(BIND_SYSTEM, user + "\n上次 JSON 格式错误，请输出合法 JSON。",
                                     events, step + "_repair")
            events[-1]["shown_refs"] = shown_refs
            events[-1]["evidence_view_prompt_tokens"] = view_tokens
            candidates = proposal.get("bindings", [])
            if not isinstance(candidates, list):
                events[-1]["validation_error"] = "bindings must be a list"
                continue
            rejected = []
            for candidate in candidates:
                try:
                    if candidate["id"] != field["id"] or candidate.get("source_ref") not in shown_refs:
                        raise ValueError("Binding field/reference was not shown in this view")
                    item = validate_binding(candidate, field, state.row, state.by_ref)
                    bindings[item["id"]] = item
                except (ValueError, KeyError, TypeError) as exc:
                    rejected.append({"candidate": candidate, "reason": str(exc)})
            if rejected:
                events[-1]["rejected_bindings"] = rejected
            events[-1]["accepted_ids"] = [field["id"]] if field["id"] in bindings else []
        return bindings

    def run(self, input_row: dict) -> dict:
        state = EvidenceState(input_row, self.chunks, self.tokenizer)
        events: list[dict] = []
        result = {"status": "invalid_plan", "final": {"answer": None}, "error": None,
                  "trace": {"events": events}}
        bindings: dict[str, dict] = {}
        try:
            plan = self._plan(state, events)
            result["trace"]["field_plan"] = plan
            if plan["formula_id"] == "unsupported":
                result.update(status="unsupported_plan", final={"answer": None})
                result["trace"]["reason"] = plan["reason"]
                return self._finish(result, state, {})
            result["status"] = "binding_error"
            needed = [f for f in plan["fields"] if f["source"] == "report"]
            if needed:
                for round_number in range(self.max_retrieval_rounds + 1):
                    self._bind(state, plan, bindings, events, round_number)
                    missing = [f for f in needed if f["id"] not in bindings]
                    if not missing:
                        break
                    # A claimed missing value in an already displayed E block gets
                    # one focused recheck before the agent accepts the refusal.
                    apparent = {f["id"]: state.candidate_refs(f) for f in missing}
                    apparent = {ident: refs for ident, refs in apparent.items() if refs}
                    if apparent:
                        self._bind(state, plan, bindings, events, round_number,
                                   hint=f"复核：这些字段所在候选证据可能已显示：{apparent}。逐字段核对。")
                        missing = [f for f in needed if f["id"] not in bindings]
                    if not missing or round_number >= self.max_retrieval_rounds:
                        break
                    found = []
                    for field in missing:
                        added_for_field = 0
                        for candidate in self.retriever.search(field, state.row, state.seen_ids, round_number + 1):
                            ref = state.add(candidate)
                            if ref:
                                found.append({"field_id": field["id"], "ref": ref,
                                              "chunk_id": candidate["chunk"]["chunk_id"],
                                              "query": candidate["query"]})
                                added_for_field += 1
                                # Expose two plausible blocks before asking the model
                                # again; one rerank mistake should not end the search.
                                if added_for_field >= 2:
                                    break
                    events.append({"step": "targeted_retrieval", "round": round_number + 1,
                                   "requested_fields": [f["id"] for f in missing], "added": found})
                    if not found:
                        break
            complete = all(f["id"] in bindings for f in needed)
            if not complete:
                result.update(status="insufficient", final={"answer": MISSING})
                result["trace"]["reason"] = "缺少字段：" + "、".join(
                    f"{f['year'] or ''}{f['metric']}" for f in needed if f["id"] not in bindings)
                return self._finish(result, state, bindings)
            result["status"] = "calculation_error"
            try:
                tool = calculate(plan, bindings, state.row)
            except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
                events.append({"step": "python_decimal_tool", "error": str(exc)})
                # One constrained binding repair; this never changes provenance rules.
                report_fields = [f for f in needed]
                if report_fields:
                    bindings = {}
                    self._bind(state, plan, bindings, events, self.max_retrieval_rounds + 1,
                               hint=f"上次计算工具报错：{exc}。重新核对全部报告字段、单位和引用。")
                tool = calculate(plan, bindings, state.row)
            events.append({"step": "python_decimal_tool", "result": tool})
            result.update(status="calculated" if plan["formula_id"] != "direct" else "direct",
                          final={"answer": tool["answer"]})
            result["trace"]["calculation"] = tool
            return self._finish(result, state, bindings)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError,
                ZeroDivisionError, OverflowError, SyntaxError, ArithmeticError) as exc:
            result["error"] = str(exc)
            return self._finish(result, state, bindings)
        except (RuntimeError, OSError) as exc:
            result.update(status="api_error", error=f"{type(exc).__name__}: {exc}")
            return self._finish(result, state, bindings)

    @staticmethod
    def _finish(result: dict, state: EvidenceState, bindings: dict) -> dict:
        plan = result["trace"].get("field_plan", {})
        fields = plan.get("fields", [])
        report = [f for f in fields if f["source"] == "report"]
        question = [f for f in fields if f["source"] == "question"]
        result["trace"]["bindings"] = bindings
        result["trace"]["evidence_completeness"] = {
            "planned_fields": len(fields), "question_fields": len(question),
            "report_fields": len(report), "verified_report_fields": len(bindings),
            "complete": bool(fields) and len(report) == len(bindings),
            "metric_type": "mechanically_validated_plan_fields_not_manual_gold_operands",
        }
        result["source_chunk_ids"] = [e["chunk_id"] for e in state.row["selected_evidence"]]
        shown_refs = {ref for event in result["trace"].get("events", [])
                      for ref in event.get("shown_refs", [])}
        result["shown_chunk_ids"] = [e["chunk_id"] for e in state.row["selected_evidence"]
                                     if e["citation"] in shown_refs]
        result["selected_evidence"] = state.row["selected_evidence"]
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-retrieval-rounds", type=int, default=MAX_RETRIEVAL_ROUNDS)
    parser.add_argument("--tokenizer-path", type=Path,
                        help="Qwen3-4B-Instruct-2507 tokenizer.json; also supports QWEN_TOKENIZER_PATH")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.max_retrieval_rounds <= 3:
        parser.error("--max-retrieval-rounds must be 0-3")
    if MAX_PLAN_INPUT_TOKENS + 1024 > MODEL_CONTEXT_TOKENS or MAX_BIND_INPUT_TOKENS + 512 > MODEL_CONTEXT_TOKENS:
        parser.error("V3 prompt budgets plus generation reserves must fit the 3200-token model context")
    try:
        tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
        tokenizer = load_tokenizer(tokenizer_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows or len({row["finqa_id"] for row in rows}) != len(rows):
        raise ValueError("No input rows or duplicate question IDs")
    for row in rows:
        _forbid_eval_fields(row)
    chunks = {item["chunk_id"]: item for item in read_jsonl(CHUNKS)}
    for row in rows:
        EvidenceState(row, chunks, tokenizer)
    if args.dry_run:
        print(json.dumps({"version": VERSION, "rows": len(rows),
                          "example_question": rows[0]["question"],
                          "planner_system": PLAN_SYSTEM}, ensure_ascii=False, indent=2))
        return
    output = args.output or (OUTPUT if args.limit is None else
                             OUTPUT.with_name(f"task5_v3_rag_smoke{args.limit}_generation.jsonl"))
    if output.exists():
        raise FileExistsError(f"V3 output already exists: {output}")
    retriever = TargetedRetriever(list(chunks.values()))
    ask = lambda messages: call_model_v3(args.endpoint, messages, args.timeout)
    agent = RagAgentV3(ask, retriever, chunks, tokenizer, args.max_retrieval_rounds)
    input_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    tokenizer_hash = source_hash(tokenizer_path)
    chunks_hash = source_hash(CHUNKS)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("x", encoding="utf-8") as handle:
        for index, row in enumerate(rows, 1):
            started = time.perf_counter()
            try:
                result = agent.run(row)
            except Exception as exc:
                # Keep a per-question trace even when an external model/search
                # call fails; an interrupted batch can still be inspected.
                result = {"status": "api_error", "final": {"answer": None},
                          "trace": {}, "error": f"{type(exc).__name__}: {exc}"}
            record = {"finqa_id": row["finqa_id"], "mode": "rag", "model": MODEL,
                      "prompt_version": VERSION, "input_sha256": input_hash,
                      "tokenizer_sha256": tokenizer_hash, "chunks_sha256": chunks_hash,
                      "latency_seconds": round(time.perf_counter() - started, 3), **result}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{index}/{len(rows)} {row['finqa_id']} {record['status']}", flush=True)
    partial.replace(output)
    print(json.dumps({"rows": len(rows), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
