"""Task 3: reproducible BM25, dense, and RRF retrieval over evidence chunks.

Only question and company_with_period enter the query. No answer or reference path
is loaded, so evaluation labels cannot leak into candidate generation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import pickle
from pathlib import Path
import re
import time
from bge_m3_encoder import BgeM3Encoder, model_fingerprint, DIMENSION, MAX_LENGTH, BATCH_SIZE

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CHUNKS = PROJECT / "data/processed/corpus/evidence_chunks.jsonl"
QUESTIONS = PROJECT / "data/raw/financecomplexqa/questions/corporate_reports"
INDEX = PROJECT / "data/processed/corpus/retrieval_index_bge_m3"
MODEL_CACHE = PROJECT / "models/embedding/fastembed_cache"
MODEL_PATH = PROJECT / "models/embedding/bge-m3-onnx"
RESULTS = PROJECT / "data/processed/eval/task3_bge_m3_rankings.jsonl"
MODEL = "BAAI/bge-m3"
QUERY_PREFIX = ""
RETRIEVAL_VIEW_VERSION = 'v4.8-company-period-table-unit-20261007'
TOP_EACH = 30
TOP_HYBRID = 20
RRF_K = 60
BM25_K1 = 1.2
BM25_B = 0.75
OUTPUT_MARKER = "### 输出格式"
ATOM = re.compile(r"(?<![A-Za-z0-9])(?:\d{4}年|\d{4}|\d+(?:\.\d+)?%|\d+(?:\.\d+)?|[A-Za-z]+\d*(?:\.[A-Za-z]+)?)")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def source_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    import jieba

    lowered = text.lower()
    words = [part.strip() for part in jieba.cut(lowered) if part.strip()]
    # Jieba can split years and stock codes. Preserve each complete numeric/code atom.
    atoms = [match.group() for match in ATOM.finditer(lowered)]
    return words + atoms


def make_query(row: dict) -> dict:
    """Intentionally whitelist fields, excluding gold and Reference_documents."""
    question = row["question"].split(OUTPUT_MARKER, 1)[0].strip()
    company = row.get("company_with_period", "").strip()
    return {"finqa_id": row["finqa_id"], "question": question,
            "company_with_period": company,
            "retrieval_query": f"{company} {question}".strip()}


def load_queries(directory: Path = QUESTIONS) -> list[dict]:
    rows = [make_query(row) for path in sorted(directory.glob("*.jsonl"))
            for row in read_jsonl(path)]
    ids = [row["finqa_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate finqa_id in question input")
    return rows


def build_bm25(chunks: list[dict]) -> dict:
    postings = defaultdict(list)
    lengths = []
    for index, chunk in enumerate(chunks):
        counts = Counter(tokenize(chunk["text"]))
        lengths.append(sum(counts.values()))
        for token, freq in counts.items():
            postings[token].append((index, freq))
    n = len(chunks)
    return {"postings": dict(postings), "lengths": lengths,
            "avgdl": sum(lengths) / n if n else 0, "n": n}


def bm25_search(index: dict, query: str, top_k: int = TOP_EACH) -> list[tuple[int, float]]:
    scores = defaultdict(float)
    n, avgdl = index["n"], index["avgdl"]
    if not n or not avgdl:
        return []
    for term, qtf in Counter(tokenize(query)).items():
        postings = index["postings"].get(term, [])
        if not postings:
            continue
        df = len(postings)
        idf = math.log1p((n - df + 0.5) / (df + 0.5))
        for doc, tf in postings:
            norm = BM25_K1 * (1 - BM25_B + BM25_B * index["lengths"][doc] / avgdl)
            scores[doc] += min(qtf, 2) * idf * tf * (BM25_K1 + 1) / (tf + norm)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def dense_search(vectors: np.ndarray, query_vector: np.ndarray,
                 top_k: int = TOP_EACH) -> list[tuple[int, float]]:
    scores = vectors @ normalize(query_vector)[0]
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [(int(i), float(scores[i])) for i in order]


def rrf(lists: list[list[tuple[int, float]]], top_k: int = TOP_HYBRID,
        k: int = RRF_K) -> list[tuple[int, float]]:
    scores = defaultdict(float)
    for ranked in lists:
        for rank, (index, _) in enumerate(ranked, start=1):
            scores[index] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]


def encoder(cache_dir: Path):
    if not (MODEL_PATH / "download_manifest.json").is_file():
        raise FileNotFoundError(
            "Local BGE-M3 missing; run scripts/download_bge_m3.py first")
    return BgeM3Encoder(MODEL_PATH)

def retrieval_view(chunk, chunks=()):
    """Metadata guides matching; unchanged source cells remain the evidence."""
    from table_structure import table_structure
    lines = [f"公司：{chunk['company']}", f"报告：{chunk.get('report_title', chunk['doc_id'])}",
             '章节：'+' / '.join(chunk.get('section_path', []))]
    if chunk.get('content_type') == 'table':
        lines += ['表题：'+str(chunk.get('table_caption') or '未注明'),
                  '原文单位线索：'+'、'.join(chunk.get('unit_mentions', []))]
        headers = table_structure(chunk)['header_rows']
        lines += ['表头原位置：'+' | '.join(map(str,r.get('display_cells', []))) for r in headers]
        # Nearby notes are candidate context, not proof of a field's unit.
        notes = [c for c in chunks if c.get('doc_id')==chunk['doc_id']
            and c.get('content_type')=='prose' and c.get('company')==chunk['company']
            and '单位' in c.get('content_text','')
            and c.get('source_start_line',0)<=chunk.get('source_end_line',0)+12
            and c.get('source_end_line',0)>=chunk.get('source_start_line',0)-12]
        lines += ['邻近原文单位说明：'+c['content_text'] for c in notes]
    lines.append(chunk['content_text'])
    return '\n'.join(lines)

def validate_index(metadata, chunks_path=CHUNKS, vectors=None):
    if metadata.get('model') != MODEL or metadata.get('dense_dimension') != DIMENSION:
        raise ValueError('Old embedding space; rebuild the BGE-M3 index')
    if metadata.get('retrieval_view_version') != RETRIEVAL_VIEW_VERSION:
        raise ValueError('Retrieval representation changed; rebuild index')
    if source_hash(chunks_path) != metadata['source_sha256']:
        raise ValueError('Chunk input changed; rebuild index')
    if model_fingerprint(MODEL_PATH) != metadata['model_sha256']:
        raise ValueError('Embedding model changed; rebuild index')
    if vectors is not None and vectors.shape != (metadata['chunk_count'], DIMENSION):
        raise ValueError('Index vector shape mismatch')


def embed_passages(model, texts: list[str]) -> np.ndarray:
    return normalize(np.stack(list(model.embed(texts, batch_size=BATCH_SIZE))))


def embed_queries(model, texts: list[str]) -> np.ndarray:
    return normalize(np.stack(list(model.embed(
        texts, batch_size=BATCH_SIZE))))


def build(chunks_path: Path = CHUNKS, index_dir: Path = INDEX) -> dict:
    started = time.perf_counter()
    chunks = list(read_jsonl(chunks_path))
    if len({x["chunk_id"] for x in chunks}) != len(chunks):
        raise ValueError("Duplicate chunk IDs")
    index_dir.mkdir(parents=True, exist_ok=True)
    bm25 = build_bm25(chunks)
    with (index_dir / "bm25.pkl").open("wb") as handle:
        pickle.dump(bm25, handle, protocol=pickle.HIGHEST_PROTOCOL)
    model = encoder(MODEL_CACHE)
    by_doc = defaultdict(list)
    for c in chunks: by_doc[c['doc_id']].append(c)
    texts = [retrieval_view(c, by_doc[c['doc_id']]) for c in chunks]
    lengths = [len(x.ids) for x in model.tokenizer.encode_batch(texts)]
    if max(lengths, default=0) > MAX_LENGTH:
        raise ValueError('Dense view too long; cannot silently truncate a financial table')
    fingerprint=model_fingerprint(MODEL_PATH)
    checkpoint=index_dir/'embedding_progress.json'
    partial=index_dir/'dense_vectors.partial.npy'
    signature=dict(source_sha256=source_hash(chunks_path),model_sha256=fingerprint,
        retrieval_view_version=RETRIEVAL_VIEW_VERSION,shape=[len(chunks),DIMENSION])
    completed=0
    if checkpoint.exists() and partial.exists():
        saved=json.loads(checkpoint.read_text(encoding='utf-8'))
        if any(saved.get(k)!=v for k,v in signature.items()):raise ValueError('Incompatible partial embedding index')
        completed=saved['completed'];vectors=np.load(partial,mmap_mode='r+')
        if not 0<=completed<=len(chunks) or vectors.shape!=(len(chunks),DIMENSION):raise ValueError('Invalid partial index')
    else:
        vectors=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(len(chunks),DIMENSION))
    for start in range(completed,len(chunks),32):
        batch = texts[start:start+32]
        vectors[start:start+len(batch)] = embed_passages(model,batch)
        vectors.flush()
        checkpoint_tmp=checkpoint.with_suffix('.json.tmp')
        checkpoint_tmp.write_text(json.dumps(dict(**signature,completed=start+len(batch))),encoding='utf-8')
        checkpoint_tmp.replace(checkpoint)
        print(json.dumps({'stage':'bge_m3_document_embedding','completed':min(start+32,len(chunks)),
                          'total':len(chunks)}, ensure_ascii=False), flush=True)
    if vectors.shape != (len(chunks), DIMENSION):
        raise ValueError(f"Unexpected dense index shape: {vectors.shape}")
    del vectors
    partial.replace(index_dir/'dense_vectors.npy')
    metadata = {"source": str(chunks_path),
                "source_sha256": source_hash(chunks_path), "chunk_count": len(chunks),
                "model": MODEL, "dense_dimension": DIMENSION,
                "model_sha256": fingerprint,
                "dense_backend": "official BAAI ONNX CPU; exact NumPy inner product",
                "retrieval_view_version": RETRIEVAL_VIEW_VERSION,
                "max_input_tokens": MAX_LENGTH, "observed_max_input_tokens": max(lengths,default=0),
                "truncated_count": 0, "query_prefix": QUERY_PREFIX,
                "normalized": True, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
                "rrf_k": RRF_K, "top_each": TOP_EACH, "top_hybrid": TOP_HYBRID,
                "build_seconds": round(time.perf_counter() - started, 3)}
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    return metadata


def rank(chunks_path: Path = CHUNKS, index_dir: Path = INDEX,
         questions_dir: Path = QUESTIONS, output: Path = RESULTS) -> dict:
    chunks = list(read_jsonl(chunks_path))
    meta = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    with (index_dir / "bm25.pkl").open("rb") as handle:
        bm25 = pickle.load(handle)
    vectors = np.load(index_dir / "dense_vectors.npy")
    validate_index(meta,chunks_path,vectors)
    if bm25["n"] != len(chunks) or len(vectors) != len(chunks):
        raise ValueError("Index/chunk count mismatch")
    queries = load_queries(questions_dir)
    model = encoder(MODEL_CACHE)
    started = time.perf_counter()
    query_vectors = embed_queries(model, [x["retrieval_query"] for x in queries])
    embed_seconds = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    if output.exists():raise FileExistsError(output)
    partial=output.with_suffix('.jsonl.partial')
    with partial.open("w", encoding="utf-8") as handle:
        for query, vector in zip(queries, query_vectors):
            t0 = time.perf_counter()
            sparse = bm25_search(bm25, query["retrieval_query"])
            dense = dense_search(vectors, vector)
            hybrid = rrf([sparse, dense])
            latency = time.perf_counter() - t0
            latencies.append(latency)

            def candidates(items):
                return [{"rank": r, "chunk_id": chunks[i]["chunk_id"],
                         "doc_id": chunks[i]["doc_id"], "score": round(score, 8)}
                        for r, (i, score) in enumerate(items, start=1)]

            record = {**query, "bm25": candidates(sparse), "dense": candidates(dense),
                      "hybrid": candidates(hybrid),
                      "embedding_model": MODEL, "retrieval_view_version": RETRIEVAL_VIEW_VERSION,
                      "retrieval_seconds": round(latency, 5)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    partial.replace(output)
    return {"question_count": len(queries), "chunk_count": len(chunks),
            "query_embedding_seconds": round(embed_seconds, 3),
            "mean_rank_seconds": round(float(np.mean(latencies)), 5),
            "p50_rank_seconds": round(float(np.percentile(latencies, 50)), 5),
            "p95_rank_seconds": round(float(np.percentile(latencies, 95)), 5),
            "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "rank", "all"])
    args = parser.parse_args()
    if args.command in ("build", "all"):
        print(json.dumps(build(), ensure_ascii=False, indent=2), flush=True)
    if args.command in ("rank", "all"):
        print(json.dumps(rank(), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
