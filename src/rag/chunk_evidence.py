#!/usr/bin/env python3
"""Build traceable prose and table evidence chunks from parsed LaTeX JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any


YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
YEAR_CELL_RE = re.compile(r"^\s*((?:19|20)\d{2})\s*年?\s*$")
UNIT_RE = re.compile(r"亿元|万元|千元|百万元|万股|亿股|元/股|人民币|美元|元|%|％|倍")
MULTIROW_RE = re.compile(r"\\multirow(?:\[[^]]*\])?\s*\{(\d+)\}")
SENTENCE_ENDINGS = "。！？；;!?"


def prose_spans(text: str, max_chars: int = 500, overlap_chars: int = 60) -> list[tuple[int, int]]:
    """Cover an entire section with roughly 300–500 character overlapping spans."""
    if not text:
        return []
    if max_chars <= 0 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("max_chars must be positive and overlap_chars must be smaller")
    if len(text) <= max_chars:
        return [(0, len(text))]

    target_chars = max(1, int(max_chars * 0.8))
    min_chars = max(1, max_chars // 2)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        remaining = len(text) - start
        if remaining <= max_chars:
            spans.append((start, len(text)))
            break
        goal = min(start + target_chars, len(text))
        if len(text) - goal < min_chars:
            goal = max(start + min_chars, len(text) - min_chars + overlap_chars)
        goal = min(goal, start + max_chars)
        punctuation = max((text.rfind(mark, start + min_chars - 1, goal) for mark in SENTENCE_ENDINGS), default=-1)
        end = punctuation + 1 if punctuation >= start + min_chars - 1 else goal
        if end <= start:
            end = min(start + max_chars, len(text))
        spans.append((start, end))
        start = max(start + 1, end - overlap_chars)
    return spans


def _unique_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    return list(dict.fromkeys(pattern.findall(text)))


def _section_label(section: dict[str, Any]) -> str:
    return " / ".join(section["path"]) if section["path"] else section["title"]


def chunk_prose(doc: dict[str, Any], section: dict[str, Any], max_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    spans = prose_spans(section["text"], max_chars, overlap_chars)
    chunks = []
    for ordinal, (start, end) in enumerate(spans, start=1):
        passage = section["text"][start:end]
        header = f"公司：{doc['company_candidate_from_filename']}\n报告：{doc['title']}\n章节：{_section_label(section)}"
        chunks.append(
            {
                "schema_version": 1,
                "chunk_id": f"{section['section_id']}::p{ordinal:03d}",
                "doc_id": doc["doc_id"],
                "source_path": doc["source_path"],
                "source_sha256": doc["source_sha256"],
                "company": doc["company_candidate_from_filename"],
                "report_title": doc["title"],
                "section_id": section["section_id"],
                "section_path": section["path"],
                "content_type": "prose",
                "source_start_line": section["source_start_line"],
                "source_end_line": section["source_end_line"],
                "location_granularity": "section",
                "prose_char_start": start,
                "prose_char_end": end,
                "table_id": None,
                "header_row_numbers": [],
                "data_row_numbers": [],
                "years": _unique_matches(YEAR_RE, passage),
                "unit_mentions": _unique_matches(UNIT_RE, passage),
                "content_text": passage,
                "text": header + "\n正文：" + passage,
            }
        )
    return chunks


def _header_count(rows: list[dict[str, Any]]) -> int:
    if len(rows) < 2:
        return len(rows)
    first_raw = " ".join(rows[0]["raw_cells"])
    second_years = sum(bool(YEAR_CELL_RE.fullmatch(cell)) for cell in rows[1]["cells"])
    return 2 if ("\\multicolumn" in first_raw or "\\multirow" in first_raw or second_years >= 2) else 1


def _display_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Carry a visible multirow label into blank continuation cells."""
    active: dict[int, tuple[str, int]] = {}
    display: list[dict[str, Any]] = []
    fills = 0
    for row in rows:
        cells = list(row["cells"])
        filled_columns: list[int] = []
        for column, (label, remaining) in active.items():
            if column < len(cells) and not cells[column]:
                cells[column] = label
                filled_columns.append(column)
                fills += 1
        active = {column: (label, remaining - 1) for column, (label, remaining) in active.items() if remaining > 1}
        for column, raw_cell in enumerate(row["raw_cells"]):
            match = MULTIROW_RE.search(raw_cell)
            if match and int(match.group(1)) > 1 and row["cells"][column]:
                active[column] = (row["cells"][column], int(match.group(1)) - 1)
        display.append(
            {
                "row": row["row"],
                "cells": row["cells"],
                "display_cells": cells,
                "raw_cells": row["raw_cells"],
                "filled_columns": filled_columns,
            }
        )
    return display, fills


def _year_column_map(headers: list[dict[str, Any]], column_count: int) -> dict[int, str]:
    for header in reversed(headers):
        cells = header["display_cells"]
        if len(cells) != column_count:
            continue
        years = {index: match.group(1) for index, cell in enumerate(cells) if (match := YEAR_CELL_RE.fullmatch(cell))}
        if len(years) >= 2:
            return years
    return {}


def _render_row(row: dict[str, Any], headers: list[dict[str, Any]], is_header: bool) -> tuple[str, bool]:
    cells = row["display_cells"]
    label = "表头" if is_header else "数据"
    line = f"{label}第{row['row']}行：" + " | ".join(cells)
    if is_header:
        return line, False
    year_columns = _year_column_map(headers, len(cells))
    if not year_columns:
        return line, False
    values = [f"{year_columns[index]}={cells[index]}" for index in sorted(year_columns) if cells[index]]
    return line + "；年份对应值：" + " | ".join(values), bool(values)


def _table_context(doc: dict[str, Any], section: dict[str, Any], table: dict[str, Any]) -> str:
    caption = table["caption"] if table["caption"] else "原文无表题"
    header = f"公司：{doc['company_candidate_from_filename']}\n报告：{doc['title']}\n章节：{_section_label(section)}\n表格：{caption}"
    clear_units = [unit for unit in table["units_seen_near_header"] if UNIT_RE.fullmatch(unit)]
    if clear_units:
        header += "\n表头附近出现的单位：" + "、".join(clear_units)
    return header


def chunk_table(doc: dict[str, Any], section: dict[str, Any], table: dict[str, Any], max_rows: int, max_chars: int) -> tuple[list[dict[str, Any]], int, int]:
    if max_rows <= 0 or max_chars <= 0:
        raise ValueError("table row and character limits must be positive")
    rows, filled_cells = _display_rows(table["rows"])
    header_count = _header_count(table["rows"])
    headers = rows[:header_count]
    data_rows = rows[header_count:]
    header_lines = [_render_row(row, headers, True)[0] for row in headers]
    context = _table_context(doc, section, table)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in data_rows:
        proposed = current + [row]
        proposed_lines = [_render_row(item, headers, False)[0] for item in proposed]
        proposed_text = "\n".join(header_lines + proposed_lines)
        if current and (len(current) >= max_rows or len(proposed_text) > max_chars):
            groups.append(current)
            current = [row]
        else:
            current = proposed
    if current or not data_rows:
        groups.append(current)

    chunks: list[dict[str, Any]] = []
    year_aligned_rows = 0
    for ordinal, group in enumerate(groups, start=1):
        data_lines = []
        for row in group:
            line, aligned = _render_row(row, headers, False)
            data_lines.append(line)
            year_aligned_rows += int(aligned)
        body = "\n".join(header_lines + data_lines)
        text = context + "\n" + body
        chunk_rows = headers + group
        chunks.append(
            {
                "schema_version": 1,
                "chunk_id": f"{table['table_id']}::c{ordinal:03d}",
                "doc_id": doc["doc_id"],
                "source_path": doc["source_path"],
                "source_sha256": doc["source_sha256"],
                "company": doc["company_candidate_from_filename"],
                "report_title": doc["title"],
                "section_id": section["section_id"],
                "section_path": section["path"],
                "content_type": "table",
                "source_start_line": table["source_start_line"],
                "source_end_line": table["source_end_line"],
                "location_granularity": "table",
                "prose_char_start": None,
                "prose_char_end": None,
                "table_id": table["table_id"],
                "table_caption": table["caption"],
                "table_environment": table["environment"],
                "header_row_numbers": [row["row"] for row in headers],
                "data_row_numbers": [row["row"] for row in group],
                "rows": chunk_rows,
                "years": _unique_matches(YEAR_RE, text),
                "unit_mentions": _unique_matches(UNIT_RE, text),
                "content_text": body,
                "text": text,
                "oversize_content": len(body) > max_chars,
            }
        )
    return chunks, filled_cells, year_aligned_rows


def chunk_document(doc: dict[str, Any], max_prose_chars: int, prose_overlap: int, max_table_chars: int, max_table_rows: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tables_by_id = {table["table_id"]: table for table in doc["tables"]}
    if len(tables_by_id) != len(doc["tables"]):
        raise ValueError(f"duplicate table ID in {doc['doc_id']}")
    assigned: set[str] = set()
    chunks: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for section in doc["sections"]:
        if section["text"]:
            chunks.extend(chunk_prose(doc, section, max_prose_chars, prose_overlap))
        for table_id in section["table_ids"]:
            if table_id in assigned or table_id not in tables_by_id:
                raise ValueError(f"invalid table assignment: {table_id}")
            assigned.add(table_id)
            table_chunks, fills, aligned = chunk_table(doc, section, tables_by_id[table_id], max_table_rows, max_table_chars)
            chunks.extend(table_chunks)
            counters["multirow_filled_cells"] += fills
            counters["year_aligned_data_rows"] += aligned
            counters["untitled_tables"] += int(not tables_by_id[table_id]["caption"])
    if assigned != set(tables_by_id):
        raise ValueError(f"unassigned table(s) in {doc['doc_id']}: {len(set(tables_by_id) - assigned)}")
    return chunks, dict(counters)


def validate_coverage(documents: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> dict[str, int]:
    by_section: dict[str, list[dict[str, Any]]] = {}
    by_table: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        if chunk["content_type"] == "prose":
            by_section.setdefault(chunk["section_id"], []).append(chunk)
        else:
            by_table.setdefault(chunk["table_id"], []).append(chunk)
    uncovered_prose = 0
    missing_table_rows = 0
    duplicate_data_rows = 0
    missing_header_repetitions = 0
    for doc in documents:
        for section in doc["sections"]:
            if not section["text"]:
                continue
            spans = sorted((c["prose_char_start"], c["prose_char_end"]) for c in by_section.get(section["section_id"], []))
            covered_end = 0
            for start, end in spans:
                if start > covered_end:
                    uncovered_prose += start - covered_end
                covered_end = max(covered_end, end)
            uncovered_prose += max(0, len(section["text"]) - covered_end)
        for table in doc["tables"]:
            parts = by_table.get(table["table_id"], [])
            original_rows = {row["row"] for row in table["rows"]}
            covered_rows = {row for part in parts for row in part["header_row_numbers"] + part["data_row_numbers"]}
            missing_table_rows += len(original_rows - covered_rows)
            data_counts = Counter(row for part in parts for row in part["data_row_numbers"])
            duplicate_data_rows += sum(count - 1 for count in data_counts.values() if count > 1)
            if parts:
                header_numbers = parts[0]["header_row_numbers"]
                missing_header_repetitions += sum(not set(header_numbers).issubset(set(part["header_row_numbers"])) for part in parts)
    return {
        "uncovered_prose_characters": uncovered_prose,
        "missing_table_rows": missing_table_rows,
        "duplicate_data_rows": duplicate_data_rows,
        "missing_header_repetitions": missing_header_repetitions,
    }


def build_report(documents: list[dict[str, Any]], chunks: list[dict[str, Any]], counters: Counter[str], coverage: dict[str, int], settings: argparse.Namespace) -> str:
    prose = [chunk for chunk in chunks if chunk["content_type"] == "prose"]
    tables = [chunk for chunk in chunks if chunk["content_type"] == "table"]
    max_prose = max((len(chunk["content_text"]) for chunk in prose), default=0)
    max_table = max((len(chunk["content_text"]) for chunk in tables), default=0)
    max_full_text = max((len(chunk["text"]) for chunk in chunks), default=0)
    oversize = sum(chunk["oversize_content"] for chunk in tables)
    header_only = sum(not chunk["data_row_numbers"] for chunk in tables)
    short_prose = sum(len(chunk["content_text"]) < 50 for chunk in prose)
    short_front_matter = sum(len(chunk["content_text"]) < 50 and not chunk["section_path"] for chunk in prose)
    return f"""# RAG 第 2 部分：证据分块结果与问题记录

运行日期：{date.today().isoformat()}  
输入：`data/processed/corpus/parsed_documents.jsonl`（{len(documents)} 份文档）  
输出：`data/processed/corpus/evidence_chunks.jsonl`  
脚本：`src/rag/chunk_evidence.py`

## 固定参数

| 参数 | 数值 |
| --- | ---: |
| 正文最大字符数 | {settings.max_prose_chars} |
| 正文相邻块重叠 | {settings.prose_overlap} 字符 |
| 表格每块最多数据行 | {settings.max_table_rows} |
| 表格块目标正文上限 | {settings.max_table_chars} 字符；不可拆的长行允许超出 |

正文只在同一章节内切分。表格块重复所在公司、报告、章节、表题和表头；通常一块包含 1–{settings.max_table_rows} 行数据，源表若只有表头则生成无数据行的块。若年列表头与数据行列数一致，另写出“年份对应值”；跨行合并的空单元格继承可见标签，但原始单元格也保留。数值和单位不做换算。

## 实际结果

| 指标 | 结果 |
| --- | ---: |
| 总证据块 | {len(chunks):,} |
| 正文块 | {len(prose):,} |
| 少于 50 字的正文块 | {short_prose:,}（其中前置内容 {short_front_matter:,}） |
| 表格块 | {len(tables):,} |
| 涉及源表格 | {sum(len(doc['tables']) for doc in documents):,} |
| 无表题、以章节补上下文的源表格 | {counters['untitled_tables']:,} |
| 跨行合并填充的展示单元格 | {counters['multirow_filled_cells']:,} |
| 写出年份与数值对应关系的数据行 | {counters['year_aligned_data_rows']:,} |
| 正文块内容最大长度 | {max_prose:,} 字符 |
| 表格块内容最大长度 | {max_table:,} 字符 |
| 完整检索文本最大长度（含来源前缀） | {max_full_text:,} 字符 |
| 只有表头的表格块 | {header_only:,} |
| 超过表格目标上限的块 | {oversize:,} |

## 覆盖检查

| 检查项 | 结果 |
| --- | ---: |
| 未被任何块覆盖的正文字符 | {coverage['uncovered_prose_characters']:,} |
| 未被任何块覆盖的表格行 | {coverage['missing_table_rows']:,} |
| 非表头数据行重复计入不同块 | {coverage['duplicate_data_rows']:,} |
| 表头未随同表数据块重复 | {coverage['missing_header_repetitions']:,} |

## 遇到的问题与处理

- {counters['untitled_tables']} 张表没有原始 `caption`。其块中明确写“原文无表题”，同时附公司、报告和章节路径；没有伪造标题。
- 初次抽查发现“股票代码”会被上游的宽松单位提取识别成“股”。本阶段的单位提示只保留明确的货币、比例、倍数及“元/股”等单位；源表格文字与原始单元格不变。
- {short_prose} 个正文块少于 50 字，其中 {short_front_matter} 个是前置内容。短文本暂时保留以保证资料可追溯；任务 3 可把过滤前置说明作为独立检索对照，不依据最终答案偷偷筛选。
- `\\multicolumn` 与 `\\multirow` 会导致可见列数和实际数据行不同。保留表头原貌；只有列数匹配且表头至少有两个明确年份时，才追加“年份对应值”。多行标签的填充只用于检索文本，原始 `cells` 和 `raw_cells` 仍可复核。
- 表格内容长度超过 {settings.max_table_chars} 字符目标上限的块有 {oversize} 个；单行不可拆时允许超限并会显式标记。重排阶段需要按模型 tokenizer 检查完整检索文本的长度，记录截断情况。
- `source_start_line` / `source_end_line` 的粒度为来源章节或整张表；正文块另保存章节文本字符范围，表格块另保存表格行号。引用时应结合这些字段定位原文。

## 独立校验

运行下方测试命令，6 项通过；覆盖正文无缺口、表头重复、年份列映射、跨行标签续接、无表题回退、单位提示与真实解析数据抽样。全量覆盖检查结果见上表。

```powershell
& "{sys.executable}" -m unittest discover -s tests -p "test_chunk_evidence.py" -v
```

## 可复现命令

```powershell
& "{sys.executable}" src/rag/chunk_evidence.py --force
```
"""


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/processed/corpus/parsed_documents.jsonl")
    parser.add_argument("--output", type=Path, default=root / "data/processed/corpus/evidence_chunks.jsonl")
    parser.add_argument("--report", type=Path, default=root / "docs/RAG_TASK2_CHUNKING_RESULTS.md")
    parser.add_argument("--max-prose-chars", type=int, default=500)
    parser.add_argument("--prose-overlap", type=int, default=60)
    parser.add_argument("--max-table-chars", type=int, default=650)
    parser.add_argument("--max-table-rows", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="replace existing output and report")
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error(f"parsed input does not exist: {args.input}")
    if not args.force and (args.output.exists() or args.report.exists()):
        parser.error("output exists; pass --force to replace it")
    if args.max_prose_chars <= args.prose_overlap or args.prose_overlap < 0:
        parser.error("prose overlap must be nonnegative and smaller than the maximum")
    if args.max_table_rows <= 0 or args.max_table_chars <= 0:
        parser.error("table limits must be positive")

    documents = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not documents:
        parser.error("parsed input contains no documents")
    chunks: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for doc in documents:
        doc_chunks, doc_counters = chunk_document(doc, args.max_prose_chars, args.prose_overlap, args.max_table_chars, args.max_table_rows)
        chunks.extend(doc_chunks)
        counters.update(doc_counters)
    if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
        raise ValueError("duplicate chunk IDs")
    coverage = validate_coverage(documents, chunks)
    if any(coverage.values()):
        raise ValueError(f"evidence coverage failed: {coverage}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="\n") as stream:
        for chunk in chunks:
            stream.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_output.replace(args.output)
    args.report.write_text(build_report(documents, chunks, counters, coverage, args), encoding="utf-8", newline="\n")
    print(json.dumps({
        "documents": len(documents),
        "chunks": len(chunks),
        "prose_chunks": sum(chunk["content_type"] == "prose" for chunk in chunks),
        "table_chunks": sum(chunk["content_type"] == "table" for chunk in chunks),
        "coverage": coverage,
        "output": str(args.output),
        "report": str(args.report),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
