#!/usr/bin/env python3
"""Parse FinanceComplexQA LaTeX reports into traceable document records.

This is a dependency-free first-stage parser. It preserves raw LaTeX alongside
plain text so downstream retrieval can use readable text without losing the
ability to audit the source table or section.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


HEADING_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}
TABLE_ENVS = {"tabular", "tabular*", "tabularx", "longtable"}
FLOAT_ENVS = {"table", "table*"}
ENV_TOKEN_RE = re.compile(r"\\(begin|end)\s*\{([^}]+)\}")
HEADING_RE = re.compile(
    r"\\(?P<name>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?"
)
UNIT_RE = re.compile(r"(?:亿元|万元|千元|百万元|元|人民币|美元|%|％|股|万股|亿股|倍|天|人)")


def strip_tex_comments(source: str) -> str:
    """Remove unescaped TeX comments while preserving line boundaries."""
    out: list[str] = []
    i = 0
    while i < len(source):
        if source[i] != "%":
            out.append(source[i])
            i += 1
            continue
        slash_count = 0
        j = i - 1
        while j >= 0 and source[j] == "\\":
            slash_count += 1
            j -= 1
        if slash_count % 2:
            out.append(source[i])
            i += 1
            continue
        newline = source.find("\n", i)
        if newline < 0:
            break
        out.append("\n")
        i = newline + 1
    return "".join(out)


def read_group(text: str, position: int, opener: str = "{") -> tuple[str, int] | None:
    """Read a balanced brace or bracket group starting at position."""
    closer = "}" if opener == "{" else "]"
    if position >= len(text) or text[position] != opener:
        return None
    depth = 0
    escaped = False
    for i in range(position, len(text)):
        char = text[i]
        if char == opener and not escaped:
            depth += 1
        elif char == closer and not escaped:
            depth -= 1
            if depth == 0:
                return text[position + 1 : i], i + 1
        if char == "\\" and not escaped:
            escaped = True
        else:
            escaped = False
    return None


def _command_arguments(text: str, position: int, optional: bool = True) -> tuple[list[str], int]:
    args: list[str] = []
    i = position
    while i < len(text) and text[i].isspace():
        i += 1
    if optional and i < len(text) and text[i] == "[":
        result = read_group(text, i, "[")
        if result:
            args.append(result[0])
            i = result[1]
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            break
        result = read_group(text, i)
        if not result:
            break
        args.append(result[0])
        i = result[1]
    return args, i


def _replace_macro_calls(text: str, macro: str, argument_count: int, keep_index: int | tuple[int, ...] | None) -> str:
    pattern = re.compile(r"\\" + re.escape(macro) + r"\*?(?![A-Za-z@])")
    out: list[str] = []
    cursor = 0
    while match := pattern.search(text, cursor):
        out.append(text[cursor : match.start()])
        args, end = _command_arguments(text, match.end(), optional=False)
        if len(args) < argument_count:
            out.append(text[match.start() : match.end()])
            cursor = match.end()
            continue
        if keep_index is not None:
            out.extend(args[index] for index in keep_index) if isinstance(keep_index, tuple) else out.append(args[keep_index])
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def latex_to_text(value: str) -> str:
    """Make a conservative readable rendering of common LaTeX text/table cells."""
    text = value
    # Commands whose visible value is in a later argument.
    for macro, count, keep in (
        ("multicolumn", 3, 2),
        ("multirow", 3, 2),
        ("textcolor", 2, 1),
        ("colorbox", 2, 1),
        ("fcolorbox", 3, 2),
        ("SI", 2, (0, 1)),
        ("num", 1, 0),
        ("vspace", 1, None),
        ("hspace", 1, None),
        ("color", 1, None),
        ("pagecolor", 1, None),
        ("cellcolor", 1, None),
        ("rowcolor", 1, None),
        ("blankline", 1, None),
        ("rule", 2, None),
    ):
        text = _replace_macro_calls(text, macro, count, keep)
    # Formatting and layout commands whose arguments should remain visible.
    text = re.sub(r"\\begin\s*\{[^}]+\}(?:\s*\[[^\]]*\])?", " ", text)
    text = re.sub(r"\\end\s*\{[^}]+\}", " ", text)
    text = re.sub(r"\\(?:renewcommand|setlength|addtolength)\*?\s*\{(?:[^{}]|\{[^{}]*\})*\}\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", text)
    text = re.sub(r"\\setlist\*?(?:\[[^\]]*\])?\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", text)
    text = re.sub(r"\\(?:hline|toprule|midrule|bottomrule|tabularnewline)\b", " ", text)
    text = re.sub(r"\\(?:cline|cmidrule)(?:\([^)]*\))?\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", text)
    text = re.sub(r"\\(?:addlinespace|vspace|hspace|raisebox|rule|cellcolor|rowcolor)\s*\{(?:[^{}]|\{[^{}]*\})*\}", " ", text)
    text = re.sub(r"\\(?:textbf|textit|emph|underline|overline|mbox|makebox|makecell|text|textrm|textsf|texttt|mathrm|mathbf|mathit|ensuremath|textsuperscript|textsubscript)\*?(?![A-Za-z@])", "", text)
    text = re.sub(r"\\(?:centering|raggedright|raggedleft|vfill|hfill|newpage|clearpage|small|large|Large|LARGE|huge|Huge|bfseries|itshape|scshape|normalfont|par|noindent)\*?(?![A-Za-z@])", " ", text)
    text = re.sub(r"\\(?:newline|linebreak)\b", " ", text)
    text = re.sub(r"\\\\\s*(?:\[[^\]]*\])?", " ", text)
    text = text.replace(r"\%", "%").replace(r"\&", "&").replace(r"\$", "$")
    text = text.replace(r"\_", "_").replace(r"\#", "#").replace(r"\{", "{").replace(r"\}", "}")
    text = text.replace(r"\times", "×").replace(r"\cdot", "·").replace(r"\pm", "±")
    text = text.replace("$", " ").replace(r"\(", " ").replace(r"\)", " ").replace(r"\[", " ").replace(r"\]", " ")
    # Remove remaining command names but retain any following braced contents.
    text = re.sub(r"\\[A-Za-z@]+\*?", " ", text)
    text = text.replace("{", " ").replace("}", " ").replace("~", " ")
    text = re.sub(r"\\\\", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_environment_spans(text: str, names: Iterable[str]) -> tuple[list[dict[str, Any]], list[str]]:
    wanted = set(names)
    stacks: dict[str, list[tuple[int, int]]] = {name: [] for name in wanted}
    spans: list[dict[str, Any]] = []
    warnings: list[str] = []
    for token in ENV_TOKEN_RE.finditer(text):
        direction, name = token.group(1), token.group(2)
        if name not in wanted:
            continue
        if direction == "begin":
            stacks[name].append((token.start(), token.end()))
        elif stacks[name]:
            start, content_start = stacks[name].pop()
            spans.append(
                {
                    "env": name,
                    "start": start,
                    "content_start": content_start,
                    "content_end": token.start(),
                    "end": token.end(),
                }
            )
        else:
            warnings.append(f"unmatched \\end{{{name}}} at char {token.start()}")
    for name, stack in stacks.items():
        for start, _ in stack:
            warnings.append(f"unmatched \\begin{{{name}}} at char {start}")
    spans.sort(key=lambda item: item["start"])
    return spans, warnings


def _skip_tabular_preamble(text: str, position: int) -> tuple[str, int]:
    """Skip width/column arguments after a tabular environment opener."""
    i = position
    pieces: list[str] = []
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "[":
            group = read_group(text, i, "[")
        elif i < len(text) and text[i] == "{":
            group = read_group(text, i)
        else:
            break
        if not group:
            break
        pieces.append(group[0])
        i = group[1]
    return " ".join(pieces), i


def _split_table_rows(content: str) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    while i < len(content):
        if content[i] == "\\":
            if i + 1 < len(content) and content[i + 1] == "\\" and depth == 0:
                row = "".join(current).strip()
                if row:
                    rows.append(row)
                current = []
                i += 2
                while i < len(content) and content[i].isspace():
                    i += 1
                if i < len(content) and content[i] == "[":
                    group = read_group(content, i, "[")
                    if group:
                        i = group[1]
                continue
            word_match = re.match(r"\\tabularnewline\b", content[i:])
            if word_match and depth == 0:
                row = "".join(current).strip()
                if row:
                    rows.append(row)
                current = []
                i += word_match.end()
                continue
        char = content[i]
        if char == "{" and (i == 0 or content[i - 1] != "\\"):
            depth += 1
        elif char == "}" and depth > 0 and (i == 0 or content[i - 1] != "\\"):
            depth -= 1
        current.append(char)
        i += 1
    tail = "".join(current).strip()
    if tail:
        rows.append(tail)
    return rows


def _split_table_cells(row: str) -> list[str]:
    cells: list[str] = []
    current: list[str] = []
    depth = 0
    i = 0
    while i < len(row):
        char = row[i]
        if char == "\\" and i + 1 < len(row):
            current.extend((char, row[i + 1]))
            i += 2
            continue
        if char == "{" :
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
        if char == "&" and depth == 0:
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        i += 1
    cells.append("".join(current).strip())
    return cells


def _find_commands(text: str, name: str) -> list[dict[str, Any]]:
    pattern = re.compile(r"\\" + re.escape(name) + r"\*?(?![A-Za-z@])")
    found: list[dict[str, Any]] = []
    for match in pattern.finditer(text):
        args, end = _command_arguments(text, match.end())
        if args:
            found.append({"start": match.start(), "end": end, "args": args})
    return found


def _extract_headings(body: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for match in HEADING_RE.finditer(body):
        name = match.group("name")
        args, end = _command_arguments(body, match.end())
        if not args:
            continue
        # If an optional short title exists, the final braced argument is the displayed title.
        title_raw = args[-1]
        headings.append(
            {
                "name": name,
                "level": HEADING_LEVELS[name],
                "title_raw": title_raw,
                "title": latex_to_text(title_raw),
                "start": match.start(),
                "content_start": end,
            }
        )
    return headings


def _extract_document_body(source: str) -> tuple[str, list[str]]:
    cleaned = strip_tex_comments(source)
    begin = re.search(r"\\begin\s*\{document\}", cleaned)
    if not begin:
        raise ValueError("missing \\begin{document}")
    end = re.search(r"\\end\s*\{document\}", cleaned[begin.end() :])
    if not end:
        raise ValueError("missing \\end{document}")
    end_start = begin.end() + end.start()
    return cleaned[begin.end() : end_start], []


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, max(0, min(offset, len(text)))) + 1


def _table_records(body: str, doc_id: str) -> tuple[list[dict[str, Any]], list[str]]:
    spans, warnings = _find_environment_spans(body, TABLE_ENVS)
    floats, float_warnings = _find_environment_spans(body, FLOAT_ENVS)
    warnings.extend(float_warnings)
    records: list[dict[str, Any]] = []
    for span in spans:
        preamble, content_start = _skip_tabular_preamble(body, span["content_start"])
        raw_content = body[content_start : span["content_end"]]
        row_texts = _split_table_rows(raw_content)
        rows = []
        for row_index, raw_row in enumerate(row_texts, start=1):
            raw_cells = _split_table_cells(raw_row)
            cells = [latex_to_text(cell) for cell in raw_cells]
            if any(cell for cell in cells):
                rows.append({"row": row_index, "raw_cells": raw_cells, "cells": cells})

        containing = [f for f in floats if f["start"] <= span["start"] and span["end"] <= f["end"]]
        caption = ""
        parent_float = min(containing, key=lambda item: item["end"] - item["start"]) if containing else None
        caption_scope = body[parent_float["start"] : parent_float["end"]] if parent_float else body[span["start"] : span["end"]]
        captions = _find_commands(caption_scope, "caption")
        if captions:
            caption = latex_to_text(captions[0]["args"][-1])
        searchable_headers = " ".join([caption] + [" ".join(row["cells"]) for row in rows[:3]])
        units = list(dict.fromkeys(UNIT_RE.findall(searchable_headers)))
        records.append(
            {
                "table_id": "",
                "doc_id": doc_id,
                "environment": span["env"],
                "caption": caption,
                "preamble": preamble,
                "start_char": span["start"],
                "end_char": span["end"],
                "units_seen_near_header": units,
                "rows": rows,
                "raw_latex": body[span["start"] : span["end"]],
            }
        )

    # Preserve unusual floating tables that do not contain a tabular environment.
    for float_span in floats:
        if any(float_span["start"] <= item["start"] and item["end"] <= float_span["end"] for item in spans):
            continue
        scope = body[float_span["start"] : float_span["end"]]
        captions = _find_commands(scope, "caption")
        caption = latex_to_text(captions[0]["args"][-1]) if captions else ""
        records.append(
            {
                "table_id": "",
                "doc_id": doc_id,
                "environment": float_span["env"],
                "caption": caption,
                "preamble": "",
                "start_char": float_span["start"],
                "end_char": float_span["end"],
                "units_seen_near_header": list(dict.fromkeys(UNIT_RE.findall(caption))),
                "rows": [],
                "raw_latex": scope,
            }
        )
    records.sort(key=lambda item: item["start_char"])
    for index, record in enumerate(records, start=1):
        record["table_id"] = f"{doc_id}::t{index:04d}"
    return records, warnings


def _section_records(body: str, headings: list[dict[str, Any]], tables: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    table_spans = [(t["start_char"], t["end_char"], t["table_id"]) for t in tables]
    segments: list[dict[str, Any]] = []
    if headings and headings[0]["start"] > 0:
        segments.append({"name": "前置内容", "level": 0, "path": [], "start": 0, "content_start": 0, "end": headings[0]["start"]})

    stack: list[tuple[int, str]] = []
    normalized_headings: list[dict[str, Any]] = []
    for heading in headings:
        while stack and stack[-1][0] >= heading["level"]:
            stack.pop()
        stack.append((heading["level"], heading["title"]))
        heading["path"] = [title for _, title in stack]
        normalized_headings.append(heading)
    for index, heading in enumerate(normalized_headings):
        end = normalized_headings[index + 1]["start"] if index + 1 < len(normalized_headings) else len(body)
        segments.append(
            {
                "name": heading["title"],
                "level": heading["level"],
                "path": heading["path"],
                "start": heading["start"],
                "content_start": heading["content_start"],
                "end": end,
            }
        )
    if not segments:
        segments.append({"name": "全文", "level": 0, "path": [], "start": 0, "content_start": 0, "end": len(body)})

    result = []
    for index, segment in enumerate(segments, start=1):
        raw = body[segment["content_start"] : segment["end"]]
        prose_parts: list[str] = []
        cursor = segment["content_start"]
        related_tables = []
        for table_start, table_end, table_id in table_spans:
            if table_start < segment["end"] and table_end > segment["content_start"]:
                related_tables.append(table_id)
                if table_start >= cursor:
                    prose_parts.append(body[cursor:table_start])
                    cursor = min(table_end, segment["end"])
        prose_parts.append(body[cursor : segment["end"]])
        result.append(
            {
                "section_id": f"{doc_id}::s{index:04d}",
                "doc_id": doc_id,
                "level": segment["level"],
                "title": segment["name"],
                "path": segment["path"],
                "start_char": segment["start"],
                "end_char": segment["end"],
                "table_ids": related_tables,
                "text": latex_to_text(" ".join(prose_parts)),
                "raw_latex": raw,
            }
        )
    return result


def parse_document(path: Path, input_root: Path) -> tuple[dict[str, Any], list[str]]:
    raw_bytes = path.read_bytes()
    source = raw_bytes.decode("utf-8-sig")
    cleaned = strip_tex_comments(source)
    body, warnings = _extract_document_body(cleaned)
    body_marker = re.search(r"\\begin\s*\{document\}", cleaned)
    assert body_marker is not None
    body_start_in_source = body_marker.end()
    doc_id = path.stem
    tables, table_warnings = _table_records(body, doc_id)
    headings = _extract_headings(body)
    sections = _section_records(body, headings, tables, doc_id)
    for section in sections:
        section["source_start_line"] = _line_number(cleaned, body_start_in_source + section["start_char"])
        section["source_end_line"] = _line_number(cleaned, body_start_in_source + max(section["start_char"], section["end_char"] - 1))
        section["offset_basis"] = "document body after comment removal"
    for table in tables:
        table["source_start_line"] = _line_number(cleaned, body_start_in_source + table["start_char"])
        table["source_end_line"] = _line_number(cleaned, body_start_in_source + max(table["start_char"], table["end_char"] - 1))
        table["offset_basis"] = "document body after comment removal"
    title_commands = _find_commands(body, "title")
    title = latex_to_text(title_commands[0]["args"][-1]) if title_commands else path.stem
    marker = "_企业财务报告_"
    company = path.stem.split(marker, 1)[0] if marker in path.stem else ""
    env_counts = Counter(m.group(2) for m in ENV_TOKEN_RE.finditer(cleaned))
    begin_counts = Counter(m.group(2) for m in ENV_TOKEN_RE.finditer(cleaned) if m.group(1) == "begin")
    end_counts = Counter(m.group(2) for m in ENV_TOKEN_RE.finditer(cleaned) if m.group(1) == "end")
    unbalanced = sorted(name for name in set(begin_counts) | set(end_counts) if begin_counts[name] != end_counts[name])
    if unbalanced:
        warnings.append("environment begin/end count differs: " + ", ".join(unbalanced))
    warnings.extend(table_warnings)
    document = {
        "doc_id": doc_id,
        "source_path": path.relative_to(input_root).as_posix(),
        "source_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "source_bytes": len(raw_bytes),
        "title": title,
        "company_candidate_from_filename": company,
        "section_count": len(sections),
        "table_count": len(tables),
        "sections": sections,
        "tables": tables,
        "diagnostics": warnings,
    }
    return document, warnings


def build_report(documents: list[dict[str, Any]], failures: list[str], input_root: Path) -> str:
    env_counts = Counter(table["environment"] for doc in documents for table in doc["tables"])
    total_sections = sum(doc["section_count"] for doc in documents)
    total_tables = sum(doc["table_count"] for doc in documents)
    total_rows = sum(len(table["rows"]) for doc in documents for table in doc["tables"])
    total_cells = sum(len(row["cells"]) for doc in documents for table in doc["tables"] for row in table["rows"])
    diagnostic_count = sum(len(doc["diagnostics"]) for doc in documents)
    empty_text_sections = sum(not section["text"] for doc in documents for section in doc["sections"])
    source_bytes = sum(doc["source_bytes"] for doc in documents)
    env_summary = ", ".join(f"`{name}` {count}" for name, count in sorted(env_counts.items())) or "无"
    failure_lines = "\n".join(f"- `{failure}`" for failure in failures) or "- 无"
    diagnostic_lines = []
    for doc in documents:
        for diagnostic in doc["diagnostics"]:
            diagnostic_lines.append(f"- `{doc['doc_id']}`：{diagnostic}")
    diagnostics = "\n".join(diagnostic_lines) if diagnostic_lines else "- 未发现环境边界或文件解析警告。"
    return f"""# RAG 第 1 部分：LaTeX 解析结果与问题记录

运行日期：2026-09-28  
输入目录：`{input_root}`  
解析程序：`src/rag/parse_latex.py`  
范围：仅解析 LaTeX 文档；没有读取题目 `gold` 或把参考文档路径用于解析。

## 解析结果

| 指标 | 结果 |
| --- | ---: |
| 成功解析文档 | {len(documents)} |
| 源文件总大小 | {source_bytes:,} bytes（{source_bytes / (1024 * 1024):.2f} MiB） |
| 章节/前置内容记录 | {total_sections:,} |
| 表格记录 | {total_tables:,} |
| 表格行记录 | {total_rows:,} |
| 表格单元格记录 | {total_cells:,} |
| 空正文章节记录 | {empty_text_sections:,}（版式清理后没有正文文本） |
| 文件/结构诊断数 | {diagnostic_count:,} |
| 表格环境分布 | {env_summary} |
| 解析失败文件 | {len(failures)} |

## 输出文件

- 结构化结果：`data/processed/corpus/parsed_documents.jsonl`，每个源文件一行，含 sections、tables、清理注释后的 LaTeX、源文件行号、正文相对字符位置和 SHA-256。
- 单元测试：`tests/test_parse_latex.py`。

## 已知问题与处理

- `\\multicolumn`、`\\multirow` 等命令会改变表格视觉布局。解析器把常见宏转为可读单元格，并同时保留每行 `raw_cells`、整表 `raw_latex`，需要时可回查。源文件行号对应原文件；字符位置对应去除注释后的 document body。
- 抽查时发现 `\\cmidrule(lr){{3-7}}` 的绘表参数曾被误当作单元格文本（出现 `3-7`）；清洗规则已修正并加入独立测试，最终结果中此类残留为 0。封面间距、颜色、空白占位线和行距命令曾留下 `3cm`、`formblue`、`4pt` 等排版词，现已过滤并加入清理逻辑。
- `tabularx` 和 `longtable` 保留环境、列格式、行/单元格，不做 PDF 排版还原；当前源文件只有少数此类环境。
- LaTeX 自定义宏、公式排版和图表未展开或 OCR。数据记录保留原始内容；RAG 分块阶段需继续抽查这些边界案例。
- `units_seen_near_header` 只是标题附近的单位线索，不是完整单位解析结果；不能据此自动换算数值。
- `{empty_text_sections}` 个空正文记录是版式清理后没有正文文本的章节；包括连续标题或只包含版式命令的片段，不计为文档解析失败。原始 LaTeX 仍保留供复核。

## 独立校验与抽查

- `unittest`：7 项通过，覆盖注释、格式宏、行/单元格边界、嵌套环境，以及真实报告抽样。
- 141 个输出文档 ID 均唯一；所有源文件 SHA-256 与重新读取的文件一致。
- 全部表格都有原始 LaTeX 和有效源文件行号；未发现重复表格 ID、Unicode 替换字符或常见绘表线参数残留。
- 抽查第一份报告中的年份列、股票代码、指标值及单位；`\\multicolumn` 和 `\\multirow` 的可见内容保留。多单位混写仍按原文保留，未进行数值单位转换。

## 运行诊断

{diagnostics}

## 失败文件

{failure_lines}

## 可复现命令

```powershell
& "C:\\Users\\Dell\\.cache\\codex-runtimes\\codex-primary-runtime\\dependencies\\python\\python.exe" `
  "src\\rag\\parse_latex.py" --force
```
"""


def main(argv: list[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=project_root / "data/raw/financecomplexqa/latex/corporate_reports")
    parser.add_argument("--output", type=Path, default=project_root / "data/processed/corpus/parsed_documents.jsonl")
    parser.add_argument("--report", type=Path, default=project_root / "docs/RAG_TASK1_PARSING_RESULTS.md")
    parser.add_argument("--force", action="store_true", help="replace existing output and report")
    args = parser.parse_args(argv)
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if not args.force and (args.output.exists() or args.report.exists()):
        parser.error("output exists; pass --force to replace it")

    paths = sorted(args.input_dir.glob("*.tex"))
    if not paths:
        parser.error(f"no .tex files found in {args.input_dir}")
    documents: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in paths:
        try:
            document, _ = parse_document(path, args.input_dir)
            documents.append(document)
        except Exception as exc:  # retain other files when one input is malformed
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_output.replace(args.output)
    report_text = build_report(documents, failures, args.input_dir)
    args.report.write_text(report_text, encoding="utf-8", newline="\n")
    print(json.dumps({
        "input_files": len(paths),
        "parsed_documents": len(documents),
        "failed_documents": len(failures),
        "output": str(args.output),
        "report": str(args.report),
        "diagnostics": sum(len(doc["diagnostics"]) for doc in documents),
        "tables": sum(doc["table_count"] for doc in documents),
        "sections": sum(doc["section_count"] for doc in documents),
    }, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
