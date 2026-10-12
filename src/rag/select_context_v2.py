"""Build compact, operand-aware contexts from frozen Top-20 rerank results.

This version never reads reference documents, answers, or error-analysis labels.
It is an exploratory selector; evaluate on a separate holdout before claiming gain.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re

from generate_calculate_v2 import messages_for
from prompt_budget import count_chat_tokens, load_tokenizer, resolve_tokenizer_path
from rerank_context import format_evidence, read_jsonl
from table_structure import table_structure


PROJECT = Path(__file__).resolve().parents[2]
RERANKED = PROJECT / "data/processed/eval/task4_reranked_contexts.jsonl"
CHUNKS = PROJECT / "data/processed/corpus/evidence_chunks.jsonl"
OUTPUT = PROJECT / "data/processed/eval/task4_v3_contexts.jsonl"
VERSION = "coverage-v3-token"
MAX_PROMPT_TOKENS = 2600
MAX_GENERATION_TOKENS = 512
MODEL_CONTEXT_TOKENS = 3200
MAX_BLOCKS = 5
DECIMAL = re.compile(r"(?<!\d)\d[\d,]*\.\d{2,}(?!\d)")
PROSE_NUMERIC_WITH_UNIT = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?\d[\d,]*(?:\.\d+)?\s*(?:亿元|百万元|万元|千元|人民币元|元/股|元|%|％|倍)"
)
PROSE_NUMERIC = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d[\d,]*(?:\.\d+)?(?![A-Za-z0-9_.])")
YEAR = re.compile(r"(?<!\d)(20\d{2})(?:\d{4}(?!\d)|(?!\d))")
FULL_DATE = re.compile(r"20\d{6}")
UNKNOWN_COMPANY = "未识别机构"


@dataclass(frozen=True)
class Slot:
    name: str
    aliases: tuple[str, ...]
    year: str | None
    date_tokens: tuple[str, ...] = ()
    kind: str = "financial"


def _question_supplies_metric_amount(question: str, year: int, *metrics: str) -> bool:
    """Only an explicit amount fills a formula operand; a percent change does not."""
    alternatives = "|".join(re.escape(metric) for metric in metrics)
    amount = r"[-+]?\d[\d,]*(?:\.\d+)?\s*(?:亿元|百万元|万元|千元|元)"
    patterns = (
        rf"{year}\s*年(?:报|末|度)?[^，,。；;]{{0,35}}(?:{alternatives})"
        rf"[^，,。；;]{{0,20}}(?:为|是|约为|达到|增至)\s*{amount}",
        rf"(?:{alternatives})[^，,。；;]{{0,20}}{year}\s*年"
        rf"[^，,。；;]{{0,20}}(?:为|是|约为|达到|增至)\s*{amount}",
    )
    return any(re.search(pattern, question) for pattern in patterns)


def required_slots(question: str) -> list[Slot]:
    """Infer report-sourced operands while honoring values and scenarios in the question."""
    q = question.lower()
    years = [int(x) for x in YEAR.findall(q)]
    current = max(years) if years else None
    slots: list[Slot] = []

    def add(name: str, aliases: tuple[str, ...], year: int | None = current,
            *, date_tokens: tuple[str, ...] = (), kind: str = "financial") -> None:
        slot = Slot(name, aliases, str(year) if year is not None else None,
                    date_tokens, kind)
        if slot not in slots:
            slots.append(slot)

    if "静态市盈率" in q or "静态pe" in q or "静态市盈率" in question:
        # Price/PB/EPS may be explicitly supplied as scenario inputs. Only request
        # report values that remain necessary for this question's actual formula.
        dates = sorted(set(x for x in FULL_DATE.findall(question) if x.startswith("2025")))
        date_range = re.search(r"(20\d{6})\s*(?:到|至|--|—|~|～)\s*(20\d{6})", question)
        period_dates = list(date_range.groups()) if date_range else dates
        prior = current - 1 if current else 2024
        if "总市值" in q or "市值" in q:
            # Static P/E from market capitalization uses annual net profit.
            add("净利润", ("归属于母公司股东的净利润", "归母净利润", "净利润"), prior)
        elif "pb" in q or "市净率" in q:
            pb_has_value = bool(re.search(r"(?:市净率|pb)[^，,。；;]{0,24}\d", q))
            if not pb_has_value:
                pb_dates = sorted(set(x for x in FULL_DATE.findall(question)
                                      if x.startswith("2025")))
                add("市净率", ("市净率", "pb", "p/b"), 2025,
                    date_tokens=(pb_dates[-1][4:],) if pb_dates else ())
            add("每股净资产", ("每股净资产", "bps"), prior)
            add("每股收益", ("每股收益", "eps"), prior)
        else:
            eps_in_question = bool(re.search(r"(?:每股收益|eps)[^，,。；;]{0,24}\d", q))
            has_explicit_change = bool(re.search(
                r"(?:涨跌为|涨跌额为|日涨跌为|涨跌幅为)\s*-?\d", q))
            is_multi_day_order = any(term in q for term in ("最高", "最低", "次高", "次低"))
            if is_multi_day_order and len(period_dates) >= 2:
                # Period extrema need the target's price path; question-supplied
                # EPS/one-day prices override their report counterparts.
                add("区间股价序列",
                    ("收盘价", "每日收盘价", "股价涨跌", "涨跌金额", "涨跌额",
                     "每日涨跌", "涨跌幅"),
                    int(period_dates[0][:4]),
                    date_tokens=(period_dates[0][4:], period_dates[-1][4:]), kind="price_series")
                if not eps_in_question:
                    add("每股收益", ("每股收益", "eps"), prior)
            elif "下一个交易日" in q or "下一交易日" in q:
                # Unless the change itself is given, use the report's next-day
                # change and the close in the question to construct that price.
                if not has_explicit_change:
                    date_token = "08/21"
                    if dates:
                        month = int(dates[0][4:6])
                        day = int(dates[0][6:8]) + 1
                        date_token = f"{month:02d}/{day:02d}"
                    add("下一交易日价格变动", ("涨跌金额", "每日涨跌金额", "日涨跌额",
                                                  "每日涨跌额", "涨跌额", "涨跌(元)",
                                                  "涨跌（元）", "日涨跌幅",
                                                  "每日涨跌幅", "涨跌幅", "收盘价"),
                        current or 2025, date_tokens=(date_token,), kind="price_change")
                if not eps_in_question:
                    add("每股收益", ("每股收益", "eps"), prior)
            elif not eps_in_question:
                add("每股收益", ("每股收益", "eps"), prior)
    elif "现金流量比率" in q:
        add("经营现金流", ("经营活动产生的现金流量净额", "经营活动现金流量净额",
                           "经营活动现金流净额", "经营现金流量净额", "经营现金流"))
    elif "现金比率" in q:
        add("货币资金", ("货币资金",))
        add("交易性金融资产", ("交易性金融资产",))
    elif "速动比率" in q:
        add("流动资产", ("流动资产合计", "流动资产（", "流动资产("))
        add("流动负债", ("流动负债合计", "流动负债（", "流动负债("))
        if "存货为" not in q:
            add("存货", ("存货",))
    elif "营业利润率" in q:
        if not re.search(r"营业利润[^，,。；;]{0,30}\d", q):
            add("营业利润", ("营业利润", "经营利润"))
        if "净利润" in q and "净利率" in q:
            add("净利润", ("净利润", "归母净利润", "归属于母公司股东的净利润"), current)
            add("销售净利率", ("销售净利率", "净利率"), current)
        elif not re.search(r"营业收入[^，,。；;]{0,30}\d", q):
            add("营业收入", ("营业收入", "营业总收入"), current)
    elif "固定资产增长率" in q:
        add("固定资产上期", ("固定资产",), current - 1 if current else None)
        if "固定资产为" not in q:
            add("固定资产当期", ("固定资产",), current)
    elif "归属母公司股东的净利润" in q and "同比增长率" in q:
        add("归母净利润上期", ("归属于母公司股东的净利润", "归母净利润"),
            current - 1 if current else None)
    elif "营业利润" in q and "复合增长率" in q and len(set(years)) >= 2:
        start_year, end_year = min(years), max(years)
        if not _question_supplies_metric_amount(question, end_year, "营业利润", "经营利润"):
            add("营业利润终点", ("营业利润", "经营利润"), end_year)
        if not _question_supplies_metric_amount(question, start_year, "营业利润", "经营利润"):
            add("营业利润起点", ("营业利润", "经营利润"), start_year)
    elif "同比增长率" in q and current is not None:
        for name, aliases in (
            ("营业收入", ("营业收入", "营业总收入")),
            ("净利润", ("归母净利润", "归属于母公司股东的净利润", "净利润")),
            ("固定资产", ("固定资产",)),
        ):
            if any(alias in q for alias in aliases):
                add(name + "上期", aliases, current - 1)
                break
    return slots


def target_entity(company: str, question: str) -> str:
    """Recover the named issuer when the corpus metadata says 未识别机构."""
    if company != UNKNOWN_COMPANY:
        return company
    cleaned = re.sub(r"^(?:假如|如果|若|倘若)\s*", "", question.strip())
    match = re.match(r"([\u4e00-\u9fffA-Za-z·()（）]{2,20}?)(?=20\d|在|的|于|，|,)", cleaned)
    return match.group(1) if match else company


def _numeric_cell(value: str) -> bool:
    return bool(re.search(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", value or ""))


def _table_header_rows(chunk: dict) -> list[dict]:
    return table_structure(chunk)["header_rows"]


def _table_data_rows(chunk: dict) -> list[dict]:
    return table_structure(chunk)["data_rows"]


def _normalize_period_label(label: str) -> str:
    text = str(label)
    chinese = re.fullmatch(r"(\d{1,2})月(\d{1,2})日", text)
    if chinese:
        return f"{int(chinese.group(1)):02d}{int(chinese.group(2)):02d}"
    return re.sub(r"\D", "", text)


def _header_columns(chunk: dict) -> dict[int, str]:
    rows = _table_header_rows(chunk)
    width = max((len(row.get("display_cells", [])) for row in rows), default=0)
    return {
        index: " ".join(str(row.get("display_cells", [])[index])
                        for row in rows if index < len(row.get("display_cells", [])))
        for index in range(width)
    }


def _year_labels(chunk: dict) -> list[tuple[str, int]]:
    """Flatten multi-row table headers while retaining their displayed order."""
    rows = _table_header_rows(chunk)
    labels: list[tuple[str, int]] = []
    for row in rows:
        for cell in row.get("display_cells", []):
            text = re.sub(r"\s+", "", str(cell))
            markers = re.findall(
                r"20\d{6}|20\d{2}(?:[-/]\d{1,2}[-/]\d{1,2}|年度|年报|年)?|"
                r"(?<!\d)\d{1,2}[-/]\d{1,2}(?!\d)|\d{1,2}月\d{1,2}日", text)
            for marker in markers:
                labels.append((marker, len(labels)))
    # Multi-row LaTeX headers sometimes repeat the same date across header rows.
    unique: list[tuple[str, int]] = []
    seen = set()
    for label, ordinal in labels:
        normalized = _normalize_period_label(label)
        if normalized not in seen:
            unique.append((label, ordinal))
            seen.add(normalized)
    return unique


def _row_numeric_cells(row: dict) -> list[tuple[int, str]]:
    cells = [str(cell) for cell in row.get("display_cells", [])]
    return [(index, cell) for index, cell in enumerate(cells)
            if _numeric_cell(cell) and not re.fullmatch(r"\d{6}\.[A-Za-z]+", cell.strip())]


def _normal_text(text: str) -> str:
    return re.sub(r"[\s（）()]", "", str(text)).lower()


def _period_value_cells(row: dict, year_labels: list[tuple[str, int]],
                        slot: Slot, header_columns: dict[int, str]) -> list[tuple[int, str, int]]:
    """Map numeric cells to periods, including one date split into subcolumns."""
    numeric = _row_numeric_cells(row)
    label_count = len(year_labels)
    if not label_count or len(numeric) < label_count:
        return []
    if len(numeric) % label_count == 0:
        repeat = len(numeric) // label_count
        expanded = []
        for ordinal in range(label_count):
            group = numeric[ordinal * repeat:(ordinal + 1) * repeat]
            alias_cells = [(index, cell) for index, cell in group
                           if any(_normal_text(alias) in _normal_text(header_columns.get(index, ""))
                                  for alias in slot.aliases)]
            # For hierarchical headers the same day can span amount and rate
            # cells; keep only the subcolumn matching this formula field.
            for index, cell in (alias_cells or group):
                expanded.append((index, cell, ordinal))
        return expanded
    # If an extra numeric ID/subtotal is present, preserve the rightmost
    # period-count cells as the prior parser did.
    return [(index, cell, ordinal)
            for ordinal, (index, cell) in enumerate(numeric[-label_count:])]


def _date_patterns(token: str) -> list[re.Pattern]:
    digits = re.sub(r"\D", "", token)
    if len(digits) < 4:
        return []
    month, day = int(digits[-4:-2]), int(digits[-2:])
    month_text, day_text = f"0?{month}", f"0?{day}"
    return [
        re.compile(rf"20\d{{2}}年\s*{month_text}月\s*{day_text}日"),
        re.compile(rf"20\d{{2}}[-/]0?{month}[-/]0?{day}"),
        re.compile(rf"20\d{{2}}{month:02d}{day:02d}"),
        re.compile(rf"(?<!\d){month_text}[-/]{day_text}(?!\d)"),
        re.compile(rf"(?<!\d){month_text}月\s*{day_text}日"),
    ]


def _date_associated_value(sentence: str, token: str) -> bool:
    """Require a date to be tied to a nearby quoted price/change value."""
    matches = []
    for pattern in _date_patterns(token):
        matches.extend(pattern.finditer(sentence))
    if not matches:
        return False
    matches.sort(key=lambda match: match.start())
    for match in matches:
        following_dates = [other.start() for other in re.finditer(
            r"20\d{6}|20\d{2}年\s*\d{1,2}月\s*\d{1,2}日|"
            r"20\d{2}[-/]\d{1,2}[-/]\d{1,2}|(?<!\d)\d{1,2}月\s*\d{1,2}日|"
            r"(?<!\d)\d{1,2}[-/]\d{1,2}(?!\d)", sentence[match.end():])]
        end = match.end() + following_dates[0] if following_dates else min(len(sentence), match.end() + 72)
        segment = sentence[match.end():min(end, match.end() + 72)]
        if not re.search(r"收盘价|收盘|股价|涨跌额|涨跌金额|涨跌幅|涨跌|价格", segment):
            continue
        if re.search(r"(?<!\d)[+-]?\d[\d,]*(?:\.\d+)?\s*(?:元/股|元|%|％)?", segment):
            return True
    return False


def _prose_metric_value_present(slot: Slot, sentence: str, *, target: str = "",
                                peer_names: set[str] | None = None) -> bool:
    """Require the metric and its value to be locally associated in prose.

    Report prose may mention a metric/year while also containing unrelated
    decimals such as table confidence scores (for example, 0.95). Financial
    amounts therefore need an explicit adjacent unit; dimensionless PE/PB may
    omit a unit but must sit close to the corresponding metric name.
    """
    aliases = [match for alias in slot.aliases for match in
               re.finditer(re.escape(alias), sentence, re.IGNORECASE)]
    if not aliases:
        return False
    dimensionless = slot.name in {"市净率", "市盈率", "静态市盈率", "pb", "pe"}
    values = (PROSE_NUMERIC if dimensionless else PROSE_NUMERIC_WITH_UNIT).finditer(sentence)
    for value in values:
        raw = value.group().strip()
        number = re.match(r"[-+]?\d[\d,]*(?:\.\d+)?", raw)
        if not number:
            continue
        normalized = number.group().replace(",", "")
        if re.fullmatch(r"20\d{2}(?:\d{4})?", normalized):
            continue
        if any(min(abs(value.start() - alias.end()), abs(alias.start() - value.end())) <=
               (32 if dimensionless else 80) for alias in aliases):
            if peer_names and target:
                companies = [(match.start(), match.end(), target)
                             for match in re.finditer(re.escape(target), sentence)]
                for peer in peer_names:
                    companies.extend((match.start(), match.end(), peer)
                                     for match in re.finditer(re.escape(peer), sentence))
                preceding = [item for item in companies if item[1] <= value.start()]
                nearest = max(preceding, key=lambda item: item[1], default=None)
                if nearest is None or nearest[2] != target or value.start() - nearest[1] > 120:
                    continue
            return True
    return False


def _coded_company_names(text: str) -> set[str]:
    """Issuer names explicitly paired with stock codes in the same source chunk."""
    return {match.group(1) for match in re.finditer(
        r"([\u4e00-\u9fffA-Za-z·]{2,16})\s*[（(]\s*\d{6}\.(?:SH|SZ|BJ)\s*[）)]",
        text, re.IGNORECASE)}


def _table_slot_match(slot: Slot, chunk: dict, company: str, entity: str | None = None) -> dict:
    structure = table_structure(chunk)
    rows = chunk.get("rows", [])
    data_rows = structure["data_rows"]
    columns = _header_columns(chunk)
    year_labels = _year_labels(chunk)
    all_header = " ".join(columns.values())
    caption = chunk.get("table_caption", "") or ""
    target = entity or company
    # For recognized reports, the report identity is authoritative when a table
    # summarizes a single issuer and omits its name in the data rows. For the
    # generic report, require the actual issuer name in a data row.
    target_rows = [row for row in data_rows
                   if target in " ".join(row.get("display_cells", []))]
    issuer_header = any(term in all_header for term in ("公司", "股票简称", "股票代码", "标的公司"))
    section_text = " ".join(chunk.get("section_path", []))
    scoped_to_target = (chunk.get("company") == company and company != UNKNOWN_COMPANY) or (
        company == UNKNOWN_COMPANY and entity and entity in (caption + " " + section_text))
    if not issuer_header and scoped_to_target:
        # A company name in the report/caption scopes every metric row unless
        # the table itself has a company column. Identification rows such as
        # “688619.SH / 罗普特” must not hide the following PB/BPS/EPS rows.
        target_rows = data_rows
    if not target_rows and not rows:
        target_rows = []
    table_scope_text = " ".join((caption, all_header)).lower()
    metric_scope_hits = [alias for alias in slot.aliases
                         if alias.lower() in table_scope_text]
    metric_label_rows = [row for row in target_rows if any(
        alias.lower() in " ".join(map(str, row.get("display_cells", []))).lower()
        for alias in slot.aliases)]
    metric_groups: list[list[dict]] = []
    for metric_row in metric_label_rows:
        group = [metric_row]
        row_cells = [str(cell) for cell in metric_row.get("display_cells", [])]
        has_value = any(_numeric_cell(cell) for cell in row_cells)
        if not has_value:
            try:
                start = next(index for index, row in enumerate(target_rows)
                             if row.get("row") == metric_row.get("row"))
            except StopIteration:
                start = -1
            for continuation in target_rows[start + 1:]:
                cells = [str(cell).strip() for cell in continuation.get("display_cells", [])]
                continuation_text = " ".join(cells).lower()
                if any(alias.lower() in continuation_text for alias in slot.aliases):
                    break
                label = re.sub(r"[\s：:（）()]", "", cells[0]).lower() if cells else ""
                is_value_or_unit_row = any(term in label for term in (
                    "原始披露值", "披露值", "原值", "数值", "金额", "单位", "量纲",
                    "折算值", "转换值", "value", "unit"))
                if is_value_or_unit_row or (not label and any(_numeric_cell(cell) for cell in cells)):
                    group.append(continuation)
                else:
                    break
        metric_groups.append(group)
    # A metric-specific row is stronger than table-level text. If the metric
    # appears only in the title/header, treat the table as metric-specific;
    # otherwise do not let a different numeric row lend its value or unit.
    if not metric_groups and metric_scope_hits:
        metric_groups = [target_rows]
    metric_rows = list({row.get("row"): row for group in metric_groups for row in group}.values())
    metric_rows.sort(key=lambda row: row.get("row", 0))
    row_metric_hits = [alias for alias in slot.aliases if any(
        alias.lower() in " ".join(map(str, row.get("display_cells", []))).lower()
        for row in metric_label_rows)]
    metric_hits = list(dict.fromkeys([*metric_scope_hits, *row_metric_hits]))
    target_numeric_rows = [row for row in metric_rows if any(
        _numeric_cell(cell) for cell in row.get("display_cells", []))]

    label_count = len(year_labels)
    normalized_labels = [(label, _normalize_period_label(label)) for label, _ in year_labels]
    matching_ordinals: list[int] = []
    if slot.date_tokens:
        token_ordinals: list[int] = []
        for token in slot.date_tokens:
            wanted = _normalize_period_label(token)[-4:]
            hits = [ordinal for ordinal, (_, digits) in enumerate(normalized_labels)
                    if wanted and digits[-4:] == wanted]
            if hits:
                token_ordinals.append(hits[0])
        # Every requested endpoint must be present. For interval extrema, all
        # trading-date columns between the endpoints must also have values.
        if len(token_ordinals) == len(slot.date_tokens) and token_ordinals:
            if slot.kind == "price_series" and len(token_ordinals) >= 2:
                matching_ordinals = list(range(min(token_ordinals), max(token_ordinals) + 1))
            else:
                matching_ordinals = sorted(set(token_ordinals))
    elif slot.year:
        matching_ordinals = [ordinal for ordinal, (_, digits) in enumerate(normalized_labels)
                             if slot.year in digits]

    matched_indices_by_row: list[tuple[dict, list[tuple[int, str, int]]]] = []
    for row in metric_rows:
        period_values = _period_value_cells(row, year_labels, slot, columns)
        if period_values:
            mapped = [entry for entry in period_values if entry[2] in matching_ordinals]
            matched_indices_by_row.append((row, mapped))
    year_row_hit = any(slot.year and slot.year in " ".join(row.get("display_cells", []))
                       for row in metric_rows)
    date_row_text = " ".join(" ".join(map(str, row.get("display_cells", [])))
                             for row in metric_rows)
    date_row_hit = bool(slot.date_tokens) and all(
        any(pattern.search(date_row_text) for pattern in _date_patterns(token))
        for token in slot.date_tokens)
    numeric_column_hit = bool(matching_ordinals) and any(
        {entry[2] for entry in mapped} == set(matching_ordinals) and all(
            index < len(row.get("display_cells", [])) and
            _numeric_cell(row.get("display_cells", [])[index])
            for index, _, _ in mapped)
        for row, mapped in matched_indices_by_row)
    if slot.date_tokens:
        # If the date is a real table column, require the corresponding numeric
        # cells. For key/value rows with the date in the metric label, bind the
        # number in that same row instead.
        period_column_hit = numeric_column_hit or (date_row_hit and not year_labels)
    elif slot.year and year_labels:
        period_column_hit = numeric_column_hit
    else:
        period_column_hit = True
    year_column_hit = bool(matching_ordinals) if slot.year and not slot.date_tokens else True
    # A metric row may spell out its own year (e.g. "2024 年营业利润") rather
    # than using a year column; keep that separate from column-based evidence.
    year_hit = (not slot.year or year_column_hit or year_row_hit)
    metric_row_hit = bool(metric_hits)
    value_hit = bool(target_numeric_rows) and period_column_hit
    header_unit = re.search(r"表头附近出现的单位：([^\n]+)", chunk.get("text", ""))
    global_unit_context = " ".join((caption, all_header,
                                    header_unit.group(1) if header_unit else ""))
    unit_context_parts = [global_unit_context]
    unit_context_parts.extend(" ".join(map(str, row.get("display_cells", [])))
                               for row in metric_label_rows)
    unit_row_numbers: set[int] = set()
    for row, mapped in matched_indices_by_row:
        for group in metric_groups:
            if row not in group:
                continue
            for index, value, _ in mapped:
                unit_context_parts.append(value)
                for related in group:
                    cells = [str(cell) for cell in related.get("display_cells", [])]
                    label = re.sub(r"[\s：:（）()]", "", cells[0]).lower() if cells else ""
                    if any(term in label for term in ("单位", "unit")):
                        unit_row_numbers.add(related.get("row"))
                        if index < len(cells):
                            unit_context_parts.append(cells[index])
    unit_context = " ".join(unit_context_parts)
    unit_matches = list(dict.fromkeys(re.findall(
        r"亿元|百万元|万元|千元|元/股|万股|亿股|人民币元|元|%|％|倍|天",
        unit_context)))
    unit_present = bool(unit_matches)
    unit_required = slot.name not in {"市净率", "市盈率", "静态市盈率", "pb", "pe"}
    structural_match = metric_row_hit and value_hit and year_hit
    # An unlabeled table value is not complete evidence for a financial
    # operand: the unit must be visible in the table's source text/metadata.
    matched = structural_match and (unit_present or not unit_required)
    return {
        "matched": matched,
        "structural_match": structural_match,
        "content_type": "table",
        "caption": caption,
        "target_data_rows": [row.get("row") for row in target_numeric_rows],
        "effective_header_row_numbers": structure["header_row_numbers"],
        "inferred_header_row_numbers": structure["inferred_header_row_numbers"],
        "effective_data_row_numbers": structure["data_row_numbers"],
        "metric_data_row_numbers": [row.get("row") for row in metric_rows],
        "unit_data_row_numbers": sorted(number for number in unit_row_numbers
                                         if number is not None),
        "matched_year_columns": {str(i): year_labels[i][0] for i in matching_ordinals
                                 if i < len(year_labels)},
        "matched_period_values": [
            {"row": row.get("row"), "column_index": index,
             "period": year_labels[ordinal][0], "value": value}
            for row, mapped in matched_indices_by_row for index, value, ordinal in mapped
            if ordinal < len(year_labels)
        ],
        "metric_aliases_found": metric_hits,
        "unit_mentions": chunk.get("unit_mentions", []),
        "unit_matches_in_source": unit_matches,
        "unit_status": ("present" if unit_present else
                        "not_required_dimensionless" if not unit_required else
                        "not_explicitly_recognized"),
        "year_column_verified": year_column_hit or year_row_hit,
        "period_columns_verified": numeric_column_hit if slot.date_tokens else None,
        "value_in_requested_period_verified": period_column_hit,
        "date_row_verified": date_row_hit,
    }


def slot_match_info(slot: Slot, chunk: dict, company: str, entity: str | None = None) -> dict:
    """Check metric/value/year in the same target table row or prose passage."""
    body = chunk.get("content_text", "")
    full_text = " ".join((chunk.get("table_caption", "") or "", body,
                          chunk.get("text", ""), " ".join(chunk.get("section_path", [])),
                          chunk.get("report_title", "")))
    target = entity or company
    report_match = chunk.get("company") == company
    if company == UNKNOWN_COMPANY:
        report_match = report_match and target in full_text
    else:
        report_match = report_match or target in full_text
    if not report_match:
        return {"matched": False, "reason": "target_company_not_in_evidence"}
    if chunk.get("content_type") == "table":
        return _table_slot_match(slot, chunk, company, entity)

    # Match against report prose itself, not the synthetic metadata prefix in
    # chunk["text"]. Otherwise the report company's name can make a peer's
    # figures appear to belong to the target company.
    source_text = " ".join((chunk.get("table_caption", "") or "", body))
    peer_names = _coded_company_names(source_text) - {target}
    for sentence in re.split(r"[。；\n]", source_text):
        if not any(alias.lower() in sentence.lower() for alias in slot.aliases):
            continue
        if slot.year and slot.year not in sentence:
            continue
        if target not in sentence:
            # A prose block can continue discussing a comparator over several
            # sentences without repeating its name. If a code-linked peer is
            # named anywhere in the passage, require the target issuer to be
            # explicit in the value sentence instead of inheriting the report
            # metadata as the subject.
            if (peer_names or
                    company == UNKNOWN_COMPANY or chunk.get("company") != company):
                continue
        if slot.date_tokens and not all(_date_associated_value(sentence, token)
                                        for token in slot.date_tokens):
            # A range label is not a price series. Each endpoint must be tied
            # to its own nearby close/change value in this same passage.
            continue
        if _prose_metric_value_present(slot, sentence, target=target,
                                       peer_names=peer_names):
            return {"matched": True, "content_type": "prose", "year": slot.year,
                    "date_tokens": list(slot.date_tokens), "source_lines": [
                        chunk.get("source_start_line"), chunk.get("source_end_line")],
                    "unit_mentions": chunk.get("unit_mentions", [])}
    return {"matched": False, "reason": "metric_year_value_not_co_located"}


def slot_match(slot: Slot, chunk: dict, company: str, entity: str | None = None) -> bool:
    return bool(slot_match_info(slot, chunk, company, entity).get("matched"))


def table_binding_match(slot: Slot, chunk: dict, company: str, entity: str,
                        value: str, unit: str) -> dict:
    """Verify a proposed operand against the target table row/year column/unit."""
    details = _table_slot_match(slot, chunk, company, entity)
    if not details.get("matched"):
        return {**details, "matched": False, "reason": "table_metric_company_or_year_not_verified"}
    try:
        expected_value = Decimal(str(value).replace(",", ""))
    except InvalidOperation:
        return {**details, "matched": False, "reason": "binding_value_is_not_numeric"}
    row_numbers = set(details.get("target_data_rows", []))
    data_rows = [row for row in chunk.get("rows", []) if row.get("row") in row_numbers]
    labels = _year_labels(chunk)
    if slot.date_tokens:
        year_markers = [label for label, _ in labels]
        chosen_ordinals = [index for index, label in enumerate(year_markers)
                           if any(_normalize_period_label(token)[-4:] in
                                  _normalize_period_label(label)[-4:] for token in slot.date_tokens)]
    elif slot.year:
        chosen_ordinals = [index for index, (label, _) in enumerate(labels)
                           if slot.year in _normalize_period_label(label)]
    else:
        chosen_ordinals = []
    candidate_cells: list[tuple[int, dict, str]] = []
    header_columns = _header_columns(chunk)
    for row in data_rows:
        cells = [str(cell) for cell in row.get("display_cells", [])]
        if chosen_ordinals and labels:
            mapped = _period_value_cells(row, labels, slot, header_columns)
            for index, cell, ordinal in mapped:
                if ordinal in chosen_ordinals:
                    candidate_cells.append((index, row, cell))
        else:
            for index, cell in enumerate(cells):
                if _numeric_cell(cell):
                    candidate_cells.append((index, row, cell))
    matched_cell = None
    for index, row, cell in candidate_cells:
        found = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", cell)
        if not found:
            continue
        try:
            actual_value = Decimal(found.group().replace(",", ""))
        except InvalidOperation:
            continue
        if actual_value == expected_value:
            matched_cell = (index, row, cell)
            break
    if matched_cell is None:
        return {**details, "matched": False, "reason": "value_not_in_verified_target_year_cell"}
    # Check only table-wide headings and the verified metric/value/unit rows.
    # `unit_mentions` and unrelated data rows can contain another metric's unit;
    # substring matching also mistakes 元 for 万元.
    column = matched_cell[0]
    specific_units = [matched_cell[2]]
    for row in chunk.get("rows", []):
        cells = row.get("display_cells", [])
        if row.get("row") in details.get("unit_data_row_numbers", []) and column < len(cells):
            specific_units.append(str(cells[column]))
    def source_units_in(text):
        return {"%" if found == "％" else "元" if found == "人民币元" else found
                for found in re.findall(
                    r"万元/股|人民币元|百万元|亿元|千元|万元|元/股|万股|亿股|元|股|[%％]|倍|天", text)}
    source_units = source_units_in(" ".join(specific_units))
    if not source_units:
        source_units = source_units_in(header_columns.get(column, ""))
    if not source_units:
        cells = matched_cell[1].get("display_cells", [])
        nearby = re.search(r"表头附近出现的单位：([^\n]+)", chunk.get("text", ""))
        source_units = source_units_in(" ".join((
            chunk.get("table_caption", "") or "", str(cells[0]) if cells else "",
            nearby.group(1) if nearby else "")))
    per_share = any("每股" in alias or alias.lower() in {"eps", "bps"}
                    or alias in {"股价", "收盘价", "收盘价格", "涨跌金额", "涨跌额"}
                    for alias in slot.aliases)
    unit_verified = ((unit == "decimal" and (not source_units or source_units == {"倍"}))
                     or unit in source_units
                     or (per_share and unit in {"元/股", "万元/股"}
                         and unit.split("/")[0] in source_units))
    if not unit_verified:
        return {**details, "matched": False, "reason": "unit_not_present_in_source_table",
                "source_cell": matched_cell[2], "unit_verified": False}
    return {"matched": True, **details, "source_row": matched_cell[1].get("row"),
            "source_column": matched_cell[0], "source_cell": matched_cell[2],
            "unit_verified": unit_verified}


def candidate_score(candidate: dict, chunk: dict, company: str,
                    uncovered: set[str], slots: list[Slot],
                    entity: str | None = None) -> tuple[int, int]:
    matched = {slot.name for slot in slots if slot_match(slot, chunk, company, entity)}
    gain = len(matched & uncovered)
    target = entity or company
    target_report = chunk.get("company") == company
    target_body = target in (chunk.get("text", "") + chunk.get("content_text", ""))
    table = chunk.get("content_type") == "table"
    structural = (30 if target_report else -100) + (12 if target_body else 0) + (8 if table else 0)
    precision = max((len(match.group().split(".")[-1]) for match in DECIMAL.finditer(
        chunk.get("content_text", ""))), default=0)
    structural += min(12, max(0, precision - 2) * 2)
    structural -= min(candidate["rerank_rank"], 20)
    return gain, structural


def _candidate_matches_target(chunk: dict, company: str, entity: str) -> bool:
    text = " ".join((chunk.get("content_text", ""), chunk.get("text", "")))
    if company == UNKNOWN_COMPANY:
        return chunk.get("company") == company and entity in text
    return chunk.get("company") == company


def _ordered_selection(selected: list[tuple[dict, dict]], slots: list[Slot],
                       company: str, entity: str) -> list[tuple[dict, dict]]:
    slot_order = {slot.name: index for index, slot in enumerate(slots)}

    def key(pair: tuple[dict, dict]) -> tuple:
        candidate, chunk = pair
        matches = [slot_order[slot.name] for slot in slots
                   if slot_match(slot, chunk, company, entity)]
        group = min(matches) if matches else len(slots)
        # Formula fields are primary groups. Inside each field group, keep a
        # report together and restore the order of its original source lines.
        return (group, chunk.get("doc_id", ""),
                chunk.get("source_start_line", 0), chunk.get("source_end_line", 0),
                candidate.get("rerank_rank", 999), chunk.get("chunk_id", ""))

    return sorted(selected, key=key)


def _render_selection(row: dict, selected: list[tuple[dict, dict]], slots: list[Slot],
                      company: str, entity: str, tokenizer,
                      max_prompt_tokens: int) -> tuple[str, list[dict], list[dict], int]:
    ordered = _ordered_selection(selected, slots, company, entity)
    parts: list[str] = []
    evidence: list[dict] = []
    for item, chunk in ordered:
        ref_number = len(evidence) + 1
        parts.append(format_evidence(ref_number, chunk))
        coverage = {slot.name: slot_match_info(slot, chunk, company, entity)
                    for slot in slots}
        structure = table_structure(chunk) if chunk.get("content_type") == "table" else None
        evidence.append({
            "citation": f"E{ref_number}", "chunk_id": item["chunk_id"],
            "doc_id": item["doc_id"], "source_path": chunk["source_path"],
            "source_start_line": chunk["source_start_line"],
            "source_end_line": chunk["source_end_line"],
            "section_path": chunk["section_path"], "content_type": chunk["content_type"],
            "table_id": chunk.get("table_id"),
            "header_row_numbers": (structure["header_row_numbers"] if structure
                                   else chunk.get("header_row_numbers", [])),
            "inferred_header_row_numbers": (structure["inferred_header_row_numbers"]
                                            if structure else []),
            "data_row_numbers": (structure["data_row_numbers"] if structure
                                 else chunk.get("data_row_numbers", [])),
            "years": chunk.get("years", []), "unit_mentions": chunk.get("unit_mentions", []),
            "rerank_rank": item["rerank_rank"], "rerank_logit": item.get("rerank_logit"),
            "covered_slots": sorted(name for name, details in coverage.items()
                                     if details.get("matched")),
            "coverage_details": coverage,
        })
    context = "\n\n".join(parts)
    candidate_row = {**row, "context_text": context, "selected_evidence": evidence}
    messages = messages_for(candidate_row, "rag")
    prompt_tokens = count_chat_tokens(tokenizer, messages)
    groups = []
    for slot in slots:
        refs = [item["citation"] for item in evidence if slot.name in item["covered_slots"]]
        groups.append({"field": slot.name, "year": slot.year,
                       "citations": refs, "covered": bool(refs)})
    if prompt_tokens > max_prompt_tokens:
        raise ValueError(f"internal selector error: rendered prompt uses {prompt_tokens} tokens, "
                         f"budget is {max_prompt_tokens}")
    return context, evidence, groups, prompt_tokens


def _refine_slots_from_retrieved_methodology(question: str, ranked: list[dict],
                                            chunks: dict[str, dict], slots: list[Slot],
                                            company: str, entity: str) -> list[Slot]:
    """Use report-described algebra when a required value is explicitly derivable."""
    q = question.lower()
    if "营业利润率" not in q or "营业收入" not in {slot.name for slot in slots}:
        return slots
    evidence_text = " ".join(chunks[item["chunk_id"]].get("text", "") for item in ranked)
    normalized = re.sub(r"\s+", "", evidence_text)
    revenue = next((slot for slot in slots if slot.name == "营业收入"), None)
    if not revenue:
        return slots
    described = ("营业收入=净利润÷销售净利率" in normalized or
                 "营业收入=净利润/销售净利率" in normalized or
                 ("营业收入本身并非直接披露" in normalized and "销售净利率" in normalized))
    # Even without a prose derivation, a same-report/year net-profit and net-
    # margin pair is enough to derive revenue from the definition of net margin.
    net_profit = Slot("净利润", ("净利润", "归母净利润", "归属于母公司股东的净利润"), revenue.year)
    net_margin = Slot("销售净利率", ("销售净利率", "净利率"), revenue.year)
    paired_docs: dict[str, set[str]] = {}
    for item in ranked:
        chunk = chunks[item["chunk_id"]]
        for probe in (net_profit, net_margin):
            if slot_match(probe, chunk, company, entity):
                paired_docs.setdefault(item["doc_id"], set()).add(probe.name)
    derivable_pair = any({"净利润", "销售净利率"}.issubset(found)
                         for found in paired_docs.values())
    direct_revenue = any(slot_match(revenue, chunks[item["chunk_id"]], company, entity)
                         for item in ranked)
    if direct_revenue or (not described and not derivable_pair):
        return slots
    replaced = [slot for slot in slots if slot.name != "营业收入"]
    replaced.extend([
        net_profit,
        net_margin,
    ])
    return replaced


def build_context(row: dict, chunks: dict[str, dict], tokenizer,
                  max_prompt_tokens: int = MAX_PROMPT_TOKENS,
                  max_blocks: int = MAX_BLOCKS) -> dict:
    forbidden = {"gold", "Reference_documents", "external_knowledge"}
    if forbidden & row.keys():
        raise ValueError(f"Evaluation field leaked into selector: {row.get('finqa_id')}")
    company = row["company_with_period"].strip()
    entity = target_entity(company, row["question"])
    if max_prompt_tokens + MAX_GENERATION_TOKENS > MODEL_CONTEXT_TOKENS:
        raise ValueError("prompt-token budget plus reserved generation exceeds the model context")
    slots = required_slots(row["question"])
    ranked = row["reranked"]
    if len({x["chunk_id"] for x in ranked}) != len(ranked):
        raise ValueError("Duplicate reranked chunk")
    for item in ranked:
        if item["chunk_id"] not in chunks:
            raise ValueError(f"Unknown chunk: {item['chunk_id']}")
        if item["doc_id"] != chunks[item["chunk_id"]]["doc_id"]:
            raise ValueError(f"Mismatched chunk document: {item['chunk_id']}")
    slots = _refine_slots_from_retrieved_methodology(row["question"], ranked, chunks,
                                                    slots, company, entity)
    remaining = ranked.copy()
    selected: list[tuple[dict, dict]] = []
    omitted: list[dict] = []
    uncovered = {slot.name for slot in slots}
    while remaining and len(selected) < max_blocks:
        candidates = sorted(
            remaining,
            key=lambda item: candidate_score(item, chunks[item["chunk_id"]], company,
                                             uncovered, slots, entity),
            reverse=True,
        ) if slots else remaining
        chosen = None
        for item in candidates:
            chunk = chunks[item["chunk_id"]]
            if not _candidate_matches_target(chunk, company, entity):
                continue
            proposed_selection = [*selected, (item, chunk)]
            try:
                context, evidence, field_groups, prompt_tokens = _render_selection(
                    row, proposed_selection, slots, company, entity, tokenizer, max_prompt_tokens)
            except ValueError as exc:
                if "uses" not in str(exc):
                    raise
                omitted.append({"chunk_id": item["chunk_id"], "reason": "qwen_prompt_token_budget"})
                remaining.remove(item)
                continue
            else:
                chosen = (item, chunk, context, evidence, field_groups, prompt_tokens)
                break
        if chosen is None:
            # All candidates either belong to another issuer or exceeded budget.
            if candidates and all(item not in remaining for item in candidates):
                break
            remaining = [item for item in remaining if _candidate_matches_target(
                chunks[item["chunk_id"]], company, entity)]
            if not remaining:
                break
            continue
        item, chunk, context, evidence, field_groups, prompt_tokens = chosen
        selected.append((item, chunk))
        gain = {slot.name for slot in slots if slot_match(slot, chunk, company, entity)} & uncovered
        uncovered -= gain
        remaining.remove(item)
        # Once all inferred operands are present, do not pad with weakly
        # related comparator prose. Unknown tasks retain the original Top-K.
        if slots and not uncovered:
            break

    context, evidence, field_groups, prompt_tokens = _render_selection(
        row, selected, slots, company, entity, tokenizer, max_prompt_tokens)
    return {
        "finqa_id": row["finqa_id"], "question": row["question"],
        "company_with_period": company, "target_entity": entity,
        "retrieval_query": row["retrieval_query"],
        "reranked": ranked, "context_text": context,
        "selected_evidence": evidence, "omitted_candidates": omitted,
        "context_chars": len(context), "prompt_tokens": prompt_tokens,
        "prompt_token_budget": max_prompt_tokens,
        "reserved_generation_tokens": MAX_GENERATION_TOKENS,
        "selector_version": VERSION,
        "required_slots": [slot.name for slot in slots],
        "field_groups": field_groups,
        "uncovered_slots": sorted(uncovered),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RERANKED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tokenizer-path", type=Path,
                        help="Qwen3-4B-Instruct-2507 tokenizer.json; also supports QWEN_TOKENIZER_PATH")
    parser.add_argument("--max-prompt-tokens", type=int, default=MAX_PROMPT_TOKENS)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_prompt_tokens < 1 or args.max_prompt_tokens + MAX_GENERATION_TOKENS > MODEL_CONTEXT_TOKENS:
        parser.error("--max-prompt-tokens plus the 512-token generation reserve must fit the 3200-token model context")
    try:
        tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
        tokenizer = load_tokenizer(tokenizer_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.output.exists():
        raise FileExistsError(f"Existing result is protected: {args.output}")
    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[:args.limit]
    chunks = {item["chunk_id"]: item for item in read_jsonl(CHUNKS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(build_context(row, chunks, tokenizer,
                                                  args.max_prompt_tokens), ensure_ascii=False) + "\n")
    partial.replace(args.output)
    print(json.dumps({"rows": len(rows), "output": str(args.output),
                      "selector_version": VERSION}, ensure_ascii=False))


if __name__ == "__main__":
    main()
