"""Source-neutral financial vocabulary and small, question-selected prompt cards.

Exact aliases are usable by validators; related terms are retrieval hints only.
The JSON contains neither benchmark questions nor answers. This module imports
no generation, retrieval, financial-semantic or field-planning module.
"""
from __future__ import annotations

import json
from pathlib import Path
import re


LEXICON_PATH = Path(__file__).resolve().parents[2] / "configs/financial_lexicon_v4.json"


def normalise_term(term: str) -> str:
    """Ignore whitespace and ASCII case only; preserve financial qualifiers."""
    if not isinstance(term, str):
        return ""
    compact = re.sub(r"\s+", "", term)
    return re.sub(r"[A-Z]", lambda match: match.group().lower(), compact)


def term_pattern(term: str) -> str:
    """Prevent PE/EPS/EBIT matching the middle of an English identifier."""
    escaped = re.escape(term)
    prefix = r"(?<![A-Za-z0-9_])" if term and term[0].isascii() and term[0].isalnum() else ""
    suffix = r"(?![A-Za-z0-9_])" if term and term[-1].isascii() and term[-1].isalnum() else ""
    return prefix + escaped + suffix


def load_lexicon(path: Path = LEXICON_PATH):
    """Fail early on ambiguous aliases, contradictory exclusions or long cards."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("metrics")
    if not isinstance(records, list) or not records:
        raise ValueError("金融词典必须包含非空 metrics 列表")
    names, index, metrics = {}, {}, {}
    kinds = {"money", "ratio", "shares", "per_share", "day"}
    for record in records:
        name = record.get("name")
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("金融词典指标名称无效")
        if name in names or normalise_term(name) in index:
            raise ValueError(f"重复指标名称: {name}")
        if record.get("unit_kind") not in kinds:
            raise ValueError(f"指标单位类型无效: {name}")
        aliases = []
        for key in ("aliases", "compatibility_aliases", "related_terms", "do_not_substitute"):
            values = record.get(key, [])
            if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"指标 {name} 的 {key} 必须为非空字符串列表")
        local = set()
        for alias in [name, *record["aliases"], *record.get("compatibility_aliases", [])]:
            normal = normalise_term(alias)
            # The canonical name may be repeated verbatim in the displayed aliases.
            if normal in local:
                if alias == name:
                    continue
                raise ValueError(f"指标 {name} 重复别名: {alias}")
            if normal in index and index[normal] != name:
                raise ValueError(f"别名冲突: {alias} 同时指向 {index[normal]} 和 {name}")
            local.add(normal)
            index[normal] = name
            aliases.append(alias)
        names[name] = record
        metrics[name] = tuple(aliases)
    for name, record in names.items():
        for excluded in record.get("do_not_substitute", []):
            if index.get(normalise_term(excluded)) == name:
                raise ValueError(f"别名与禁止混同项矛盾: {name}/{excluded}")
    cards = payload.get("planning_cards", [])
    ids = set()
    for card in cards:
        if not isinstance(card.get("id"), str) or card["id"] in ids:
            raise ValueError("计算语义卡 ID 缺失或重复")
        ids.add(card["id"])
        if not isinstance(card.get("text"), str) or not card["text"] or len(card["text"]) > 180:
            raise ValueError(f"计算语义卡须为 180 字符以内: {card['id']}")
        for key in ("triggers", "formula_ids"):
            if not isinstance(card.get(key, []), list) or any(not isinstance(term, str) or not term for term in card.get(key, [])):
                raise ValueError(f"计算语义卡 {card['id']} 的 {key} 无效")
        if not isinstance(card.get("priority", 0), int):
            raise ValueError(f"计算语义卡 {card['id']} priority 无效")
    return payload, names, index, metrics


LEXICON, RECORDS, _ALIAS_INDEX, METRICS = load_lexicon()
VERSION = LEXICON["version"]


def canonical(metric: str) -> str:
    """Resolve a complete term; unknown phrases keep their original identity."""
    if not isinstance(metric, str):
        return metric
    return _ALIAS_INDEX.get(normalise_term(metric), metric.strip())


def aliases(metric: str) -> list[str]:
    key = canonical(metric)
    return list(METRICS.get(key, (metric,)))


def metric_card(metric: str) -> str:
    """Short typed definition for the requested metric, never a numeric answer."""
    record = RECORDS.get(canonical(metric))
    if record is None:
        return ""
    names = {"money": "金额", "ratio": "比率", "shares": "股数", "per_share": "每股金额", "day": "天数"}
    card = f"{record['name']}[{names[record['unit_kind']]}]：{record['definition']}"
    forbidden = record.get("do_not_substitute", [])[:3]
    if forbidden:
        card += "；不可替换为" + "/".join(forbidden)
    if record.get("compatibility_aliases"):
        card += "；兼容表述须核对原文口径"
    return card


def planning_guidance(question: str, formula: str | None = None) -> str:
    """Select at most two cards; longest specific concepts have higher priority."""
    candidates = []
    for order, card in enumerate(LEXICON.get("planning_cards", [])):
        formula_match = formula is not None and formula in card.get("formula_ids", [])
        text_match = any(re.search(term_pattern(term), question, re.I) for term in card.get("triggers", []))
        if formula_match or text_match:
            candidates.append((-card.get("priority", 0), order, card))
    selected = []
    used_groups = set()
    for _, _, card in sorted(candidates, key=lambda item: item[:2]):
        group = card.get("exclusive_group")
        if group and group in used_groups:
            continue
        selected.append(card["text"])
        if group:
            used_groups.add(group)
        if len(selected) == 2:
            break
    return "\n".join(selected)
