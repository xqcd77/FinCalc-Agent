"""Question-only coverage protection over independently scored candidates.

Financial requirements and verified numeric spans are the existing runtime
selector rules. No reference-document labels, gold answers or qrels are read.
Whole source blocks are retained; this module only changes ordering.
"""
from __future__ import annotations
import re

from select_context_v4 import report_requirements, strict_match, issuer_names
from select_context_v2 import target_entity
from table_delivery_v48 import CompanionIndex

VERSION = "financial-dependency-coverage-guard-v49r3-20261012"


def calculation_requirements(row):
    """Use runtime dependency blueprints; retrieve only missing report inputs.

    The PS fallback declares revenue as a dependency but leaves the fiscal year
    unspecified when the question provides no report-year basis. It never
    guesses the missing annual revenue or a date-specific market value.
    """
    from planning_blueprints_v4 import blueprint
    from question_conditions_v4 import question_conditions, date_mentions
    from financial_semantics_v4 import canonical
    q = row["question"]
    target = row.get("target_entity") or target_entity(row["company_with_period"], q)
    bp = blueprint(q, target, ())
    plan = (bp or {}).get("plan") or {}
    if plan.get("fields"):
        requirements = [f for f in plan["fields"] if f.get("source") == "report"]
    else:
        requirements = report_requirements(row)
    if re.search(r"(?:市销率|(?<![A-Za-z])PS(?![A-Za-z]))[^。；;?？]{0,25}(?:是多少|为多少|是多大)", q, re.I):
        supplied = {canonical(f["metric"]) for f in question_conditions(q, target)}
        if "营业收入" not in supplied and not any(canonical(f["metric"]) == "营业收入" for f in requirements):
            requirements.append(dict(id="revenue", metric="营业收入", year=None, source="report"))
        days = {d[2] for d in date_mentions(q)}
        if "总市值" not in supplied and not any(canonical(f["metric"]) == "总市值" for f in requirements):
            f = dict(id="market_cap", metric="总市值", year=None, source="report")
            if len(days) == 1:
                f.update(date=next(iter(days)), year=next(iter(days))[:4])
            requirements.append(f)
    unique = {}
    for f in requirements:
        if f.get("metric") and f.get("source", "report") == "report":
            unique.setdefault((canonical(f["metric"]), f.get("year"), f.get("date")), f)
    return list(unique.values())


def guard_order(row, ranked, chunks, names=None):
    if {"gold", "Reference_documents", "external_knowledge"} & row.keys():
        raise ValueError("Evaluation-only fields entered evidence ranking")
    company = row["company_with_period"]
    target = row.get("target_entity") or target_entity(company, row["question"])
    requirements = calculation_requirements({**row, "target_entity": target})
    fields = {}
    for f in requirements:
        if f.get("metric") and f.get("source", "report") == "report":
            fields.setdefault((f["metric"], f.get("year"), f.get("date")), f)
    names = issuer_names(chunks) if names is None else names
    index = chunks if isinstance(chunks, CompanionIndex) else CompanionIndex(chunks)
    choices = []
    seen = set()
    for position, item in enumerate(ranked):
        ident = item["chunk_id"]
        if ident in seen:
            raise ValueError("Duplicate candidate")
        seen.add(ident)
        chunk = index[ident]
        if chunk["doc_id"] != item["doc_id"]:
            raise ValueError("Candidate document identity mismatch")
        companions = {c["chunk_id"]: c for c in index.companions(chunk)}
        supported = set()
        # Candidate company is the reporting issuer; target may be a named peer
        # within that report. Match the numeric row owner using existing rules.
        if chunk.get("company") == company:
            for key, f in fields.items():
                match = strict_match(f, chunk, company, target, names, companions)
                if match.get("matched"):
                    supported.add(key)
        choices.append({"item": item, "position": position, "supported": supported,
                        "is_table": chunk.get("content_type") == "table"})
    covered, ordered = set(), []
    while choices:
        # Maximise NEW required evidence coverage, then retain corroborating
        # numeric evidence even when the first candidate already covers a role.
        # A single matching (possibly conflicting) value must not push all other
        # same-metric/year evidence behind merely topical prose.
        choices.sort(key=lambda c: (
            -len(c["supported"] - covered),
            -len(c["supported"]),
            -int(c["is_table"] and bool(c["supported"])),
            c["position"],
        ))
        chosen = choices.pop(0)
        gained = chosen["supported"] - covered
        covered |= chosen["supported"]
        record = dict(chosen["item"], rerank_rank=len(ordered) + 1,
                      evidence_guard_version=VERSION,
                      unguarded_rank=chosen["position"] + 1,
                      guarded_support=[dict(metric=k[0], year=k[1], date=k[2])
                                       for k in sorted(chosen["supported"], key=str)],
                      newly_covered_count=len(gained))
        ordered.append(record)
    audit = {"version": VERSION, "target_entity": target,
             "requirements": list(fields.values()), "covered_requirements": len(covered),
             "required_count": len(fields), "uses_eval_labels": False}
    return ordered, audit
