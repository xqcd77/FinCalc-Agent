"""One quotation contract for training views, runtime display and bounded repair.

No financial value, whitespace inside a quotation, or LaTeX reference is rewritten.
The raw upstream tables remain in dataset source sidecars.
"""
from copy import deepcopy

VERSION = 'source-text-quotation-v1-20261007'


def training_view(chunk):
    """Remove authored Markdown boundary pipes; keep original cells verbatim."""
    result = deepcopy(chunk)
    if result.get('content_type') == 'table':
        for key in ('content_text', 'text'):
            result[key] = '\n'.join(
                line.strip()[1:-1].strip()
                if line.strip().startswith('|') and line.strip().endswith('|')
                else line for line in result.get(key, '').splitlines())
    result['evidence_representation_version'] = VERSION
    return result


def quotation_body(chunk):
    """Only this exact source string is accepted by the provenance validator."""
    return chunk['content_text'].strip()


def repair_boundary_pipes(binding, evidence):
    """Repair a unique exact table substring; all other validators still run.

    Never rewrite refs, internal separators, numbers, units, dates or entities.
    This is formatting recovery, not a replacement for missing evidence.
    """
    result = deepcopy(binding)
    if not isinstance(binding, dict):
        return binding, None
    chunk = evidence.get(binding.get('source_ref'), {})
    quote = binding.get('quote')
    source = chunk.get('content_text', '')
    if chunk.get('content_type') != 'table' or not isinstance(quote, str) or quote in source:
        return result, None
    if '\n' in quote or '|' not in quote:
        return result, None
    cleaned = quote.strip().strip('|').strip()
    if not cleaned or '|' not in cleaned or source.count(cleaned) != 1:
        return result, None
    result['quote'] = cleaned
    return result, dict(kind='unique_exact_boundary_pipe_repair', original_quote=quote,
                        exact_source_quote=cleaned, source_ref=binding['source_ref'])


def citation_feedback(binding, evidence):
    chunk = evidence.get(binding.get('source_ref'), {})
    source = chunk.get('content_text', '')
    quote = binding.get('quote', '')
    _, repaired = repair_boundary_pipes(binding, evidence)
    lines = ([repaired['exact_source_quote']] if repaired else
             [line for line in source.splitlines()
              if str(binding.get('value', '')) and str(binding['value']) in line][:2])
    return dict(attempted_quote=str(quote)[:220], exact_quote_present=bool(quote and quote in source),
                original_source_lines=[line[:320] for line in lines],
                instruction='原文行仅用于修复引用；仍须核对公司、指标、年份、单位。逐字复制，不补首尾竖线，不改写LaTeX标签。')
