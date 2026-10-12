"""Conservative metric row views; source text and quotes remain untouched.

The view limits numeric rows to their longest financial metric label.  A
special structural recovery supports LaTeX tables whose multi-column metric
groups were parsed onto consecutive rows with a blank repeated issuer cell.
It requires matching explicit group widths and repeated period header rows;
table captions alone never establish which financial metric a row contains.
"""
from __future__ import annotations

from copy import deepcopy
import re

from financial_semantics_v4 import canonical, metric_spans
from table_structure import table_structure, is_period_header_row


_PERIOD = re.compile(r"20\d{2}(?:年(?:度|报|中|末)?|中|末|H1|FY)?", re.I)
_NUMBER = re.compile(r"^\s*[-+]?\d[\d,]*(?:\.\d+)?(?:\s*[^\d]*)?$", re.I)
_UNIT_LABEL = re.compile(r"单位|量纲|原始披露值|披露值|原值|数值|金额|折算值|转换值|value|unit", re.I)


def _cells(row):
    return [str(x).strip() for x in row.get('display_cells', row.get('cells', []))]


def _metrics(text):
    return list(dict.fromkeys(item[2] for item in metric_spans(text)))


def _annotate(row, metric, issuer=None):
    row['metric_scope_v4'] = metric
    if issuer:
        row['issuer_scope_v4'] = issuer
    else:
        # Keep ordinary "原始披露值"/"单位" labels unchanged: the existing
        # verifier uses them to join the metric's value-and-unit group.
        return
    for key in ('cells', 'display_cells'):
        cells = list(row.get(key, []))
        if not cells:
            continue
        label = str(cells[0]).strip()
        if issuer and not label:
            label = issuer
        if metric not in _metrics(label):
            label += ('；' if label else '') + metric
        cells[0] = label
        row[key] = cells


def _split_metric_groups(structure):
    """Recover only explicitly spanned, repeated-period source groups."""
    headers = structure['header_rows']
    for header in headers:
        groups = []
        for raw in header.get('raw_cells', []):
            span = re.search(r'\\multicolumn\{(\d+)\}', str(raw))
            metrics = _metrics(str(raw))
            if span and len(metrics) == 1:
                groups.append((metrics[0], int(span.group(1))))
        if len(groups) < 2 or len({width for _, width in groups}) != 1:
            continue
        width = groups[0][1]
        period_rows = []
        for row in headers:
            labels = [x for x in _cells(row)[1:] if x]
            if len(labels) == width and all(_PERIOD.fullmatch(x) for x in labels):
                period_rows.append(labels)
        if len(period_rows) != len(groups) or len({tuple(x) for x in period_rows}) != 1:
            continue
        return [metric for metric, _ in groups], width
    return None


def _explicit_horizontal_group_view(chunk, wanted):
    """Project a group only when its width and repeated periods are explicit.

    Example: 公司|流动资产（三列）|非流动资产（三列）, followed
    by 公司|2017年末|2018年末|2019年末|2017年末|2018年末|2019年末.
    This changes parser cells only. Source text and quote bytes stay intact.
    """
    structure = table_structure(chunk)
    width_names = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6,
                   '七': 7, '八': 8, '九': 9, '十': 10}
    for header in structure['header_rows']:
        groups = []
        for cell in _cells(header)[1:]:
            match = re.fullmatch(r'(.+?)[（(]([一二三四五六七八九十]|\d+)列[）)]', cell)
            if not match:
                groups = []
                break
            count = width_names.get(match[2], int(match[2]) if match[2].isdigit() else 0)
            groups.append((canonical(match[1]), count))
        if len(groups) < 2 or len({name for name, _ in groups}) != len(groups):
            continue
        candidates = [i for i, (metric, _) in enumerate(groups) if metric == wanted]
        if len(candidates) != 1:
            continue
        total = sum(width for _, width in groups)
        period = next((r for r in structure['header_rows']
                       if len(_cells(r)) == total + 1
                       and all(_PERIOD.fullmatch(c) for c in _cells(r)[1:])), None)
        if period is None:
            continue
        slices = []
        offset = 1
        for _, width in groups:
            slices.append((offset, offset + width))
            offset += width
        periods = _cells(period)
        if len({tuple(periods[a:b]) for a, b in slices}) != 1:
            continue
        start, end = slices[candidates[0]]
        result = deepcopy(chunk)
        kept = []
        for row in result['rows']:
            cells = _cells(row)
            if row['row'] == header['row']:
                continue
            if len(cells) != total + 1:
                return chunk  # Do not guess a missing or shifted column.
            projected = [cells[0]] + cells[start:end]
            if row['row'] != period['row']:
                projected[0] += '；' + wanted
                row['metric_scope_v4'] = wanted
            row['cells'] = projected
            row['display_cells'] = projected
            kept.append(row)
        result['rows'] = kept
        result['header_row_numbers'] = [period['row']]
        result['data_row_numbers'] = [r['row'] for r in kept if r['row'] != period['row']]
        result['explicit_horizontal_group_v4'] = dict(metric=wanted, columns=[start, end-1])
        result['parser_view_source_columns_v4'] = [0, *range(start, end)]
        return result
    return chunk


def metric_scoped_table(chunk, field_metric, target):
    """Return a deepcopy with only source-supported metric rows eligible.

    Only ``rows`` is changed.  All original text, provenance, header/data row
    number metadata, and cell values in retained rows remain available; metric
    and inherited issuer labels are structural annotations for the verifier.
    """
    result = deepcopy(chunk)
    if not result.get('rows'):
        return result
    wanted = canonical(field_metric)
    result = _explicit_horizontal_group_view(result, wanted)
    structure = table_structure(result)
    data = structure['data_rows']
    header_ids = set(structure['header_row_numbers'])
    split = _split_metric_groups(structure)
    scopes = {}
    inherited_issuer = None
    ordinal = 0
    if split:
        groups, width = split
        for row in data:
            cells = _cells(row)
            if len(cells) != width + 1 or not all(_NUMBER.fullmatch(x) for x in cells[1:]):
                inherited_issuer = None
                ordinal = 0
                continue
            if cells[0]:
                inherited_issuer = cells[0]
                ordinal = 0
            if inherited_issuer is None or ordinal >= len(groups):
                continue
            scopes[row.get('row')] = (groups[ordinal], inherited_issuer)
            ordinal += 1

    # Explicit row labels take priority over captions and report-level titles.
    previous_metric = None
    explicit_scopes = False
    for row in data:
        number = row.get('row')
        if number in scopes:
            continue
        cells = _cells(row)
        label = cells[0] if cells else ''
        row_metrics = _metrics(label)
        # A dictionary cannot enumerate every financial statement line. An
        # exact source row label is sufficient to scope that same requested
        # metric; sibling known rows must not hide an unknown exact label.
        # No substring/alias expansion or values from a caption are accepted.
        if label and canonical(label) == wanted:
            row_metrics = [wanted]
        if len(row_metrics) == 1:
            previous_metric = row_metrics[0]
            scopes[number] = (previous_metric, None)
            explicit_scopes = True
        elif len(row_metrics) > 1:
            previous_metric = None
            # Ambiguous multi-metric label cannot identify one operand row.
            scopes[number] = ('__ambiguous_metric__', None)
            explicit_scopes = True
        elif previous_metric and (_UNIT_LABEL.search(label) or not label):
            scopes[number] = (previous_metric, None)
        elif label:
            previous_metric = None

    if not scopes:
        return result
    kept = []
    for row in result['rows']:
        number = row.get('row')
        if number in header_ids or is_period_header_row(row):
            kept.append(row)
            continue
        scope = scopes.get(number)
        if scope:
            metric, issuer = scope
            if metric != wanted or (issuer and target not in issuer):
                continue
            _annotate(row, wanted, issuer)
            kept.append(row)
        elif not explicit_scopes and not split:
            kept.append(row)
        # When explicit metric rows exist, unknown numeric rows must not borrow
        # a metric from a broad table caption.  Non-data source rows are kept.
        elif number not in structure['data_row_numbers']:
            kept.append(row)
    result['rows'] = kept
    return result


def annual_period_view(chunk, field):
    """Exclude explicitly interim columns for annual/year-end fields.

    Parser-only masks preserve column indices and original quotation text.
    A mere year substring must not equate a half-year flow with a full-year flow.
    Explicit period/date requests retain the existing separate date contract.
    """
    if not field.get('year') or field.get('date') or field.get('period'):
        return chunk
    result = deepcopy(chunk)
    headers = table_structure(result)['header_rows']
    width = max((len(r.get('display_cells', [])) for r in headers), default=0)
    excluded = []
    for index in range(1, width):
        label = ' '.join(str(r.get('display_cells', [])[index]) for r in headers
                         if index < len(r.get('display_cells', [])))
        if re.search(r'半年度|半年|年中|中期|季度|H[12]|Q[1-4]|1\s*[-至]\s*6\s*月|06[-/]30|6月30日', label, re.I):
            excluded.append(index)
    if not excluded:
        return result
    for row in result.get('rows', []):
        for key in ('cells', 'display_cells'):
            cells = list(row.get(key, []))
            for index in excluded:
                if index < len(cells):
                    cells[index] = '非年度列（不作为本字段操作数）'
            row[key] = cells
    result['excluded_interim_columns_v4'] = excluded
    return result
