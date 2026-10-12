"""Recover table headers that span several source rows.

Some LaTeX tables split period labels over a second row while the corpus parser
records that row as data.  The helpers here preserve original row numbers but
expose an effective header/data view to both context formatting and coverage.
"""

from __future__ import annotations

import re
from exact_cpu_cache import ExactCPUCache, signature

_STRUCTURE_CACHE = ExactCPUCache(1024)


_PERIOD_CELL = re.compile(
    r"(?:20\d{2}(?:\s*(?:[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}|年度|年报|年))?|"
    r"\d{1,2}\s*(?:月\s*\d{1,2}\s*日|[-/]\s*\d{1,2}))\Z"
)


def is_period_header_row(row: dict) -> bool:
    """True for rows containing only two or more year/date labels and blanks."""
    cells = [str(cell).strip() for cell in row.get("display_cells", [])]
    nonempty = [cell for cell in cells if cell]
    return len(nonempty) >= 2 and all(_PERIOD_CELL.fullmatch(cell) for cell in nonempty)


def table_structure(chunk: dict) -> dict:
    key = signature({k: chunk.get(k) for k in ('rows', 'header_row_numbers', 'data_row_numbers')})
    return _STRUCTURE_CACHE.get_or_compute(key, lambda: _table_structure_uncached(chunk))


def _table_structure_uncached(chunk: dict) -> dict:
    """Return source rows split into effective header and data rows.

    Explicit parser metadata remains authoritative for ordinary rows. A row
    made exclusively of period labels is promoted to a header even when the
    parser incorrectly placed its number in ``data_row_numbers``.
    """
    rows = chunk.get("rows", [])
    by_number = {row.get("row"): row for row in rows if row.get("row") is not None}
    explicit_headers = set(chunk.get("header_row_numbers", []))
    inferred_headers = {number for number, row in by_number.items()
                        if number not in explicit_headers and is_period_header_row(row)}
    header_numbers = explicit_headers | inferred_headers
    raw_data_numbers = set(chunk.get("data_row_numbers", []))
    data_numbers = raw_data_numbers - header_numbers
    # Keep data rows in order even if parser metadata is sparse or malformed.
    headers = [by_number[number] for number in sorted(header_numbers) if number in by_number]
    data = [by_number[number] for number in sorted(data_numbers) if number in by_number]
    return {
        "header_rows": headers,
        "data_rows": data,
        "header_row_numbers": sorted(number for number in header_numbers if number in by_number),
        "inferred_header_row_numbers": sorted(number for number in inferred_headers if number in by_number),
        "data_row_numbers": sorted(number for number in data_numbers if number in by_number),
    }
