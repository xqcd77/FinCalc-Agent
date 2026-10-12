"""Deliver complete field table rows, all headers and source unit companions.

Views are projections only: source chunks and validator quotation strings never
change. If exact row recovery fails, retain the original whole table.
"""
from copy import deepcopy
import re
from table_structure import table_structure

VERSION = 'v4.8-atomic-field-table-view-20261007'

class CompanionIndex(dict):
    """Run-immutable corpus, indexed without changing source iteration order."""
    def __init__(self, chunks):
        super().__init__(chunks)
        self.groups = {}
        for c in self.values():
            self.groups.setdefault((c.get('doc_id'), c.get('company')), []).append(c)
        from exact_cpu_cache import ExactCPUCache
        self.companion_cache = ExactCPUCache(1024)

    def companions(self, table):
        from exact_cpu_cache import signature
        # The entire corpus is immutable for the lifetime of this run/index.
        return self.companion_cache.get_or_compute(signature(table), lambda:
            _table_companions(table, self.groups.get((table.get('doc_id'), table.get('company')), ())))

def table_companions(table, chunks):
    if isinstance(chunks, CompanionIndex): return chunks.companions(table)
    return _table_companions(table, chunks.values())

def _table_companions(table, candidates):
    if table.get('content_type') != 'table': return []
    companions = []
    own_headers = table_structure(table)['header_row_numbers']
    for c in candidates:
        if c['chunk_id']==table['chunk_id'] or c.get('doc_id')!=table['doc_id']: continue
        if c.get('company')!=table.get('company'): continue
        if c.get('content_type')=='table' and table.get('table_id') and c.get('table_id')==table['table_id']:
            # Chunking normally repeats all headers. Only recover an actually missing header.
            if not own_headers and table_structure(c)['header_row_numbers']: companions.append(c)
        elif c.get('content_type')=='prose' and re.search(r'单位\s*(?:为|是|[:：])',c.get('content_text','')):
            if c.get('source_start_line',0)<=table.get('source_end_line',0)+12 and c.get('source_end_line',0)>=table.get('source_start_line',0)-12:
                companions.append(c)
    return sorted(companions,key=lambda c:(c.get('source_start_line',0),c['chunk_id']))

def projected_table(chunk, fields, company, target, companions):
    if chunk.get('content_type')!='table' or not fields: return chunk
    from select_context_v4 import strict_match
    keep=set()
    for f in fields:
        match=strict_match(f,chunk,company,target,(),companions)
        if match.get('matched'):
            keep.update(match.get('target_data_rows',[]))
            keep.update(match.get('unit_data_row_numbers',[]))
            keep.update(x['row'] for x in match.get('matched_period_values',[]))
    structure=table_structure(chunk)
    if not keep: return chunk
    numbers={r['row'] for r in structure['header_rows']}|keep
    # Carry nearby label-only parent rows for multirow metrics/issuer ownership.
    rows=chunk.get('rows',[])
    for i,r in enumerate(rows):
        if r.get('row') in keep and i:
            prev=rows[i-1]
            if not any(re.search(r'\d[\d,]*\.\d+',str(x)) for x in prev.get('display_cells',[])[1:]):
                numbers.add(prev['row'])
            cells=r.get('display_cells',[])
            if cells and not str(cells[0]).strip():
                # Multirow company ownership can be carried by a numeric row
                # (e.g. revenue above a blank-name profit row). Keep that anchor.
                for anchor in reversed(rows[:i]):
                    acells=anchor.get('display_cells',[])
                    name=str(acells[0]).strip() if acells else ''
                    if name:
                        if target and target in name:numbers.add(anchor['row'])
                        break
    lines=chunk['content_text'].splitlines();selected=[];found=set()
    for line in lines:
        marker=re.match(r'(?:表头|数据)第(\d+)行[:：]',line)
        if marker:
            number=int(marker.group(1))
            if number in numbers:selected.append(line);found.add(number)
            continue
        matches=[r['row'] for r in rows if ' | '.join(map(str,r.get('display_cells',[]))) in line]
        if matches:
            if any(n in numbers for n in matches):selected.append(line);found.update(matches)
        else:
            # Captions, unit statements and footnotes remain verbatim.
            selected.append(line)
    if not numbers.issubset(found): return chunk
    view=deepcopy(chunk)
    view['rows']=[r for r in rows if r['row'] in numbers]
    view['data_row_numbers']=[n for n in structure['data_row_numbers'] if n in numbers]
    view['content_text']='\n'.join(selected)
    view['delivery_view_version']=VERSION
    return view

def format_field_evidence(number,chunk,fields,company,target,companions):
    from rerank_context import format_evidence
    view=projected_table(chunk,fields,company,target,companions)
    return format_evidence(number,view)

def bundle_audit(chunk, companions):
    if chunk.get('content_type')!='table': return None
    s=table_structure(chunk)
    return dict(table_id=chunk.get('table_id'),header_rows=s['header_row_numbers'],
        data_rows=s['data_row_numbers'],unit_mentions=chunk.get('unit_mentions',[]),
        companion_chunk_ids=[c['chunk_id'] for c in companions],
        policy='keep all header columns and complete target rows; never crop numerical cells')
