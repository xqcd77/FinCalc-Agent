"""V4 issuer-aware selection and atomic table/unit-note delivery. No gold inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
from copy import deepcopy
from financial_semantics_v4 import facts, supports, canonical, aliases, metric_spans, RATIO_METRICS, UNIT
from table_semantics_v4 import metric_scoped_table
from field_plan_v4 import infer_formula, FORMULAS, ROLE_METRICS, formula_roles
from evidence_prompt_v4 import method_excerpt
from select_context_v2 import Slot, required_slots, target_entity, slot_match_info
from prompt_budget import count_text_tokens
from tokenizer_v4 import native_tokenizer
from rerank_context import format_evidence
from generate_calculate import read_jsonl
from hybrid_retrieval import CHUNKS, PROJECT
from table_delivery_v48 import table_companions, format_field_evidence, bundle_audit

VERSION='v4.8-bge-m3-atomic-table-and-cot-20261007'

def issuer_names(chunks):
    names={c.get('company','') for c in chunks.values()}-{'' ,'未识别机构'}
    for c in chunks.values():
        for row in c.get('rows',[]):
            for cell in row.get('display_cells',[])[:2]:
                m=re.search(r'([\u4e00-\u9fffA-Za-z·]{2,20})\s*[（(]\s*(?:股票代码[:：]\s*)?\d{4,6}(?:\.(?:SH|SZ|HK|BJ))?\s*[）)]',str(cell),re.I)
                if m:names.add(m.group(1))
    return sorted(names)

def unit_notes(table, chunks):
    """Only nearby, same-document source notes; no unit guessed from table magnitude."""
    a,b=table.get('source_start_line',0),table.get('source_end_line',0)
    return [c for c in chunks.values() if c.get('doc_id')==table['doc_id']
        and c.get('content_type')=='prose' and '单位' in c.get('content_text','')
        and c.get('source_start_line',0)<=b+12 and c.get('source_end_line',0)>=a-12]

def explicit_note_unit(field, target, note, table=None):
    text=note.get('content_text','');year=field.get('year')
    # The issuer+period+unit must be one explicit note, not merely present in its block.
    for clause in re.split('[；;。\n]',text):
        scoped=bool(table and target in (table.get('table_caption') or '')
            and note.get('source_start_line',0)<=table.get('source_start_line',0)
            and note.get('source_end_line',0)>=table.get('source_end_line',0)
            and not re.search(r'同业|可比|对比',table.get('table_caption') or ''))
        if not any(term in clause for term in aliases(field['metric'])) and not scoped:continue
        patterns=[re.escape(target)+r'[^；;。]{0,65}?'+re.escape(year or '')+r'[^；;。]{0,35}?单位(?:为|是|[:：])\s*('+UNIT+r')']
        for pattern in patterns:
            found=re.search(pattern,clause)
            if found:
                return ('元' if found.group(1)=='人民币元' else found.group(1),found.group())
        if scoped and year:
            found=re.search(re.escape(year)+r'[^。；;]{0,45}?单位(?:为|是|[:：])\s*('+UNIT+r')',clause)
            if found:return (found.group(1),found.group())
    return None

def augmented_table(table, field, target, companions):
    copy=metric_scoped_table(table,field['metric'],target);support=[]
    from table_semantics_v4 import annual_period_view
    copy=annual_period_view(copy,field)
    # V3 recognises 公司 headers but misses 企业名称. Parser-only annotation;
    # source quotes and displayed original table remain untouched.
    for row in copy.get('rows',[]):
        for key in ('display_cells','cells'):
            if row.get(key) and str(row[key][0]).strip() in {'企业名称','企业简称','标的名称'}:
                row[key][0]='公司/'+str(row[key][0])
    for ref,note in companions.items():
        found=explicit_note_unit(field,target,note,table)
        if found:support.append((ref,found))
    units={x[1][0] for x in support}
    if len(units)==1:
        copy['table_caption']=(copy.get('table_caption') or '')+'；核验单位说明：'+next(iter(units))
    if canonical(field['metric']) in RATIO_METRICS and not copy.get('unit_mentions'):
        copy['table_caption']=(copy.get('table_caption') or '')+'；指标定义为无量纲倍数'
    return copy,support

def strict_match(field, chunk, company, target, names=(), companions=None):
    if chunk.get('content_type')=='table':
        key=canonical(field['metric'])
        # A 毛利率 column is not a 毛利 amount; 现金流/收入 is not cash flow.
        text=chunk.get('content_text','')+' '+(chunk.get('table_caption') or '')
        if key in __import__('financial_semantics_v4').METRICS and key not in {m[2] for m in metric_spans(text)}:
            return {'matched':False,'reason':'table_metric_only_substring_of_another_metric'}
        if key=='经营现金流' and re.search(r'现金流(?:净额|量净额)?\s*(?:/|÷|收入比)|经营现金流收入比',text):
            return {'matched':False,'reason':'cashflow_ratio_column_is_not_cashflow_amount'}
        aug,support=augmented_table(chunk,field,target,companions or {})
        result=slot_match_info(Slot(field['id'],tuple(aliases(field['metric'])),field.get('year'),
            date_tokens=(field['date'],) if field.get('date') else ()),aug,company,target)
        if result.get('matched'):
            def value_cell(cell):
                text=re.split(r'[；;]|年份对应值',str(cell),1)[0].replace('\\','').strip()
                unit=r'(?:'+UNIT+r'|\(\s*(?:'+UNIT+r')\s*\)|（\s*(?:'+UNIT+r')\s*）)'
                return bool(re.fullmatch(r'[-+]?\d[\d,]*(?:\.\d+)?\s*'+unit+r'?',text))
            period=result.get('matched_period_values',[])
            if period:
                actual=any(c['column_index']>0 and value_cell(c['value']) for c in period)
            else:
                rows=set(result.get('target_data_rows',[]))
                actual=any(value_cell(cell) for r in aug.get('rows',[]) if r.get('row') in rows
                           for cell in r.get('display_cells',[])[1:])
            if not actual:return {'matched':False,'reason':'target_metric_has_no_numeric_value_cell'}
        return {**result,'unit_note_refs':[x[0] for x in support]}
    matched=[f for f in facts(chunk.get('content_text',''),target,names,default_target=chunk.get('company')==target,extra_metrics=(field['metric'],)) if supports(field,f)]
    return {'matched':bool(matched),'facts':matched,'reason':None if matched else 'issuer_metric_year_value_not_verified'}

def report_requirements(row):
    question=row['question']; entity=row.get('target_entity') or target_entity(row['company_with_period'],question)
    slots=required_slots(question)
    fields=[dict(id=s.name,metric=canonical(s.aliases[0]),year=s.year,source='report') for s in slots]
    known=[f for f in facts(question,entity,default_target=True) if f['unit']!='%' or
        re.search(r'(?:为|是|达到)\s*$',question[max(0,f['start']-10):f['start']])]
    fields=[f for f in fields if not any(supports(f,k) for k in known)]
    formula=infer_formula(question)
    if formula in {'growth_rate','fixed_asset_turnover','current_asset_turnover','asset_turnover','inventory_turnover',
                   'cash_interest_coverage','revenue_from_margin','debt_equity_from_debt_ratio','debt_ratio_from_multiplier','cash_ratio'}:
        required=[f for f in formula_roles(question,formula) if f['metric']]
        return [f for f in required if not any(supports(f,k) for k in known)]
    years=re.findall(r'20\d{2}',question);end=max(years) if years else None
    if formula and formula not in {'cagr','cagr_changed','growth_rate'}:
        for role in FORMULAS[formula][0]:
            metric=ROLE_METRICS.get(role)
            if not metric:continue
            year=str(int(end)-1) if role.startswith('start_') and end else end
            if any(f['metric']==metric and f['year']==year for f in known):continue
            candidate=dict(id=role,metric=metric,year=year,source='report')
            if not any(f['metric']==metric and f['year']==year for f in fields):fields.append(candidate)
    if formula=='current_ratio_changed':
        for role,metric in [('quick','速动比率'),('inventory','存货')]:
            fields.append(dict(id=role,metric=metric,year=end,source='report'))
    if formula=='inventory_days_from_cycle':
        for f in formula_roles(question,'inventory_days_from_cycle_balances'):
            if f['metric'] and not any(supports(f,k) for k in known) and not any(x['metric']==f['metric'] and x['year']==f['year'] for x in fields):fields.append(f)
    return fields

def build_context(row,chunks,tokenizer,max_tokens=5800,max_blocks=10):
    if {'gold','Reference_documents','external_knowledge'}&row.keys():raise ValueError('Evaluation labels leaked into V4 selector')
    entity=row.get('target_entity') or target_entity(row['company_with_period'],row['question'])
    names=issuer_names(chunks);fields=report_requirements(row)
    ranked=[];seen=set()
    for c in row.get('reranked',[]):
        if c['chunk_id'] in seen:continue
        if c['chunk_id'] not in chunks or c['doc_id']!=chunks[c['chunk_id']]['doc_id']:raise ValueError('Unknown/mismatched rerank chunk')
        if chunks[c['chunk_id']]['company']!=row['company_with_period']:continue
        ranked.append(c);seen.add(c['chunk_id'])
    choices=[]
    for item in ranked:
        c=chunks[item['chunk_id']];notes=table_companions(c,chunks)
        comps={n['chunk_id']:n for n in notes}
        matches={f['id']:strict_match(f,c,row['company_with_period'],entity,names,comps) for f in fields}
        choices.append((item,c,notes,matches))
    selected=[];covered=set();omitted=[]
    remaining=choices.copy()
    while remaining and len(selected)<max_blocks:
        remaining.sort(key=lambda x:(-len({k for k,v in x[3].items() if v['matched']}-covered),
            -int(x[1].get('content_type')=='table' and any(v['matched'] for v in x[3].values())),x[0].get('rerank_rank',100)))
        item,c,notes,matches=remaining.pop(0)
        gain={k for k,v in matches.items() if v['matched']}
        if fields and not gain:continue
        group=[c]+notes
        ids={s['chunk_id'] for s in selected};group=[g for g in group if g['chunk_id'] not in ids]
        proposed=selected+group
        rendered='\n\n'.join(format_field_evidence(i+1,x,fields,row['company_with_period'],entity,
            {y['chunk_id']:y for y in proposed}) for i,x in enumerate(proposed))
        if len(proposed)>max_blocks or count_text_tokens(tokenizer,rendered)>max_tokens:
            omitted.append({'chunk_id':c['chunk_id'],'reason':'atomic_bundle_token_or_block_budget'});continue
        selected=proposed;covered|=gain
        if fields and covered=={f['id'] for f in fields}:break
        if not fields and len(selected)>=5:break
    # Keep relevant methodology even when it does not contain a directly bindable amount.
    # Selection is restricted to actually retrieved, same-issuer candidates.
    methods=[]
    for item,c,_,_ in choices:
        note=method_excerpt(row['question'],c)
        if note:methods.append((-note['score'],item.get('rerank_rank',100),c))
    for _,_,c in sorted(methods,key=lambda x:(x[0],x[1]))[:2]:
        if any(x['chunk_id']==c['chunk_id'] for x in selected):continue
        proposed=selected+[c]
        rendered='\n\n'.join(format_field_evidence(i+1,x,fields,row['company_with_period'],entity,
            {y['chunk_id']:y for y in proposed}) for i,x in enumerate(proposed))
        if len(proposed)<=max_blocks and count_text_tokens(tokenizer,rendered)<=max_tokens:selected=proposed
        else:omitted.append({'chunk_id':c['chunk_id'],'reason':'method_evidence_token_or_block_budget'})
    selected.sort(key=lambda c:(c['doc_id'],c.get('source_start_line',0),c.get('source_end_line',0),c['chunk_id']))
    evidence=[];context=[]
    for i,c in enumerate(selected,1):
        evidence.append(dict(citation=f'E{i}',chunk_id=c['chunk_id'],doc_id=c['doc_id'],source_path=c['source_path'],
            source_start_line=c.get('source_start_line'),source_end_line=c.get('source_end_line'),content_type=c.get('content_type')))
        context.append(format_field_evidence(i,c,fields,row['company_with_period'],entity,
            {f'E{j}':x for j,x in enumerate(selected,1)}))
    return {**row,'target_entity':entity,'context_text':'\n\n'.join(context),'selected_evidence':evidence,
        'selector_version':VERSION,'required_fields_v4':fields,'uncovered_fields_v4':sorted({f['id'] for f in fields}-covered),
        'omitted_candidates':omitted,'evidence_tokens':count_text_tokens(tokenizer,'\n\n'.join(context)),
        'issuer_names':names,'table_delivery_audit':[bundle_audit(c,table_companions(c,chunks))
            for c in selected if c.get('content_type')=='table']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=PROJECT/'data/processed/eval/task4_v4_8_reranked.jsonl')
    p.add_argument('--output',type=Path,default=PROJECT/'data/processed/eval/task4_v4_8_contexts.jsonl')
    p.add_argument('--tokenizer-path',type=Path)
    args=p.parse_args();tok=native_tokenizer(args.tokenizer_path);chunks={c['chunk_id']:c for c in read_jsonl(CHUNKS)}
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    partial=args.output.with_suffix('.jsonl.partial')
    with partial.open('x',encoding='utf-8') as h:
        for row in read_jsonl(args.input):h.write(json.dumps(build_context(row,chunks,tok),ensure_ascii=False)+'\n')
    partial.replace(args.output);print(args.output)

if __name__=='__main__':main()
