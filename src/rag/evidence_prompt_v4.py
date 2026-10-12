"""Sourced method snippets and field/year views; never reads answers or labels."""
from __future__ import annotations
import json
import re
from decimal import Decimal

from financial_semantics_v4 import facts, supports, canonical, metric_spans
from generate_calculate import UNITS
from rerank_context import format_evidence
from financial_knowledge_v4 import metric_card
from table_delivery_v48 import format_field_evidence


METHOD_MARKERS=('计算公式','计算逻辑','计算框架','通用公式','平均余额','平均固定资产',
                '平均流动资产','平均占用','期初与期末','期初和期末','推算','勾稽',
                '严口径','扣非','剔除','调整后','期初基准值','完整会计年度','完整财年','365天','360天','365 天','360 天')


def method_excerpt(question, chunk, max_chars=550):
    """Match methodology even when the source says 计算逻辑 instead of 计算公式."""
    if chunk.get('content_type')!='prose':
        # Some reports place the explicit day convention in a table footnote.
        # Expose only that original method line, never infer it from a numeric cell.
        body=chunk.get('content_text','')
        target=next((t for t in ('应收账款周转天数','应付账款周转天数','现金循环周期','存货周转天数') if t in question),None)
        if chunk.get('content_type')=='table' and target:
            lines=[line for line in body.splitlines() if target in line and re.search(r'(?<!\d)(?:360|365)\s*(?:天|日)',line) and re.search(r'计算|基准|年度',line)]
            if lines:return dict(text='\n'.join(lines)[:max_chars],score=12,source_start_line=chunk.get('source_start_line'),source_end_line=chunk.get('source_end_line'))
        return None
    body=chunk.get('content_text','')
    markers=[body.find(w) for w in METHOD_MARKERS if w in body]
    if not markers:return None
    target_terms=re.findall(r'现金流量利息保障倍数|固定资产周转率|流动资产周转率|总资产周转率|存货周转率|应收账款周转天数|应付账款周转天数|现金循环周期|应收账款周转率|应付账款周转率|营业利润率|净资产收益率|营运资本周转率|市盈率|市净率|复合增长率|[\u4e00-\u9fff]{2,8}增长率',question)
    matched=[t for t in target_terms if t in body]
    metrics={m[2] for m in metric_spans(question)}
    shared=metrics & {m[2] for m in metric_spans(body)}
    if not matched and not shared:return None
    anchor=min(markers+[body.find(t) for t in matched])
    if '周转天数' in question or '现金循环周期' in question:
        basis=re.search(r'(?<!\d)(?:360|365)\s*(?:天|日)',body)
        if basis:anchor=basis.start()
    start=max(body.rfind('。',0,anchor)+1,body.rfind('\n',0,anchor)+1)
    end=min(len(body),start+max_chars)
    return dict(text=body[start:end],score=10*len(matched)+len(shared),
                source_start_line=chunk.get('source_start_line'),source_end_line=chunk.get('source_end_line'))


def field_facts(field, row, refs, by_ref):
    result=[];seen=set()
    from select_context_v2 import target_entity
    target=row.get('target_entity') or target_entity(row['company_with_period'],row['question'])
    for ref in refs:
        c=by_ref[ref]
        if c.get('content_type')!='prose':continue
        for f in facts(c.get('content_text',''),target,row.get('issuer_names',()),
                       default_target=c.get('company')==target,extra_metrics=(field['metric'],)):
            if not supports(field,f):continue
            key=(ref,f['metric'],f['year'],f['value'],f['unit'])
            if key in seen:continue
            seen.add(key)
            original=bool(re.search(r'原始披露(?:值)?(?:为|是)?\s*$',c['content_text'][max(0,f['start']-25):f['start']]))
            result.append({**f,'source_ref':ref,'original_disclosure':original})
    return result


def compact_facts(items, limit=10):
    return [dict(source_ref=f['source_ref'],company=f['subject'],metric=f['metric'],year=f['year'],
                 value=f['value'],unit=f['unit'],original_disclosure=f['original_disclosure']) for f in items[:limit]]


def unit_conflicts(items):
    groups={}
    for f in items:
        if f['unit'] not in {'元','万元','亿元','百万元','千元'}:continue
        key=(f['source_ref'],f['metric'],f['year'])
        groups.setdefault(key,[]).append(f)
    warnings=[]
    for key,group in groups.items():
        values=[Decimal(f['value'])*UNITS[f['unit']] for f in group]
        if len(values)>1 and max(values)-min(values)>max(abs(v) for v in values)*Decimal('0.01'):
            warnings.append(dict(source_ref=key[0],metric=key[1],year=key[2],
                reason='同块同指标同年金额折算相差超过1%，原文存在冲突；核对原始披露与换算表述，不得按数量级猜值',
                candidates=compact_facts(group)))
    return warnings


def binding_user(row,field,refs,by_ref,hint=''):
    from select_context_v2 import target_entity
    items=field_facts(field,row,refs,by_ref)
    target=row.get('target_entity') or target_entity(row['company_with_period'],row['question'])
    view='\n\n'.join(format_field_evidence(int(r[1:]),by_ref[r],[field],row['company_with_period'],target,
        by_ref) for r in refs)
    user=(f"目标公司：{row.get('target_entity') or target_entity(row['company_with_period'],row['question'])}\n问题：{row['question']}\n"
          f"只绑定此字段：{json.dumps(field,ensure_ascii=False)}\n"
          f"本次请求：指标={field['metric']}；年份={field.get('year') or '未指定'}；交易日={field.get('date') or '无'}。先核对年份，再取数字。")
    card=metric_card(field['metric'])
    if card:user+='\n字段词义与口径：'+card
    # Expose the conservative row mapping to the model as well as the
    # validator. Original E-block text remains the only quotation source.
    from select_context_v4 import augmented_table, strict_match
    target=row.get('target_entity') or target_entity(row['company_with_period'],row['question'])
    table_locations=[]
    for ref in refs:
        c=by_ref[ref]
        if c.get('content_type')!='table':continue
        match=strict_match(field,c,row['company_with_period'],target,row.get('issuer_names',()),by_ref)
        if not match.get('matched'):continue
        scoped,_=augmented_table(c,field,target,by_ref)
        mapped={r['row']:r for r in scoped.get('rows',[]) if r.get('metric_scope_v4')}
        originals={r['row']:r for r in c.get('rows',[])}
        for cell in match.get('matched_period_values',[]):
            number,index=cell['row'],cell['column_index']
            if number not in mapped or index<=0:continue
            original=originals[number].get('display_cells',originals[number].get('cells',[]))
            source_indices=scoped.get('parser_view_source_columns_v4')
            source_index=source_indices[index] if source_indices and index<len(source_indices) else index
            if source_index>=len(original):continue
            table_locations.append(dict(source_ref=ref,metric=mapped[number]['metric_scope_v4'],
                year=field.get('year'),source_row=number,source_column=source_index+1,source_cell=original[source_index]))
            if len(table_locations)>=4:break
        if len(table_locations)>=4:break
    if table_locations:
        user+='\n表格结构解析定位（指标行/列组已核验，仅供定位；quote仍从原始E块逐字复制，保留负号）：'+json.dumps(table_locations,ensure_ascii=False)
    if items:user+='\n原文解析的字段候选（仅定位辅助，引用仍须逐字来自下面E块）：'+json.dumps(compact_facts(items),ensure_ascii=False)
    warnings=unit_conflicts(items)
    if warnings:user+='\n原文金额冲突警示：'+json.dumps(warnings,ensure_ascii=False)
    user+='\n证据视图：\n'+(view or '无可显示的匹配证据')+'\n'+hint
    return user


def binding_repair_hint(field, proposal, rejects, evidence=None):
    request=dict(id=field['id'],metric=canonical(field['metric']),year=field.get('year'),date=field.get('date'))
    old=[{k:b.get(k) for k in ('id','value','unit','source_ref')} for b in proposal.get('bindings',[]) if isinstance(b,dict)][:2]
    errors=[dict(reason=r['reason'][:240],category=r['category'],
                 actual_facts=r.get('details',{}).get('actual_facts',[])[:2],
                 expected_candidates=r.get('details',{}).get('expected_candidates',[])[:2],
                 source_conflict=r.get('details',{}).get('source_conflict',False)) for r in rejects[:2]]
    from evidence_contract_v4 import citation_feedback
    citations=[citation_feedback(r['candidate'],evidence or {}) for r in rejects[:1]
               if isinstance(r.get('candidate'),dict) and r.get('category')=='provenance']
    return ('修复请求字段：'+json.dumps(request,ensure_ascii=False)+'\n上次绑定：'+json.dumps(old,ensure_ascii=False)+
            '\n具体错误及数值归属：'+json.dumps(errors,ensure_ascii=False)+
            ('\n引用差异与原文：'+json.dumps(citations,ensure_ascii=False) if citations else '')+
            '\n若错误是年份/指标不匹配，必须重新选择对应年份/指标的原始数值；删除逗号、改单位或复制原答案不能修复归属错误。引用只能取本次显示E块。')
