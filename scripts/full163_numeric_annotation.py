"""Offline full-source review packets and evaluation-only numeric operand qrels.

Explicit assistant decisions are stored separately; this script never invents
missing operand labels from generated model plans or ranking positions.
"""
import argparse, collections, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/numeric_evidence_annotation_full163_20261012'
CORPUS = ROOT / 'data/processed/corpus/evidence_chunks.jsonl'
QDIR = ROOT / 'data/raw/financecomplexqa/questions/corporate_reports'
ROLE = {
 'OP':'营业利润','NP':'净利润','ANP':'归属于母公司股东的净利润',
 'R':'营业收入','TR':'营业总收入','COST':'营业成本','TA':'资产总计',
 'CA':'流动资产合计','NCA':'非流动资产合计','TL':'负债合计','CL':'流动负债合计','NCL':'非流动负债合计',
 'EQ':'所有者权益合计','PEQ':'归属于母公司所有者权益合计','MIN':'少数股东权益',
 'EM':'权益乘数','DAR':'资产负债率','DER':'产权比率','ROE':'净资产收益率',
 'BPS':'每股净资产','EPS':'每股收益','MC':'总市值','PRICE':'收盘价','PCT':'涨跌幅','CHANGE':'涨跌',
 'OCFO':'经营活动现金流量净额','FCFO':'筹资活动现金流量净额','CASH':'货币资金','TFA':'交易性金融资产',
 'AR':'应收账款','AP':'应付账款','INV':'存货','FA':'固定资产','GM':'销售毛利率','GP':'毛利',
 'CR':'流动比率','QR':'速动比率','QUICK':'速动资产','DSO':'应收账款周转天数','DPO':'应付账款周转天数',
 'FV':'公允价值变动收益','INVEST':'投资收益','NOI':'营业外收支净额','NR':'非经常性损益',
 'TAX':'所得税','PAIDTAX':'支付的所得税','INT':'利息费用','DA':'折旧与摊销','EBIT':'息税前利润',
 'TIER1':'一级资本净额','LOANS':'贷款总额','FLOAT':'流通A股','CCC':'现金循环周期','WC':'营运资本','MARGIN':'销售净利率',
}

def decisions():
    result={}
    for line in (OUT/'requirements_decisions.tsv').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'):continue
        key,fs,note=line.split('|',2)
        fields=[]
        for s in fs.split(';'):
            if not s:continue
            code,period=s.split('@')
            fields.append(dict(id=s,code=code,metric=ROLE[code],period=period,source='report',
                **({'date':period} if len(period)==8 else {'year':period})))
        result['finreport_'+key]=dict(fields=fields,formula_note=note)
    return result

def scan():
    sys.path.insert(0,str(ROOT/'src/rag'))
    from select_context_v4 import strict_match,issuer_names
    from evidence_prompt_v4 import field_facts
    qs,cs=load();byid={c['chunk_id']:c for c in cs};names=issuer_names(byid)
    specs=decisions();assert set(specs)=={q['finqa_id'] for q in qs}
    out=[]
    for q in qs:
        entity=q['company_with_period'];doc=Path(q['Reference_documents'].replace('\\','/')).stem
        source=[c for c in cs if c['doc_id']==doc]
        ev={f'E{i}':c for i,c in enumerate(source)}
        row=dict(question=q['question'],company_with_period=entity,target_entity=entity,issuer_names=names)
        fs=[]
        for f in specs[q['finqa_id']]['fields']:
            candidates=[]
            for ref,c in ev.items():
                if c['content_type']=='table':
                    m=strict_match(f,c,entity,entity,names,ev)
                    for x in m.get('matched_period_values',[]) if m.get('matched') else []:
                        candidates.append(dict(chunk_id=c['chunk_id'],location=x,
                            headers=[r for r in c.get('rows',[]) if r['row'] in c.get('header_row_numbers',[])],
                            row=next((r for r in c.get('rows',[]) if r['row']==x.get('row')),None),
                            source_lines=[c['source_start_line'],c['source_end_line']],kind='table'))
                else:
                    for x in field_facts(f,row,[ref],ev):
                        candidates.append(dict(chunk_id=c['chunk_id'],location=x,source_lines=[c['source_start_line'],c['source_end_line']],kind='prose'))
            fs.append(dict(**f,candidates=candidates))
        out.append(dict(finqa_id=q['finqa_id'],question=q['question'].split('###')[0].strip(),
                        entity=entity,reference_doc_id=doc,external_knowledge=q['external_knowledge'],
                        formula_note=specs[q['finqa_id']]['formula_note'],fields=fs))
    p=OUT/'source_locator_candidates.jsonl'
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf-8')
    print(json.dumps(dict(questions=len(out),fields=sum(len(x['fields']) for x in out),missing=sum(not f['candidates'] for x in out for f in x['fields']))))

def compact(start,count,missing_only=False):
    for i,q in enumerate(jl(OUT/'source_locator_candidates.jsonl')):
        if not start<=i<start+count:continue
        print('\nQ',q['finqa_id'][10:],q['question'])
        for f in q['fields']:
            if missing_only and f['candidates']:continue
            uniq={}
            for c in f['candidates']:
                x=c['location'];val=x.get('value');unit=x.get('unit','')
                key=(str(val),str(unit));uniq.setdefault(key,[]).append(c)
            vs=[]
            for v,xs in uniq.items():
                c=xs[0];x=c['location'];loc=f"r{x.get('row')}c{x.get('column_index')}" if c['kind']=='table' else str(x.get('quote',''))[:55]
                vs.append((v,len(xs),c['chunk_id'].split('::',1)[1],loc))
            print(' F',f['id'],f['metric'],'VALUES',vs)

def table_review(start,count):
    _,cs=load();byid={c['chunk_id']:c for c in cs}
    for i,q in enumerate(jl(OUT/'source_locator_candidates.jsonl')):
        if not start<=i<start+count:continue
        print('\nQ',q['finqa_id'][10:],q['question'])
        for f in q['fields']:
            ts=[c for c in f['candidates'] if c['kind']=='table']
            if not ts:
                ps=[c for c in f['candidates'] if c['kind']=='prose']
                print(' F',f['id'],'NO_TABLE;PROSE',[(c['chunk_id'].split('::',1)[1],c['location'].get('value'),c['location'].get('unit'),c['location'].get('quote','')[:90]) for c in ps[:7]])
            else:
                print(' F',f['id'],[(c['chunk_id'].split('::',1)[1],c['location'],byid[c['chunk_id']].get('table_caption',''),byid[c['chunk_id']].get('unit_mentions',[])) for c in ts])

def grid_scan():
    from numeric_annotation_source_grid import index,metric_role,period_match
    qs,cs=load();facts=index(cs,qs);specs=decisions();packets=[]
    for q in qs:
        fs=[]
        doc=Path(q['Reference_documents'].replace('\\','/')).stem
        for f in specs[q['finqa_id']]['fields']:
            xs=[x for x in facts if x['entity']==q['company_with_period'] and metric_role(x['metric_text'],f['code']) and period_match(x['period_text'],f['period'])]
            xs.sort(key=lambda x:(not x['chunk_id'].startswith(doc+'::'),len(x['cell']),x['chunk_id'],x['row'],x['column']))
            fs.append(dict(**f,source_candidates=xs))
        packets.append(dict(finqa_id=q['finqa_id'],question=q['question'].split('###')[0].strip(),entity=q['company_with_period'],reference_doc_id=doc,formula_note=specs[q['finqa_id']]['formula_note'],fields=fs))
    (OUT/'independent_grid_candidates.jsonl').write_text(''.join(json.dumps(q,ensure_ascii=False)+'\n' for q in packets),encoding='utf-8')
    print(json.dumps(dict(questions=len(packets),operands=sum(len(q['fields']) for q in packets),missing=sum(not f['source_candidates'] for q in packets for f in q['fields']),unknown_unit=sum(any(x['unit'] is None for x in f['source_candidates']) for q in packets for f in q['fields']))))

def grid_review(start,count):
    for i,q in enumerate(jl(OUT/'independent_grid_candidates.jsonl')):
        if not start<=i<start+count:continue
        print('\nQ',q['finqa_id'][10:],q['entity'],q['formula_note'])
        for f in q['fields']:
            xs=f['source_candidates'];uniq={}
            for x in xs:
                uniq.setdefault((x['value'],x['unit']),x)
            print(f['id'],[(v,x['chunk_id'].split('::',1)[1],f"r{x['row']}c{x['column']}",x['header_text'][:65],x['row_label'][:35]) for v,x in uniq.items()])

def check_anchors():
    from decimal import Decimal
    from numeric_annotation_source_grid import numeric
    qs,cs=load();qq={q['finqa_id']:q for q in qs};cc={c['chunk_id']:c for c in cs};ss=decisions()
    anchors={};errors=[]
    for lineno,line in enumerate((OUT/'source_anchor_decisions.tsv').read_text(encoding='utf-8').splitlines(),1):
        if not line.strip() or line.startswith('#'):continue
        tail,fid,suf,rr,col,unit,val=line.split('|');qid='finreport_'+tail;doc=Path(qq[qid]['Reference_documents'].replace('\\','/')).stem;cid=doc+'::'+suf
        key=(qid,fid)
        if key in anchors:errors.append((lineno,'duplicate',key))
        anchors[key]=dict(chunk_id=cid,row=int(rr),column=int(col),unit=unit,value=val)
        if cid not in cc:errors.append((lineno,'missing_chunk',cid));continue
        c=cc[cid]
        if int(rr):
            row=next((r for r in c.get('rows',[]) if r['row']==int(rr)),None)
            if not row or int(col)>=len(row.get('display_cells',[])):errors.append((lineno,'coordinate_missing',key));continue
            actual=numeric(row['display_cells'][int(col)])
            if actual is None or Decimal(actual)!=Decimal(val):errors.append((lineno,'numeric_mismatch',key,row['display_cells'][int(col)],val))
        else:
            clean=re.sub(r'[,，\s]','',c['content_text'])
            if val.replace(',','') not in clean:errors.append((lineno,'prose_value_absent',key,val))
    missing=[]
    for qid,d in ss.items():
        for f in d['fields']:
            if (qid,f['id']) not in anchors:missing.append((qid,f['id']))
    orphan=[key for key in anchors if key[1] not in {f['id'] for f in ss[key[0]]['fields']}]
    report=dict(requirement_questions=len(ss),source_anchors=len(anchors),anchor_errors=errors,missing_anchors=missing,orphan_anchors=orphan)
    (OUT/'annotation_progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


def jl(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]

def load():
    qs = sorted([q for p in sorted(QDIR.glob('*.jsonl')) for q in jl(p)], key=lambda q:q['finqa_id'])
    cs = jl(CORPUS)
    entities={'00175':'阿特斯','00176':'上港集团','00177':'甘李药业','00178':'华电国际','00180':'今世缘','00181':'齐鲁银行','00183':'石头科技'}
    for q in qs:
        if q['finqa_id'][10:] in entities:q['company_with_period']=entities[q['finqa_id'][10:]]
    return qs, cs

def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    files = list(sorted(QDIR.glob('*.jsonl'))) + [CORPUS,
        ROOT/'data/processed/eval/task3_bge_m3_rankings.jsonl',
        ROOT/'data/processed/eval/task4_v4_8_contexts.jsonl',
        ROOT/'data/processed/eval/task4_v49_guarded_contexts_20261012.jsonl']
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    p=OUT/'input_manifest.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Frozen annotation source changed')
    p.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def preview(start, count):
    qs,cs=load()
    prior={q['finqa_id']:q for q in jl(ROOT/'reports/retrieval_semantic_review_163_20261008/assistant_top10_qrels.jsonl')}
    for q in qs[start:start+count]:
        doc=Path(q['Reference_documents'].replace('\\','/')).stem
        source=[c for c in cs if c['doc_id']==doc]
        print('\nQ',q['finqa_id'],q['question'].split('###')[0].strip().replace('\n',' '))
        print('PRIOR_NEED',prior[q['finqa_id']]['needed_evidence'])
        print('METHOD',q['external_knowledge'].replace('\n',' ')[:650])
        tables=[c for c in source if c['content_type']=='table']
        for c in tables:
            print('TABLE',c['chunk_id'].split('::',1)[1],'L',c['source_start_line'],c.get('table_caption',''),'/'.join(c.get('section_path',[])))
            rows=c.get('rows',[])
            chosen=[r for r in rows if q['company_with_period'] in '|'.join(map(str,r.get('display_cells',r.get('cells',[]))))]
            for r in rows:
                if chosen and r['row'] not in c.get('header_row_numbers',[]) and r not in chosen:continue
                cells=r.get('display_cells',r.get('cells',[]))
                print('r'+str(r['row']), ' | '.join(f'{i}:{x}' for i,x in enumerate(cells)))
        if not tables: print('NO_TABLES')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['freeze','preview','scan','compact','tables','grid','grid-review','check-anchors']);ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=10);ap.add_argument('--missing-only',action='store_true')
    a=ap.parse_args()
    if a.action=='freeze':freeze();print('Frozen source artifacts for163 questions')
    elif a.action=='preview':preview(a.start,a.count)
    elif a.action=='scan':scan()
    elif a.action=='tables':table_review(a.start,a.count)
    elif a.action=='grid':grid_scan()
    elif a.action=='grid-review':grid_review(a.start,a.count)
    elif a.action=='check-anchors':check_anchors()
    else:compact(a.start,a.count,a.missing_only)

if __name__=='__main__':main()
