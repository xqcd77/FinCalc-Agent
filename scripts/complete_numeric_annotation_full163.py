"""Evaluation-only export of explicit assistant source annotations.

Candidate discovery uses source numbers, never rankings or generated plans.
Only explicit source anchors and reviewed alternatives become positive qrels.
"""
import argparse
import collections
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

import full163_numeric_annotation as a

UNIT_SCALE = {'元': Decimal(1), '万元': Decimal(10000),
              '百万元': Decimal(1000000), '亿元': Decimal(100000000),
              '股': Decimal(1), '万股': Decimal(10000),
              '百万股': Decimal(1000000), '亿股': Decimal(100000000),
              '元/股': Decimal(1), '%': Decimal('0.01'),
              'decimal': Decimal(1), '天': Decimal(1)}


def write_json(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_jl(p, xs):
    p.write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in xs), encoding='utf-8')


def anchors():
    qs, cs = a.load(); qq = {q['finqa_id']:q for q in qs}
    out = []
    for l in (a.OUT/'source_anchor_decisions.tsv').read_text(encoding='utf-8').splitlines():
        if not l or l.startswith('#'): continue
        tail, fid, suf, row, col, unit, value = l.split('|')
        qid = 'finreport_'+tail
        doc = Path(qq[qid]['Reference_documents'].replace('\\','/')).stem
        out.append(dict(finqa_id=qid, field_id=fid, chunk_id=doc+'::'+suf,
                        row=int(row), column=int(col), value=value, unit=unit))
    return out


def normal_text(t):
    return re.sub(r'[,，\s{}\\]', '', t)


def alternatives():
    qs, cs = a.load(); qq = {q['finqa_id']:q for q in qs}
    # All 3,590 corpus chunks, including peer-report appearances, are searched.
    normalized = [(c, normal_text(c['content_text'])) for c in cs]
    out=[]
    for x in anchors():
        v=Decimal(x['value']); unit=x['unit']
        values={format(v, 'f')}
        if unit in {'元','万元','百万元','亿元'}:
            values |= {format(v*UNIT_SCALE[unit]/UNIT_SCALE[u], 'f')
                       for u in ['元','万元','百万元','亿元']}
        if unit in {'股','万股','百万股','亿股'}:
            values |= {format(v*UNIT_SCALE[unit]/UNIT_SCALE[u], 'f')
                       for u in ['股','万股','百万股','亿股']}
        if unit in {'%','decimal'}:
            values |= {format(v*UNIT_SCALE[unit]/UNIT_SCALE[u], 'f') for u in ['%','decimal']}
        vals=set(s.rstrip('0').rstrip('.') if '.' in s else s for s in values)
        for c,t in normalized:
            if c['chunk_id']==x['chunk_id']:continue
            occurrences=[]
            for s in vals:
                pat=r'(?<![\d.])'+re.escape(s)+r'0*(?![\d.])' if '.' in s else r'(?<![\d.])'+re.escape(s)+r'(?:\.0+)?(?![\d.])'
                for m in re.finditer(pat,t):
                    occurrences.append(t[max(0,m.start()-120):m.end()+90])
            if occurrences:
                out.append(dict(finqa_id=x['finqa_id'], field_id=x['field_id'],
                                entity=qq[x['finqa_id']]['company_with_period'],
                                metric=a.ROLE[x['field_id'].split('@')[0]],
                                expected_value=x['value'], expected_unit=unit,
                                chunk_id=c['chunk_id'], content_type=c['content_type'],
                                contexts=list(dict.fromkeys(occurrences))[:5]))
    write_jl(a.OUT/'alternative_candidates.jsonl',out)
    print(json.dumps(dict(candidates=len(out), fields=len({(x['finqa_id'],x['field_id']) for x in out}))))


def review(start,count):
    xs=a.jl(a.OUT/'alternative_candidates.jsonl')
    groups=collections.defaultdict(list)
    for x in xs:groups[x['finqa_id']].append(x)
    ids=[q['finqa_id'] for q in a.load()[0]]
    for qid in ids[start:start+count]:
        print('\nQ',qid[10:])
        for x in groups[qid]:
            print(x['field_id'],x['expected_value'],x['expected_unit'],x['chunk_id'],
                  ' // '.join(x['contexts']))


def export():
    a.freeze()
    a.check_anchors()
    checks=json.loads((a.OUT/'annotation_progress.json').read_text(encoding='utf-8'))
    assert not checks['anchor_errors'] and not checks['orphan_anchors'], checks
    qs,cs=a.load();cc={c['chunk_id']:c for c in cs};specs=a.decisions()
    aa={(x['finqa_id'],x['field_id']):x for x in anchors()}
    candidate_rows=a.jl(a.OUT/'alternative_candidates.jsonl')
    reviewed=a.jl(a.OUT/'alternative_review_decisions.jsonl')
    assert len(reviewed)==len(candidate_rows)==494
    assert {r['candidate_index'] for r in reviewed}==set(range(494))
    alts=collections.defaultdict(list)
    for r in reviewed:
        x=candidate_rows[r['candidate_index']]
        assert all(r[k]==x[k] for k in ['finqa_id','field_id','chunk_id'])
        assert r['chunk_id'] in cc
        if r['relevant']:alts[(r['finqa_id'],r['field_id'])].append(r)
    notes=json.loads((a.OUT/'case_review_notes.json').read_text(encoding='utf-8'))
    cases=[];qrels=[]
    for q in qs:
        qid=q['finqa_id'];s=specs[qid];fields=[]
        for f in s['fields']:
            key=qid,f['id'];x=aa.get(key)
            sources=[]
            if x:
                c=cc[x['chunk_id']]
                row=next((r for r in c.get('rows',[]) if r['row']==x['row']),None)
                witness=dict(**x,kind='primary_manual_anchor',
                    source_path=c['source_path'],source_sha256=c['source_sha256'],
                    source_start_line=c['source_start_line'],source_end_line=c['source_end_line'],
                    table_caption=c.get('table_caption'),
                    quote=' | '.join(row['display_cells']) if row else c['content_text'],
                    headers=[r for r in c.get('rows',[]) if r['row'] in c.get('header_row_numbers',[])])
                if qid=='finreport_00006':
                    witness['context_companion_chunk_ids']=[c['doc_id']+'::t0003::c001']
                sources.append(witness)
            for r in alts[key]:
                c=cc[r['chunk_id']]
                sources.append(dict(chunk_id=c['chunk_id'],kind='reviewed_alternative',
                    source_path=c['source_path'],source_sha256=c['source_sha256'],
                    source_start_line=c['source_start_line'],source_end_line=c['source_end_line'],
                    quote=c['content_text'],table_caption=c.get('table_caption'),
                    context_companion_chunk_ids=r.get('context_companion_chunk_ids',[]),
                    review_reason=r['reason']))
            ids=sorted({s['chunk_id'] for s in sources})
            fields.append(dict(**f,entity=q['company_with_period'],
                availability='source_anchored' if x else 'source_scope_gap',
                source_value=x['value'] if x else None,source_unit=x['unit'] if x else None,
                accepted_chunk_ids=ids,source_witnesses=sources,
                precision_policy='preserve_literal_source_precision_no_imputed_digits'))
            for cid in ids:
                qrels.append(dict(finqa_id=qid,field_id=f['id'],metric=f['metric'],period=f['period'],
                    entity=q['company_with_period'],chunk_id=cid,relevance=1,
                    reviewer='assistant_source_review',independent_human_review=False))
        cases.append(dict(schema_version='numeric-operand-qrels-full163-v1',finqa_id=qid,
            question=q['question'].split('###')[0].strip(),entity=q['company_with_period'],
            reference_doc_id=Path(q['Reference_documents'].replace('\\','/')).stem,
            formula_note=s['formula_note'],required_operands=fields,
            status='retrieval_not_applicable' if not fields else ('source_scope_gap' if any(not f['accepted_chunk_ids'] for f in fields) else 'reviewed_source_anchored'),
            review_notes=notes.get(qid,[]),reviewer='assistant_source_review',independent_human_review=False,
            label_basis='question_and_full_original_latex_no_generated_plan_or_gold_answer_used'))
    assert len(cases)==163 and len({c['finqa_id'] for c in cases})==163
    assert sum(len(c['required_operands']) for c in cases)==350
    write_jl(a.OUT/'question_numeric_annotations.jsonl',cases)
    write_jl(a.OUT/'numeric_operand_chunk_qrels.jsonl',qrels)
    summary=dict(questions_reviewed=len(cases),report_dependent_questions=sum(bool(c['required_operands']) for c in cases),
        retrieval_not_applicable_questions=[c['finqa_id'] for c in cases if not c['required_operands']],
        required_operands=350,source_anchored_operands=len(aa),source_scope_gaps=checks['missing_anchors'],
        primary_anchors=len(aa),alternative_candidates_reviewed=len(reviewed),
        alternative_positives=sum(r['relevant'] for r in reviewed),
        alternative_negatives=sum(not r['relevant'] for r in reviewed),
        positive_question_operand_chunk_pairs=len(qrels),
        questions_with_review_notes=sum(bool(c['review_notes']) for c in cases),
        reviewer='assistant_source_review',independent_human_review=False,
        qrels_exhaustiveness='verified_source_anchor_and_exact_unit_conversion_pool; not exhaustive rounded_or_derived_equivalence gold',
        integrity=checks)
    write_json(a.OUT/'annotation_summary.json',summary)
    freeze_labels={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        a.OUT/'requirements_decisions.tsv',a.OUT/'source_anchor_decisions.tsv',
        a.OUT/'alternative_candidates.jsonl',a.OUT/'alternative_review_decisions.jsonl',
        a.OUT/'case_review_notes.json',a.OUT/'question_numeric_annotations.jsonl',
        a.OUT/'numeric_operand_chunk_qrels.jsonl']}
    write_json(a.OUT/'qrels_manifest.json',freeze_labels)
    print(json.dumps(summary,ensure_ascii=False))


def evaluate():
    # Freeze labels before opening any ranked results for metric calculation.
    frozen=json.loads((a.OUT/'qrels_manifest.json').read_text(encoding='utf-8'))
    for name,digest in frozen.items():assert hashlib.sha256((a.OUT/name).read_bytes()).hexdigest()==digest
    cases=a.jl(a.OUT/'question_numeric_annotations.jsonl')
    rs={r['finqa_id']:r for r in a.jl(a.ROOT/'data/processed/eval/task3_bge_m3_rankings.jsonl')}
    old={r['finqa_id']:r for r in a.jl(a.ROOT/'data/processed/eval/task4_v4_8_contexts.jsonl')}
    r1={r['finqa_id']:r for r in a.jl(a.ROOT/'data/processed/eval/task4_v49_guarded_contexts_20261012.jsonl')}
    ids={c['finqa_id'] for c in cases}
    assert set(rs)==set(old)==set(r1)==ids
    stages=['BM25','BGE-M3','RRF','old_reranked','R1_large_same_candidates','R1_unguarded_same_candidates','R1_coverage_guard']
    audits=[];aggregates={}
    for stage in stages:
        all_hits=available_hits=required=available=complete=0;rr_sum=0;applicable=0
        for c in cases:
            qid=c['finqa_id'];fs=c['required_operands']
            if stage in ['BM25','BGE-M3','RRF']:
                xs=rs[qid][{'BM25':'bm25','BGE-M3':'dense','RRF':'hybrid'}[stage]]
            elif stage=='old_reranked':xs=old[qid]['reranked']
            elif stage=='R1_large_same_candidates':xs=sorted(r1[qid]['reranked'],key=lambda x:x['large_rank'])
            elif stage=='R1_unguarded_same_candidates':xs=sorted(r1[qid]['reranked'],key=lambda x:x['unguarded_rank'])
            else:xs=r1[qid]['reranked']
            ranked=[x['chunk_id'] for x in xs[:10]];top=set(ranked)
            assert len(top)==len(ranked), (qid,stage,'duplicate_ranked_chunk')
            eligible=bool(fs)
            hits={f['id']:bool(top.intersection(f['accepted_chunk_ids'])) for f in fs}
            known=[f for f in fs if f['accepted_chunk_ids']]
            positive={cid for f in fs for cid in f['accepted_chunk_ids']}
            first=next((i for i,cid in enumerate(ranked,1) if cid in positive),None)
            rr=1/first if first else 0.0
            if eligible:
                applicable+=1;required+=len(fs);available+=len(known)
                all_hits+=sum(hits.values());available_hits+=sum(hits[f['id']] for f in known)
                complete+=all(hits.values());rr_sum+=rr
            audits.append(dict(finqa_id=qid,stage=stage,applicable=eligible,top10=ranked,
                required_operand_count=len(fs),source_anchored_operand_count=len(known),
                hit_operand_count=sum(hits.values()),operand_hits=hits,
                numeric_evidence_recall_at10=sum(hits.values())/len(fs) if fs else None,
                all_required_numeric_operands_covered_at10=all(hits.values()) if fs else None,
                first_relevant_chunk_rank_at10=first,evidence_chunk_rr_at10=rr if eligible else None,
                review_notes=c['review_notes'],source_scope_gaps=[f['id'] for f in fs if not f['accepted_chunk_ids']]))
        aggregates[stage]=dict(numeric_evidence_recall_at10=dict(hits=all_hits,denominator=required,value=all_hits/required),
            available_numeric_evidence_recall_at10=dict(hits=available_hits,denominator=available,value=available_hits/available),
            evidence_chunk_mrr_at10=dict(rr_sum=rr_sum,denominator=applicable,value=rr_sum/applicable),
            all_required_numeric_operands_covered_at10=dict(complete_questions=complete,denominator=applicable,value=complete/applicable))
    def delta(before,after):
        b=aggregates[before];n=aggregates[after]
        return dict(before=before,after=after,
            numeric_evidence_recall_pp=(n['numeric_evidence_recall_at10']['value']-b['numeric_evidence_recall_at10']['value'])*100,
            complete_numeric_evidence_pp=(n['all_required_numeric_operands_covered_at10']['value']-b['all_required_numeric_operands_covered_at10']['value'])*100,
            evidence_chunk_mrr_delta=n['evidence_chunk_mrr_at10']['value']-b['evidence_chunk_mrr_at10']['value'])
    comparison=dict(annotation_questions=163,report_dependent_questions=160,required_operands=350,
        corpus_present_operands=348,k=10,stages=aggregates,
        paired_deltas=[delta('old_reranked','R1_coverage_guard'),delta('R1_unguarded_same_candidates','R1_coverage_guard'),delta('R1_large_same_candidates','R1_coverage_guard')],
        definitions=dict(numeric_evidence_recall_at10='micro average required report numeric operands retrieved in first 10 distinct chunks / 350 requirements; 2 absent-source operands remain misses',
            available_numeric_evidence_recall_at10='same hits / 348 operands actually located in frozen source corpus',
            evidence_chunk_mrr_at10='mean 1/rank of first verified numeric-operand chunk within top 10 over 160 report-dependent questions; 0 when no hit',
            all_required_numeric_operands_covered_at10='160 report-dependent questions whose complete independently annotated numeric requirement list is covered; not a semantic faithfulness or executable-answer guarantee'),
        limitations=['Assistant source review, not independently human-adjudicated gold.',
            '3 questions use only their own numbers and are retrieval N/A, not misses.',
            '2 paid-income-tax inputs absent; accrual income-tax expense not substituted.',
            'Source precision, corruption, incompatible scope and official-answer disputes preserved in per-case notes.',
            'Qrels use verified primary and exact/unit-converted alternative occurrences; not exhaustive for all rounded or derived equivalent evidence.',
            'Source-lineage numeric recall can count split-table fragments; delivered company/year/unit completeness is a separate metric.',
            'Old reranker versus R1 changes candidate pool and scoring; only same-candidate before/after isolates coverage protection.',
            '163-question development regression set has intentional training overlap; no independent generalization claim.'])
    write_json(a.OUT/'full163_retrieval_comparison.json',comparison)
    write_jl(a.OUT/'per_question_retrieval_metrics.jsonl',audits)
    with (a.OUT/'retrieval_comparison.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.writer(h);w.writerow(['stage','numeric_evidence_recall_at10','hits','required_operands','available_operand_recall_at10','evidence_chunk_mrr_at10','mrr_questions','complete_numeric_evidence_at10','complete_questions'])
        for name,r in aggregates.items():w.writerow([name,r['numeric_evidence_recall_at10']['value'],r['numeric_evidence_recall_at10']['hits'],350,r['available_numeric_evidence_recall_at10']['value'],r['evidence_chunk_mrr_at10']['value'],160,r['all_required_numeric_operands_covered_at10']['value'],r['all_required_numeric_operands_covered_at10']['complete_questions']])
    print(json.dumps(comparison,ensure_ascii=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['alternatives','review','export','evaluate']);p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=10)
    x=p.parse_args()
    if x.action=='alternatives':alternatives()
    elif x.action=='review':review(x.start,x.count)
    elif x.action=='export':export()
    else:evaluate()


if __name__=='__main__':main()
