"""Offline V4 evaluation: labels are loaded only here, not by the agent."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
from evaluate_rag import PROJECT, QUESTIONS, REFUSAL, four_places, keyed, numeric, rows
from evaluate_v3 import independently_labeled_coverage


def evaluate(input_path, prefix, operand_labels=None, baseline=None):
    questions=keyed((r for p in sorted(QUESTIONS.glob('*.jsonl')) for r in rows(p)), 'questions')
    outputs=keyed(rows(input_path),'V4 outputs')
    if not outputs or not outputs.keys()<=questions.keys():raise ValueError('Unknown/empty question IDs')
    if len({(o.get('model'),o.get('mode'),o.get('prompt_version')) for o in outputs.values()})!=1:
        raise ValueError('Mixed experiment settings in one file')
    audits=[];errors=Counter();repairs=Counter()
    for ident,o in outputs.items():
        trace=o.get('trace') or {};events=trace.get('events',[]);plan=trace.get('field_plan') or {}
        gold=questions[ident]['gold'];answer=(o.get('final') or {}).get('answer')
        g,p=numeric(gold),numeric(answer)
        correct=(p is not None and four_places(p)==four_places(g)) if g is not None else answer==REFUSAL
        fields=plan.get('fields',[]);reports=[f for f in fields if f.get('source')=='report']
        for e in events:
            if e.get('error_category'):errors[e['error_category']]+=1
            for rejected in e.get('rejected_bindings',[]):errors[rejected['category']]+=1
            if e.get('json_normalisation'):repairs['lossless_json_closure']+=1
            if e.get('normalisation'):repairs['binding_numeric_unit_split']+=1
            if 'repair' in e.get('step',''):repairs['repair_calls']+=1
        audits.append(dict(finqa_id=ident,gold=str(gold),prediction=answer,status=o['status'],correct=bool(correct),
            gold_numeric=g is not None,plan_valid=bool(fields) and plan.get('formula_id')!='unsupported',
            planned_report_fields=len(reports),verified_report_fields=len(trace.get('bindings') or {}),
            planned_evidence_complete=(trace.get('evidence_completeness') or {}).get('complete') is True,
            model_calls=sum(bool(e.get('call_attempted')) for e in events),
            retrieval_rounds=sum(e.get('step')=='targeted_retrieval' for e in events),
            shown_chunk_ids=o.get('shown_chunk_ids',[]),error=o.get('error')))
    n=len(audits);rate=lambda a,b:round(a/b,6) if b else None
    valid=[a for a in audits if a['plan_valid']];complete=[a for a in valid if a['planned_evidence_complete']]
    calculated=[a for a in audits if a['status'] in {'direct','calculated'} and a['gold_numeric']]
    independent=independently_labeled_coverage(operand_labels,outputs,questions)
    complete_ids=[d['finqa_id'] for d in independent.get('details',[]) if d['complete'] and numeric(questions[d['finqa_id']]['gold']) is not None]
    summary=dict(question_count=n,subset_run=n!=len(questions),model=next(iter(outputs.values())).get('model'),
        mode=next(iter(outputs.values())).get('mode'),status_counts=dict(Counter(a['status'] for a in audits)),
        final_correct_n=sum(a['correct'] for a in audits),final_accuracy=rate(sum(a['correct'] for a in audits),n),
        field_plan_valid_rate=rate(len(valid),n),
        planned_report_field_coverage=dict(verified=sum(a['verified_report_fields'] for a in valid),
            planned=sum(a['planned_report_fields'] for a in valid),rate=rate(sum(a['verified_report_fields'] for a in valid),sum(a['planned_report_fields'] for a in valid))),
        complete_planned_evidence_rate=rate(len(complete),len(valid)),
        calculation_plan_pass_rate_given_complete_evidence=rate(sum(a['status'] in {'calculated','direct'} for a in complete),len(complete)),
        numeric_accuracy_given_valid_calculation=dict(correct=sum(a['correct'] for a in calculated),denominator=len(calculated),rate=rate(sum(a['correct'] for a in calculated),len(calculated))),
        independent_operand_coverage=independent,
        independent_false_refusal_given_complete_evidence=dict(available=bool(complete_ids),
            n=sum((outputs[i].get('final') or {}).get('answer')==REFUSAL for i in complete_ids),denominator=len(complete_ids),
            rate=rate(sum((outputs[i].get('final') or {}).get('answer')==REFUSAL for i in complete_ids),len(complete_ids))),
        independent_failure_given_complete_evidence=dict(n=sum(outputs[i]['status'] not in {'calculated','direct'} for i in complete_ids),denominator=len(complete_ids),
            rate=rate(sum(outputs[i]['status'] not in {'calculated','direct'} for i in complete_ids),len(complete_ids))),
        validation_error_events=dict(errors),repair_events=dict(repairs),
        mean_model_calls=rate(sum(a['model_calls'] for a in audits),n),mean_targeted_retrieval_rounds=rate(sum(a['retrieval_rounds'] for a in audits),n),
        limitations=['Planned coverage depends on the model checklist; it is not independent required-field recall.',
            'Without independently reviewed operand labels, true evidence-complete false-refusal rate is unavailable.',
            'These 163 questions and 22 audited failures participated in development; accuracy is exploratory.',
            'The official gold is unchanged; known source/gold disputes require a separate sensitivity report.'])
    if baseline:
        old=keyed(rows(baseline),'baseline');pairs=[]
        for a in audits:
            i=a['finqa_id']
            if i not in old:continue
            if old[i].get('mode')!=outputs[i].get('mode'):raise ValueError('Cannot compare RAG with no-RAG as a version delta')
            prior=numeric((old[i].get('final') or {}).get('answer'));g=numeric(questions[i]['gold'])
            ok=(prior is not None and four_places(prior)==four_places(g)) if g is not None else (old[i].get('final') or {}).get('answer')==REFUSAL
            pairs.append((ok,a['correct']))
        summary['paired_baseline']=dict(n=len(pairs),baseline_correct=sum(a for a,b in pairs),v4_correct=sum(b for a,b in pairs),
            accuracy_delta=rate(sum(int(b)-int(a) for a,b in pairs),len(pairs)),
            transitions=dict(Counter(('correct' if a else 'wrong')+'_to_'+('correct' if b else 'wrong') for a,b in pairs)),
            baseline_models=sorted({o.get('model','unspecified') for o in old.values()}))
    metrics=prefix.with_name(prefix.name+'_metrics.json');detail=prefix.with_name(prefix.name+'_question_audit.jsonl')
    if metrics.exists() or detail.exists():raise FileExistsError('Choose a fresh output prefix')
    prefix.parent.mkdir(parents=True,exist_ok=True)
    metrics.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    detail.write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audits),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=PROJECT/'data/processed/eval/task5_v4_qwen35_rag_generation.jsonl')
    p.add_argument('--output-prefix',type=Path,default=PROJECT/'reports/task6_v4_qwen35_rag')
    p.add_argument('--operand-labels',type=Path);p.add_argument('--baseline',type=Path)
    a=p.parse_args();print(json.dumps(evaluate(a.input,a.output_prefix,a.operand_labels,a.baseline),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
