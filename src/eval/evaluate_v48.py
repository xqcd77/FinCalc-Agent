"""Unified offline retrieval, evidence, planning, calculation and QLoRA metrics.

All rates expose their denominator. Source-verified automatic operand labels are
reported on their certified subset, never as human gold for all 163 questions.
"""
from __future__ import annotations
import argparse,json,math,re,statistics,sys
from collections import Counter
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src/rag'))
from evaluate_rag import QUESTIONS,REFUSAL,rows,keyed,numeric,four_places
from financial_semantics_v4 import canonical
from solution_steps_v48 import validate_steps
from trajectory_metrics import repeated_tools

KS=(1,3,5,8,10,20,30)

def rate(n,d,definition,**extra):
    return dict(n=int(n),denominator=int(d),value=round(n/d,6) if d else None,
                available=bool(d),definition=definition,**extra)

def load_optional(path):return keyed(rows(path),str(path)) if path and Path(path).is_file() else {}

def ranking_metrics(questions,records,field,labels):
    used=[i for i in records if i in questions]
    result=dict(questions=len(used),candidate_unit='source chunks',by_k={})
    for k in KS:
        doc_hit=doc_precision=doc_rr=doc_f1=0.;field_hit=field_total=complete=label_n=0
        chunk_prec=ndcg=chunk_rr=0.
        for i in used:
            candidates=records[i].get(field,[])[:k]
            if len({c['chunk_id'] for c in candidates})!=len(candidates):raise ValueError('Duplicate ranked chunks')
            doc=Path(questions[i]['Reference_documents'].replace('\\','/')).stem
            docs=list(dict.fromkeys(c['doc_id'] for c in candidates))
            hit=doc in docs;doc_hit+=hit;doc_precision+=sum(d==doc for d in docs)/len(docs) if docs else 0
            if hit:doc_f1+=2/(len(docs)+1)
            if hit:doc_rr+=1/(docs.index(doc)+1)
            case=labels.get(i)
            if not case:continue
            fs=[f for f in case['fields'] if f['source']=='report']
            if not fs:continue
            ids={c['chunk_id'] for c in candidates};relevant={c for f in fs for c in f['acceptable_chunk_ids']}
            label_n+=1;covered=sum(bool(ids&set(f['acceptable_chunk_ids'])) for f in fs)
            field_hit+=covered;field_total+=len(fs);complete+=covered==len(fs)
            first=next((j+1 for j,c in enumerate(candidates) if c['chunk_id'] in relevant),None)
            chunk_rr+=1/first if first else 0
            chunk_prec+=sum(c['chunk_id'] in relevant for c in candidates)/len(candidates) if candidates else 0
            dcg=sum((c['chunk_id'] in relevant)/math.log2(j+2) for j,c in enumerate(candidates))
            ideal=sum(1/math.log2(j+2) for j in range(min(k,len(relevant))))
            ndcg+=dcg/ideal if ideal else 0
        n=len(used)
        result['by_k'][str(k)]=dict(
            document_recall=rate(doc_hit,n,'Reference document hit among the first k chunks; one official reference document per question'),
            document_precision_macro=dict(value=round(doc_precision/n,6) if n else None,denominator=n,
                definition='Mean relevant unique documents / retrieved unique documents in the first k chunks'),
            document_mrr=dict(value=round(doc_rr/n,6) if n else None,denominator=n,
                definition='Reciprocal rank after document deduplication within the first k chunks'),
            required_report_field_recall=rate(field_hit,field_total,'Certified report operands covered by at least one acceptable candidate chunk'),
            complete_required_evidence_rate=rate(complete,label_n,'All certified report operands covered in the first k chunks'),
            relevant_chunk_precision_macro=dict(value=round(chunk_prec/label_n,6) if label_n else None,denominator=label_n),
            evidence_chunk_mrr=dict(value=round(chunk_rr/label_n,6) if label_n else None,denominator=label_n,
                definition='First source-certified required numeric evidence chunk reciprocal rank within top k; automatic labelled subset only'),
            chunk_ndcg=dict(value=round(ndcg/label_n,6) if label_n else None,denominator=label_n,
                definition='Binary relevance of source-verified operand chunks; all alternative support chunks are relevant'))
        result['by_k'][str(k)]['document_f1_macro']=dict(value=round(doc_f1/n,6) if n else None,denominator=n,
            definition='Mean per-question document F1 with one official relevant document; missing retrieval scores zero')
    return result

def semantic_key(f):return (canonical(f['metric']),f.get('year'),f.get('date'),f.get('source'))

def label_coverage(case,ids,output=None,actually_shown=False):
    results=[]
    for f in case['fields']:
        if f['source']=='question':results.append(True);continue
        hit=bool(set(ids)&set(f['acceptable_chunk_ids']))
        if hit and actually_shown:
            hit=False
            refs={e['citation']:e['chunk_id'] for e in output.get('selected_evidence',[])}
            for event in (output.get('trace') or {}).get('events',[]):
                shown={refs[r] for r in event.get('shown_refs',[]) if r in refs}
                messages=event.get('messages',[])
                text='\n'.join(m.get('content','') for m in messages if m.get('role')=='user')
                if '证据视图：\n' not in text:continue
                text=text.split('证据视图：\n',1)[1]
                if any(a['chunk_id'] in shown and a['quote'] in text for a in f.get('accepted_quotes',[])):
                    hit=True;break
        results.append(hit)
    return results

def context_metrics(contexts,labels):
    required=covered=complete=eligible=labelled=label_complete=label_fields=label_covered=0
    token_sizes=[];tables=table_headers=0;omissions=Counter()
    for ident,row in contexts.items():
        fs=row.get('required_fields_v4',[]);missing=set(row.get('uncovered_fields_v4',[]))
        if fs:
            required+=len(fs);covered+=sum(f['id'] not in missing for f in fs)
            eligible+=1;complete+=not missing
        if row.get('evidence_tokens') is not None:token_sizes.append(int(row['evidence_tokens']))
        for t in row.get('table_delivery_audit',[]):
            if t:tables+=1;table_headers+=bool(t.get('header_rows'))
        omissions.update(x.get('reason','unknown') for x in row.get('omitted_candidates',[]))
        if ident in labels:
            case=labels[ident];ids={e['chunk_id'] for e in row.get('selected_evidence',[])}
            text=row.get('context_text','');hits=[]
            for f in case['fields']:
                hits.append(f['source']=='question' or any(a['chunk_id'] in ids and a['quote'] in text for a in f.get('accepted_quotes',[])))
            labelled+=1;label_fields+=len(hits);label_covered+=sum(hits);label_complete+=all(hits)
    return dict(
        registry_field_coverage=rate(covered,required,'Selector registered roles covered; registry checklist may include alternatives and is not independent gold'),
        registry_context_completeness=rate(complete,eligible,'No uncovered registered roles after context assembly'),
        certified_context_field_coverage=rate(label_covered,label_fields,'Certified fields with exact source quote actually present in assembled context; question conditions counted separately from retrieval'),
        certified_context_completeness=rate(label_complete,labelled,'All source-certified operand quotations present in assembled context'),
        delivered_table_header_presence=rate(table_headers,tables,'Delivered table bundles with explicit or inferred header rows'),
        context_tokens_max=max(token_sizes) if token_sizes else None,context_tokens_mean=statistics.mean(token_sizes) if token_sizes else None,
        omitted_candidate_reasons=dict(omissions))

def training_metrics(directory):
    if not directory:return dict(available=False,reason='No training directory supplied')
    directory=Path(directory)
    if not directory.is_dir():raise ValueError('Training directory does not exist')
    result=dict(available=True,scope='Teacher-forced completion-only training metrics; not financial QA accuracy')
    for name in ('train_results.json','eval_results.json','run_config.json','evaluation_scope.json'):
        p=directory/name
        if p.is_file():result[p.stem]=json.loads(p.read_text(encoding='utf-8'))
    events=list(rows(directory/'events.jsonl')) if (directory/'events.jsonl').is_file() else []
    result['validation_history']=[dict(epoch=e.get('epoch'),step=e.get('step'),metrics=e.get('metrics',{}))
        for e in events if e.get('event')=='validation_metrics']
    result['peak_gpu_gib']=max((float(e['peak_gpu_gib']) for e in events if e.get('peak_gpu_gib') is not None),default=None)
    result['checkpoint_saves']=[dict(step=e.get('step'),checkpoint=e.get('checkpoint')) for e in events if e.get('event')=='checkpoint_saved']
    result['training_completed']=any(e.get('event')=='training_complete' for e in events)
    result['errors']=[e.get('error') for e in events if e.get('event')=='fatal_error']
    return result

def evaluate(input_path,prefix,rankings_path=None,contexts_path=None,labels_path=None,baseline=None,questions_dir=QUESTIONS,training_dir=None):
    questions=keyed((q for p in sorted(Path(questions_dir).glob('*.jsonl')) for q in rows(p)),'questions')
    outputs=load_optional(input_path);rankings=load_optional(rankings_path);contexts=load_optional(contexts_path)
    if input_path and not outputs:raise ValueError('Generation input missing or empty')
    if not (outputs or rankings or contexts):raise ValueError('No evaluation artifacts')
    for data in (outputs,rankings,contexts):
        if not data.keys()<=questions.keys():raise ValueError('Unknown question ID in artifacts')
    settings={(o.get('model'),o.get('mode'),o.get('prompt_version')) for o in outputs.values()}
    if len(settings)>1:raise ValueError('Mixed generation experiment settings')
    label_data=json.loads(Path(labels_path).read_text(encoding='utf-8')) if labels_path and Path(labels_path).is_file() else {}
    labels={c['finqa_id']:c for c in label_data.get('cases',[]) if c.get('certified') is True}
    if len(labels)!=sum(c.get('certified') is True for c in label_data.get('cases',[])):raise ValueError('Duplicate operand labels')
    retrieval={}
    for field in ('bm25','dense','hybrid'):
        if rankings:retrieval[field]=ranking_metrics(questions,rankings,field,labels)
    if contexts:retrieval['reranked']=ranking_metrics(questions,contexts,'reranked',labels)
    audits=[];counts=Counter();abs_errors=[];rel_errors=[];latencies=[];error_codes=Counter()
    calls=prompt_tokens=completion_tokens=0
    for i,o in outputs.items():
        q=questions[i];trace=o.get('trace') or {};events=trace.get('events',[]);plan=trace.get('field_plan') or {}
        answer=(o.get('final') or {}).get('answer');g=numeric(q['gold']);p=numeric(answer)
        gold_refusal=q['gold']==REFUSAL;pred_refusal=answer==REFUSAL
        correct=(p is not None and four_places(p)==four_places(g)) if g is not None else answer==q['gold']
        numeric_correct=bool(g is not None and p is not None and four_places(p)==four_places(g))
        valid=bool(plan.get('fields')) and plan.get('formula_id') not in (None,'unsupported')
        calculated=o['status'] in {'calculated','direct'}
        planned=[f for f in plan.get('fields',[]) if f.get('source')=='report'];bindings=trace.get('bindings') or {}
        verified=sum(f['id'] in bindings for f in planned)
        complete=valid and verified==len(planned)
        tool_repetition=repeated_tools(o,contexts.get(i))
        counts['logged_tool_calls']+=tool_repetition['logged_tool_calls']
        counts['identifiable_tool_calls']+=tool_repetition['identifiable_tool_calls']
        counts['unidentifiable_tool_calls']+=tool_repetition['unidentifiable_tool_calls']
        counts['repeated_tool_calls']+=tool_repetition['repeated_tool_calls']
        counts['questions_with_repeated_tools']+=tool_repetition['repeated_tool_calls']>0
        counts['correct_planned_sufficient']+=bool(correct and complete)
        a=dict(finqa_id=i,gold=str(q['gold']),prediction=answer,status=o['status'],correct=bool(correct),
            plan_valid=valid,planned_report_fields=len(planned),verified_report_fields=verified,
            model_planned_evidence_complete=complete,refused=pred_refusal,
            formula_id=plan.get('formula_id'),error=o.get('error'),
            correct_and_planned_evidence_sufficient=bool(correct and complete),
            tool_repetition=tool_repetition,
            errors=[dict(step=e.get('step'),category=e.get('error_category'),message=e.get('validation_error') or e.get('error'))
                for e in events if e.get('validation_error') or e.get('error')])
        counts.update(dict(correct=bool(correct),numeric_gold=g is not None,numeric_correct=numeric_correct,
            numeric_prediction=g is not None and p is not None,plan_valid=valid,calculated=calculated,
            planned_fields=len(planned),verified_fields=verified,complete_plan=complete,
            calculated_complete=calculated and complete,calculated_numeric=calculated and g is not None,
            calculated_numeric_correct=calculated and numeric_correct,pred_refusal=pred_refusal,
            gold_refusal=gold_refusal,correct_refusal=pred_refusal and gold_refusal,
            answerable_refusal=pred_refusal and g is not None,final_numeric_parse=p is not None,
            final_format=isinstance(answer,str) and (p is not None or pred_refusal)))
        if g is not None:
            counts['numeric_em_unrounded']+=p is not None and p==g
            if p is not None:
                err=abs(p-g);abs_errors.append(err)
                counts['numeric_abs_tol_1e4']+=err<=Decimal('0.0001')
                counts['numeric_relative_tol_1pct']+=err<=max(Decimal('0.0001'),abs(g)*Decimal('.01'))
                if g!=0:rel_errors.append(err/abs(g))
                a.update(absolute_error=str(err),relative_error=str(err/abs(g)) if g else None)
        called=[e for e in events if e.get('call_attempted')]
        calls+=len(called)
        for e in called:
            usage=e.get('usage') or {};prompt_tokens+=int(usage.get('prompt_tokens',e.get('prompt_tokens',0)) or 0)
            completion_tokens+=int(usage.get('completion_tokens',0) or 0)
            if 'completion_tokens' in usage:counts['calls_with_completion_usage']+=1
            raw=e.get('raw_response');obj=None
            if isinstance(raw,str):
                try:
                    obj=json.loads(raw);counts['raw_json_valid']+=isinstance(obj,dict)
                except json.JSONDecodeError:pass
            if isinstance(obj,dict):
                counts['steps_present']+=bool(obj.get('solution_steps'))
                try:validate_steps(obj,required=True);counts['steps_valid']+=1
                except ValueError:pass
                proposals=obj.get('bindings',[])
                if isinstance(proposals,list):counts['binding_proposals']+=len(proposals)
            counts['binding_accepted']+=len(e.get('accepted_ids',[]))
            if str(e.get('step','')).startswith('plan'):
                counts['plan_calls']+=1
                counts['plan_call_pass']+=plan_event_pass(e)
            if e.get('error_category'):error_codes[e['error_category']]+=1
            for rejected in e.get('rejected_bindings',[]):error_codes[rejected.get('category','unknown')]+=1
        first_plan=next((e for e in called if e.get('step')=='plan'),None)
        counts['first_plan_attempted']+=first_plan is not None
        counts['first_plan_pass']+=bool(first_plan and plan_event_pass(first_plan))
        repair_calls=[e for e in called if 'repair' in str(e.get('step',''))]
        counts['questions_with_repair']+=bool(repair_calls);counts['repair_final_correct']+=bool(repair_calls and correct)
        counts['targeted_retrieval_rounds']+=sum(e.get('step')=='targeted_retrieval' for e in events)
        counts['planning_retrieval_searches']+=sum(e.get('step')=='planning_evidence_search' for e in events)
        counts['questions_with_targeted_retrieval']+=any(e.get('step')=='targeted_retrieval' for e in events)
        counts['targeted_retrieval_final_correct']+=correct and any(e.get('step')=='targeted_retrieval' for e in events)
        counts['accepted_binding_fields']+=len(bindings)
        counts['binding_source_quote_present']+=sum(bool(b.get('source_ref') and b.get('quote')) for b in bindings.values())
        counts['binding_unit_present']+=sum(bool(b.get('unit')) for b in bindings.values())
        counts['binding_unit_companions']+=sum(bool(b.get('unit_ref')) for b in bindings.values())
        counts['binding_companion_quotes']+=sum(bool(b.get('unit_ref') and b.get('unit_quote')) for b in bindings.values())
        counts['auto_binding_recoveries']+=sum(e.get('step')=='deterministic_binding_recovery' for e in events)
        if o.get('latency_seconds') is not None:latencies.append(float(o['latency_seconds']))
        case=labels.get(i)
        if case:
            counts['certified_questions']+=1
            selected=label_coverage(case,o.get('source_chunk_ids',[]))
            shown=label_coverage(case,o.get('shown_chunk_ids',[]),o,True)
            a.update(certified_selected_evidence_complete=all(selected),certified_shown_evidence_complete=all(shown),
                certified_fields=len(shown),certified_shown_fields=sum(shown))
            counts['certified_fields']+=len(shown);counts['certified_shown_fields']+=sum(shown)
            counts['certified_selected_complete']+=all(selected);counts['certified_shown_complete']+=all(shown)
            counts['certified_complete_refusal']+=all(shown) and pred_refusal
            counts['certified_complete_calculated']+=all(shown) and calculated
            counts['correct_certified_sufficient']+=all(shown) and correct
            a['correct_and_certified_evidence_sufficient']=bool(all(shown) and correct)
            required={semantic_key(f) for f in case['fields']};actual={semantic_key(f) for f in plan.get('fields',[])}
            counts['dependency_contract_match']+=valid and actual==required
            a['dependency_contract_match']=valid and actual==required
        else:a['certified_evidence_label_available']=False
        audits.append(a)
    n=len(outputs)
    evidence_sufficiency=dict(
        planned_evidence_sufficiency_rate=rate(counts['complete_plan'],n,
            'Trajectories with all final model-planned report fields verified / all evaluated trajectories; depends on plan correctness'),
        correct_and_sufficient_trajectory_rate=rate(counts['correct_planned_sufficient'],n,
            'Official answer correct AND all final model-planned report fields verified / all evaluated trajectories'),
        answer_accuracy_given_sufficient_evidence=rate(counts['correct_planned_sufficient'],counts['complete_plan'],
            'Official answer correct among trajectories with complete verified model-planned inputs'),
        certified_visible_evidence_sufficiency_rate=rate(counts['certified_shown_complete'],counts['certified_questions'],
            'All certified required input quotes actually delivered / evaluated source-certified trajectories'),
        certified_correct_and_sufficient_trajectory_rate=rate(counts['correct_certified_sufficient'],counts['certified_questions'],
            'Official answer correct AND all certified input quotes actually delivered / evaluated source-certified trajectories'),
        certified_answer_accuracy_given_sufficient_evidence=rate(counts['correct_certified_sufficient'],counts['certified_shown_complete'],
            'Official answer correct among source-certified trajectories with complete delivered input evidence'),
        scope='One final question trajectory; model-planned full-set proxy and automatic certified subset reported separately; no semantic Faithfulness claim')
    final=dict(
        accuracy=rate(counts['correct'],n,'Official answer accuracy; numeric answers rounded half-up to four decimal places'),
        accuracy_over_expected_set=rate(counts['correct'],len(questions),'Correct answers / full benchmark; uncompleted rows count as wrong'),
        numeric_exact_match_4dp=rate(counts['numeric_correct'],counts['numeric_gold'],'Numeric EM after Decimal four-place normalization'),
        numeric_exact_match_unrounded=rate(counts['numeric_em_unrounded'],counts['numeric_gold'],'Decimal value equality without rounding'),
        numeric_abs_tolerance_1e4=rate(counts['numeric_abs_tol_1e4'],counts['numeric_gold'],'Absolute error <= 0.0001; missing answers fail'),
        numeric_relative_tolerance_1pct=rate(counts['numeric_relative_tol_1pct'],counts['numeric_gold'],'Error <= max(0.0001, 1% of abs(gold)); missing answers fail'),
        numeric_mae_on_answers=str(sum(abs_errors)/len(abs_errors)) if abs_errors else None,
        numeric_median_relative_error_on_nonzero_gold=str(statistics.median(rel_errors)) if rel_errors else None,
        final_answer_format_valid=rate(counts['final_format'],n,'String numeric answer or exact refusal phrase'),
        answerable_false_refusal_rate=rate(counts['answerable_refusal'],counts['numeric_gold'],'Refusal on numeric-gold questions'))
    precision=counts['correct_refusal']/counts['pred_refusal'] if counts['pred_refusal'] else None
    recall=counts['correct_refusal']/counts['gold_refusal'] if counts['gold_refusal'] else None
    final['refusal_precision']=rate(counts['correct_refusal'],counts['pred_refusal'],'Refusal precision')
    final['refusal_recall']=rate(counts['correct_refusal'],counts['gold_refusal'],'Refusal recall')
    final['refusal_f1']=2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else None
    if precision is not None and recall is not None and precision+recall==0:final['refusal_f1']=0.
    evidence=dict(
        planned_field_binding_coverage=rate(counts['verified_fields'],counts['planned_fields'],'Verified bindings / model-planned report fields; not gold operand recall'),
        planned_evidence_completeness=rate(counts['complete_plan'],counts['plan_valid'],'All model-planned report fields verified'),
        certified_required_field_coverage=rate(counts['certified_shown_fields'],counts['certified_fields'],'Source-certified fields actually shown with exact accepted quote'),
        certified_selected_evidence_completeness=rate(counts['certified_selected_complete'],counts['certified_questions'],'All source-certified fields in selected chunks'),
        certified_shown_evidence_completeness=rate(counts['certified_shown_complete'],counts['certified_questions'],'All source-certified fields actually visible in model binding prompts'),
        false_refusal_given_certified_complete_evidence=rate(counts['certified_complete_refusal'],counts['certified_shown_complete'],'False refusal conditioned on actually shown certified evidence'),
        binding_source_quote_presence=rate(counts['binding_source_quote_present'],counts['accepted_binding_fields'],'Accepted final bindings with source_ref and quote; structural provenance, not independent semantic faithfulness'),
        binding_unit_presence=rate(counts['binding_unit_present'],counts['accepted_binding_fields'],'Accepted bindings with explicit unit'),
        companion_unit_quote_presence=rate(counts['binding_companion_quotes'],counts['binding_unit_companions'],'Bindings using separate unit evidence with unit_quote'),
        targeted_retrieval_final_correct_rate=rate(counts['targeted_retrieval_final_correct'],counts['questions_with_targeted_retrieval'],'Final correctness on questions with supplemental search; not causal improvement'))
    planning=dict(valid_plan_rate=rate(counts['plan_valid'],n,'Final nonempty executable plan passed runtime validation'),
        first_plan_pass_rate=rate(counts['first_plan_pass'],counts['first_plan_attempted'],'First plan response passed, without planning repair'),
        plan_call_pass_rate=rate(counts['plan_call_pass'],counts['plan_calls'],'Nonempty plans accepted per planning call'),
        dependency_contract_match_rate=rate(counts['dependency_contract_match'],counts['certified_questions'],'Exact metric/year/date/source operand set on certified subset'),
        raw_json_object_valid_rate=rate(counts['raw_json_valid'],calls,'Model raw response JSON object parseability before repair'),
        solution_steps_presence_rate=rate(counts['steps_present'],calls,'Responses containing solution_steps'),
        solution_steps_schema_valid_rate=rate(counts['steps_valid'],calls,'Step schema and declared ID validity; not a semantic proof'),
        binding_proposal_acceptance_rate=rate(counts['binding_accepted'],counts['binding_proposals'],'Accepted field bindings / generated binding proposals'),
        repair_final_correct_rate=rate(counts['repair_final_correct'],counts['questions_with_repair'],'Final correctness among questions with model repair calls'))
    calculation=dict(execution_success_rate=rate(counts['calculated'],n,'Successful deterministic tool execution or direct sourced value'),
        execution_pass_given_planned_complete=rate(counts['calculated_complete'],counts['complete_plan'],'Tool pass when all model-planned fields are verified'),
        execution_pass_given_certified_complete=rate(counts['certified_complete_calculated'],counts['certified_shown_complete'],'Tool pass with source-certified complete visible evidence'),
        numeric_em_given_execution=rate(counts['calculated_numeric_correct'],counts['calculated_numeric'],'Numeric correctness conditional on successful tool execution'))
    summary=dict(version='v4.8-unified-rag-qlora-metrics',question_count=n,expected_questions=len(questions),
        generation_completion_rate=rate(n,len(questions),'Completed generation rows / benchmark questions'),
        retrieval_only=not outputs,experiment=dict(zip(('model','mode','prompt_version'),next(iter(settings)))) if settings else {},
        retrieval=retrieval,context=context_metrics(contexts,labels),final=final,evidence=evidence,
        evidence_sufficiency=evidence_sufficiency,planning=planning,calculation=calculation,
        qlora_training=training_metrics(training_dir),
        efficiency=dict(model_calls=calls,mean_model_calls=calls/n if n else None,
            repeated_tool_call_rate=rate(counts['repeated_tool_calls'],counts['identifiable_tool_calls'],
                'Same question: exact same identifiable tool name and arguments after first occurrence / identifiable tool calls; legacy reconstruction, not wasted-call causality'),
            tool_argument_capture_coverage=rate(counts['identifiable_tool_calls'],counts['logged_tool_calls'],
                'Tool events whose arguments can be reconstructed / all logged tool calls'),
            logged_tool_calls=counts['logged_tool_calls'],unidentifiable_tool_calls=counts['unidentifiable_tool_calls'],
            trajectories_with_repeated_tools=rate(counts['questions_with_repeated_tools'],n,
                'Evaluated question trajectories with at least one identifiable exact repeated tool call'),
            mean_targeted_retrieval_rounds=counts['targeted_retrieval_rounds']/n if n else None,
            planning_retrieval_searches=counts['planning_retrieval_searches'],deterministic_binding_recoveries=counts['auto_binding_recoveries'],
            prompt_tokens=prompt_tokens,completion_tokens=completion_tokens,completion_usage_calls=counts['calls_with_completion_usage'],
            latency_p50=statistics.median(latencies) if latencies else None,
            latency_p95=sorted(latencies)[math.ceil(.95*len(latencies))-1] if latencies else None),
        status_counts=dict(Counter(o['status'] for o in outputs.values())),validation_error_events=dict(error_codes),
        failed_status_counts=dict(Counter(a['status'] for a in audits if not a['correct'])),
        evidence_label_scope=dict(labelled_questions=len(labels),method='automatic source/program/gold-consistent subset',
            human_reviewed_questions=sum(c.get('reviewed_by_human') is True for c in labels.values()),
            unresolved_questions=label_data.get('unresolved_questions'),limitations=label_data.get('limitations',[])),
        limitations=['Model-planned field coverage is not independent required-field recall.',
            'Certified operand qrels are automatic and share registry/validation code; no human full-set claim.',
            'Faithfulness-style source checks do not prove the financial formula is correct.',
            'Operand qrels cover verified support in official reference documents; other correct reports are not exhaustively annotated.',
            'The 163 questions participated in development; these are development regression results.'])
    if baseline:
        prior=load_optional(baseline);transitions=Counter();paired=[]
        for a in audits:
            if a['finqa_id'] not in prior:continue
            b=prior[a['finqa_id']];g=numeric(questions[a['finqa_id']]['gold']);p=numeric((b.get('final') or {}).get('answer'))
            ok=(p is not None and four_places(p)==four_places(g)) if g is not None else (b.get('final') or {}).get('answer')==questions[a['finqa_id']]['gold']
            paired.append((ok,a['correct']));transitions[f'{"correct" if ok else "wrong"}_to_{"correct" if a["correct"] else "wrong"}']+=1
        summary['paired_comparison']=dict(n=len(paired),baseline_correct=sum(x for x,y in paired),current_correct=sum(y for x,y in paired),
            accuracy_delta=sum(int(y)-int(x) for x,y in paired)/len(paired) if paired else None,transitions=dict(transitions),
            limitation='Compare same base model, quantization, decoding and data; otherwise delta is confounded.')
        current_configs=[o.get('experiment_config',{}) for o in outputs.values()]
        prior_configs=[o.get('experiment_config',{}) for o in prior.values()]
        summary['paired_comparison']['precision_verified_equal']=bool(current_configs and prior_configs and
            all(c.get('precision') not in (None,'unknown') for c in current_configs+prior_configs) and
            len({c['precision'] for c in current_configs+prior_configs})==1)
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    metric_path=prefix.with_name(prefix.name+'_metrics.json');audit_path=prefix.with_name(prefix.name+'_question_audit.jsonl')
    if metric_path.exists() or audit_path.exists():raise FileExistsError('Choose a new evaluation output prefix')
    metric_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    audit_path.write_text(''.join(json.dumps(a,ensure_ascii=False,default=str)+'\n' for a in audits),encoding='utf-8')
    return summary

def plan_event_pass(event):
    if event.get('validation_error'):return False
    if 'plan_accepted' in event:return event['plan_accepted'] is True
    try:obj=json.loads(event.get('raw_response',''))
    except (TypeError,json.JSONDecodeError):return False
    return bool(isinstance(obj,dict) and obj.get('fields') and obj.get('formula_id') not in (None,'unsupported'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,help='Generation JSONL; omit for retrieval-only evaluation')
    ap.add_argument('--rankings',type=Path,default=ROOT/'data/processed/eval/task3_bge_m3_rankings.jsonl')
    ap.add_argument('--contexts',type=Path,default=ROOT/'data/processed/eval/task4_v4_8_contexts.jsonl')
    ap.add_argument('--labels',type=Path,default=ROOT/'data/processed/eval/evidence_labels_v48.json')
    ap.add_argument('--baseline',type=Path)
    ap.add_argument('--training-dir',type=Path,help='Optional training output directory for loss, perplexity, checkpoint and memory metrics')
    ap.add_argument('--output-prefix',type=Path,required=True)
    args=ap.parse_args();s=evaluate(args.input,args.output_prefix,args.rankings,args.contexts,args.labels,args.baseline,training_dir=args.training_dir)
    print(json.dumps(dict(question_count=s['question_count'],retrieval_only=s['retrieval_only'],
        accuracy=s['final']['accuracy'],output_prefix=str(args.output_prefix)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
