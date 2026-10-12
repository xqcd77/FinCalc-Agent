"""V4: semantic planning, typed repair routes, and evidence bundle delivery.

Uses the existing local retrieval index and a user-started OpenAI-compatible model
server. No evaluation labels, audit files or gold answers are read by this agent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import sys
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib import request, error
from agent_v3 import EvidenceState, TargetedRetriever, PROJECT, CHUNKS, MAX_RETRIEVAL_ROUNDS
from field_plan_v4 import PLAN_SYSTEM, BIND_SYSTEM, validate_plan, validate_binding, calculate, schema_hint, infer_formula
from question_conditions_v4 import condition_ledger, task_guidance, missing_question_condition, resolve_question_entity, question_facts, date_mentions
from financial_semantics_v4 import V4Error
from select_context_v4 import build_context, strict_match, unit_notes, report_requirements, VERSION as SELECTOR_VERSION
from evidence_prompt_v4 import method_excerpt, binding_user, binding_repair_hint, field_facts
from financial_semantics_v4 import canonical
from financial_knowledge_v4 import planning_guidance
from plan_contract_v4 import repair_feedback, repair_actions, planning_issues
from planning_blueprints_v4 import blueprint, day_basis
from rerank_context import format_evidence
from prompt_budget import count_chat_tokens, ensure_prompt_fits, resolve_tokenizer_path
from tokenizer_v4 import native_tokenizer
from solution_steps_v48 import validate_steps
from stage_prompt_contract_v49 import planning_input, append_planning_options
from table_delivery_v48 import table_companions,format_field_evidence,CompanionIndex
from exact_cpu_cache import ExactCPUCache, signature
from eval_execution import ResultJournal, bounded_results, path_hashes
from generate_calculate import read_jsonl, MISSING

VERSION='rag-agent-v4.8-bge-m3-atomic-table-and-cot-20261007'
MODEL=os.environ.get('QWEN_MODEL_ID','Qwen/Qwen3.5-4B')
INPUT=PROJECT/'data/processed/eval/task4_v4_8_contexts.jsonl'
OUTPUT=PROJECT/'data/processed/eval/task5_v4_8_qwen35_rag_generation.jsonl'

def parse_json(raw):
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip(),flags=re.I)
    try: obj=json.loads(text);fixed=False
    except json.JSONDecodeError:
        # Only close unclosed containers. Never complete a string, number, quote or field.
        stack=[];inside=False;escaped=False
        for ch in text:
            if inside:
                if escaped:escaped=False
                elif ch=='\\':escaped=True
                elif ch=='"':inside=False
            elif ch=='"':inside=True
            elif ch in '{[':stack.append(ch)
            elif ch in '}]':
                if not stack or (stack.pop(),ch) not in {('{','}'),('[',']')}:
                    raise V4Error('format','JSON括号不匹配')
        if inside or not stack:raise V4Error('format','JSON不完整，需要重新输出')
        try:obj=json.loads(text+''.join('}' if x=='{' else ']' for x in reversed(stack)));fixed=True
        except json.JSONDecodeError as exc:raise V4Error('format','JSON不能无损闭合') from exc
    if not isinstance(obj,dict):raise V4Error('format','需要JSON对象')
    return obj,fixed

def call_model(endpoint,messages,timeout,model=MODEL):
    is_plan=messages[0]['content']==PLAN_SYSTEM
    body=json.dumps(dict(model=model,messages=messages,temperature=0,
        max_tokens=1024 if is_plan else 768,response_format={'type':'json_object'}),ensure_ascii=False).encode()
    req=request.Request(endpoint.rstrip('/')+'/chat/completions',body,{'Content-Type':'application/json'})
    try:
        with request.urlopen(req,timeout=timeout) as h:response=json.load(h)
    except error.HTTPError as exc:raise RuntimeError(f'Model HTTP {exc.code}: {exc.read(500).decode("utf-8","replace")}') from exc
    if response.get('model') and response['model']!=model:raise RuntimeError('Server returned a different model')
    return response['choices'][0]['message']['content'],response.get('usage',{})

@dataclass(frozen=True)
class BindingRequest:
    view: str
    refs: tuple
    user: str
    token_count: int
    evidence_signature: str

class EvidenceStateV4(EvidenceState):
    def __init__(self,row,chunks,tokenizer):
        super().__init__(row,chunks,tokenizer)
        self.entity=resolve_question_entity(row['question'],self.entity)
        # Binding prompts and validators read row metadata; use the same issuer
        # across planning, evidence selection and numerical provenance checks.
        self.row['target_entity']=self.entity
        self._field_requests=ExactCPUCache(32)
        self._matches=ExactCPUCache(128)

    def _match(self,field,chunk):
        # Include all visible companions, complete field and company metadata.
        # Additions/rollback/ID/period/unit changes produce a different key.
        key=signature((field,chunk,self.row['company_with_period'],self.entity,
                       self.row.get('issuer_names',()),self.by_ref))
        return self._matches.get_or_compute(key,lambda: strict_match(field,chunk,
            self.row['company_with_period'],self.entity,self.row.get('issuer_names',()),self.by_ref))

    def candidate_refs(self,field):
        return [ref for ref,c in self.by_ref.items() if self._match(field,c).get('matched')]

    def view_for_field(self,field,hint=''):
        prepared=self.request_for_field(field,hint)
        return prepared.view,list(prepared.refs),prepared.token_count

    def request_for_field(self,field,hint=''):
        key=signature((field,hint,self.row,self.by_ref,self.entity))
        return self._field_requests.get_or_compute(key,lambda:
            self._request_for_field_uncached(field,hint,key))

    def _request_for_field_uncached(self,field,hint,key):
        groups=[]
        for ref,c in self.by_ref.items():
            match=self._match(field,c)
            if not match['matched']:continue
            companions=table_companions(c,self.chunks)
            refs=[ref]+match.get('unit_note_refs',[])+[r for r,other in self.by_ref.items()
                if any(other['chunk_id']==n['chunk_id'] for n in companions)]
            groups.append((c.get('content_type')!='table',c.get('source_start_line',0),refs))
        chosen=[];chosen_user=None;chosen_tokens=None
        for _,_,group in sorted(groups):
            refs=list(dict.fromkeys(chosen+group))
            ordered=sorted(refs,key=lambda r:(self.by_ref[r]['doc_id'],self.by_ref[r].get('source_start_line',0),int(r[1:])))
            user=binding_user(self.row,field,ordered,self.by_ref,hint)
            tokens=count_chat_tokens(self.tokenizer,[dict(role='system',content=BIND_SYSTEM),dict(role='user',content=user)])
            if tokens<=2300:chosen=ordered;chosen_user=user;chosen_tokens=tokens
        view='\n\n'.join(format_field_evidence(int(r[1:]),self.by_ref[r],[field],
            self.row['company_with_period'],self.entity,self.by_ref) for r in chosen)
        if chosen_user is None:
            chosen_user=binding_user(self.row,field,chosen,self.by_ref,hint)
            chosen_tokens=count_chat_tokens(self.tokenizer,[dict(role='system',content=BIND_SYSTEM),dict(role='user',content=chosen_user)])
        return BindingRequest(view,tuple(chosen),chosen_user,chosen_tokens,key)

class RagAgentV4:
    def __init__(self,ask,retriever,chunks,tokenizer,max_retrieval_rounds=2,mode='rag'):
        self.ask,self.retriever,self.chunks,self.tokenizer=ask,retriever,chunks,tokenizer
        self.max_retrieval_rounds=max_retrieval_rounds
        if mode not in {'rag','no_rag'}:raise ValueError('Unknown mode')
        self.mode=mode

    @staticmethod
    def _calendar_notes(state):
        q=state.row['question'];m=re.search(r'(20\d{6})到(20\d{6})',q)
        if re.search(r'下一个交易日|下一交易日',q):
            given=[f for f in question_facts(q,state.entity) if f['metric']=='股价' and f.get('date')]
            if not given:return []
            base=min(given,key=lambda f:f['start'])['date']
            candidate=datetime.strptime(base,'%Y-%m-%d')+timedelta(days=1)
            while candidate.weekday()>=5:candidate+=timedelta(days=1)
            day=candidate.strftime('%Y-%m-%d')
            field=dict(id='next_calendar_price',metric='股价',year=day[:4],date=day,source='report')
            for ref,c in state.by_ref.items():
                if c.get('content_type')=='table' and strict_match(field,c,state.row['company_with_period'],state.entity,state.row.get('issuer_names',()),state.by_ref).get('matched'):
                    return [(ref,'已核验下一交易日：'+day+'；基准日：'+base+'。下一候选工作日确有目标公司行情，未凭缺行推断休市。')]
            return []
        if not m or not any(w in q for w in ('最高','最低','次高','次低')):return []
        begin,end=(datetime.strptime(x,'%Y%m%d') for x in m.groups())
        if not 0<=(end-begin).days<=14:return []
        expected=[(begin+timedelta(days=i)).strftime('%Y-%m-%d') for i in range((end-begin).days+1) if (begin+timedelta(days=i)).weekday()<5]
        for metric,label in [('股价','收盘价'),('涨跌幅','涨跌幅'),('涨跌金额','涨跌金额')]:
            if metric!='股价' and not any(f['metric']=='股价' and f.get('date') and f['date']<expected[0] for f in question_facts(q,state.entity)):continue
            byref={}
            for ref,c in state.by_ref.items():
                if c.get('content_type')!='table':continue
                for day in expected:
                    field=dict(id='calendar_value',metric=metric,year=day[:4],date=day,source='report')
                    if strict_match(field,c,state.row['company_with_period'],state.entity,state.row.get('issuer_names',()),state.by_ref).get('matched'):
                        byref.setdefault(ref,[]).append(day)
            covered={d for dates in byref.values() for d in dates}
            if set(expected)==covered:
                return [(ref,'区间工作日'+label+'覆盖完整；已核验'+label+'日期：'+json.dumps(dates,ensure_ascii=False)+'。逐日值仍须绑定该表目标公司行/日期列，禁止套用对照公司。') for ref,dates in byref.items()]
        return []

    @staticmethod
    def _valuation_cap_notes(state):
        q=state.row['question']
        if '静态市盈率' not in q or not any(c['metric']=='总市值' for c in question_facts(q,state.entity)):return []
        notes=[]
        for ref,c in state.by_ref.items():
            text=c.get('content_text','')[:550]
            if c.get('company')!=state.row['company_with_period'] or c.get('content_type')!='prose' or state.entity not in text:continue
            if re.search(r'静态市盈率.*总市值\s*/\s*上一年度归属于母公司',text):notes.append((ref,text))
        return notes

    @staticmethod
    def _fiscal_notes(state):
        q=state.row['question'];dates=date_mentions(q)
        if not dates or not re.search(r'市盈率|市净率|市销率',q):return []
        year=str(min(int(d[2][:4]) for d in dates)-1)
        metric='每股净资产' if '市净率' in q else '每股收益' if '市盈率' in q else '营业收入'
        field=dict(id='fiscal_basis',metric=metric,year=year,source='report')
        for ref,c in state.by_ref.items():
            if strict_match(field,c,state.row['company_with_period'],state.entity,state.row.get('issuer_names',()),state.by_ref).get('matched'):
                return [(ref,'本报告目标公司已核验完整财年字段：'+metric+'，'+year+'年度；这里只确定期间，原始值仍须单独证据绑定。')]
        return []

    def _prepare_planning_evidence(self,state,events):
        """At most two targeted searches for missing methodology/calendar."""
        if self.mode!='rag' or self.retriever is None or not self.max_retrieval_rounds:return
        q=state.row['question'];queries=[]
        notes=[n['text'] for c in state.by_ref.values() if (n:=method_excerpt(q,c))]
        if ('周转天数' in q or '现金循环周期' in q) and not day_basis(q,notes):
            metric=next((m for m in ('应收账款周转天数','应付账款周转天数','现金循环周期','存货周转天数') if m in q),'周转天数')
            queries.append(dict(id='method_probe',metric=metric,year=None,source='report',aliases=['计算方法','360天 365天']))
        if ((re.search(r'20\d{6}到20\d{6}',q) and any(w in q for w in ('最高','最低','次高','次低'))) or re.search(r'下一个交易日|下一交易日',q)) and not self._calendar_notes(state):
            queries.append(dict(id='calendar_probe',metric='股价',year=None,source='report',aliases=['收盘价','涨跌幅','涨跌金额','每日涨跌','交易日行情表']))
        if '静态市盈率' in q and any(c['metric']=='总市值' for c in question_facts(q,state.entity)) and not self._valuation_cap_notes(state):
            queries.append(dict(id='valuation_method_probe',metric='市盈率',year=None,source='report',aliases=['总市值','归母净利润','上一年度','计算方法']))
        for field in queries[:2]:
            log=dict(step='planning_evidence_search',field_id=field['id'],added_refs=[],candidate_chunk_ids=[],max_searches=2)
            events.append(log)
            candidates=self.retriever.search(field,state.row,state.seen_ids,1)
            for candidate in candidates[:20]:
                c=candidate['chunk'];log['candidate_chunk_ids'].append(c['chunk_id'])
                if c.get('company')!=state.row['company_with_period']:continue
                if field['id']=='method_probe':
                    note=method_excerpt(q,c)
                    if not note or not day_basis(q,[note['text']]):continue
                elif field['id']=='valuation_method_probe':
                    if c.get('content_type')!='prose' or state.entity not in c.get('content_text','') or not re.search(r'静态市盈率.*总市值\s*/\s*上一年度归属于母公司',c.get('content_text','')[:550]):continue
                elif c.get('content_type')!='table' or not any(t in str(c.get('table_caption',''))+c.get('content_text','') for t in ('收盘价','股价','涨跌')):continue
                ref=state.add(candidate)
                if ref:log['added_refs'].append(ref)
                if len(log['added_refs'])>=2:break
            log['method_or_calendar_still_required']=not (day_basis(q,[n['text'] for c in state.by_ref.values() if (n:=method_excerpt(q,c))]) if field['id']=='method_probe' else self._valuation_cap_notes(state) if field['id']=='valuation_method_probe' else self._calendar_notes(state))

    def _ask(self,system,user,events,step,shown_refs=(),prepared=None):
        messages=[dict(role='system',content=system),dict(role='user',content=user)]
        budget=2100 if system==PLAN_SYSTEM else 2300
        if prepared is not None and (system!=BIND_SYSTEM or user!=prepared.user or tuple(shown_refs)!=prepared.refs):
            raise ValueError('Prepared binding request does not match dispatched prompt')
        tokens=prepared.token_count if prepared is not None else count_chat_tokens(self.tokenizer,messages)
        event=dict(step=step,call_attempted=False,prompt_tokens=tokens,prompt_token_budget=budget,
            shown_refs=list(shown_refs),messages_sha256=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest(),
            messages=messages)
        events.append(event)
        if tokens>budget:raise V4Error('context_budget',f'{step} prompt uses {tokens} tokens; budget is {budget}')
        event['call_attempted']=True
        raw,usage=self.ask(messages);event.update(raw_response=raw,usage=usage)
        if usage.get('prompt_tokens') is not None and usage['prompt_tokens']!=tokens:
            raise RuntimeError('Server tokenizer/template differs from local prompt counter')
        obj,fixed=parse_json(raw)
        try:event['solution_steps']=validate_steps(obj)
        except ValueError as exc:
            # Explanation errors never invent a binding or change a valid executable plan.
            event['solution_steps_error']=str(exc)
        event['solution_steps_present']=bool(obj.get('solution_steps'))
        if fixed:event['json_normalisation']='closed_containers_without_inventing_content'
        return obj

    def _plan(self,state,events,repair=None):
        missing=missing_question_condition(state.row['question'])
        if missing:
            events.append(dict(step='question_condition_precheck',call_attempted=False,reason=missing))
            return dict(formula_id='unsupported',reason=missing,fields=[])
        self._prepare_planning_evidence(state,events)
        # Include sourced methodology for planning derived fields; no gold/external_knowledge.
        methods=[(ref,method_excerpt(state.row['question'],c)) for ref,c in state.by_ref.items()]
        methods=sorted([(ref,note) for ref,note in methods if note],key=lambda x:-x[1]['score'])
        notes=self._calendar_notes(state)+self._valuation_cap_notes(state)+self._fiscal_notes(state)+[(ref,note['text']) for ref,note in methods]
        notes=notes[:3]
        base,initial_hint,optional=planning_input(state.row['question'],state.entity)
        previous=repair;err=(repair or {}).get('tool_error','');last_error=ValueError(err)
        repeats={}
        for attempt in range(3):
            user=base
            if previous:
                feedback='\n'+repair_feedback(previous,last_error,state.row['question'],state.entity)
                repeated=repeats.get(str(last_error),0)>=2
                if count_chat_tokens(self.tokenizer,[dict(role='system',content=PLAN_SYSTEM),dict(role='user',content=user+feedback)])>2100:
                    # Prior raw response is persisted in events. Replan from the
                    # full question and targeted diagnostics rather than clipping JSON.
                    details=getattr(last_error,'details',{})
                    feedback='\n重新规划。具体错误：'+str(last_error)+'\n'+repair_actions(last_error,state.row['question'])+'\n多项检查：'+json.dumps(planning_issues(previous,state.row['question'],state.entity,4),ensure_ascii=False,separators=(',',':'))
                if repeated:
                    feedback='\n同类错误已连续出现两次，丢弃上一计划结构，依据问题和新公式骨架重新规划。具体错误：'+str(last_error)+'\n'+repair_actions(last_error,state.row['question'])+'\n多项检查：'+json.dumps(planning_issues(previous,state.row['question'],state.entity,4),ensure_ascii=False,separators=(',',':'))
                user+=feedback
            elif attempt:
                user+='\n上次回复未解析成功：'+str(last_error)+'。重新输出完整紧凑JSON；fields最多12项，省略aliases/报告值，quote只取必要原文，禁止输出解释或复制截断字符串。'
            # Keep complete JSON/error; never truncate a plan into a misleading fragment.
            shown=[];shown_notes=[]
            current_hint=initial_hint
            for ref,note in notes:
                addition='\n已检索方法说明：'+json.dumps([(ref,note)],ensure_ascii=False)
                next_hint=schema_hint(state.row['question'],shown_notes+[note],entity=state.entity)
                candidate=user.replace(current_hint,next_hint,1)+addition
                if count_chat_tokens(self.tokenizer,[dict(role='system',content=PLAN_SYSTEM),dict(role='user',content=candidate)])<=2100:
                    user=candidate;shown.append(ref);shown_notes.append(note);current_hint=next_hint
            user=append_planning_options(user,optional,self.tokenizer)
            try:
                proposal=self._ask(PLAN_SYSTEM,user,events,'plan' if attempt==0 and repair is None else 'plan_repair',
                    shown)
                previous=proposal
                events[-1]['planning_issues']=planning_issues(proposal,state.row['question'],state.entity)
                bp=blueprint(state.row['question'],state.entity,shown_notes)
                events[-1]['planning_blueprint']=bp['family'] if bp else None
                plan=validate_plan(proposal,state.row['question'],shown_notes,entity=state.entity)
                events[-1]['planning_entity']=state.entity
                if plan.get('contract_normalisations'):
                    events[-1]['contract_normalisations']=plan['contract_normalisations']
                if plan['formula_id']=='unsupported':
                    reason=plan['reason']
                    if 'missing_question_condition' in reason:return plan
                    raise V4Error('semantic','缺少报告的直接值时应列出基础字段及推导关系；'+reason)
                events[-1]['plan_accepted']=True
                return plan
            except (ValueError,KeyError,TypeError,SyntaxError) as exc:
                last_error=exc;err=str(exc);events[-1].update(validation_error=err,error_category=getattr(exc,'category','schema'),
                                             validation_details=getattr(exc,'details',{}))
                repeats[err]=repeats.get(err,0)+1
                if getattr(exc,'details',{}).get('contract_normalisations'):
                    events[-1]['contract_normalisations']=exc.details['contract_normalisations']
        raise V4Error('planning',f'三次计划仍未通过：{err}')

    def _bind_field(self,state,field,events,round_number):
        hint='';last_category=None;last_rejects=[];last_refs=[]
        for attempt in range(2):
            prepared=state.request_for_field(field,hint)
            view,refs=prepared.view,list(prepared.refs)
            if not refs and state.candidate_refs(field):
                events.append(dict(step='context_budget_error',field_id=field['id'],candidate_refs=state.candidate_refs(field)))
                return None,'context_budget'
            user=prepared.user
            last_refs=refs
            try:
                proposal=self._ask(BIND_SYSTEM,user,events,f'bind_round_{round_number}_{field["id"]}'+('_repair' if attempt else ''),refs,prepared=prepared)
                candidates=proposal.get('bindings',[])
                if not isinstance(candidates,list):raise V4Error('format','bindings必须为列表')
                if not candidates:
                    events[-1]['missing_ids']=[field['id']]
                    if attempt==0 and refs:
                        hint='这些已显示块通过字段候选核验。先逐项检查公司、指标、年份、数字；确实无直接值则missing。';continue
                    if last_category:break
                    return None,('model_missing_on_verified_candidates' if refs else None)
                rejects=[]
                for b in candidates:
                    try:
                        if not isinstance(b,dict):raise V4Error('format','binding必须为JSON对象')
                        if b.get('id')!=field['id'] or b.get('source_ref') not in refs:raise V4Error('provenance','字段ID或引用不是本次显示内容')
                        from evidence_contract_v4 import repair_boundary_pipes
                        displayed={r:state.by_ref[r] for r in refs}
                        repaired,repair=repair_boundary_pipes(b,displayed)
                        if not isinstance(repaired.get('quote'),str) or repaired['quote'] not in view:
                            raise V4Error('provenance','引文不在本次实际投递的证据正文中')
                        if repaired.get('unit_quote') and repaired['unit_quote'] not in view:
                            raise V4Error('provenance','单位引用不在本次实际投递的证据正文中')
                        item=validate_binding(repaired,field,state.row,displayed)
                        if repair:events[-1]['citation_format_repair']=repair
                        events[-1]['accepted_ids']=[field['id']]
                        if item.get('normalisation'):events[-1]['normalisation']=item['normalisation']
                        return item,None
                    except (ValueError,KeyError,TypeError) as exc:
                        last_category=getattr(exc,'category','format')
                        rejects.append(dict(candidate=b,reason=str(exc),category=last_category,details=getattr(exc,'details',{})))
                events[-1]['rejected_bindings']=rejects
                last_rejects=rejects
                hint=binding_repair_hint(field,proposal,rejects,{r:state.by_ref[r] for r in refs})
            except (ValueError,KeyError,TypeError) as exc:
                last_category=getattr(exc,'category','format')
                events[-1].update(validation_error=str(exc),error_category=last_category)
                hint='上次具体错误：'+str(exc)+'；请重新输出完整合法JSON，保持证据约束。'
        # Narrow deterministic recovery for a proven wrong-year selection only.
        # Never recover missing/format failures, dates, conflicts, or ambiguous amounts.
        wrong_year=any(r['category']=='semantic' and any(f.get('year')!=field.get('year') and f.get('metric')==canonical(field['metric'])
                       for f in r.get('details',{}).get('actual_facts',[])) for r in last_rejects)
        if wrong_year and not field.get('date'):
            candidates=field_facts(field,state.row,last_refs,state.by_ref)
            unique={(f['value'],f['unit']) for f in candidates}
            if len(unique)==1:
                f=candidates[0]
                b=dict(id=field['id'],value=f['value'],unit=f['unit'],source_ref=f['source_ref'],quote=f['quote'])
                try:
                    item=validate_binding(b,field,state.row,{r:state.by_ref[r] for r in last_refs})
                    events.append(dict(step='verified_year_fact_recovery',field_id=field['id'],shown_refs=last_refs,
                                       reason='two_failed_year_bindings_and_one_unique_verified_candidate',binding=item))
                    return item,None
                except (ValueError,KeyError,TypeError):pass
        return None,last_category or 'format'

    def run(self,input_row):
        # Re-evaluate all existing rerank candidates when an old V3 context is supplied.
        if self.mode=='no_rag':
            if {'gold','Reference_documents','external_knowledge'}&input_row.keys():raise ValueError('Evaluation labels leaked')
            row={**input_row,'selected_evidence':[],'context_text':'','reranked':[],'issuer_names':[]}
        else:
            row=build_context(input_row,self.chunks,self.tokenizer) if input_row.get('selector_version')!=SELECTOR_VERSION else input_row
        state=EvidenceStateV4(row,self.chunks,self.tokenizer);events=[];bindings={}
        result=dict(status='invalid_plan',final={'answer':None},error=None,trace={'events':events,'version':VERSION})
        try:
            plan=self._plan(state,events);result['trace']['field_plan']=plan
            if plan['formula_id']=='unsupported':
                result.update(status='insufficient_question',final={'answer':MISSING},error=plan['reason'])
                return self._finish(result,state,bindings)
            expression_repaired=False
            while True:
                result['status']='binding_validation_error'
                needed=[f for f in plan['fields'] if f['source']=='report'];failures={}
                for round_number in range(self.max_retrieval_rounds+1):
                    for f in needed:
                        if f['id'] in bindings:continue
                        item,category=self._bind_field(state,f,events,round_number)
                        if item:bindings[f['id']]=item;failures.pop(f['id'],None)
                        elif category:failures[f['id']]=category
                    missing=[f for f in needed if f['id'] not in bindings]
                    if not missing:break
                    # Formatting/provenance failures on visible candidates are not evidence absence.
                    searchable=[f for f in missing if not state.candidate_refs(f)]
                    if self.mode=='no_rag' or round_number==self.max_retrieval_rounds or not searchable:break
                    added=[];candidate_log=[]
                    for f in searchable:
                        candidates=self.retriever.search(f,state.row,state.seen_ids,round_number+1)
                        scored=[]
                        for i,candidate in enumerate(candidates,1):
                            chunk=candidate['chunk'];notes=table_companions(chunk,self.chunks)
                            match=strict_match(f,chunk,state.row['company_with_period'],state.entity,state.row.get('issuer_names',()),{n['chunk_id']:n for n in notes})
                            candidate_log.append(dict(field_id=f['id'],rank=i,chunk_id=chunk['chunk_id'],query=candidate['query'],verified_match=bool(match['matched']),rrf_score=candidate.get('rrf_score'),rerank_score=candidate.get('rerank_score')))
                            scored.append((not match['matched'],i,candidate,notes))
                        count=0
                        for _,_,candidate,notes in sorted(scored,key=lambda x:(x[0],x[1])):
                            group=[candidate]+[dict(chunk=n,query=candidate['query'],round=round_number+1) for n in notes]
                            before=len(state.row['selected_evidence']);saved=[]
                            for c in group:
                                ref=state.add(c)
                                if ref:saved.append(dict(field_id=f['id'],ref=ref,chunk_id=c['chunk']['chunk_id'],query=c['query']))
                            # Atomic rollback if any required companion could not be added.
                            required={c['chunk']['chunk_id'] for c in group}
                            if not required<=state.seen_ids:
                                for e in state.row['selected_evidence'][before:]:state.by_ref.pop(e['citation']);state.seen_ids.remove(e['chunk_id'])
                                state.row['selected_evidence']=state.row['selected_evidence'][:before]
                                state.row['context_text']='\n\n'.join(format_evidence(int(r[1:]),c) for r,c in state.by_ref.items())
                                continue
                            added.extend(saved);count+=int(bool(saved))
                            if count>=2:break
                    events.append(dict(step='targeted_retrieval',round=round_number+1,requested_fields=[f['id'] for f in searchable],added=added,candidates=candidate_log))
                    if not added:break
                missing=[f for f in needed if f['id'] not in bindings]
                if missing:
                    reasons={f['id']:failures.get(f['id'],'evidence_missing') for f in missing}
                    categories=set(reasons.values())
                    result.update(status='binding_validation_error' if categories-{'evidence_missing'} else 'insufficient',
                        final={'answer':None if categories-{'evidence_missing'} else MISSING},error=json.dumps(reasons,ensure_ascii=False))
                    result['trace']['missing_fields']=reasons;break
                try:
                    result['status']='calculation_error'
                    tool=calculate(plan,bindings,state.row)
                    events.append(dict(step='python_decimal_tool',result=tool))
                    result.update(status='direct' if plan['formula_id']=='direct' else 'calculated',final={'answer':tool['answer']})
                    result['trace']['calculation']=tool;break
                except (ValueError,ArithmeticError,TypeError,KeyError) as exc:
                    events.append(dict(step='python_decimal_tool',error=str(exc),error_category='expression_or_numeric_domain'))
                    if expression_repaired:result.update(status='calculation_error',error=str(exc));break
                    old=plan;plan=self._plan(state,events,repair={**plan,'tool_error':str(exc)})
                    result['trace']['field_plan']=plan;expression_repaired=True
                    if plan['formula_id']=='unsupported':
                        result.update(status='unsupported_plan',error=plan['reason']);break
                    # A changed plan cannot retain stale bindings, even if its field IDs happen to agree.
                    bindings={}
        except (ValueError,TypeError,KeyError,SyntaxError,ArithmeticError) as exc:
            result.update(error=str(exc),error_category=getattr(exc,'category','validation'))
        except (RuntimeError,OSError) as exc:result.update(status='api_error',error=str(exc))
        return self._finish(result,state,bindings)

    @staticmethod
    def _finish(result,state,bindings):
        fields=result['trace'].get('field_plan',{}).get('fields',[])
        report=[f for f in fields if f['source']=='report']
        result['trace']['bindings']=bindings
        result['trace']['evidence_completeness']=dict(planned_fields=len(fields),question_fields=len(fields)-len(report),
            report_fields=len(report),verified_report_fields=len(bindings),complete=bool(fields) and len(bindings)==len(report),
            metric_type='semantic_verified_plan_fields_not_independent_gold_operands')
        formula=infer_formula(state.row['question'],state.entity)
        if formula in {'growth_rate','fixed_asset_turnover','current_asset_turnover','asset_turnover','inventory_turnover',
                       'cash_interest_coverage','revenue_from_margin','debt_equity_from_debt_ratio','debt_ratio_from_multiplier','cash_ratio'}:
            required=report_requirements(state.row)
            verified=[f for f in required if any(x['id'] in bindings and canonical(x['metric'])==canonical(f['metric'])
                      and x.get('year')==f.get('year') for x in fields)]
            result['trace']['registered_required_field_coverage']=dict(required=required,verified=len(verified),
                complete=bool(fields) and len(verified)==len(required),metric_type='registry_roles_not_independent_human_gold')
        result['selected_evidence']=state.row['selected_evidence']
        result['source_chunk_ids']=[e['chunk_id'] for e in result['selected_evidence']]
        shown={r for e in result['trace']['events'] for r in e.get('shown_refs',[])}
        result['shown_chunk_ids']=[e['chunk_id'] for e in result['selected_evidence'] if e['citation'] in shown]
        return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=INPUT);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--endpoint',default='http://127.0.0.1:8000/v1');p.add_argument('--model',default=MODEL);p.add_argument('--tokenizer-path',type=Path)
    p.add_argument('--mode',choices=['rag','no_rag'],default='rag')
    p.add_argument('--timeout',type=int,default=360);p.add_argument('--limit',type=int);p.add_argument('--dry-run',action='store_true');p.add_argument('--max-retrieval-rounds',type=int,default=2)
    p.add_argument('--run-label',default='base',help='Record base or adapter/checkpoint identity for comparison')
    p.add_argument('--precision-label',default='unknown',help='Record actual server precision, e.g. bf16 or nf4')
    p.add_argument('--skip-evaluation',action='store_true')
    p.add_argument('--evaluation-prefix',type=Path)
    p.add_argument('--training-dir',type=Path,help='Optional adapter training output directory to include in offline evaluation')
    p.add_argument('--question-workers',type=int,choices=[1,2],default=1,
        help='2 overlaps independent CPU question preparation; model requests remain serial. Enable after paired checks.')
    p.add_argument('--resume',action='store_true',help='Resume only an exactly fingerprint-matching result journal')
    p.add_argument('--retry-api-errors',action='store_true',help='With --resume, archive/retry only api_error; retain genuine failures')
    args=p.parse_args()
    if args.limit is not None and args.limit<1:p.error('--limit must be positive')
    if args.retry_api_errors and not args.resume:p.error('--retry-api-errors requires --resume')
    if not 0<=args.max_retrieval_rounds<=3:p.error('retrieval rounds must be 0..3')
    tok=native_tokenizer(args.tokenizer_path);chunks=CompanionIndex({c['chunk_id']:c for c in read_jsonl(CHUNKS)})
    rows=list(read_jsonl(args.input));rows=rows[:args.limit] if args.limit else rows
    if not rows or len({r['finqa_id'] for r in rows})!=len(rows):raise ValueError('Missing/duplicate question IDs')
    for row in rows:
        if {'gold','Reference_documents','external_knowledge'}&row.keys():raise ValueError('Evaluation labels leaked into generation')
        EvidenceStateV4(row,chunks,tok)
    if args.dry_run:print(json.dumps(dict(version=VERSION,rows=len(rows),model=args.model),ensure_ascii=False));return
    if args.mode=='no_rag' and args.output==OUTPUT:args.output=OUTPUT.with_name(OUTPUT.name.replace('_rag_','_no_rag_'))
    if not args.resume and (args.output.exists() or args.output.with_suffix('.jsonl.partial').exists()):
        raise FileExistsError('Output exists; use --resume with a matching new-format run, or choose a new output')
    # Bind every journal to content hashes, never just a human run label.
    # Old servers without run-identity fail closed; upgrade the serving script.
    try:
        with request.urlopen(args.endpoint.rstrip('/')+'/run-identity',timeout=args.timeout) as h:
            server_identity=json.load(h)
    except error.HTTPError as exc:
        raise RuntimeError('Model server must expose /v1/run-identity; deploy the updated BF16 LoRA serving script') from exc
    if server_identity.get('model')!=args.model:raise ValueError('Server identity/model mismatch')
    from hybrid_retrieval import INDEX, MODEL_PATH as EMBEDDING_PATH
    from rerank_context import MODEL_PATH as RERANKER_PATH
    identity=dict(server=server_identity,source=path_hashes(PROJECT/'src/rag',{'.py'}),
        configs=path_hashes(PROJECT/'configs',{'.json'}),
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        corpus_sha256=hashlib.sha256(CHUNKS.read_bytes()).hexdigest(),
        tokenizer=path_hashes(resolve_tokenizer_path(args.tokenizer_path).parent,{'.json','.jinja','.txt','.model'}),
        index=path_hashes(INDEX) if args.mode=='rag' else None,
        embedding=path_hashes(EMBEDDING_PATH) if args.mode=='rag' else None,
        reranker=path_hashes(RERANKER_PATH) if args.mode=='rag' else None,
        parameters=dict(model=args.model,mode=args.mode,timeout=args.timeout,
            run_label=args.run_label,precision=args.precision_label,
            temperature=0,plan_max_tokens=1024,binding_max_tokens=768,
            max_retrieval_rounds=args.max_retrieval_rounds,question_workers=args.question_workers,
            cpu_cache_enabled=os.environ.get('FINCALC_EXACT_CPU_CACHE','1')!='0'))
    model_request_lock=threading.Lock()
    def ask(messages):
        with model_request_lock:return call_model(args.endpoint,messages,args.timeout,args.model)
    retriever=TargetedRetriever(list(chunks.values())) if args.mode=='rag' else None
    agent=RagAgentV4(ask,retriever,chunks,tok,args.max_retrieval_rounds,args.mode)
    # Immutable input artifacts are hashed once, not reread after every question.
    artifact_hashes = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        tokenizer_sha256=hashlib.sha256(resolve_tokenizer_path(args.tokenizer_path).read_bytes()).hexdigest(),
        chunks_sha256=hashlib.sha256(CHUNKS.read_bytes()).hexdigest())
    def run_question(row):
        start=time.perf_counter()
        try:result=agent.run(row)
        except Exception as exc:result=dict(status='api_error',final={'answer':None},trace={},error=f'{type(exc).__name__}: {exc}')
        output=dict(finqa_id=row['finqa_id'],mode=args.mode,model=args.model,prompt_version=VERSION,
            experiment_config=dict(run_label=args.run_label,precision=args.precision_label,temperature=0,
                plan_max_tokens=1024,binding_max_tokens=768,max_retrieval_rounds=args.max_retrieval_rounds,
                question_workers=args.question_workers),
            **artifact_hashes,latency_seconds=round(time.perf_counter()-start,3),**result)
        return output
    journal=ResultJournal(args.output,[r['finqa_id'] for r in rows],identity,args.resume,args.retry_api_errors)
    try:
        pending=journal.pending(rows)
        print(f'Resume retained {len(journal.records)}/{len(rows)}; pending {len(pending)}; CPU workers {args.question_workers}; GPU requests serial',flush=True)
        for output in bounded_results(pending,run_question,args.question_workers):
            journal.append(output)
            print(f'{len(journal.records)}/{len(rows)} {output["finqa_id"]} {output["status"]}',flush=True)
        journal.finalize()
    finally:journal.close()
    if not args.skip_evaluation:
        # Evaluation is a separate process, after all predictions are saved.
        # The generation agent never receives official answers or operand qrels.
        prefix=args.evaluation_prefix or PROJECT/'reports'/args.output.stem/'evaluation'
        command=[sys.executable,str(PROJECT/'src/eval/evaluate_v48.py'),
            '--input',str(args.output),'--contexts',str(args.input),'--output-prefix',str(prefix)]
        if args.training_dir:command.extend(['--training-dir',str(args.training_dir)])
        subprocess.run(command,check=True)

if __name__=='__main__':main()
