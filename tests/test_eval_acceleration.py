"""Exact CPU/request parity, cross-question isolation and interrupted resume."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/rag'))
from agent_v4 import EvidenceStateV4
from exact_cpu_cache import ExactCPUCache
from eval_execution import ResultJournal, bounded_results
from financial_semantics_v4 import metric_spans, METRICS, term_pattern
from rerank_context import BGERerankerLarge
from table_delivery_v48 import CompanionIndex, table_companions
from tokenizer_v4 import native_tokenizer
from evidence_prompt_v4 import binding_user
from field_plan_v4 import BIND_SYSTEM


def load(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


@unittest.skipUnless((ROOT/"reports/eval_acceleration_implementation_20261012/before/agent_v4.py").exists() and (ROOT/"models/qwen3.5-9b-audit-tokenizer/tokenizer.json").exists(), "Optional historical parity artifacts/model tokenizer are not distributed")
class AccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('pre_acceleration_agent',
            ROOT/'reports/eval_acceleration_implementation_20261012/before/agent_v4.py')
        cls.old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.old
        spec.loader.exec_module(cls.old)
        cls.chunks = {c['chunk_id']:c for c in load(ROOT/'data/processed/corpus/evidence_chunks.jsonl')}
        cls.index = CompanionIndex(cls.chunks)
        cls.contexts = {r['finqa_id']:r for r in load(ROOT/'data/processed/eval/task4_v4_8_contexts.jsonl')}
        cls.outputs = load(ROOT/'data/processed/eval/task5_lora_1920_r32_best_epoch3_bf16_rag_20261010.jsonl')
        cls.tok = native_tokenizer(ROOT/'models/qwen3.5-9b-audit-tokenizer/tokenizer.json')

    def test_all_available_report_fields_exact_prompt_and_token_ids(self):
        checked = 0
        for result in self.outputs:
            row = self.contexts[result['finqa_id']]
            old = self.old.EvidenceStateV4(copy.deepcopy(row),self.chunks,self.tok)
            new = EvidenceStateV4(copy.deepcopy(row),self.index,self.tok)
            for field in result.get('trace',{}).get('field_plan',{}).get('fields',[]):
                if field.get('source') != 'report': continue
                for hint in ('', '核对年份与原文单位；仅修复当前字段。'):
                    expected = old.view_for_field(field,hint)
                    actual = new.request_for_field(field,hint)
                    self.assertEqual(expected,(actual.view,list(actual.refs),actual.token_count))
                    user = binding_user(old.row,field,expected[1],old.by_ref,hint)
                    self.assertEqual(user,actual.user)
                    messages = [dict(role='system',content=BIND_SYSTEM),dict(role='user',content=user)]
                    rendered = self.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    self.assertEqual(len(self.tok.encode(rendered,add_special_tokens=False)),actual.token_count)
                    self.assertEqual(actual,new.request_for_field(field,hint))
                    checked += 1
        self.assertGreater(checked,100)
        print(f'Exact field request comparisons: {checked}',flush=True)

    def test_term_matching_original_order_and_copy_isolation(self):
        for chunk in list(self.chunks.values())[::31]:
            text = chunk.get('content_text',''); candidates=[]
            for name,terms in METRICS.items():
                for term in terms:
                    candidates.extend((m.start(),m.end(),name) for m in re.finditer(term_pattern(term),text,re.I))
            accepted=[]
            for item in sorted(candidates,key=lambda x:(-(x[1]-x[0]),x[0])):
                if not any(item[0]<x[1] and x[0]<item[1] for x in accepted):accepted.append(item)
            self.assertEqual(sorted(accepted),metric_spans(text))
            metric_spans(text).clear()
            self.assertEqual(sorted(accepted),metric_spans(text))

    def test_indexed_companions_equal_full_scan(self):
        for chunk in self.chunks.values():
            if chunk.get('content_type')=='table':
                self.assertEqual(table_companions(chunk,self.chunks),table_companions(chunk,self.index))

    def test_request_cache_invalidates_field_evidence_and_company(self):
        result = self.outputs[0]; row=copy.deepcopy(self.contexts[result['finqa_id']])
        field=next(f for f in result['trace']['field_plan']['fields'] if f['source']=='report')
        state=EvidenceStateV4(row,self.index,self.tok)
        first=state.request_for_field(field)
        changed=dict(field,id=field['id']+'_new',year='1999')
        self.assertNotEqual(first.evidence_signature,state.request_for_field(changed).evidence_signature)
        saved=state.by_ref.copy(); state.by_ref.clear()
        empty=state.request_for_field(field)
        self.assertFalse(empty.refs); self.assertNotEqual(first.evidence_signature,empty.evidence_signature)
        state.by_ref.update(saved)
        self.assertEqual(first,state.request_for_field(field))
        state.entity='另一家公司'
        self.assertNotEqual(first.evidence_signature,state.request_for_field(field).evidence_signature)

    def test_reranker_cache_requires_exact_order_and_batch(self):
        ranker=BGERerankerLarge.__new__(BGERerankerLarge)
        ranker.score_cache=ExactCPUCache(128);calls=[]
        def score(query,documents,batch):
            calls.append((query,tuple(documents),batch));return list(range(len(documents)))
        ranker._rerank_uncached=score
        ranker.rerank('q',['a','b'],4).clear()
        self.assertEqual([0,1],ranker.rerank('q',['a','b'],4))
        ranker.rerank('q',['b','a'],4);ranker.rerank('q',['a','b'],2);ranker.rerank('q2',['a','b'],4)
        self.assertEqual(4,len(calls));self.assertEqual(1,ranker.score_cache.hits)

    def test_resume_retains_refusal_retries_only_api_error_and_orders_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.jsonl';ids=['a','b','c'];identity={'adapter':'x','template':'t'}
            j=ResultJournal(path,ids,identity)
            j.append(dict(finqa_id='b',status='insufficient'));j.append(dict(finqa_id='a',status='api_error',error='network'))
            j.close()
            j=ResultJournal(path,ids,identity,True,True)
            self.assertEqual(['a','c'],[r['finqa_id'] for r in j.pending([dict(finqa_id=i) for i in ids])])
            j.append(dict(finqa_id='c',status='calculated'));j.append(dict(finqa_id='a',status='calculated'))
            j.finalize();j.close()
            self.assertEqual(ids,[r['finqa_id'] for r in load(path)])
            self.assertEqual('insufficient',load(path)[1]['status'])
            with self.assertRaisesRegex(ValueError,'fingerprint'):
                ResultJournal(path,ids,{'adapter':'different'},True)
            j=ResultJournal(path,ids,identity,True);self.assertFalse(j.pending([dict(finqa_id=i) for i in ids]));j.finalize();j.close()

    def test_resume_interrupted_last_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'run.jsonl';j=ResultJournal(path,['a','b'],{'base':'x'})
            j.append(dict(finqa_id='a',status='calculated'));j.close()
            with path.with_suffix('.jsonl.partial').open('ab') as h:h.write(b'{"finqa_id":"b",')
            j=ResultJournal(path,['a','b'],{'base':'x'},True)
            self.assertEqual(['a'],list(j.records));j.append(dict(finqa_id='b',status='calculated'));j.finalize();j.close()
            self.assertTrue(path.with_suffix('.interrupted_tail.bin').exists())

    def test_two_workers_isolate_questions_and_keep_request_serial(self):
        lock=threading.Lock();active=0;peak=0;inflight=0;max_inflight=0
        def run(row):
            nonlocal active,peak,inflight,max_inflight
            active+=1;peak=max(peak,active)
            steps=[]
            for step in range(3):
                time.sleep(.005)
                with lock:
                    inflight+=1;max_inflight=max(max_inflight,inflight)
                    time.sleep(.005);steps.append(step);inflight-=1
            active-=1
            return row['finqa_id'],steps
        results=list(bounded_results([dict(finqa_id=str(i)) for i in range(7)],run,2))
        self.assertEqual(7,len(results));self.assertEqual(2,peak);self.assertEqual(1,max_inflight)
        self.assertTrue(all(steps==[0,1,2] for _,steps in results))


if __name__=='__main__':unittest.main()
