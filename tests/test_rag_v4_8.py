"""Contract regressions for evidence delivery, CoT supervision and metric denominators."""
import copy,json,sys,tempfile,unittest
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src/rag'),str(ROOT/'src/eval')]
from table_delivery_v48 import projected_table,table_companions,format_field_evidence
from solution_steps_v48 import validate_steps
from hybrid_retrieval import QUERY_PREFIX,DIMENSION,validate_index
from evaluate_v48 import evaluate,ranking_metrics,plan_event_pass

class TableDeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples=[json.loads(s) for s in (ROOT/'tests/fixtures/table_delivery_v48.metadata.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_real_multirow_header_and_exact_quotes_preserved_without_mutation(self):
        m=next(m for m in self.samples if m['family']=='bind_complex_annual_interim_headers')
        f=m['fields'][0];b=m['teacher_output']['bindings'][0];c=m['evidence'][b['source_ref']];before=copy.deepcopy(c)
        view=projected_table(c,[f],m['row']['company_with_period'],m['company'],m['evidence'])
        headers=set(c['header_row_numbers'])
        self.assertTrue(headers<={r['row'] for r in view['rows']})
        for r in c['rows']:
            if r['row'] in headers:self.assertIn(r,view['rows'])
        text=format_field_evidence(int(b['source_ref'][1:]),c,[f],m['row']['company_with_period'],m['company'],m['evidence'])
        self.assertIn(b['quote'],text)
        self.assertEqual(c,before)

    def test_unit_companion_cannot_cross_company_or_document(self):
        c=dict(chunk_id='t',company='甲',doc_id='d',content_type='table',rows=[],source_start_line=10,source_end_line=20)
        note=dict(chunk_id='u',company='甲',doc_id='d',content_type='prose',content_text='单位：万元',source_start_line=8,source_end_line=9)
        others={'u':note,'peer':dict(note,chunk_id='peer',company='乙'),'other':dict(note,chunk_id='other',doc_id='other')}
        self.assertEqual([x['chunk_id'] for x in table_companions(c,others)],['u'])

    def test_complete_unit_continuation_row_not_discarded(self):
        from unittest.mock import patch
        c=dict(content_type='table',rows=[dict(row=i,display_cells=cells) for i,cells in [(1,['指标','2023','2024']),(2,['营业收入','100','120']),(3,['单位','万元','万元']),(4,['成本','70','80'])]],
            header_row_numbers=[1],data_row_numbers=[2,3,4],content_text='表头第1行：指标 | 2023 | 2024\n数据第2行：营业收入 | 100 | 120\n数据第3行：单位 | 万元 | 万元\n数据第4行：成本 | 70 | 80')
        with patch('select_context_v4.strict_match',return_value=dict(matched=True,target_data_rows=[2],matched_period_values=[],unit_data_row_numbers=[3])):
            v=projected_table(c,[dict(metric='营业收入')],'甲','甲',{})
        self.assertEqual([r['row'] for r in v['rows']],[1,2,3])
        self.assertIn('万元',v['content_text']);self.assertEqual(len(c['rows']),4)

    def test_every_positive_training_quote_and_companion_unit_is_visible(self):
        for m in self.samples:
            if m['task_stage'].startswith('planner'):continue
            text='\n\n'.join(format_field_evidence(int(r[1:]),c,m['fields'],m['row']['company_with_period'],m['company'],m['evidence']) for r,c in m['evidence'].items())
            for b in m['teacher_output']['bindings']:
                self.assertIn(b['quote'],text,m['sample_id'])
                if b.get('unit_quote'):self.assertIn(b['unit_quote'],text,m['sample_id'])

class SolutionStepTests(unittest.TestCase):
    def test_step_id_cannot_reference_missing_or_fabricated_field(self):
        obj=dict(fields=[dict(id='assets')],solution_steps=[dict(check='source',detail='核对资产',field_ids=['assets']),dict(check='units',detail='统一万元',field_ids=['debt'])])
        with self.assertRaises(ValueError):validate_steps(obj,required=True)
        obj['solution_steps'][1]['field_ids']=[]
        self.assertEqual(len(validate_steps(obj,required=True)),2)

    def test_optional_legacy_steps_do_not_invent_explanation(self):
        self.assertEqual(validate_steps(dict(fields=[])),[])
        with self.assertRaises(ValueError):validate_steps(dict(fields=[]),required=True)

    def test_malformed_model_declarations_are_rejected_with_value_error(self):
        steps=[dict(check='source',detail='核对',field_ids=[]),dict(check='units',detail='单位',field_ids=[])]
        for change in ({'fields':None},{'missing':[[]]},{'bindings':'bad'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_steps(dict(solution_steps=steps,**change),required=True)

    def test_dense_model_contract_and_prefix(self):
        self.assertEqual(QUERY_PREFIX,'');self.assertEqual(DIMENSION,1024)
        with self.assertRaises(ValueError):validate_index(dict(model='BAAI/bge-small-zh-v1.5',dense_dimension=512),Path('missing'),None)

    def test_dense_encoder_uses_named_pooled_output_not_first_graph_output(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import numpy as np
        from bge_m3_encoder import BgeM3Encoder
        class Session:
            def get_inputs(self):return [SimpleNamespace(name=n) for n in ('input_ids','attention_mask')]
            def get_outputs(self):return [SimpleNamespace(name=n) for n in ('token_embeddings','sentence_embedding')]
            def run(self,names,feed):
                self.names=names
                return [np.ones((feed['input_ids'].shape[0],1024),dtype=np.float32)]
        s=Session()
        class Options:
            def add_session_config_entry(self,*args):pass
        from tokenizers import Tokenizer, models
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]')).save(str(p/'tokenizer.json'))
            (p/'config.json').write_text('{"pad_token_id":0}',encoding='utf-8')
            fake_ort=SimpleNamespace(SessionOptions=Options,InferenceSession=lambda *args,**kwargs:s)
            with patch.dict(sys.modules,{'onnxruntime':fake_ort}):
                e=BgeM3Encoder(p)
                v=list(e.embed(['营业收入2023']))[0]
        self.assertEqual(s.names,['sentence_embedding'])
        self.assertEqual(v.shape,(1024,));self.assertAlmostEqual(float(np.linalg.norm(v)),1.,places=5)

class MetricTests(unittest.TestCase):
    def test_document_hit_does_not_imply_operand_recall_empty_retrieval_fails(self):
        qs={'a':dict(Reference_documents='doc.tex'),'b':dict(Reference_documents='doc.tex')}
        records={'a':dict(hybrid=[dict(chunk_id='unrelated',doc_id='doc')]),'b':dict(hybrid=[])}
        labels={'a':dict(fields=[dict(source='report',acceptable_chunk_ids=['needed'])])}
        m=ranking_metrics(qs,records,'hybrid',labels)['by_k']['1']
        self.assertEqual(m['document_recall']['value'],.5)
        self.assertEqual(m['required_report_field_recall']['value'],0)
        self.assertEqual(m['document_recall']['denominator'],2)

    def test_numeric_em_missing_answers_denominator_and_unavailable_qrels(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);q=p/'q';q.mkdir()
            (q/'q.jsonl').write_text('\n'.join(json.dumps(dict(finqa_id=i,gold=g,Reference_documents='doc.tex')) for i,g in [('a',.125),('b',.5)]),encoding='utf-8')
            (p/'o.jsonl').write_text('\n'.join(json.dumps(dict(finqa_id=i,status='calculated' if i=='a' else 'invalid_plan',final=dict(answer='.1250' if i=='a' else None))) for i in ('a','b')),encoding='utf-8')
            m=evaluate(p/'o.jsonl',p/'e',questions_dir=q)
            self.assertEqual(m['final']['accuracy']['value'],.5)
            self.assertEqual(m['final']['numeric_exact_match_4dp']['denominator'],2)
            self.assertIsNone(m['evidence']['certified_required_field_coverage']['value'])
            self.assertFalse(m['evidence']['certified_required_field_coverage']['available'])
            self.assertTrue((p/'e_question_audit.jsonl').is_file())

    def test_first_plan_pass_not_inferred_from_later_success(self):
        self.assertFalse(plan_event_pass(dict(raw_response='{}')))
        self.assertFalse(plan_event_pass(dict(raw_response=json.dumps(dict(fields=[dict(id='a')],formula_id='custom')),validation_error='wrong financial formula')))
        self.assertTrue(plan_event_pass(dict(plan_accepted=True)))

if __name__=='__main__':unittest.main()
