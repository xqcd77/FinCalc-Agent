"""Behavior regressions for semantic V4; scripted models are not accuracy experiments."""
import json
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/rag'))
from financial_semantics_v4 import V4Error, facts, supports
from field_plan_v4 import validate_plan, validate_binding, calculate
from agent_v4 import RagAgentV4, parse_json
from select_context_v4 import build_context
from expression_contract_v4 import validate_expression, calculate_typed
from prompt_budget import count_chat_tokens
from decimal import Decimal
from test_rag_v3 import make_chunk, make_row, ScriptedModel, FakeRetriever, TOKENIZER

def row_with_rank(question,chunks):
    row=make_row(question)
    row['reranked']=[dict(chunk_id=c['chunk_id'],doc_id=c['doc_id'],rerank_rank=i+1) for i,c in enumerate(chunks)]
    return row

class SemanticV4Tests(unittest.TestCase):
    def test_same_question_amount_cannot_be_cashflow_and_liability(self):
        q='假如甲公司2023年的流动负债为26.55亿元，2023年现金流量比率是多少？'
        p=dict(formula_id='cash_flow_ratio',answer_unit='decimal',fields=[
            dict(id='operating_cash_flow',metric='经营现金流',year='2023',source='question',value='26.55',unit='亿元',quote='26.55亿元'),
            dict(id='current_liabilities',metric='流动负债',year='2023',source='question',value='26.55',unit='亿元',quote='2023年的流动负债为26.55亿元')])
        with self.assertRaisesRegex(V4Error,'指标'):validate_plan(p,q)

    def test_growth_requires_previous_year(self):
        q='假如甲公司2024年营业收入为100万元，2024年同比增长率是多少？'
        p=dict(formula_id='growth_rate',answer_unit='decimal',fields=[
            dict(id='current',metric='营业收入',year='2024',source='question',value='100',unit='万元',quote='2024年营业收入为100万元'),
            dict(id='previous',metric='营业收入',year='2024',source='report')])
        with self.assertRaisesRegex(V4Error,'上年'):validate_plan(p,q)

    def test_current_ratio_does_not_accept_quick_formula(self):
        q='甲公司2024年流动比率是多少？'
        p=dict(formula_id='quick_ratio',answer_unit='decimal',fields=[dict(id=i,metric=m,year='2024',source='report') for i,m in [('current_assets','流动资产'),('inventory','存货'),('current_liabilities','流动负债')]])
        with self.assertRaisesRegex(V4Error,'不匹配'):validate_plan(p,q)

    def test_cost_binding_cannot_borrow_gross_profit_number(self):
        c=make_chunk('gross','甲公司2024年毛利为473万元。营业成本可由毛利和毛利率推算。',year='2024')
        f=dict(id='cost',metric='营业成本',year='2024',source='report',aliases=['营业成本'])
        b=dict(id='cost',value='473',unit='万元',source_ref='E1',quote='毛利为473万元')
        with self.assertRaisesRegex(V4Error,'指标'):validate_binding(b,f,make_row('甲公司2024年营业成本是多少？',c),{'E1':c})

    def test_peer_with_stock_code_label_is_not_covered(self):
        text='紫光股份有限公司（股票代码：000938.SZ）营业收入从2022年的100亿元升至2023年的200亿元。'
        self.assertFalse(facts(text,'甲公司',['紫光股份'],True))

    def test_comparator_subject_carries_to_next_sentence(self):
        text='阳光电源（300274.SZ）处于快速发展阶段。该公司2022年流动资产为100亿元。'
        self.assertFalse(facts(text,'甲公司',['阳光电源'],True))

    def test_value_with_matching_unit_is_normalised(self):
        c=make_chunk('profit','甲公司2024年营业利润为31.54亿元。',year='2024')
        f=dict(id='operating_profit',metric='营业利润',year='2024',source='report',aliases=['营业利润'])
        b=dict(id='operating_profit',value='31.54亿元',unit='亿元',source_ref='E1',quote='营业利润为31.54亿元')
        result=validate_binding(b,f,make_row('甲公司2024年营业利润是多少？',c),{'E1':c})
        self.assertEqual(result['value'],'31.54')
        self.assertTrue(result['normalisation'])

    def test_round_cent_ratio_and_wrong_arity_rejected_before_execution(self):
        for expr in ['round_cent(a / b, 4)','round_cent(a / b)']:
            with self.assertRaises(V4Error):validate_expression(expr,{'a':(1,0),'b':(1,0)})

    def test_registered_360_day_contract_executes(self):
        self.assertEqual(validate_expression('360 * a / b',{'a':(1,0),'b':(1,0)}),(0,0))
        self.assertEqual(calculate_typed('360 * a / b',{'a':Decimal(1),'b':Decimal(2)},{'a':(1,0),'b':(1,0)})[0],Decimal(180))

    def test_json_closure_never_invents_string_content(self):
        obj,fixed=parse_json('{"bindings":[],"missing":[]')
        self.assertTrue(fixed);self.assertEqual(obj,{'bindings':[],'missing':[]})
        with self.assertRaises(V4Error):parse_json('{"quote":"unfinished')

    def test_plan_repair_preserves_specific_error_under_prompt_budget(self):
        q='假如甲公司2024年营业收入为100万元，2024年同比增长率是多少？'
        c=make_chunk('prior','甲公司2023年营业收入为80万元。',year='2023')
        bad=dict(formula_id='growth_rate',answer_unit='decimal',fields=[dict(id='current',metric='营业收入',year='2024',source='question',value='100',unit='万元',quote='2024年营业收入为100万元'),dict(id='previous',metric='营业收入',year='2024',source='report')])
        good=deepcopy(bad);good['fields'][1]['year']='2023'
        responses=[bad,good,dict(bindings=[dict(id='previous',value='80',unit='万元',source_ref='E1',quote='2023年营业收入为80万元')],missing=[])]
        seen=[]
        def model(messages):seen.append(messages);return json.dumps(responses.pop(0),ensure_ascii=False),{}
        result=RagAgentV4(model,FakeRetriever(),{'prior':c},TOKENIZER).run(row_with_rank(q,[c]))
        self.assertEqual(result['status'],'calculated',result)
        self.assertEqual(result['final']['answer'],'0.25')
        feedback=seen[1][1]['content']
        self.assertIn('上年',feedback)
        self.assertTrue('上次计划' in feedback or '重新规划。具体错误：' in feedback)
        self.assertLessEqual(count_chat_tokens(TOKENIZER,seen[1]),2100)

    def test_format_error_does_not_trigger_retrieval_or_refusal(self):
        q='假如甲公司2024年营业收入为100亿元，2024年营业利润率是多少？'
        c=make_chunk('profit','甲公司2024年营业利润为31.54亿元。',year='2024')
        p=dict(formula_id='operating_margin',answer_unit='decimal',fields=[dict(id='revenue',metric='营业收入',year='2024',source='question',value='100',unit='亿元',quote='2024年营业收入为100亿元'),dict(id='operating_profit',metric='营业利润',year='2024',source='report')])
        model=ScriptedModel([p,dict(bindings=[dict(id='operating_profit',value='31.54亿元',unit='亿元',source_ref='E1',quote='2024年营业利润为31.54亿元')],missing=[])])
        retriever=FakeRetriever();result=RagAgentV4(model,retriever,{'profit':c},TOKENIZER).run(row_with_rank(q,[c]))
        self.assertEqual(result['status'],'calculated',result);self.assertEqual(result['final']['answer'],'0.3154')
        self.assertEqual(retriever.calls,[])

    def test_production_input_rejects_gold(self):
        c=make_chunk('source','甲公司2022年营业收入为1万元。')
        row=row_with_rank('甲公司2022年营业收入是多少？',[c]);row['gold']=1
        with self.assertRaises(ValueError):build_context(row,{'source':c},TOKENIZER)

    def test_custom_cannot_bypass_current_ratio_semantics(self):
        p=dict(formula_id='custom',answer_unit='decimal',expression='(a - b) / c',fields=[
            dict(id=i,metric=m,year='2024',source='report') for i,m in [('a','流动资产'),('b','存货'),('c','流动负债')]])
        with self.assertRaisesRegex(V4Error,'运算关系'):validate_plan(p,'甲公司2024年的流动比率是多少？')

    def test_custom_correct_simple_ratio_remains_supported(self):
        p=dict(formula_id='custom',answer_unit='decimal',expression='a / b',fields=[
            dict(id=i,metric=m,year='2024',source='report') for i,m in [('a','流动资产'),('b','流动负债')]])
        self.assertEqual(validate_plan(p,'甲公司2024年的流动比率是多少？')['formula_id'],'custom')

    def test_question_supplied_condition_cannot_be_refused_as_absent(self):
        with self.assertRaisesRegex(V4Error,'已给出'):
            validate_plan(dict(formula_id='unsupported',reason='missing_question_condition:营业收入',fields=[]),
                '假如甲公司2024年营业收入为100万元，营业利润率是多少？')

    def test_visible_complete_evidence_refusal_is_binding_failure(self):
        q='假如甲公司2024年营业收入为100亿元，营业利润率是多少？'
        c=make_chunk('p','甲公司2024年营业利润为31.54亿元。',year='2024')
        p=dict(formula_id='operating_margin',answer_unit='decimal',fields=[
            dict(id='revenue',metric='营业收入',year='2024',source='question',value='100',unit='亿元',quote='2024年营业收入为100亿元'),
            dict(id='operating_profit',metric='营业利润',year='2024',source='report')])
        model=ScriptedModel([p,{'bindings':[],'missing':['operating_profit']},{'bindings':[],'missing':['operating_profit']}])
        ret=FakeRetriever();out=RagAgentV4(model,ret,{'p':c},TOKENIZER).run(row_with_rank(q,[c]))
        self.assertEqual(out['status'],'binding_validation_error',out)
        self.assertIn('model_missing_on_verified_candidates',out['error']);self.assertEqual(ret.calls,[])

    def test_true_missing_field_triggers_targeted_search(self):
        q='假如甲公司2024年营业收入为100亿元，营业利润率是多少？'
        c=make_chunk('p','甲公司2024年营业利润为31.54亿元。',year='2024')
        p=dict(formula_id='operating_margin',answer_unit='decimal',fields=[
            dict(id='revenue',metric='营业收入',year='2024',source='question',value='100',unit='亿元',quote='2024年营业收入为100亿元'),
            dict(id='operating_profit',metric='营业利润',year='2024',source='report')])
        b=dict(bindings=[dict(id='operating_profit',value='31.54',unit='亿元',source_ref='E1',quote='2024年营业利润为31.54亿元')],missing=[])
        ret=FakeRetriever([c]);out=RagAgentV4(ScriptedModel([p,{'bindings':[],'missing':['operating_profit']},b]),ret,{'p':c},TOKENIZER).run(row_with_rank(q,[]))
        self.assertEqual(out['status'],'calculated',out);self.assertEqual(ret.calls,[('operating_profit',1)])
        self.assertEqual(out['final']['answer'],'0.3154')

    def test_no_rag_never_uses_selected_evidence_or_retrieval(self):
        q='假如甲公司2024年营业收入为100亿元，营业利润率是多少？'
        c=make_chunk('p','甲公司2024年营业利润为31.54亿元。',year='2024')
        p=dict(formula_id='operating_margin',answer_unit='decimal',fields=[
            dict(id='revenue',metric='营业收入',year='2024',source='question',value='100',unit='亿元',quote='2024年营业收入为100亿元'),
            dict(id='operating_profit',metric='营业利润',year='2024',source='report')])
        ret=FakeRetriever([c]);out=RagAgentV4(ScriptedModel([p,{'bindings':[],'missing':['operating_profit']}]),ret,{'p':c},TOKENIZER,mode='no_rag').run(row_with_rank(q,[c]))
        self.assertEqual(out['source_chunk_ids'],[]);self.assertEqual(ret.calls,[])
        self.assertEqual(out['status'],'insufficient')

    def test_runtime_formula_repair_rebinds_all_report_fields(self):
        c=make_chunk('p','甲公司2024年资产总计为10万元，流动负债为10万元。',year='2024')
        q='甲公司2024年自定义资产指标是多少？'
        p=dict(formula_id='custom',expression='a / (a - b)',answer_unit='decimal',fields=[
            dict(id='a',metric='资产总计',year='2024',source='report'),dict(id='b',metric='流动负债',year='2024',source='report')])
        a=dict(bindings=[dict(id='a',value='10',unit='万元',source_ref='E1',quote='2024年资产总计为10万元')],missing=[])
        b=dict(bindings=[dict(id='b',value='10',unit='万元',source_ref='E1',quote='流动负债为10万元')],missing=[])
        fixed={**p,'expression':'a / b'}
        model=ScriptedModel([p,a,b,fixed,a,b]);out=RagAgentV4(model,FakeRetriever(),{'p':c},TOKENIZER).run(row_with_rank(q,[c]))
        self.assertEqual(out['status'],'calculated',out);self.assertEqual(out['final']['answer'],'1')
        self.assertEqual(model.calls,6)
        self.assertEqual(sum(e['step']=='plan_repair' for e in out['trace']['events']),1)

    def test_table_ratio_column_is_not_cashflow_amount(self):
        from select_context_v4 import strict_match
        c=make_chunk('p','表头第1行：年度 | 经营现金流净额\n表头第2行：年度 | / 营业收入\n数据第3行：2023 | 0.3350',year='2023')
        c.update(content_type='table',table_caption='经营现金流收入比')
        f=dict(id='operating_cash_flow',metric='经营现金流',year='2023')
        self.assertFalse(strict_match(f,c,'甲公司','甲公司')['matched'])

    def test_margin_table_is_not_gross_profit_amount(self):
        from select_context_v4 import strict_match
        c=make_chunk('p','表头第1行：年度 | 销售毛利率（%）\n数据第2行：2024 | 72.02',year='2024')
        c.update(content_type='table',table_caption='销售毛利率')
        self.assertFalse(strict_match(dict(id='gross_profit',metric='毛利',year='2024'),c,'甲公司','甲公司')['matched'])

    def test_oversized_plan_prompt_records_budget_error_without_index_crash(self):
        def forbidden(messages):raise AssertionError('Oversized prompt reached API')
        out=RagAgentV4(forbidden,FakeRetriever(),{},TOKENIZER).run(row_with_rank('甲公司2024年流动比率是多少？'+('补充说明。'*1000),[]))
        self.assertEqual(out['status'],'invalid_plan',out)
        self.assertTrue(out['error']);self.assertEqual(len(out['trace']['events']),3)
        self.assertFalse(any(e.get('call_attempted') for e in out['trace']['events']))

    def test_single_period_formula_rejects_other_year(self):
        p=dict(formula_id='current_ratio',answer_unit='decimal',fields=[dict(id='current_assets',metric='流动资产',year='2023',source='report'),dict(id='current_liabilities',metric='流动负债',year='2024',source='report')])
        with self.assertRaisesRegex(V4Error,'应使用2024年'):validate_plan(p,'甲公司2024年的流动比率是多少？')

    def test_intermediate_metric_does_not_override_equity_multiplier_target(self):
        from field_plan_v4 import infer_formula
        self.assertEqual(infer_formula('假如甲公司2024年非流动资产为100亿元，权益乘数是多少？'),'equity_multiplier')

if __name__=='__main__':unittest.main()
