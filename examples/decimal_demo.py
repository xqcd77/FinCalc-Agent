"""Execute a cited financial calculation without a model or GPU."""
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/rag'))
from field_plan_v4 import calculate
from decimal import Decimal
row = {
    'question': '甲公司2024年营业利润率是多少？',
    'company_with_period': '甲公司2024年',
    'context_text': '[E1] 甲公司2024年营业收入120万元，营业利润18万元。',
    'selected_evidence': [{'citation':'E1','chunk_id':'synthetic_annual'}],
}
plan = {
    'formula_id':'operating_margin', 'answer_unit':'decimal',
    'fields':[
        {'id':'operating_profit','metric':'营业利润','source':'report','year':'2024'},
        {'id':'revenue','metric':'营业收入','source':'report','year':'2024'},
    ],
}
bindings = {
    'operating_profit':{'id':'operating_profit','value':'18','unit':'万元','source_ref':'E1','quote':'营业利润18万元'},
    'revenue':{'id':'revenue','value':'120','unit':'万元','source_ref':'E1','quote':'营业收入120万元'},
}
result = calculate(plan, bindings, row)
assert Decimal(result['answer']) == Decimal('0.15')
print(json.dumps({'question':row['question'],'plan':plan,'bindings':bindings,'result':result}, ensure_ascii=False, indent=2))
