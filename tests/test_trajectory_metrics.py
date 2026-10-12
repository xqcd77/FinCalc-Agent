import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/eval'))
from trajectory_metrics import repeated_tools

class ToolRepetitionTests(unittest.TestCase):
    def test_same_decimal_input_repeats_but_model_repair_is_not_tool(self):
        event=dict(step='python_decimal_tool',result=dict(expression='a/b',answer_unit='decimal',
            operands=[dict(id='a',value='20',unit='元'),dict(id='b',value='10',unit='元')]))
        output=dict(trace=dict(events=[event,dict(step='plan_repair',call_attempted=True),event]))
        m=repeated_tools(output)
        self.assertEqual((2,2,1),(m['logged_tool_calls'],m['identifiable_tool_calls'],m['repeated_tool_calls']))

    def test_changed_seen_evidence_is_new_retrieval_input(self):
        field=dict(id='a',metric='营业收入',year='2024',source='report')
        plan=dict(step='plan',plan_accepted=True,planning_entity='甲公司')
        search=dict(step='targeted_retrieval',requested_fields=['a'],
            candidates=[dict(field_id='a',query='甲公司 2024 营业收入')],added=[])
        first={**search,'added':[dict(field_id='a',chunk_id='new') ]}
        output=dict(trace=dict(field_plan=dict(fields=[field]),events=[plan,first,search,search]))
        m=repeated_tools(output,dict(company_with_period='甲公司2024',selected_evidence=[]))
        self.assertEqual(3,m['identifiable_tool_calls'])
        self.assertEqual(1,m['repeated_tool_calls'])

    def test_failed_calculation_without_arguments_stays_unknown(self):
        m=repeated_tools(dict(trace=dict(events=[dict(step='python_decimal_tool',error='zero division')])))
        self.assertEqual(1,m['unidentifiable_tool_calls'])
        self.assertEqual(0,m['identifiable_tool_calls'])

if __name__=='__main__':unittest.main()
