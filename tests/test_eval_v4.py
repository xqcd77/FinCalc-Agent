"""Meaningful metric checks: complete evidence, refusal, and paired-mode guard."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/eval'))
import evaluate_v4

class EvalV4Tests(unittest.TestCase):
    def test_binding_failure_is_not_counted_as_numeric_success_or_literal_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);q=base/'q';q.mkdir()
            (q/'q.jsonl').write_text(json.dumps(dict(finqa_id='a',gold='0.5'))+'\n',encoding='utf-8')
            output=dict(finqa_id='a',model='test',mode='rag',prompt_version='v4',status='binding_validation_error',
                final={'answer':None},trace={'field_plan':{'formula_id':'ratio','fields':[{'source':'report'}]},
                'bindings':{},'events':[]})
            src=base/'out.jsonl';src.write_text(json.dumps(output)+'\n',encoding='utf-8')
            with patch.object(evaluate_v4,'QUESTIONS',q),patch.object(evaluate_v4,'independently_labeled_coverage',return_value={'details':[{'finqa_id':'a','complete':True}]}):
                metrics=evaluate_v4.evaluate(src,base/'eval')
            self.assertEqual(metrics['final_accuracy'],0)
            self.assertEqual(metrics['numeric_accuracy_given_valid_calculation']['denominator'],0)
            self.assertEqual(metrics['independent_false_refusal_given_complete_evidence']['rate'],0)
            self.assertEqual(metrics['independent_failure_given_complete_evidence']['rate'],1)

    def test_paired_version_delta_rejects_different_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);q=base/'q';q.mkdir()
            (q/'q.jsonl').write_text(json.dumps(dict(finqa_id='a',gold='0.5'))+'\n',encoding='utf-8')
            out=dict(finqa_id='a',model='test',mode='rag',prompt_version='v4',status='calculated',final={'answer':'0.5'},trace={})
            src=base/'out.jsonl';src.write_text(json.dumps(out)+'\n',encoding='utf-8')
            old=base/'baseline.jsonl';old.write_text(json.dumps({**out,'mode':'no_rag'})+'\n',encoding='utf-8')
            with patch.object(evaluate_v4,'QUESTIONS',q),patch.object(evaluate_v4,'independently_labeled_coverage',return_value={}):
                with self.assertRaisesRegex(ValueError,'no-RAG'):evaluate_v4.evaluate(src,base/'eval',baseline=old)

if __name__=='__main__':unittest.main()
