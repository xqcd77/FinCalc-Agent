"""A reduced dataset must be runnable without weakening completeness checks."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from qlora_dataset_gate import validate_dataset_counts

class DatasetGateTests(unittest.TestCase):
    def fixture(self,n):
        return ({'counts':{'total':n,'splits':{'train':n*4//5,'validation':n//5}}},
                {'status':'passed','total_samples':n,'passed':n,'failed':0})
    def test_reduced_and_historical_dataset_sizes(self):
        for n in (1500,2000):
            m,v=self.fixture(n)
            self.assertEqual(validate_dataset_counts(m,v,n*4//5,n//5,True),m['counts']['splits'])
    def test_missing_actual_row_rejected(self):
        m,v=self.fixture(1500)
        with self.assertRaises(ValueError):validate_dataset_counts(m,v,1199,300,True)
    def test_old_2000_verification_cannot_approve_1500(self):
        m,_=self.fixture(1500);_,v=self.fixture(2000)
        with self.assertRaises(ValueError):validate_dataset_counts(m,v,1200,300,True)
    def test_failed_individual_audit_rejected(self):
        m,v=self.fixture(1500);v.update(passed=1499,failed=1)
        with self.assertRaises(ValueError):validate_dataset_counts(m,v,1200,300,True)
    def test_invalid_declaration_rejected(self):
        m,v=self.fixture(1500)
        for counts in ({'total':1500,'splits':{'train':1200}},
                       {'total':2000,'splits':{'train':1200,'validation':300}},
                       {'total':1500,'splits':{'train':1200,'validation':True}}):
            with self.subTest(counts=counts),self.assertRaises(ValueError):
                validate_dataset_counts({'counts':counts},v,1200,300,True)
    def test_historical_split_key(self):
        m,v=self.fixture(2000);m['counts']['split']=m['counts'].pop('splits')
        self.assertEqual(validate_dataset_counts(m,v,1600,400,True),{'train':1600,'validation':400})

if __name__=='__main__':unittest.main()
