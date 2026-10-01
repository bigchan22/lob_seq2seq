"""Targeted paired-support/blocked-result checks; no project prediction reads."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from consolidate import check_pair,bootstrap,read_exports,HASHES
class ConsolidationChecks(unittest.TestCase):
    def test_different_support_is_never_paired(self):
        a={k:'x' for k in HASHES};a.update(row_key_label_digest='a',rows=10)
        for k in HASHES+['row_key_label_digest','rows']:
            b=dict(a);b[k]='different'
            with self.assertRaises(ValueError):check_pair(a,b)
    def test_fold_resampling_preserves_weighting(self):
        a=np.array([1,2,3,4,5,6]);b=np.array([10,20,30,40,50,60])
        lo,hi=bootstrap([(-.02*a,a),(-.02*b,b)])
        self.assertAlmostEqual(lo,-.02);self.assertAlmostEqual(hi,-.02)
    def test_missing_exports_stay_missing(self):
        a,b,issues=read_exports(None,'a5000',{},None)
        self.assertEqual(a,[]);self.assertEqual(b,{});self.assertTrue(issues)
if __name__=='__main__':unittest.main(verbosity=2)
