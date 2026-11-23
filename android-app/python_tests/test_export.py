import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import sign_gravity_model as m


class ExportTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            sentinel = Path(t)/'sentinel'; sentinel.write_text('keep')
            with self.assertRaises(FileExistsError):
                m.main(['export-android','--out',t])
            self.assertEqual(sentinel.read_text(),'keep')

    def test_demo_wrong_cohort_stops_without_output(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)/'model'
            with self.assertRaisesRegex(ValueError,'725'):
                m.export_android(pd.DataFrame({'label':['삼키다']}),np.zeros((1,532)),{},['삼키다','아니다','상처'],out)
            self.assertFalse(out.exists())

    def test_committed_numeric_model_matches_python_probabilities(self):
        root=Path(__file__).resolve().parents[1]
        model=json.loads((root/'models/sign_model.json').read_text(encoding='utf-8-sig'))
        parity=json.loads((root/'models/parity_vectors.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(model['trained_trials'],725)
        self.assertEqual(model['feature_count'],532)
        self.assertEqual(set(model['signs']),{'삼키다','아니다','상처'})
        for v in parity['vectors']:
            blob=(root/'models'/v['csv']).read_bytes()
            self.assertEqual(m.sha256(blob),v['sha256'])
            _,hands=m.read_trial(blob)
            x=m.trial_features(hands)
            np.testing.assert_allclose(x,v['features'],rtol=0,atol=1e-10)
            z=(x-np.array(model['mean']))/np.array(model['scale'])
            logits=np.array(model['coef'])@z+model['intercept']
            p=np.exp(logits-logits.max());p/=p.sum()
            np.testing.assert_allclose(p,v['probabilities'],rtol=0,atol=1e-10)


if __name__=='__main__': unittest.main()
