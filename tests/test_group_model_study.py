"""Synthetic correctness tests; no fixture score is a measured sign result."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

import group_model_study as s
import sensor_preprocessing as p


def hand_captures(reverse=False):
    values = np.tile(np.arange(30)[:,None],(1,12)).astype(float)
    if reverse:
        values = values[::-1].copy()
    return {h:p.HandCapture('account','subject1','session','trial','a','training',h,'XIAO','v1',
        'raw.csv','0'*64,'2026-11-16T00:00:00Z','2026-11-16T00:00:04Z',False,
        np.arange(30)*100.+(0 if h=='left' else 900000),values.copy()) for h in ['left','right']}


class FeatureTests(unittest.TestCase):
    def test_local_clock_origins_do_not_change_features(self):
        hands = hand_captures()
        original = s.trial_features(hands,p.Recipe())
        hands['right'] = replace(hands['right'],times_ms=hands['right'].times_ms+90000)
        self.assertEqual(original,s.trial_features(hands,p.Recipe()))

    def test_time_order_distinguished_when_whole_statistics_are_equal(self):
        a,b = s.trial_features(hand_captures(),p.Recipe()),s.trial_features(hand_captures(True),p.Recipe())
        np.testing.assert_allclose([a[c] for c in p.FEATURES],[b[c] for c in p.FEATURES])
        self.assertNotEqual(a['phase0_left_ax_mean'],b['phase0_left_ax_mean'])
        self.assertEqual(len(a),384)

    def test_temporal_bins_partition_all_samples_and_check_time(self):
        masks = s.phase_masks(np.arange(30)*100)
        np.testing.assert_array_equal(np.sum(masks,axis=0),np.ones(30))
        with self.assertRaises(ValueError):
            s.phase_masks([0,100,100,300,400,500])
        with self.assertRaises(ValueError):
            s.phase_masks([0,1,2,3,4,1000])

    def test_missing_hand_is_rejected_not_filled_with_zeros(self):
        with self.assertRaisesRegex(ValueError,'Both hands'):
            s.trial_features({'left':hand_captures()['left']},p.Recipe())

    def test_features_do_not_change_inputs_and_sensor_selection_is_correct(self):
        hands = hand_captures()
        before = {h:c.values.copy() for h,c in hands.items()}
        tables = {name:pd.DataFrame([s.trial_features(hands,recipe)]) for name,recipe in s.RECIPES.items()}
        X,columns = s.matrices(tables)
        self.assertEqual(len(X),16)
        self.assertEqual(X['raw__whole__imu'].shape,(1,48))
        self.assertEqual(X['raw__temporal__all'].shape,(1,384))
        self.assertTrue(all('_fsr' not in c for c in columns['raw__temporal__imu']))
        for hand in hands:
            np.testing.assert_array_equal(hands[hand].values,before[hand])


class EvaluationTests(unittest.TestCase):
    def test_fast_knn_exactly_matches_sklearn_including_ties_and_zero_distance(self):
        rng = np.random.default_rng(99)
        train = rng.integers(-2,3,(70,4)).astype(float)
        test = np.vstack([train[:8],rng.integers(-2,3,(16,4))])
        y = np.array(['a','b','c','d','e']*14)
        configs = [c for c in s.configurations(['raw__whole__imu']) if c['algorithm']=='knn']
        with threadpool_limits(limits=1):
            pred = s.knn_predictions(train,y,test,configs,verify=True)
            for cfg in configs:
                reference = KNeighborsClassifier(n_neighbors=cfg['k'],p=cfg['p'],weights=cfg['weights'],
                    algorithm='brute',n_jobs=1).fit(train,y).predict(test)
                np.testing.assert_array_equal(reference,pred[cfg['id']])

    def test_nested_evaluation_never_passes_outer_people_into_search(self):
        rows = [dict(file=f'{person}-{label}-{i}.csv',person=f'subject{person}',session_id=f's{person}',
            trial_id=f'{label}-{i}',label=label) for person in range(1,11) for label in ['a','b'] for i in range(4)]
        cohort = pd.DataFrame(rows)
        rng = np.random.default_rng(55)
        X = {f'raw__{layout}__imu':rng.normal(size=(len(cohort),4)) for layout in ['whole','temporal']}
        configs=[]
        for name in X:
            for cfg in [dict(algorithm='knn',k=1,p=1,weights='uniform'),dict(algorithm='svm',C=1.,gamma='scale')]:
                configs.append(dict(id=len(configs),representation=name,layout=name.split('__')[1],**cfg))
        searches=[]
        original=s.search
        def audit(frame, *args, **kwargs):
            searches.append(frame.copy())
            return original(frame,*args,**kwargs)
        with tempfile.TemporaryDirectory() as temporary,threadpool_limits(limits=1):
            out=Path(temporary)
            with patch.object(s,'search',side_effect=audit),redirect_stdout(io.StringIO()):
                summary=s.evaluate(cohort,X,configs,out)
            logs=json.loads((out/'split_manifest.json').read_text(encoding='utf-8'))
            predictions=pd.read_csv(out/'predictions.csv')
            self.assertEqual(len(summary),7)
            for log,searched in zip(logs,searches):
                self.assertEqual(set(searched.person),set(log['train_people']))
                self.assertFalse(set(searched.person)&set(log['test_people']))
                seen=[]
                for inner in log['inner']:
                    self.assertFalse(set(inner['train_indices'])&set(log['test_indices']))
                    self.assertFalse(set(inner['validation_indices'])&set(log['test_indices']))
                    self.assertFalse(set(inner['train_people'])&set(inner['validation_people']))
                    seen.extend(inner['validation_indices'])
                self.assertEqual(sorted(seen),sorted(log['train_indices']))
            for result in summary:
                part=predictions[predictions.method.eq(result['method'])]
                self.assertEqual(result['correct'],int(part.label.eq(part.prediction).sum()))
                self.assertEqual(part.file.nunique(),len(cohort))


if __name__=='__main__':
    unittest.main()
