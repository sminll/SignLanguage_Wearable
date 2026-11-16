"""Synthetic invariant checks, never reported as real recognition accuracy."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import pandas as pd
from tools import sensor_compensation as c
from tools import study_sensor_compensation as study
from tools.study_sensor_compensation import rest_links, normalize_person


class CompensationTests(unittest.TestCase):
    def setUp(self):
        self.values = np.random.default_rng(8).uniform(100, 800, (50, 12))

    def test_original_stats_are_channel_major(self):
        got = c.hand_features(self.values)
        expected = [getattr(np, s)(self.values[:, i]) for i in range(12) for s in c.STATS]
        np.testing.assert_allclose(got, expected)

    def test_fsr_gain_change_removed_without_changing_imu(self):
        before = self.values.copy()
        scaled = before.copy()
        scaled[:, :6] *= 3
        np.testing.assert_allclose(c.hand_features(before, scale_fsr=True),
                                   c.hand_features(scaled, scale_fsr=True))
        np.testing.assert_array_equal(self.values, before)
        self.assertFalse(np.allclose(c.hand_features(before), c.hand_features(scaled)))

    def test_low_and_zero_pressure_never_amplified_by_small_denominator(self):
        zero = self.values.copy()
        zero[:, :6] = 0
        features = c.hand_features(zero, scale_fsr=True)
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_array_equal(features[:24], 0)
        tiny = zero.copy()
        tiny[:, :6] = .01
        self.assertLess(c.hand_features(tiny, scale_fsr=True)[0], .001)

    def test_imu_norm_is_rotation_invariant_but_axes_are_not(self):
        q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(3, 3)))
        rotated = self.values.copy()
        rotated[:, 6:9] = rotated[:, 6:9] @ q
        rotated[:, 9:12] = rotated[:, 9:12] @ q
        np.testing.assert_allclose(c.hand_features(self.values, magnitude_imu=True),
                                   c.hand_features(rotated, magnitude_imu=True))
        self.assertFalse(np.allclose(c.hand_features(self.values), c.hand_features(rotated)))

    def test_paired_input_order_and_feature_sizes(self):
        hands = {'right': self.values*2, 'left': self.values}
        for rep in c.REPRESENTATIONS:
            feature = c.paired_features(hands, rep)
            np.testing.assert_allclose(feature[:len(feature)//2],
                                       c.hand_features(self.values, **c.REPRESENTATIONS[rep]))
        self.assertEqual(len(c.paired_features(hands, 'baseline_all')), 96)
        self.assertEqual(len(c.paired_features(hands, 'imu_magnitude_all')), 64)
        self.assertEqual(len(c.paired_features(hands, 'imu_magnitude_only')), 16)
        with self.assertRaisesRegex(ValueError, 'Both hands'):
            c.paired_features({'left': self.values}, 'baseline_all')

    def test_nonfinite_and_negative_fsr_fail_without_zero_fill(self):
        for value in [np.nan, np.inf, -1]:
            bad = self.values.copy()
            bad[0, 0] = value
            with self.assertRaises(ValueError):
                c.hand_features(bad)

    def test_robust_statistics_reduce_isolated_spike_effect(self):
        base = np.ones((50, 12))*100
        spike = base.copy()
        spike[0, 0] = 4000
        np.testing.assert_allclose(c.hand_features(base, robust=True),
                                   c.hand_features(spike, robust=True))
        self.assertFalse(np.allclose(c.hand_features(base), c.hand_features(spike)))

    def test_rest_matching_never_crosses_person_session_hand_version_or_future(self):
        trial = dict(file='trial', account='app', person='subject8', session_id='s',
                     board_type='XIAO', hand='left', calibration_version='v',
                     start='2026-11-15T01:00:00Z')
        rest = dict(**trial, is_rest=True, rows=40, span_ms=3900,
                    calibration_verified=False, end='2026-11-15T00:59:00Z')
        rest['file'] = 'rest'
        self.assertEqual(rest_links([trial], [rest])[0]['status'], 'metadata_match_but_unverified')
        for key, value in [('person','subject9'), ('session_id','other'), ('hand','right'),
                           ('calibration_version','other'), ('account','other'),
                           ('end','2026-11-15T01:01:00Z'), ('end','2026-11-14T01:00:00Z')]:
            with self.subTest(key=key, value=value):
                altered = {**rest, key:value}
                self.assertEqual(rest_links([trial], [altered])[0]['status'], 'no_matching_previous_rest')
        self.assertEqual(normalize_person('subject-8'), 'subject8')
        self.assertEqual(normalize_person('subject-ABCDE'), 'subject-ABCDE')

    def test_source_commit_comes_from_manifest_when_legacy_feature_csv_omits_it(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name, folder in study.COHORTS.items():
                directory = root/folder
                directory.mkdir()
                record = dict(file='same.csv', sha256='hash', person='subject1')
                if name == 'three_sign':
                    record['source_commit'] = 'pinned'
                pd.DataFrame([record]).to_csv(directory/'trial_features.csv', index=False)
                pd.DataFrame([{'person':'subject1'}]).to_csv(directory/'outer_predictions.csv', index=False)
                pd.DataFrame([{'person':'subject1'}]).to_csv(directory/'outer_folds.csv', index=False)
                (directory/'evaluation.json').write_text('{}')
                (directory/'inventory_summary.json').write_text(json.dumps({'commit':'pinned'}))
            frames, _, _, hashes, unique = study.load_inputs(root)
            self.assertEqual(len(unique), 1)
            self.assertEqual(frames['six_sign'].source_commit.iloc[0], 'pinned')
            self.assertTrue(any(p.endswith('inventory_summary.json') for p in hashes))

    def test_every_variant_keeps_trials_and_holds_out_the_entire_person(self):
        rows, features, references, folds = [], {}, [], []
        for person in range(1, 5):
            who = f'subject{person}'
            for trial, label in enumerate(['a', 'b']):
                path = f'{who}/{trial}'
                rows.append(dict(person=who, file=path, trial_id=path, session_id=who, label=label))
                features[path] = {rep: np.array([person, trial], dtype=float) for rep in c.REPRESENTATIONS}
                for method in ['all_sensors', 'imu_only']:
                    references.append(dict(person=who, file=path, method=method, prediction='a'))
            for method in ['all_sensors', 'imu_only']:
                folds.append(dict(person=who, method=method, k=3, p=1, weights='uniform'))

        class SpyModel:
            def fit(self, x, y):
                self.fit_people = set(x[:, 0])
                return self

            def predict(self, x):
                if self.fit_people.intersection(x[:, 0]):
                    raise AssertionError('Outer person entered fit')
                return np.array(['a']*len(x))

        with patch.object(study.base, 'make_model', side_effect=lambda cfg: SpyModel()) as factory:
            predictions = study.evaluate_cohort(pd.DataFrame(rows), features,
                                                 pd.DataFrame(references), pd.DataFrame(folds))
        self.assertEqual(factory.call_count, 4*len(c.REPRESENTATIONS))
        self.assertEqual(len(predictions), len(rows)*len(c.REPRESENTATIONS))
        self.assertTrue(all(r.person not in r.training_people.split('|') for r in predictions.itertuples()))


if __name__ == '__main__':
    unittest.main()
