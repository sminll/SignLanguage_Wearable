"""Synthetic mechanics checks only; these scores are never experimental results."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools import diagnose_six_sign_knn as diag
from tools import evaluate_six_sign_knn as base


def raw_trial():
    rows = []
    for hand in ['left', 'right']:
        for i in range(50):
            row = dict(subject_id='subject-1', session_id='session', trial_id='trial',
                       label=base.LABELS[0], hand=hand, source='REAL', board_type='XIAO',
                       record_kind='training', sample_seq=i, time_ms=100*i + (90000 if hand == 'right' else 0))
            row.update({s: float(i) for s in base.SENSORS})
            rows.append(row)
    return pd.DataFrame(rows)


def fixture_features():
    rows = []
    for p in range(1, 5):
        for sign, label in enumerate(base.LABELS):
            for trial in range(12):
                identity = f'{p}-{sign}-{trial}'
                row = dict(person=f'subject{p}', label=label, session_id='shared-session',
                           trial_id=identity, file=identity+'.csv', sha256=identity,
                           date_kst='synthetic', start_kst=f'2026-01-01T00:00:{trial:02d}')
                row.update({c: float(sign*20+p+trial/100) for c in base.ALL_COLS + diag.TEMPORAL_COLS})
                rows.append(row)
    return pd.DataFrame(rows)


class TemporalDiagnosticTests(unittest.TestCase):
    def test_temporal_features_preserve_order_lost_by_global_statistics(self):
        forward = raw_trial()
        reverse = forward.copy()
        for hand in ['left', 'right']:
            ids = reverse.index[reverse.hand.eq(hand)]
            reverse.loc[ids, base.SENSORS] = reverse.loc[ids, base.SENSORS].to_numpy()[::-1]
        a, b = diag.temporal_features(forward), diag.temporal_features(reverse)
        for column in base.ALL_COLS:
            self.assertAlmostEqual(a[column], b[column])
        self.assertNotEqual(a['left_bin0_ax_mean'], b['left_bin0_ax_mean'])
        self.assertEqual(a['left_bin4_ax_max'], 49)
        self.assertEqual(a['left_bin0_ax_min'], 0)

    def test_independent_clock_offsets_and_scales_do_not_change_features(self):
        frame = raw_trial()
        shifted = frame.copy()
        shifted.loc[shifted.hand.eq('right'), 'time_ms'] *= 3
        shifted.loc[shifted.hand.eq('left'), 'time_ms'] += 700000
        a, b = diag.temporal_features(frame), diag.temporal_features(shifted)
        for col in diag.TEMPORAL_COLS:
            self.assertAlmostEqual(a[col], b[col])
        self.assertEqual([len(c) for c in diag.TEMPORAL.values()], [240, 240, 480])

    def test_missing_temporal_bin_fails_instead_of_silent_fill(self):
        frame = raw_trial()
        frame = frame[~((frame.hand.eq('left')) & frame.sample_seq.between(10, 19))]
        with self.assertRaisesRegex(ValueError, 'insufficient samples'):
            diag.temporal_features(frame)

    def test_chronological_class_blocks_cover_each_whole_trial_once(self):
        frame = fixture_features()
        folds = diag.within_person_folds(frame)
        self.assertTrue(folds.between(0, 4).all())
        for (_, _), group in frame.groupby(['person', 'label']):
            values = folds.loc[group.sort_values('start_kst').index].to_numpy()
            self.assertTrue((np.diff(values) >= 0).all())
            self.assertEqual(set(values), set(range(5)))
        for who, group in frame.groupby('person'):
            for fold in range(5):
                train = group[folds.loc[group.index].ne(fold)]
                test = group[folds.loc[group.index].eq(fold)]
                diag.trial_split_check(train, test, same_person=True)
                self.assertEqual(set(test.label), set(base.LABELS))

    def test_different_protocols_enforce_different_person_rules(self):
        frame = fixture_features()
        train = frame[frame.person.ne('subject1')]
        test = frame[frame.person.eq('subject1')]
        diag.trial_split_check(train, test, same_person=False)
        with self.assertRaises(ValueError):
            diag.trial_split_check(train, test, same_person=True)
        with self.assertRaises(ValueError):
            diag.trial_split_check(frame, test, same_person=False)

    def test_temporal_inner_selection_holds_out_people(self):
        frame = fixture_features()
        original = base.make_model
        checked = []

        class Tracked:
            def __init__(self, cfg):
                self.model = original(cfg)

            def fit(self, x, y):
                self.people = set(frame.loc[x.index, 'person'])
                self.model.fit(x, y)
                return self

            def predict(self, x):
                assert self.people.isdisjoint(frame.loc[x.index, 'person'])
                checked.append(True)
                return self.model.predict(x)

        configs = [dict(config_id=i, sensors=s, k=3, p=1, weights='uniform')
                   for i, s in enumerate(diag.TEMPORAL)]
        with patch.object(base, 'make_model', Tracked):
            chosen, _ = diag.select_inner_temporal(frame, configs)
        self.assertEqual(len(checked), 12)
        self.assertEqual(set(chosen), set(base.METHODS))

    def test_temporal_outer_selection_never_receives_outer_person(self):
        frame = fixture_features()
        seen = []

        def fake_inner(train, configs):
            seen.append(set(train.person))
            choices = {s: dict(config_id=i, sensors=s, k=3, p=1, weights='uniform', selection_score=0)
                       for i, s in enumerate(diag.TEMPORAL)}
            choices['nested_selection'] = choices['imu_only']
            return choices, []

        with patch.object(diag, 'select_inner_temporal', fake_inner):
            rows, _, _ = diag.evaluate_temporal_nested(frame)
        for who, fit_people in zip(base.people_order(frame.person), seen):
            self.assertNotIn(who, fit_people)
        diag.validate_predictions(rows, frame)
        self.assertFalse(rows.duplicated(['method', 'file']).any())


if __name__ == '__main__':
    unittest.main()
