"""Synthetic mechanics tests; not included in measured recognition accuracy."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools import evaluate_three_sign_candidates as tri
from tools import evaluate_six_sign_knn as base


def fixture():
    rows = []
    for p in range(1, 7):
        labels = tri.LABELS if p <= 4 else ([tri.LABELS[1]] if p == 5 else [tri.LABELS[0], tri.LABELS[2]])
        for label in labels:
            number = tri.LABELS.index(label)
            identity = f'{p}-{number}'
            row = dict(person=f'subject{p}', session_id='shared', trial_id=identity, file=identity+'.csv',
                       label=label, sha256=identity, source_commit='synthetic-only', date_kst='synthetic-only')
            row.update({c: float(100*number+p) for c in base.ALL_COLS})
            rows.append(row)
    return pd.DataFrame(rows)


def configs():
    return [dict(config_id=i, sensors=s, k=3, p=1, weights='uniform')
            for i, s in enumerate(base.FEATURES)]


class ThreeSignCandidateTests(unittest.TestCase):
    def test_partial_people_retained_and_shared_session_allowed(self):
        frame = fixture()
        tri.validate(frame)
        self.assertEqual(tri.complete_people(frame), {'subject1','subject2','subject3','subject4'})
        tri.check_split(frame[frame.person.ne('subject5')], frame[frame.person.eq('subject5')])
        with self.assertRaisesRegex(ValueError, 'person'):
            tri.check_split(frame, frame[frame.person.eq('subject5')])

    def test_duplicate_identity_and_missing_training_class_fail(self):
        frame = fixture()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            tri.validate(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        train = frame[(frame.person.ne('subject5')) & (frame.label.ne(tri.LABELS[0]))]
        with self.assertRaisesRegex(ValueError, 'missing training labels'):
            tri.check_split(train, frame[frame.person.eq('subject5')])

    def test_inner_scoring_complete_people_only_partial_people_still_train(self):
        frame = fixture()
        real = base.make_model
        seen_fit, seen_test = [], []

        class Recorded:
            def __init__(self, config):
                self.model = real(config)

            def fit(self, x, y):
                self.people = set(frame.loc[x.index, 'person'])
                seen_fit.append(self.people)
                self.model.fit(x, y)
                return self

            def predict(self, x):
                people = set(frame.loc[x.index, 'person'])
                assert self.people.isdisjoint(people)
                seen_test.append(people)
                return self.model.predict(x)

        with patch.object(base, 'make_model', Recorded):
            chosen, board = tri.select_inner(frame, configs())
        self.assertEqual(len(seen_fit), 12)
        self.assertTrue(all({'subject5','subject6'} <= p for p in seen_fit))
        self.assertTrue(all(p <= tri.complete_people(frame) for p in seen_test))
        self.assertEqual(set(chosen), set(base.METHODS))

    def test_outer_tuning_excludes_test_person_and_all_partial_trials_evaluated(self):
        frame = fixture()
        seen = []

        def fake_inner(train, grid, labels=None):
            seen.append(set(train.person))
            chosen = {c['sensors']: dict(**c, selection_score=0) for c in configs()}
            chosen['nested_selection'] = chosen['imu_only']
            return chosen, []

        with patch.object(tri, 'select_inner', fake_inner):
            rows, folds, _ = tri.evaluate(frame, configs())
        for who, fit_people in zip(base.people_order(frame.person), seen):
            self.assertNotIn(who, fit_people)
        self.assertEqual(rows.groupby('method').size().tolist(), [15]*4)
        self.assertFalse(rows.duplicated(['method','file']).any())
        for who in ['subject5','subject6']:
            self.assertFalse(folds[folds.person.eq(who)].complete_test_person.any())

    def test_summary_does_not_invent_missing_labels(self):
        frame = fixture()
        frame['prediction'] = frame.label
        frame['correct'] = True
        result = tri.summary(frame)
        self.assertEqual((result['correct'], result['trials']), (15, 15))
        self.assertEqual(result['complete_people_subset']['trials'], 12)
        partial = next(p for p in result['people'] if p['person'] == 'subject5')
        self.assertEqual(partial['labels'], [tri.LABELS[1]])
        self.assertEqual(partial['trials'], 1)
        self.assertEqual(sum(map(sum, result['confusion_matrix'])), 15)

    def test_raw_adapter_keeps_zero_fsr_and_does_not_mutate_source(self):
        records = []
        for hand in ['left','right']:
            for i in range(50):
                row = dict(subject_id='subject-1', session_id='shared', trial_id='uuid', label='삼키다',
                           hand=hand, source='REAL', board_type='XIAO', record_kind='training',
                           time='2026-11-01T12:00:00Z', time_ms=100*i+(90000 if hand=='right' else 0),
                           sample_seq=i)
                row.update({s: 0.0 if s.startswith('fsr') else float(i) for s in base.SENSORS})
                records.append(row)
        frame = pd.DataFrame(records)
        before = frame.copy(deep=True)
        result = tri.extract(frame)
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(result['label'], tri.LABELS[0])
        self.assertEqual(result['left_fsr1_mean'], 0)
        self.assertEqual(result['left_ax_mean'], result['right_ax_mean'])
        with self.assertRaisesRegex(ValueError, 'Both hands'):
            tri.extract(frame[frame.hand.eq('left')])


if __name__ == '__main__':
    unittest.main()
