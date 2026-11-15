"""Synthetic fixtures test evaluation mechanics; never experimental accuracy."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from tools import evaluate_six_sign_knn as knn


def sensor_trial(label='괜찮아', right=True):
    rows = []
    for hand in ['left', 'right'] if right else ['left']:
        for i in range(50):
            row = dict(subject_id='subject-8', session_id='shared-session', trial_id='uuid',
                       label=label, hand=hand, source='REAL', board_type='XIAO',
                       record_kind='training', time_ms=i * 100 + (0 if hand == 'left' else 90000),
                       sample_seq=i)
            row.update({s: (0.0 if s.startswith('fsr') else float(i)) for s in knn.SENSORS})
            rows.append(row)
    return pd.DataFrame(rows)


def cohort():
    rows = []
    for p in range(1, 5):
        for i, label in enumerate(knn.LABELS):
            row = {c: float(p + i * 20 + j / 100) for j, c in enumerate(knn.ALL_COLS)}
            row.update(person=f'subject{p}', session_id='app-session-shared-across-people',
                       trial_id=f'{p}-{i}', file=f'{p}-{i}.csv', sha256=f'{p}-{i}',
                       label=label, date_kst='synthetic-test-only')
            rows.append(row)
    return pd.DataFrame(rows)


def three_configs():
    return [dict(config_id=i, sensors=s, k=3, p=1, weights='uniform')
            for i, s in enumerate(knn.FEATURES)]


class SixSignEvaluationTests(unittest.TestCase):
    def test_aliases_and_subject_identity(self):
        self.assertEqual(knn.person('subject-08'), 'subject8')
        for label in ['괜찮아', '괜찮다']:
            self.assertEqual(knn.extract_features(sensor_trial(label))['label'], '괜찮다')
        for label in ['아니다', '아니다 (아니, 부인(하다))']:
            self.assertEqual(knn.extract_features(sensor_trial(label))['label'], '아니다')

    def test_independent_boot_clocks_and_population_std(self):
        result = knn.extract_features(sensor_trial())
        self.assertAlmostEqual(result['left_ax_std'], np.std(np.arange(50), ddof=0))
        self.assertEqual(result['left_ax_mean'], result['right_ax_mean'])
        self.assertEqual(result['left_fsr1_max'], 0)  # Zero FSR is preserved.
        self.assertEqual([len(v) for v in knn.FEATURES.values()], [48, 48, 96])

    def test_missing_hand_never_zero_filled(self):
        with self.assertRaisesRegex(ValueError, 'Both hands'):
            knn.extract_features(sensor_trial(right=False))

    def test_invalid_sensor_or_clock_rejected(self):
        for column, value in [('ax', np.nan), ('time_ms', -1)]:
            frame = sensor_trial()
            frame.loc[20, column] = value
            with self.assertRaises(ValueError):
                knn.extract_features(frame)

    def test_shared_session_is_allowed_but_people_overlap_is_not(self):
        frame = cohort()
        train, test = frame[frame.person.ne('subject1')], frame[frame.person.eq('subject1')]
        knn.check_split(train, test)
        with self.assertRaisesRegex(ValueError, 'person'):
            knn.check_split(frame, test)
        frame.loc[6, 'trial_id'] = frame.loc[0, 'trial_id']
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            knn.validate_cohort(frame)

    def test_scaler_learns_only_fit_data(self):
        frame = cohort()
        train = frame[frame.person.ne('subject4')]
        cfg = three_configs()[0]
        cols = knn.FEATURES[cfg['sensors']]
        model = knn.make_model(cfg).fit(train[cols], train.label)
        np.testing.assert_allclose(model.steps[0][1].mean_, train[cols].mean().to_numpy())
        before = model.steps[0][1].mean_.copy()
        model.predict(frame[frame.person.eq('subject4')][cols] * 1e9)
        np.testing.assert_array_equal(before, model.steps[0][1].mean_)

    def test_inner_models_never_fit_inner_test_people(self):
        frame = cohort()
        original = knn.make_model
        checks = []

        class RecordedModel:
            def __init__(self, cfg):
                self.inner = original(cfg)

            def fit(self, x, y):
                self.people = set(frame.loc[x.index, 'person'])
                self.inner.fit(x, y)
                return self

            def predict(self, x):
                test_people = set(frame.loc[x.index, 'person'])
                assert self.people.isdisjoint(test_people)
                checks.append(test_people)
                return self.inner.predict(x)

        with patch.object(knn, 'make_model', RecordedModel):
            selected, _ = knn.select_configs(frame, three_configs())
        self.assertEqual(len(checks), 12)
        self.assertEqual(set(selected), set(knn.METHODS))

    def test_outer_selection_sees_only_training_people_and_tests_once(self):
        frame = cohort()
        seen = []

        def fake_select(train, configs):
            seen.append(set(train.person))
            board = [dict(**cfg, selection_score=0.0, inner_folds=[]) for cfg in configs]
            return {**{cfg['sensors']: cfg for cfg in board}, 'nested_selection': board[0]}, board

        with patch.object(knn, 'select_configs', fake_select):
            rows, folds, _ = knn.evaluate_nested(frame, three_configs())
        self.assertEqual(len(seen), 4)
        for i, held in enumerate(knn.people_order(frame.person)):
            self.assertNotIn(held, seen[i])
        self.assertEqual(len(folds), 16)
        self.assertEqual(rows.groupby('method').size().tolist(), [24] * 4)
        self.assertFalse(rows.duplicated(['method', 'file']).any())

    def test_fixed_search_size_and_tie_break(self):
        self.assertEqual(len(knn.configurations()), 36)
        frame = cohort()
        with patch.object(knn, 'balanced_accuracy_score', return_value=0.5):
            winners, _ = knn.select_configs(frame, list(reversed(three_configs())))
        self.assertEqual(winners['nested_selection']['config_id'], 0)

    def test_accuracy_from_actual_prediction_counts(self):
        rows = cohort()[['person', 'label']].copy()
        rows['prediction'] = rows.label
        rows.loc[0, 'prediction'] = knn.LABELS[1]
        rows['correct'] = rows.label.eq(rows.prediction)
        result = knn.result_summary(rows)
        self.assertEqual((result['correct'], result['trials']), (23, 24))
        self.assertAlmostEqual(result['accuracy'], 23 / 24)
        cm = np.array(result['confusion_matrix'])
        self.assertEqual((cm.sum(), np.trace(cm), cm[0, 1]), (24, 23, 1))


if __name__ == '__main__':
    unittest.main()
