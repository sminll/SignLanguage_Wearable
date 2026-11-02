import csv
import io
import tempfile
import unittest
import hashlib
import contextlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from dual_hand_features import PairingPolicy, build_paired_features, mode_features
from train_dual_knn import compare_cohort, main as train_main
from train_sign_knn import SENSORS, build_features
import receive_sensor as receiver
from prepare_sensor_v2 import FIELDS
from sensor_store_v3 import V3_FIELDS, V3Store


def trial(subject='subject1', label='가위', number=1, mode='left_only'):
    rows = []
    for hand in ('left', 'right'):
        for seq, t in enumerate(np.arange(0, 5000, 100), 1):
            time = float(t + (30 if hand == 'right' else 0))
            row = dict(subject_id=subject, session_id=f'{subject}_{label}_session',
                       trial_id=f'{label}{number}', label=label, hand=hand, source='REAL',
                       sample_seq=(number - 1) * 50 + seq,
                       time_ms=10000 + t if hand == 'left' else 90000 + t,
                       host_elapsed_ms=(number - 1) * 8000 + time,
                       trial_elapsed_ms=time, gesture_type=mode)
            row.update({c: float((100 if label == '가위' else 500) + t / 100) for c in SENSORS})
            rows.append(row)
    return pd.DataFrame(rows)


class PairingTests(unittest.TestCase):
    def test_independent_device_clocks_and_inactive_hand_preserved(self):
        raw = trial()
        raw.loc[raw.hand.eq('right'), SENSORS] = 7
        features, quality = build_paired_features(raw)
        self.assertEqual(len(features), 1)
        self.assertEqual(quality.iloc[0].matched_samples, 50)
        self.assertEqual(quality.iloc[0].max_pair_delta_ms, 30)
        self.assertEqual(features.iloc[0].right_fsr1_mean, 7)
        self.assertEqual(mode_features(features, 'both').shape, (1, 96))
        self.assertEqual(mode_features(features, 'left').shape, (1, 48))
        self.assertEqual(mode_features(features, 'passive').iloc[0].fsr1_mean, 7)

    def test_right_only_active_hand(self):
        raw = trial(mode='right_only')
        raw.loc[raw.hand.eq('right'), SENSORS] = 123
        features, _ = build_paired_features(raw)
        self.assertEqual(mode_features(features, 'active').iloc[0].fsr1_mean, 123)

    def test_missing_hand_not_zero_filled(self):
        raw = trial().query('hand == "left"')
        features, quality = build_paired_features(raw)
        self.assertTrue(features.empty)
        self.assertEqual(quality.iloc[0].reason, 'missing_or_invalid_hand')
        single, _ = build_features(raw)
        self.assertEqual(len(single), 1)  # old one-hand analysis remains possible

    def test_small_dropout_accepted_but_large_gap_rejected(self):
        raw = trial()
        features, _ = build_paired_features(raw.drop(raw.index[55:57]))
        self.assertEqual(len(features), 1)
        features, quality = build_paired_features(raw.drop(raw.index[60:71]))
        self.assertTrue(features.empty)
        self.assertEqual(quality.iloc[0].reason, 'large_gap')

    def test_partial_window_rejected(self):
        features, quality = build_paired_features(trial().query('trial_elapsed_ms < 1000'))
        self.assertTrue(features.empty)
        self.assertEqual(quality.iloc[0].reason, 'partial_trial_window')

    def test_device_reset_and_invalid_trial_timing_rejected(self):
        for kind, reason in [('reset', 'device_clock_reset_or_invalid'),
                             ('mismatch', 'clock_trial_mismatch')]:
            raw = trial()
            if kind == 'reset': raw.loc[30, 'time_ms'] = 0
            else: raw.loc[0, 'host_elapsed_ms'] = 8000
            features, quality = build_paired_features(raw)
            self.assertTrue(features.empty)
            self.assertEqual(quality.iloc[0].reason, reason)

    def test_timing_gap_disagreement_and_missing_imu_rejected(self):
        raw = trial()
        raw.loc[raw.hand.eq('left'), 'time_ms'] *= 2
        features, quality = build_paired_features(raw)
        self.assertTrue(features.empty)
        self.assertEqual(quality.iloc[0].reason, 'device_host_duration_mismatch')
        raw = trial()
        raw['ax'] = np.nan
        features, quality = build_paired_features(raw)
        self.assertTrue(features.empty)
        self.assertEqual(quality.iloc[0].reason, 'entire_sensor_missing')

    def test_legacy_timing_requires_explicit_opt_in(self):
        raw = trial().drop(columns=['gesture_type', 'host_elapsed_ms', 'trial_elapsed_ms'])
        features, quality = build_paired_features(raw)
        self.assertTrue(features.empty)
        self.assertEqual(quality.iloc[0].reason, 'timing_unknown')
        features, quality = build_paired_features(raw, PairingPolicy(allow_legacy_timing=True))
        self.assertEqual(len(features), 1)
        self.assertEqual(quality.iloc[0].timing, 'legacy_unverified')

    def test_conflicts_and_duplicates_rejected(self):
        for kind, reason in [('label', 'conflicting_session_metadata'),
                             ('mode', 'conflicting_session_metadata'),
                             ('duplicate', 'duplicate_sample_key')]:
            raw = trial()
            if kind == 'label': raw.loc[0, 'label'] = '다른수어'
            if kind == 'mode': raw.loc[0, 'gesture_type'] = 'both'
            if kind == 'duplicate': raw = pd.concat([raw, raw.iloc[:1]])
            features, quality = build_paired_features(raw)
            self.assertTrue(features.empty)
            self.assertEqual(quality.iloc[0].reason, reason)

    def test_fold_plan_identical_and_subjects_disjoint(self):
        raw = pd.concat([trial(s, label, n) for s in ('subject1', 'subject2')
                         for label in ('가위', '머리') for n in range(1, 5)])
        features, _ = build_paired_features(raw)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            scores, notes = compare_cohort(features, 'test', ['left', 'right', 'both'], out, 'subject')
            self.assertEqual(len(scores), 3)
            self.assertFalse(notes)
            folds = pd.read_csv(out / 'test_folds.csv')
            for _, fold in folds.groupby('fold'):
                self.assertFalse(set(fold.query('role == "train"').subject_id)
                                 & set(fold.query('role == "test"').subject_id))
            left = pd.read_csv(out / 'test_left_predictions.csv')
            both = pd.read_csv(out / 'test_both_predictions.csv')
            self.assertTrue(left[['session_id', 'trial_id', 'fold']].equals(both[['session_id', 'trial_id', 'fold']]))

    def test_no_automatic_trial_fallback_for_one_participant(self):
        features, _ = build_paired_features(pd.concat([trial(label=x, number=n)
            for x in ('가위', '머리') for n in range(1, 5)]))
        with tempfile.TemporaryDirectory() as folder:
            scores, notes = compare_cohort(features, 'test', ['left', 'both'], Path(folder), 'subject')
            self.assertFalse(scores)
            self.assertIn('participants', notes[0]['note'])

    def test_cli_generates_comparisons_without_changing_input(self):
        raw = pd.concat([trial(s, label, n) for s in ('subject1', 'subject2')
                         for label in ('가위', '머리') for n in range(1, 5)])
        with tempfile.TemporaryDirectory() as folder:
            input_path = Path(folder) / 'input.csv'
            output_path = Path(folder) / 'results'
            raw.to_csv(input_path, index=False)
            before = hashlib.sha256(input_path.read_bytes()).digest()
            with patch('sys.argv', ['train_dual_knn.py', '--input', str(input_path),
                                   '--output', str(output_path)]), contextlib.redirect_stdout(io.StringIO()):
                train_main()
            result = pd.read_csv(output_path / 'comparisons.csv')
            self.assertEqual(len(result), 6)
            self.assertEqual(set(result['mode']), {'left', 'right', 'both', 'active', 'passive'})
            self.assertEqual(hashlib.sha256(input_path.read_bytes()).digest(), before)
            with patch('sys.argv', ['train_dual_knn.py', '--input', str(input_path),
                                   '--output', str(output_path)]):
                with self.assertRaises(FileExistsError): train_main()


def payload(timed=True):
    row = dict(source='REAL', subject_id='subject1', session_id='test_session_123',
               label='가위', trial_id='가위1', hand='left', seq=1, time_ms=12345)
    row.update({c: 1.0 for c in SENSORS})
    if timed:
        row.update(gesture_type='left_only', host_elapsed_ms=100, trial_elapsed_ms=100)
    return row


class ReceiverTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.master = self.root / 'v2.csv'
        with self.master.open('w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(FIELDS)
        self.original = self.master.read_bytes()
        self.store = V3Store(self.root / 'v3.csv')
        self.patches = [patch.object(receiver, 'MASTER', self.master),
                        patch.object(receiver, 'v3_store', self.store),
                        patch.object(receiver, '_seen', {}), patch.object(receiver, '_count', 0)]
        for item in self.patches: item.start()
        self.client = receiver.app.test_client()

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        self.folder.cleanup()

    def test_new_schema_separate_and_dedup_after_reload(self):
        data = payload()
        self.assertEqual(self.client.post('/sensor', json=data).json['status'], 'saved')
        self.assertEqual(self.master.read_bytes(), self.original)
        self.assertEqual(self.client.post('/sensor', json=data).json['status'], 'duplicate')
        self.store.loaded = False
        self.assertEqual(self.client.post('/sensor', json=data).json['status'], 'duplicate')
        data['hand'] = 'right'
        self.assertEqual(self.client.post('/sensor', json=data).json['status'], 'saved')
        with self.store.path.open() as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, V3_FIELDS)
            self.assertEqual(len(list(reader)), 2)

    def test_schema_and_session_conflicts(self):
        data = payload()
        self.client.post('/sensor', json=data)
        for key, value in [('gesture_type', 'both'), ('fsr1', 2)]:
            changed = dict(data, **{key: value})
            self.assertEqual(self.client.post('/sensor', json=changed).status_code, 409)
        self.assertEqual(self.client.post('/sensor', json=payload(False)).status_code, 409)

    def test_legacy_posts_still_use_old_format(self):
        data = payload(False)
        self.assertEqual(self.client.post('/sensor', json=data).status_code, 200)
        self.assertFalse(self.store.path.exists())
        with self.master.open() as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, FIELDS)
            self.assertEqual(len(list(reader)), 1)
        self.assertEqual(self.client.post('/sensor', json=payload()).status_code, 409)

    def test_partial_or_invalid_metadata_cannot_be_silently_dropped(self):
        for changes in ({'trial_elapsed_ms': 101}, {'host_elapsed_ms': 9000},
                        {'gesture_type': 'invalid'}, {'trial_elapsed_ms': None}):
            self.assertEqual(self.client.post('/sensor', json=dict(payload(), **changes)).status_code, 400)
        data = payload()
        del data['trial_elapsed_ms']
        self.assertEqual(self.client.post('/sensor', json=data).status_code, 400)
        self.assertFalse(self.store.path.exists())
        self.assertEqual(self.master.read_bytes(), self.original)


if __name__ == '__main__':
    unittest.main()
