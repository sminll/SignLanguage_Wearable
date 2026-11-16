import csv
import contextlib
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import evaluate_app_kfold as evaluator


FIELDS = [
    'source', 'record_kind', 'subject_id', 'session_id', 'trial_id', 'label',
    'hand', 'fsr1', 'fsr2', 'fsr3', 'fsr4', 'fsr5', 'fsr6',
]


def recording(person='subject1', session='session-a', trial='trial-1',
              label='안녕', hands=('right', 'left'), samples=8):
    rows = []
    for hand in hands:
        for frame in range(samples):
            offset = 100 if hand == 'left' else 0
            row = dict(source='REAL', record_kind='training', subject_id=person,
                       session_id=session, trial_id=trial, label=label, hand=hand)
            row.update({f'fsr{i}': offset + frame + (i - 1) * 10
                        for i in range(1, 7)})
            rows.append(row)
    return rows


def sample_trials():
    result = []
    for person in ('subject1', 'subject2'):
        for label_index, label in enumerate(('안녕', '감사')):
            for number in range(20):
                base = label_index * 100 + number + (person == 'subject2') * 2
                result.append(evaluator.Trial(
                    person=person, session_id=f'{person}-session-{number % 2}',
                    trial_id=f'{label}-{number}', label=label,
                    right=tuple(float(base + i) for i in range(24)),
                    left=tuple(float(base + i * 2) for i in range(24)),
                ))
    return result


class GitFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')

    def git(self, *args):
        return subprocess.check_output(
            ['git', '-C', str(self.repo), *args], stderr=subprocess.STDOUT,
            text=True,
        ).strip()

    def add_recording(self, filename, rows):
        path = self.repo / 'sensor-data' / 'training' / 'account' / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def commit(self):
        self.git('add', 'sensor-data')
        self.git('-c', 'user.name=Kfold Test', '-c',
                 'user.email=kfold-test@example.invalid',
                 'commit', '-qm', 'Synthetic input fixture')
        return self.git('rev-parse', 'HEAD')


class IdentityAndFeatureTests(unittest.TestCase):
    def test_confirmed_numeric_aliases_and_invalid_identifiers(self):
        self.assertEqual(evaluator.canonical_id('subject-8'), 'subject8')
        self.assertEqual(evaluator.canonical_id('subject8'), 'subject8')
        self.assertEqual(evaluator.canonical_id('subject-10'), 'subject10')
        for value in ('', 'person1', 'subject0', 'subject-0', 'subject--1', '참가자'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluator.canonical_id(value)

    def test_feature_modes_preserve_hand_order_and_do_not_include_identity(self):
        trial = evaluator.Trial(
            person='subject1', session_id='one', trial_id='trial', label='안녕',
            right=tuple(range(24)), left=tuple(range(100, 124)),
        )
        np.testing.assert_array_equal(evaluator.feature_matrix([trial], 'right'),
                                      np.arange(24, dtype=float).reshape(1, 24))
        np.testing.assert_array_equal(evaluator.feature_matrix([trial], 'left'),
                                      np.arange(100, 124, dtype=float).reshape(1, 24))
        expected = np.r_[np.arange(24), np.arange(100, 124)].reshape(1, 48)
        np.testing.assert_array_equal(evaluator.feature_matrix([trial], 'both'), expected)


class ArchiveInputTests(GitFixture):
    def test_id_aliases_pair_hands_and_participants_sort_numerically(self):
        for number in (10, 2, 1):
            self.add_recording(f'{number}.csv', recording(person=f'subject-{number}'))
        alias_rows = recording(person='subject-8', hands=('left',))
        alias_rows += recording(person='subject8', hands=('right',))
        self.add_recording('alias.csv', alias_rows)
        self.commit()

        trials, _ = evaluator.load_trials(self.repo, 'HEAD')

        self.assertEqual([trial.person for trial in trials],
                         ['subject1', 'subject2', 'subject8', 'subject10'])
        alias = trials[2]
        np.testing.assert_allclose(alias.right[:4],
                                   [3.5, np.sqrt(5.25), 0, 7])
        np.testing.assert_allclose(alias.left[:4],
                                   [103.5, np.sqrt(5.25), 100, 107])

    def test_pair_identity_includes_person_and_session(self):
        for index, (person, session) in enumerate([
            ('subject1', 'session-a'), ('subject2', 'session-a'),
            ('subject1', 'session-b'),
        ]):
            self.add_recording(f'{index}.csv', recording(
                person=person, session=session, trial='same-trial-id'))
        self.commit()

        trials, _ = evaluator.load_trials(self.repo, 'HEAD')

        self.assertEqual(len(trials), 3)
        self.assertEqual(len({(t.person, t.session_id, t.trial_id) for t in trials}), 3)

    def test_conflicting_labels_for_same_trial_are_rejected(self):
        rows = recording(hands=('left',), label='안녕')
        rows += recording(hands=('right',), label='감사')
        self.add_recording('conflict.csv', rows)
        self.commit()

        with self.assertRaises(ValueError):
            evaluator.load_trials(self.repo, 'HEAD')

    def test_missing_hand_and_insufficient_finite_samples_are_excluded(self):
        self.add_recording('valid.csv', recording(trial='valid'))
        self.add_recording('missing.csv', recording(trial='missing', hands=('left',)))
        self.add_recording('short.csv', recording(trial='short', samples=7))
        bad = recording(trial='nonfinite')
        bad[0]['fsr1'] = 'nan'
        self.add_recording('nonfinite.csv', bad)
        self.add_recording('excluded.csv', recording(trial='test-label', label='디스코드'))
        self.commit()

        trials, _ = evaluator.load_trials(self.repo, 'HEAD')

        self.assertEqual([trial.trial_id for trial in trials], ['valid'])

    def test_loading_is_read_only_and_uses_committed_content(self):
        path = self.add_recording('valid.csv', recording())
        commit = self.commit()
        original = path.read_bytes()
        path.write_bytes(original + b'\n')
        sentinel = self.repo / 'keep.txt'
        sentinel.write_text('unrelated work', encoding='utf-8')
        before = {p.relative_to(self.repo): hashlib.sha256(p.read_bytes()).digest()
                  for p in (path, sentinel)}
        status = self.git('status', '--porcelain')

        trials, _ = evaluator.load_trials(self.repo, commit)

        self.assertEqual(len(trials), 1)
        self.assertEqual(self.git('rev-parse', 'HEAD'), commit)
        self.assertEqual(self.git('status', '--porcelain'), status)
        self.assertEqual(before, {
            p.relative_to(self.repo): hashlib.sha256(p.read_bytes()).digest()
            for p in (path, sentinel)
        })

    def test_explicit_exclusion_manifest_checks_the_recorded_hash(self):
        blocked = self.add_recording('blocked.csv', recording(trial='blocked'))
        self.add_recording('kept.csv', recording(trial='kept'))
        self.commit()
        entry = {'path': blocked.relative_to(self.repo).as_posix(),
                 'sha256': hashlib.sha256(blocked.read_bytes()).hexdigest()}
        manifest = self.root / 'exclusions.json'
        manifest.write_text(json.dumps({'files': [entry]}), encoding='utf-8')

        trials, _ = evaluator.load_trials(self.repo, 'HEAD', exclusion_manifest=manifest)

        self.assertEqual([trial.trial_id for trial in trials], ['kept'])
        entry['sha256'] = '0' * 64
        manifest.write_text(json.dumps({'files': [entry]}), encoding='utf-8')
        with self.assertRaises(ValueError):
            evaluator.load_trials(self.repo, 'HEAD', exclusion_manifest=manifest)

    def test_exclusion_manifest_rejects_paths_absent_from_pinned_commit(self):
        self.add_recording('valid.csv', recording())
        commit = self.commit()
        manifest = self.root / 'stale-exclusions.json'
        manifest.write_text(json.dumps({'files': [{
            'path': 'sensor-data/training/account/not-in-this-commit.csv',
            'sha256': '0' * 64,
        }]}), encoding='utf-8')

        with self.assertRaises(ValueError):
            evaluator.load_trials(self.repo, commit, exclusion_manifest=manifest)


class EvaluationSafetyTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'results'
            out.mkdir()
            marker = out / 'keep.txt'
            marker.write_text('existing result', encoding='utf-8')

            with self.assertRaises(FileExistsError):
                evaluator.run_evaluation(sample_trials(), out, max_folds=2,
                                         seeds=(42, 43))

            self.assertEqual(marker.read_text(encoding='utf-8'), 'existing result')

    def test_rare_class_cap_disjoint_predictions_and_train_only_scaling(self):
        # Five rare-class trials leave four in development after a 20% holdout.
        # An extreme feature makes fitting the scaler on all records observable.
        majority = [trial for trial in sample_trials() if trial.label == '안녕']
        minority = [trial for trial in sample_trials() if trial.label == '감사'][:5]
        trials = majority + minority
        trials[0] = replace(trials[0], right=(100000.0, *trials[0].right[1:]))
        lookup = {(t.person, t.session_id, t.trial_id): t for t in trials}

        def read_rows(out, filename):
            with (out / filename).open(encoding='utf-8-sig', newline='') as stream:
                return list(csv.DictReader(stream))

        def identity(row):
            return row['person'], row['session_id'], row['trial_id']

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'results'
            with contextlib.redirect_stdout(io.StringIO()):
                evaluator.run_evaluation(trials, out, max_folds=20, seeds=(42, 43))
            partitions = read_rows(out, 'split.csv')
            dev = {identity(r) for r in partitions if r['partition'] == 'development'}
            test = {identity(r) for r in partitions if r['partition'] == 'test'}
            self.assertFalse(dev & test)
            self.assertEqual(dev | test, set(lookup))
            self.assertEqual(len(partitions), len(trials))

            cv_summary = read_rows(out, 'cv_summary.csv')
            self.assertEqual({int(r['folds']) for r in cv_summary}, {2, 3, 4})
            self.assertEqual(len(cv_summary), 3 * 3)
            cv_predictions = read_rows(out, 'cv_predictions.csv')
            test_predictions = read_rows(out, 'test_predictions.csv')
            summaries = read_rows(out, 'test_summary.csv')
            for mode in ('right', 'left', 'both'):
                cv_rows = [r for r in cv_predictions if r['mode'] == mode]
                self.assertEqual(len(cv_rows), len(dev))
                self.assertEqual({identity(r) for r in cv_rows}, dev)
                self.assertEqual(len({identity(r) for r in cv_rows}), len(cv_rows))
                self.assertEqual({int(r['seed']) for r in cv_rows}, {42})
                self.assertTrue(all(r['label'] == lookup[identity(r)].label for r in cv_rows))

                held_out = [r for r in test_predictions if r['mode'] == mode]
                self.assertEqual({identity(r) for r in held_out}, test)
                self.assertEqual(len(held_out), len(test))
                correct = sum(r['label'] == r['prediction'] for r in held_out)
                summary = next(r for r in summaries if r['mode'] == mode)
                self.assertEqual(int(summary['trials']), len(test))
                self.assertEqual(int(summary['correct']), correct)
                self.assertAlmostEqual(float(summary['accuracy']), correct / len(test))

            normalization = [r for r in read_rows(out, 'normalization.csv')
                             if r['mode'] == 'both']
            development_matrix = evaluator.feature_matrix([lookup[key] for key in dev], 'both')
            all_matrix = evaluator.feature_matrix(trials, 'both')
            self.assertFalse(np.allclose(development_matrix.mean(0), all_matrix.mean(0)))
            self.assertEqual(len(normalization), 48)
            np.testing.assert_allclose(
                [float(r['mean']) for r in normalization], development_matrix.mean(0))
            np.testing.assert_allclose(
                [float(r['variance']) for r in normalization], development_matrix.var(0))

    def test_group_audit_excludes_complete_sessions_and_people(self):
        trials = []
        for person in ('subject1', 'subject2'):
            for session in ('shared-session-name-1', 'shared-session-name-2'):
                for label in ('안녕', '감사'):
                    identifier = len(trials)
                    trials.append(evaluator.Trial(
                        person, session, f'trial-{identifier}', label,
                        (float(identifier),) + (1.0,) * 23, (2.0,) * 24,
                    ))
        truth = np.array([t.label for t in trials])
        matrix = evaluator.feature_matrix(trials, 'both')
        calls = []

        class RecordingModel:
            def fit(self, values, labels):
                self.training_ids = {int(row[0]) for row in values}
                return self

            def predict(self, values):
                test_ids = {int(row[0]) for row in values}
                calls.append((self.training_ids, test_ids))
                return np.array([trials[int(row[0])].label for row in values])

        with patch.object(evaluator, 'make_model', side_effect=lambda _: RecordingModel()):
            summaries, folds = evaluator.group_audit(
                trials, matrix, truth, sorted(set(truth)), neighbors=1)

        self.assertEqual(len(calls), 6)  # Four person/session groups, then two people.
        for index, (training_ids, test_ids) in enumerate(calls):
            self.assertFalse(training_ids & test_ids)
            self.assertEqual(training_ids | test_ids, set(range(8)))
            if index < 4:
                train_groups = {(trials[i].person, trials[i].session_id) for i in training_ids}
                test_groups = {(trials[i].person, trials[i].session_id) for i in test_ids}
            else:
                train_groups = {trials[i].person for i in training_ids}
                test_groups = {trials[i].person for i in test_ids}
            self.assertFalse(train_groups & test_groups)
        for kind, expected_folds in [('session', 4), ('person', 2)]:
            summary = next(row for row in summaries if row['group_kind'] == kind)
            self.assertEqual(summary['folds'], expected_folds)
            self.assertEqual(summary['trials'], 8)
            self.assertEqual(summary['correct'], 8)
            self.assertEqual(summary['missing_label_trials'], 0)
            self.assertEqual(sum(row['trials'] for row in folds
                                 if row['group_kind'] == kind), 8)
        self.assertEqual(summaries[0]['no_same_person_gesture_train_trials'], 0)
        self.assertEqual(summaries[1]['no_same_person_gesture_train_trials'], 8)


if __name__ == '__main__':
    unittest.main()
