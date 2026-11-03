"""Evaluate a frozen model on a selected participant, then fit a separate model.

Only the explicitly listed sessions are used. Raw files and the baseline
model are read-only. Re-running requires a new --output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train_sign_knn import ROOT, INPUT, IDENTITY, build_features

OLD_SESSIONS = {
    's20261101_182103_064_126986817': '갑자기',
    's20261101_183011_302_913375856': '머리',
    's20261101_183906_217_632359246': '아프다',
}
NEW_SESSIONS = {
    's20261102_170426_237_905791937': '아프다',
    's20261102_171310_095_913375856': '머리',
    's20261102_171939_434_632359246': '갑자기',
}
BASELINE = ROOT / 'data/knn_results/2026-11-01_XIAO/knn_model.joblib'
SUBJECT9_SESSIONS = {
    's20261102_172935_951_278498219': '아프다',
    's20261102_173437_914_546881519': '머리',
    's20261102_174004_674_957506835': '갑자기',
}
SUBJECT2_SESSIONS = {
    's20261102_180121_657_485375649': '아프다',
    's20261102_181920_142_800280469': '머리',
    's20261102_182501_457_141886339': '갑자기',
}
SUBJECT2_RETAKE_SESSIONS = {
    's20261102_183549_659_421761283': '아프다',
    's20261102_184120_876_915735525': '머리',
    's20261102_184858_984_792207329': '갑자기',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def select_sessions(raw, sessions, subject):
    selected = raw.loc[raw['session_id'].isin(sessions)].copy()
    if set(selected['session_id']) != set(sessions):
        raise ValueError('Requested sessions are missing.')
    if not selected['subject_id'].eq(subject).all() or not selected['hand'].eq('left').all():
        raise ValueError('Unexpected participant or hand.')
    if not selected['source'].eq('REAL').all():
        raise ValueError('Non-real data in selected sessions.')
    if selected.duplicated(['session_id', 'hand', 'sample_seq']).any():
        raise ValueError('Duplicate sample keys.')
    rest = selected.loc[selected['label'].eq('rest')].copy()
    actions = selected.loc[~selected['label'].eq('rest')].copy()
    for session, label in sessions.items():
        block = actions.loc[actions['session_id'].eq(session)]
        if set(block['label']) != {label}:
            raise ValueError('Unexpected action labels.')
        if set(block['trial_id']) != {f'{label}{k}' for k in range(1, 21)}:
            raise ValueError('Expected 20 action trials per session.')
        full = selected.loc[selected['session_id'].eq(session)]
        seq = sorted(pd.to_numeric(full['sample_seq']).astype(int))
        if seq != list(range(1, len(seq) + 1)):
            raise ValueError('Missing or invalid sample sequence.')
    features, rejected = build_features(actions)
    if len(features) != 60 or not rejected.empty:
        raise ValueError('Expected 60 valid action trials without short trials.')
    return actions, rest, features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', choices=['subject4', 'subject2', 'subject2_retake', 'subject9'], default='subject4')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.subject == 'subject9':
        raise ValueError('subject9는 사용자 요청으로 데이터와 모델을 삭제하고 학습에서 제외했습니다.')
    if args.subject == 'subject2':
        raise ValueError('오늘 subject2 세 세션은 사용자 요청으로 삭제·학습 제외했습니다. 이전 ESP32 데이터는 보존돼 있습니다.')
    subject = 'subject2' if args.subject == 'subject2_retake' else args.subject
    test_sessions = NEW_SESSIONS if subject == 'subject4' else SUBJECT2_RETAKE_SESSIONS
    baseline_path = BASELINE if subject == 'subject4' else (
        ROOT / 'data/knn_results/2026-11-02_subject1_to_subject4/combined_model/knn_model.joblib')
    expected_baseline_trials = 45 if subject == 'subject4' else 120
    old_sessions = dict(OLD_SESSIONS)
    baseline_subjects = ['subject1']
    if subject != 'subject4':
        old_sessions.update(NEW_SESSIONS)
        baseline_subjects.append('subject4')
    if args.output is None:
        name = '2026-11-02_subject1_to_subject4' if subject == 'subject4' else f'2026-11-02_subject1_subject4_to_{args.subject}'
        args.output = ROOT / 'data/knn_results' / name
    if args.output.exists():
        raise ValueError('Output already exists; choose a fresh --output directory.')
    raw_bytes = INPUT.read_bytes()
    raw = pd.read_csv(io.BytesIO(raw_bytes), encoding='utf-8-sig', low_memory=False)
    old_actions, _, old_features = select_sessions(raw, OLD_SESSIONS, 'subject1')
    if subject != 'subject4':
        previous_actions, _, previous_features = select_sessions(raw, NEW_SESSIONS, 'subject4')
        old_actions = pd.concat([old_actions, previous_actions], ignore_index=True)
        old_features = pd.concat([old_features, previous_features], ignore_index=True)
    actions, rest, features = select_sessions(raw, test_sessions, subject)
    baseline_bytes = baseline_path.read_bytes()
    baseline_hash = sha(baseline_bytes)
    bundle = joblib.load(io.BytesIO(baseline_bytes))
    pipeline = bundle['pipeline']
    columns = bundle['columns']
    classes = list(bundle['classes'])
    if set(classes) != set(test_sessions.values()):
        raise ValueError('Baseline classes do not match requested actions.')
    baseline_n = pipeline.named_steps['knn'].n_samples_fit_
    if baseline_n != expected_baseline_trials:
        raise ValueError('Unexpected baseline training size.')
    if subject != 'subject4' and set(bundle['training_sessions']) != set(old_sessions):
        raise ValueError('Baseline training sessions mismatch; refusing possible leakage.')
    if set(old_sessions) & set(test_sessions):
        raise ValueError('Train/test session overlap.')
    # Prediction only: no fit, transform fitting, or parameter changes here.
    truth = features['label'].astype(str)
    predicted = pipeline.predict(features[columns])
    accuracy = accuracy_score(truth, predicted)
    correct = int(np.sum(truth.to_numpy() == predicted))
    report = classification_report(truth, predicted, labels=classes, zero_division=0)
    matrix = confusion_matrix(truth, predicted, labels=classes)
    if sha(baseline_path.read_bytes()) != baseline_hash:
        raise ValueError('Baseline changed during evaluation.')

    evaluation_dir = args.output / 'external_evaluation'
    combined_dir = args.output / 'combined_model'
    evaluation_dir.mkdir(parents=True)
    combined_dir.mkdir()
    actions.to_csv(evaluation_dir / 'action_samples.csv', index=False, encoding='utf-8-sig')
    rest.to_csv(evaluation_dir / 'rest_samples_excluded.csv', index=False, encoding='utf-8-sig')
    features.to_csv(evaluation_dir / 'trial_features.csv', index=False, encoding='utf-8-sig')
    predictions = features[IDENTITY + ['n_frames']].copy()
    predictions['predicted_label'] = predicted
    predictions['correct'] = truth.to_numpy() == predicted
    predictions.to_csv(evaluation_dir / 'predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
        evaluation_dir / 'confusion_matrix.csv', encoding='utf-8-sig')
    evaluation = (
        '고정 모델 외부 평가 / Frozen baseline external evaluation\n'
        f'Baseline: {baseline_path.relative_to(ROOT)}\n'
        f'Baseline SHA256: {baseline_hash}\n'
        f'Baseline training: {baseline_subjects}, {baseline_n} trials\n'
        f'Evaluation: {subject}, 2026-11-02, {len(features)} action trials\n'
        f'Action samples: {len(actions)}; rest samples excluded: {len(rest)}\n'
        f'Accuracy: {correct}/{len(features)} = {accuracy:.2%}\n\n{report}\n'
        'New-participant/new-day check on ONE participant, not population accuracy.\n'
        'Rest rejection and full-word recognition were not evaluated.\n'
        'These trials become training data AFTER this frozen evaluation; they are not\n'
        'an independent test set for the combined model.\n'
    )
    (evaluation_dir / 'evaluation.txt').write_text(evaluation, encoding='utf-8')
    manifest = {
        'input_snapshot_sha256': sha(raw_bytes), 'baseline_sha256': baseline_hash,
        'baseline_training_trials': int(baseline_n),
        'training_sessions_old': old_sessions, 'evaluation_sessions': test_sessions,
        'evaluation_actions': len(actions), 'rest_excluded': len(rest),
        'test_trials': len(features), 'correct': correct, 'accuracy': accuracy,
        'baseline_predictions_before_combined_fit': True,
    }
    (evaluation_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # Only now fit a fresh pipeline on all selected action trials. No evaluation
    # metric is reported for the combined candidate without independent data.
    combined = pd.concat([old_features, features], ignore_index=True)
    expected_combined_trials = len(old_features) + 60
    if len(combined) != expected_combined_trials or combined.duplicated(IDENTITY).any():
        raise ValueError('Unexpected combined features.')
    candidate = clone(pipeline)
    candidate.fit(combined[columns], combined['label'].astype(str))
    joblib.dump({'pipeline': candidate, 'columns': columns, 'classes': classes,
                 'training_sessions': [*old_sessions, *test_sessions],
                 'training_trials': expected_combined_trials, 'independent_evaluation': None,
                 'baseline_sha256': baseline_hash}, combined_dir / 'knn_model.joblib')
    combined.to_csv(combined_dir / 'trial_features.csv', index=False, encoding='utf-8-sig')
    combined.groupby(['subject_id', 'label']).size().unstack(fill_value=0).to_csv(
        combined_dir / 'subject_label_counts.csv', encoding='utf-8-sig')
    (combined_dir / 'README.md').write_text(
        '# Combined model candidate\n\n'
        f'Training participants: {baseline_subjects + [subject]}; {expected_combined_trials} action trials; k=3.\n'
        'Fresh imputer/scaler/KNN fitted on all selected action trials. No rest class.\n'
        'Not independently evaluated; do not use the old 93.3% or frozen-baseline\n'
        'external score as this model\'s accuracy. Baseline remains unchanged.\n', encoding='utf-8')
    assert sha(baseline_path.read_bytes()) == baseline_hash
    assert candidate.named_steps['knn'].n_samples_fit_ == expected_combined_trials
    assert len(old_actions) == (2978 if subject == 'subject4' else 5959)
    print(evaluation)
    print(f'Combined candidate saved: {expected_combined_trials} trials, {len(baseline_subjects)+1} participants; independent accuracy NOT evaluated.')
    print('Output:', args.output)


if __name__ == '__main__':
    main()
