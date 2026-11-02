"""Compare single-hand and two-hand KNN on identical paired trials/splits."""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dual_hand_features import KEY, PairingPolicy, build_paired_features, mode_features
from train_sign_knn import ROOT


def make_model():
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                         StandardScaler(), KNeighborsClassifier(n_neighbors=3))


def comparison_splits(cohort, split):
    """Return one shared fold plan for EVERY feature mode in this cohort."""
    if cohort.label.nunique() < 2:
        return [], 'Need at least two sign labels; no classification accuracy.'
    if split == 'subject':
        if cohort.subject_id.nunique() < 2:
            return [], 'Need at least two participants for held-out-subject evaluation.'
        folds = list(LeaveOneGroupOut().split(cohort, cohort.label, groups=cohort.subject_id))
    else:
        if cohort.label.value_counts().min() < 4:
            return [], 'Need >=4 paired trials per class for exploratory trial split.'
        train, test = train_test_split(np.arange(len(cohort)), test_size=0.25,
                                       random_state=42, stratify=cohort.label)
        folds = [(train, test)]
    if any(len(train) < 3 or set(cohort.iloc[test].label) - set(cohort.iloc[train].label)
           for train, test in folds):
        return [], 'A fold has too little training data or a class absent from training; no aggregate score.'
    return folds, ''


def compare_cohort(cohort, name, modes, out, split):
    cohort = cohort.reset_index(drop=True)
    if cohort.empty:
        return [], [{'cohort': name, 'note': 'No eligible trials.'}]
    folds, note = comparison_splits(cohort, split)
    classes = sorted(cohort.label.unique())
    scores, notes = [], [{'cohort': name, 'note': note}] if note else []
    fold_records = []
    for number, (train, test) in enumerate(folds):
        for role, indices in [('train', train), ('test', test)]:
            selected = cohort.iloc[indices][KEY + ['label']].copy()
            selected['fold'], selected['role'] = number, role
            fold_records.extend(selected.to_dict('records'))
    pd.DataFrame(fold_records, columns=KEY + ['label', 'fold', 'role']).to_csv(
        out / f'{name}_folds.csv', index=False, encoding='utf-8-sig')
    for mode in modes:
        X = mode_features(cohort, mode)
        predictions = []
        for number, (train, test) in enumerate(folds):
            model = make_model()
            model.fit(X.iloc[train], cohort.iloc[train].label)
            pred = model.predict(X.iloc[test])
            rows = cohort.iloc[test][KEY + ['label']].copy()
            rows['predicted_label'], rows['fold'] = pred, number
            predictions.append(rows)
        if predictions:
            prediction = pd.concat(predictions, ignore_index=True)
            prediction.to_csv(out / f'{name}_{mode}_predictions.csv', index=False, encoding='utf-8-sig')
            matrix = confusion_matrix(prediction.label, prediction.predicted_label, labels=classes)
            pd.DataFrame(matrix, index=classes, columns=classes).to_csv(
                out / f'{name}_{mode}_confusion.csv', encoding='utf-8-sig')
            scores.append({'cohort': name, 'mode': mode, 'split': split,
                           'trials': len(cohort), 'evaluation_predictions': len(prediction),
                           'folds': len(folds), 'features': X.shape[1],
                           'accuracy': accuracy_score(prediction.label, prediction.predicted_label)})
        if cohort.label.nunique() >= 2 and len(cohort) >= 3:
            candidate = make_model()
            candidate.fit(X, cohort.label)
            joblib.dump({'pipeline': candidate, 'columns': list(X.columns), 'classes': classes,
                         'mode': mode, 'cohort': name, 'schema': 'paired_v3',
                         'requires_known_gesture_type': mode in ('active','passive'),
                         'training_trials': len(cohort), 'requires_both_hands_for_quality_check': True,
                         'evaluation_note': 'Full-data candidate; scores in comparisons.csv are from separate CV fold models.'},
                        out / f'{name}_{mode}_model.joblib')
    return scores, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'data/sensor_dataset_v3.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--split', choices=['subject', 'trial'], default='subject')
    parser.add_argument('--date', help='Server receipt date YYYY-MM-DD')
    parser.add_argument('--tolerance-ms', type=float, default=150)
    parser.add_argument('--max-gap-ms', type=float, default=500)
    parser.add_argument('--min-span-ms', type=float, default=3000)
    parser.add_argument('--min-match-fraction', type=float, default=0.7)
    parser.add_argument('--allow-legacy-timing', action='store_true')
    args = parser.parse_args()
    if args.date:
        datetime.strptime(args.date, '%Y-%m-%d')
    data = args.input.read_bytes()
    raw = pd.read_csv(io.BytesIO(data), encoding='utf-8-sig', low_memory=False)
    if args.date:
        raw = raw.loc[raw.time.astype('string').str.startswith(args.date, na=False)].copy()
    policy = PairingPolicy(tolerance_ms=args.tolerance_ms, max_gap_ms=args.max_gap_ms,
                           min_span_ms=args.min_span_ms, min_match_fraction=args.min_match_fraction,
                           allow_legacy_timing=args.allow_legacy_timing)
    features, quality = build_paired_features(raw, policy)
    out = args.output or ROOT / 'data/knn_results' / ('paired_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    features.to_csv(out / 'paired_features.csv', index=False, encoding='utf-8-sig')
    quality.to_csv(out / 'pair_quality.csv', index=False, encoding='utf-8-sig')
    scores, notes = [], []
    if not features.empty:
        # One pair is one sample. The opposite hand never goes into a separate fold.
        for name, cohort, modes in [
            ('all_paired', features, ['left', 'right', 'both']),
            ('one_hand_signs', features.loc[features.gesture_type.isin(['left_only', 'right_only'])],
             ['active', 'passive', 'both'])]:
            results, warnings = compare_cohort(cohort, name, modes, out, args.split)
            scores.extend(results)
            notes.extend(warnings)
    else:
        notes.append({'note': 'No quality-approved left/right pairs; no model or accuracy created.'})
    pd.DataFrame(scores, columns=['cohort', 'mode', 'split', 'trials', 'evaluation_predictions', 'folds', 'features', 'accuracy']).to_csv(
        out / 'comparisons.csv', index=False, encoding='utf-8-sig')
    (out / 'manifest.json').write_text(json.dumps({
        'input': str(args.input), 'input_sha256': hashlib.sha256(data).hexdigest(),
        'policy': asdict(policy), 'split': args.split, 'date_filter': args.date,
        'paired_trials': len(features), 'quality_groups': len(quality), 'notes': notes,
        'caveat': 'Trial split is exploratory, not new-person accuracy. Legacy timing is unverified. Gesture type is not a prediction feature, but active/passive baselines assume the active hand is known beforehand.'},
        ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Accepted paired trials: {len(features)} / {len(quality)}')
    print(pd.DataFrame(scores).to_string(index=False) if scores else notes)
    print('Results:', out)


if __name__ == '__main__':
    main()
