"""Read-only, trial-level FSR K-fold evaluation of a pinned training-data commit.

This evaluates already-recorded participants, not unseen-user generalization.
No production model is saved or replaced. See docs/kfold-validation.md.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


DEFAULT_EXCLUDED_LABELS = frozenset({'좆까', '디스코드', '디스코드 테스트'})
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
MODES = ('right', 'left', 'both')
STATS = ('mean', 'std', 'min', 'max')
FEATURES = tuple(f'{hand}_fsr{i}_{stat}' for hand in ('right', 'left')
                 for i in range(1, 7) for stat in STATS)
METRICS = ('trials', 'correct', 'accuracy', 'macro_f1', 'balanced_accuracy')
IDENTITY = ('person', 'session_id', 'trial_id', 'label')


@dataclass(frozen=True)
class Trial:
    person: str
    session_id: str
    trial_id: str
    label: str
    right: tuple[float, ...]
    left: tuple[float, ...]

    def identity(self):
        return {key: getattr(self, key) for key in IDENTITY}


def canonical_id(value):
    match = re.fullmatch(r'subject-?([0-9]+)', str(value).strip(), re.I)
    if not match or int(match[1]) < 1:
        raise ValueError(f'Not a positive numeric subject ID: {value!r}')
    return f'subject{int(match[1])}'


def load_trials(repo, ref, min_samples=8, excluded_labels=DEFAULT_EXCLUDED_LABELS,
                exclusion_manifest=None):
    """Pair hands by canonical person/session/trial; aggregate finite FSR values.

    Default scope intentionally includes all cohorts, matching the exploratory
    all-data comparison. Operational training exclusions are only applied when
    their path/hash manifest is explicitly supplied.
    """
    if min_samples < 2:
        raise ValueError('min_samples must be at least 2')
    repo = Path(repo)
    commit = subprocess.check_output(
        ['git', '-C', str(repo), 'rev-parse', '--verify', '--end-of-options',
         f'{ref}^{{commit}}'], text=True).strip()
    archive = subprocess.check_output(
        ['git', '-C', str(repo), 'archive', commit, 'sensor-data/training'])
    blocked = {}
    if exclusion_manifest is not None:
        entries = json.loads(Path(exclusion_manifest).read_text(encoding='utf-8'))['files']
        blocked = {entry['path']: entry['sha256'] for entry in entries}
    groups, identities = {}, {}
    files, rejected, skipped = [], [], Counter()
    required = {'subject_id', 'session_id', 'trial_id', 'label', 'hand', 'source',
                *(f'fsr{i}' for i in range(1, 7))}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith('.csv'):
                continue
            blob = tar.extractfile(member).read()
            digest = hashlib.sha256(blob).hexdigest()
            excluded = member.name in blocked
            if excluded and digest != blocked[member.name]:
                raise ValueError(f'Exclusion hash mismatch: {member.name}')
            files.append(dict(path=member.name, sha256=digest, excluded=excluded))
            if excluded:
                skipped['manifest_files'] += 1
                continue
            reader = csv.DictReader(io.StringIO(blob.decode('utf-8-sig')))
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f'Missing CSV columns: {member.name}')
            for row in reader:
                if (row['source'] or '').strip().upper() != 'REAL':
                    skipped['non_real_rows'] += 1
                    continue
                if (row.get('record_kind', 'training') or '').strip() != 'training':
                    skipped['non_training_rows'] += 1
                    continue
                label = (row['label'] or '').strip()
                if label in excluded_labels:
                    skipped['excluded_label_rows'] += 1
                    continue
                try:
                    person = canonical_id(row['subject_id'])
                except ValueError:
                    skipped['invalid_subject_rows'] += 1
                    continue
                session, trial, hand = ((row[key] or '').strip() for key in
                                        ('session_id', 'trial_id', 'hand'))
                if not session or not trial or not label or hand not in ('right', 'left'):
                    skipped['invalid_identity_rows'] += 1
                    continue
                identity = (person, session, trial)
                if identity in identities and identities[identity] != label:
                    raise ValueError(f'Conflicting labels for trial {identity}')
                identities[identity] = label
                key = (*identity, label)
                if key not in groups:
                    groups[key] = {h: [[] for _ in range(6)] for h in ('right', 'left')}
                for i in range(6):
                    try:
                        value = float(row[f'fsr{i+1}'])
                    except (TypeError, ValueError):
                        value = np.nan
                    if np.isfinite(value):
                        groups[key][hand][i].append(value)
                    else:
                        skipped['nonfinite_sensor_values'] += 1
    unmatched = set(blocked) - {entry['path'] for entry in files}
    if unmatched:
        raise ValueError(f'Exclusion paths absent from source commit: {sorted(unmatched)}')
    trials = []
    for key in sorted(groups, key=lambda k: (int(k[0][7:]), *k[1:])):
        hands = groups[key]
        if any(len(values) < min_samples for sensors in hands.values() for values in sensors):
            rejected.append(dict(zip(IDENTITY, key), reason='incomplete_hand_or_fsr'))
            continue
        features = {}
        for hand, sensors in hands.items():
            features[hand] = tuple(float(value) for values in sensors for value in
                                   (np.mean(values), np.std(values, ddof=0),
                                    np.min(values), np.max(values)))
        trials.append(Trial(*key, features['right'], features['left']))
    if not trials:
        raise ValueError('No complete paired FSR trials')
    return trials, dict(commit=commit, ref=ref, files=files, rejected=rejected,
                        skipped=dict(skipped), min_samples=min_samples,
                        excluded_labels=sorted(excluded_labels),
                        exclusion_manifest=str(exclusion_manifest) if exclusion_manifest else None,
                        excluded_files=sum(entry['excluded'] for entry in files),
                        operational_exclusions_applied=bool(blocked))


def feature_matrix(trials, mode):
    if mode not in MODES:
        raise ValueError(f'Unknown input mode: {mode}')
    values = [(t.right if mode == 'right' else t.left if mode == 'left'
               else t.right + t.left) for t in trials]
    result = np.asarray(values, dtype=float)
    if result.shape != (len(trials), 48 if mode == 'both' else 24) or not np.isfinite(result).all():
        raise ValueError('Expected finite 24-feature hands / 48-feature paired trials')
    return result


def make_model(neighbors=3, normalized=True):
    steps = [('scaler', StandardScaler())] if normalized else []
    return Pipeline(steps + [('knn', KNeighborsClassifier(
        n_neighbors=neighbors, weights='uniform', metric='euclidean',
        algorithm='brute', n_jobs=1))])


def metrics(truth, prediction, labels):
    cm = confusion_matrix(truth, prediction, labels=labels)
    supported = cm.sum(axis=1) > 0
    recalls = np.diag(cm)[supported] / cm.sum(axis=1)[supported]
    return dict(trials=len(truth), correct=int(np.trace(cm)),
                accuracy=float(np.mean(truth == prediction)),
                macro_f1=float(f1_score(truth, prediction, labels=labels,
                                       average='macro', zero_division=0)),
                balanced_accuracy=float(np.mean(recalls)))


def write_csv(path, fields, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_cv(X, y, labels, mode, folds, seeds, neighbors):
    runs, fold_rows, first_predictions = [], [], None
    for seed in seeds:
        predicted = np.empty(len(y), dtype=object)
        seen = np.zeros(len(y), dtype=int)
        fold_ids = np.zeros(len(y), dtype=int)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train, validation) in enumerate(splitter.split(X, y), 1):
            if len(train) < neighbors:
                raise ValueError('Not enough training trials for the fixed KNN neighbor count')
            assert set(train).isdisjoint(validation)
            model = make_model(neighbors).fit(X[train], y[train])
            predicted[validation] = model.predict(X[validation])
            assert np.allclose(model.named_steps['scaler'].mean_, X[train].mean(axis=0))
            seen[validation] += 1
            fold_ids[validation] = fold
            fold_rows.append(dict(mode=mode, folds=folds, seed=seed, fold=fold,
                                  train_trials=len(train),
                                  **metrics(y[validation], predicted[validation], labels)))
        assert np.all(seen == 1)
        runs.append(dict(mode=mode, folds=folds, seed=seed, **metrics(y, predicted, labels)))
        if first_predictions is None:
            first_predictions = (predicted.copy(), fold_ids.copy())
    accuracies = [r['accuracy'] for r in runs]
    summary = dict(mode=mode, folds=folds, repeats=len(seeds),
                   **{key: float(np.mean([r[key] for r in runs]))
                      for key in ('accuracy', 'macro_f1', 'balanced_accuracy')},
                   repeat_accuracy_std=float(np.std(accuracies, ddof=1)) if len(seeds) > 1 else 0.0,
                   fold_accuracy_std=float(np.std([r['accuracy'] for r in fold_rows], ddof=1)),
                   min_accuracy=min(accuracies), max_accuracy=max(accuracies))
    return summary, runs, fold_rows, first_predictions


def group_audit(trials, X, y, labels, neighbors):
    """Optional whole-corpus audits; these are not the fixed test-set comparison."""
    rows, summaries = [], []
    people = np.array([t.person for t in trials])
    for kind in ('session', 'person'):
        groups = np.array([f'{t.person}|{t.session_id}' if kind == 'session'
                           else t.person for t in trials])
        if len(set(groups)) < 2:
            continue
        predicted = np.empty(len(y), object)
        seen = np.zeros(len(y), int)
        missing_total = 0
        no_same_person_gesture = 0
        for fold, (train, test) in enumerate(LeaveOneGroupOut().split(X, y, groups), 1):
            assert set(groups[train]).isdisjoint(groups[test])
            if kind == 'person':
                assert set(people[train]).isdisjoint(people[test])
            if len(train) < neighbors:
                raise ValueError('Group audit has too few training trials')
            model = make_model(neighbors).fit(X[train], y[train])
            predicted[test] = model.predict(X[test])
            seen[test] += 1
            missing = set(y[test]) - set(y[train])
            absent = int(sum(label in missing for label in y[test]))
            missing_total += absent
            train_pairs = {(people[i], y[i]) for i in train}
            uncovered = sum((people[i], y[i]) not in train_pairs for i in test)
            no_same_person_gesture += uncovered
            rows.append(dict(group_kind=kind, fold=fold, group=groups[test[0]],
                             train_trials=len(train), missing_label_trials=absent,
                             no_same_person_gesture_train_trials=uncovered,
                             **metrics(y[test], predicted[test], labels)))
        assert np.all(seen == 1)
        summaries.append(dict(group_kind=kind, folds=len(set(groups)),
                              missing_label_trials=missing_total,
                              no_same_person_gesture_train_trials=no_same_person_gesture,
                              **metrics(y, predicted, labels)))
    return summaries, rows


def run_evaluation(trials, out, max_folds=20, seeds=DEFAULT_SEEDS, test_size=.2,
                   split_seed=20261116, neighbors=3, audit_groups=False, source=None):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a fresh output directory: {out}')
    if max_folds < 2 or neighbors < 1 or not 0 < test_size < 1:
        raise ValueError('Require max_folds>=2, neighbors>=1 and 0<test_size<1')
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide nonempty, distinct repetition seeds')
    keys = [(t.person, t.session_id, t.trial_id) for t in trials]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate person/session/trial identities')
    Xall = feature_matrix(trials, 'both')
    y = np.array([t.label for t in trials])
    labels = sorted(set(y))
    if len(labels) < 2 or min(Counter(y).values()) < 2:
        raise ValueError('Require at least two classes and two trials per class')
    development, test = train_test_split(np.arange(len(trials)), test_size=test_size,
                                        stratify=y, random_state=split_seed)
    assert set(development).isdisjoint(test)
    fold_limit = min(max_folds, min(Counter(y[development]).values()))
    if fold_limit < 2:
        raise ValueError('Too few development trials in the rarest class for K-fold')
    folds_to_try = list(range(2, fold_limit + 1))
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out/'trial_features.csv', (*IDENTITY, *FEATURES),
              [dict(t.identity(), **dict(zip(FEATURES, t.right + t.left))) for t in trials])
    test_set = set(test)
    write_csv(out/'split.csv', ('partition', *IDENTITY),
              [dict(partition='test' if i in test_set else 'development', **t.identity())
               for i, t in enumerate(trials)])
    scores, runs, folds, selected, selected_predictions = [], [], [], {}, {}
    rank = lambda row: (row['accuracy'], row['macro_f1'], -row['folds'])
    with threadpool_limits(limits=1):
        for mode in MODES:
            X = feature_matrix(trials, mode)[development]
            for count in folds_to_try:
                result, repeated, detail, predictions = evaluate_cv(
                    X, y[development], labels, mode, count, seeds, neighbors)
                scores.append(result)
                runs.extend(repeated)
                folds.extend(detail)
                if mode not in selected or rank(result) > rank(selected[mode]):
                    selected[mode] = result
                    selected_predictions[mode] = predictions
        recommended = max(MODES, key=lambda mode: rank(selected[mode]))
        # Freeze the choice before scoring the reserved test partition.
        (out/'selection.json').write_text(json.dumps(
            dict(selected=selected, recommended=recommended,
                 rule='mean OOF accuracy, then Macro F1, then smaller fold count'),
            ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        final, test_predictions, oof, scalers, raw_baselines = [], [], [], [], []
        for mode in MODES:
            X = feature_matrix(trials, mode)
            model = make_model(neighbors).fit(X[development], y[development])
            prediction = model.predict(X[test])
            final.append(dict(mode=mode, selected_folds=selected[mode]['folds'],
                              train_trials=len(development),
                              cv_accuracy=selected[mode]['accuracy'],
                              cv_macro_f1=selected[mode]['macro_f1'],
                              **metrics(y[test], prediction, labels)))
            for i, index in enumerate(test):
                test_predictions.append(dict(mode=mode, **trials[index].identity(),
                                             prediction=prediction[i],
                                             correct=int(prediction[i] == y[index])))
            pp, ff = selected_predictions[mode]
            for i, index in enumerate(development):
                oof.append(dict(mode=mode, folds=selected[mode]['folds'], seed=seeds[0],
                                fold=int(ff[i]), **trials[index].identity(),
                                prediction=pp[i], correct=int(pp[i] == y[index])))
            columns = FEATURES[:24] if mode == 'right' else FEATURES[24:] if mode == 'left' else FEATURES
            scaler = model.named_steps['scaler']
            assert np.allclose(scaler.mean_, X[development].mean(axis=0))
            for i, feature in enumerate(columns):
                scalers.append(dict(mode=mode, feature=feature, mean=scaler.mean_[i],
                                    variance=scaler.var_[i], std=float(np.sqrt(scaler.var_[i])),
                                    scale=scaler.scale_[i]))
            # Same preselected splits, without scaling; not used for selection.
            accuracies = []
            for seed in seeds:
                p = np.empty(len(development), object)
                splitter = StratifiedKFold(n_splits=selected[mode]['folds'],
                                          shuffle=True, random_state=seed)
                for tr, va in splitter.split(X[development], y[development]):
                    raw = make_model(neighbors, normalized=False).fit(
                        X[development][tr], y[development][tr])
                    p[va] = raw.predict(X[development][va])
                accuracies.append(float(np.mean(p == y[development])))
            raw_baselines.append(dict(mode=mode, folds=selected[mode]['folds'],
                                      raw_accuracy=float(np.mean(accuracies)),
                                      scaled_accuracy=selected[mode]['accuracy']))
        audits, audit_folds = group_audit(trials, Xall, y, labels, neighbors) if audit_groups else ([], [])
    write_csv(out/'cv_summary.csv', tuple(scores[0]), scores)
    write_csv(out/'cv_runs.csv', tuple(runs[0]), runs)
    write_csv(out/'cv_folds.csv', tuple(folds[0]), folds)
    write_csv(out/'cv_predictions.csv', ('mode', 'folds', 'seed', 'fold', *IDENTITY,
                                        'prediction', 'correct'), oof)
    write_csv(out/'test_summary.csv', tuple(final[0]), final)
    write_csv(out/'test_predictions.csv', ('mode', *IDENTITY, 'prediction', 'correct'), test_predictions)
    write_csv(out/'normalization.csv', ('mode', 'feature', 'mean', 'variance', 'std', 'scale'), scalers)
    write_csv(out/'normalization_comparison.csv', tuple(raw_baselines[0]), raw_baselines)
    if audits:
        write_csv(out/'group_audits.csv', tuple(audits[0]), audits)
        write_csv(out/'group_folds.csv', tuple(audit_folds[0]), audit_folds)
    # For inspection only: these are final-development scalers, not the CV scalers.
    scaler = StandardScaler().fit(Xall[development])
    normalized = scaler.transform(Xall)
    write_csv(out/'normalized_features.csv', ('partition', *IDENTITY, *FEATURES),
              [dict(partition='test' if i in test_set else 'development', **t.identity(),
                    **dict(zip(FEATURES, normalized[i]))) for i, t in enumerate(trials)])
    train_sessions = {(trials[i].person, trials[i].session_id) for i in development}
    shared = sum((trials[i].person, trials[i].session_id) in train_sessions for i in test)
    manifest = dict(source=source or {}, trials=len(trials), participants=len({t.person for t in trials}),
                    labels=labels, development_trials=len(development), test_trials=len(test),
                    test_size=test_size, split_seed=split_seed, seeds=list(seeds),
                    folds=folds_to_try, neighbors=neighbors, distance='euclidean',
                    normalization='StandardScaler fitted on each training partition only',
                    selected=selected, recommended=recommended, test=final, group_audits=audits,
                    shared_session_test_trials=shared, sklearn_version=sklearn.__version__,
                    numpy_version=np.__version__, models_saved=False,
                    limitations=[
                        'Trial-level splits can share participants and sessions; not unseen-user accuracy.',
                        'The reserved set is internal and may have appeared in earlier analyses.',
                        'The highest fold score is exploratory; fold count is not a KNN parameter.',
                        'Session audits also change person-by-gesture coverage, not only wearing conditions.',
                        'normalized_features.csv uses final-development scaling; do not pre-scale CV with it.',
                        'Fewer than two repeats gives repeat_accuracy_std=0, not evidence of stability.',
                        'This all-data scope is distinct from operational cohort/placement exclusions.',
                    ])
    (out/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ref', default='origin/training-data')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-folds', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--test-size', type=float, default=.2)
    parser.add_argument('--split-seed', type=int, default=20261116)
    parser.add_argument('--neighbors', type=int, default=3)
    parser.add_argument('--min-samples', type=int, default=8)
    parser.add_argument('--exclude-label', action='append', default=[])
    parser.add_argument('--exclusion-manifest', type=Path)
    parser.add_argument('--audit-groups', action='store_true',
                        help='Also evaluate whole-session and whole-person exclusion with both hands')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Choose a fresh output directory; existing results are never overwritten')
    trials, source = load_trials(args.repo, args.ref, args.min_samples,
                                 DEFAULT_EXCLUDED_LABELS | set(args.exclude_label),
                                 args.exclusion_manifest)
    result = run_evaluation(trials, args.out, args.max_folds, args.seeds,
                            args.test_size, args.split_seed, args.neighbors,
                            args.audit_groups, source)
    print(f"Commit {source['commit']}: {len(trials)} trials, {result['participants']} people, "
          f"{len(result['labels'])} labels; folds {result['folds']}")
    for row in result['test']:
        print(f"{row['mode']}: selected folds={row['selected_folds']}, "
              f"CV={row['cv_accuracy']:.2%}, test={row['correct']}/{row['trials']} "
              f"({row['accuracy']:.2%}), balanced={row['balanced_accuracy']:.2%}")
    print(f"Shared-session test trials: {result['shared_session_test_trials']}/{result['test_trials']}; "
          'do not report this as new-session/new-person accuracy.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
