"""Exploratory same-person and temporal-feature diagnostics on pinned real trials.

This design follows inspection of earlier outer predictions. Results are exploratory
internal reanalysis, not a fresh external evaluation, even with nested tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

try:
    from . import evaluate_six_sign_knn as base
except ImportError:
    import evaluate_six_sign_knn as base

BINS = 5
FIXED = dict(config_id=4, sensors='imu_only', k=5, p=1, weights='uniform')
TEMPORAL_COLS = [f'{h}_bin{b}_{s}_{stat}' for h in ['left', 'right']
                 for b in range(BINS) for s in base.SENSORS for stat in base.STATS]
TEMPORAL = {'imu_only': [c for c in TEMPORAL_COLS if '_fsr' not in c],
            'fsr_only': [c for c in TEMPORAL_COLS if '_fsr' in c],
            'all_sensors': TEMPORAL_COLS}
MAPS = {'global': base.FEATURES, 'five_bins': TEMPORAL}


def temporal_features(frame):
    # Reuse all strict identity/hand/sensor checks from the original adapter.
    row = base.extract_features(frame)
    for hand in ['left', 'right']:
        part = frame[frame.hand.eq(hand)]
        times = part.time_ms.to_numpy(float)
        # Per-hand relative progress only. Never subtract one board's clock from
        # the other; this does not assert synchronized sample-wise hand capture.
        progress = (times - times[0]) / (times[-1] - times[0])
        assignment = np.minimum((progress * BINS).astype(int), BINS - 1)
        for b in range(BINS):
            chunk = part.iloc[np.flatnonzero(assignment == b)]
            if len(chunk) < 3:
                raise ValueError(f'{hand} bin{b}: insufficient samples; review cohort before analysis')
            for sensor in base.SENSORS:
                values = chunk[sensor].to_numpy(float)
                for stat in base.STATS:
                    row[f'{hand}_bin{b}_{sensor}_{stat}'] = float(getattr(np, stat)(values))
    return row


def load_features(previous, repo):
    meta = json.loads((previous / 'inventory_summary.json').read_text(encoding='utf-8'))
    inventory = pd.read_csv(previous / 'inventory_trial_inventory.csv', keep_default_na=False)
    expected = pd.read_csv(previous / 'cohort_manifest.csv', keep_default_na=False)
    old_features = pd.read_csv(previous / 'trial_features.csv', keep_default_na=False)
    if meta['alias_map'] != base.ALIASES or meta['parsing_errors']:
        raise ValueError('Inventory rules need review')
    hashes = {m['path']: m['sha256'] for m in meta['manifest']}
    if len(hashes) != len(inventory) or hashes != dict(zip(inventory.file, inventory.sha256)):
        raise ValueError('Inventory and pinned hash manifest differ')
    allowed = set(expected.file)
    if allowed != set(inventory[inventory.basic_pass.astype(str).str.lower().eq('true')].file):
        raise ValueError('Prior cohort differs from audited valid trials')
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive',
                                       meta['commit'], 'sensor-data/training'])
    records = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in inventory.itertuples():
            raw = tar.extractfile(entry.file).read()
            if hashlib.sha256(raw).hexdigest() != entry.sha256:
                raise ValueError(f'Raw file changed: {entry.file}')
            if entry.file not in allowed:
                continue
            row = temporal_features(pd.read_csv(io.BytesIO(raw), keep_default_na=False))
            for field in ['person', 'session_id', 'trial_id', 'label']:
                if row[field] != getattr(entry, field):
                    raise ValueError(f'Identity mismatch: {field}')
            row.update(file=entry.file, sha256=entry.sha256, date_kst=entry.date_kst,
                       start_kst=entry.start_kst)
            records.append(row)
    features = pd.DataFrame(records).sort_values(['person', 'session_id', 'trial_id']).reset_index(drop=True)
    base.validate_cohort(features)
    if set(features.file) != allowed:
        raise ValueError('Temporal feature generation changed the cohort')
    actual = features.set_index('file').sort_index()
    for frame in [expected, old_features]:
        prior = frame.set_index('file').sort_index()
        if not actual[base.IDENTITY[:-4] + ['sha256', 'label', 'date_kst']].equals(
                prior[base.IDENTITY[:-4] + ['sha256', 'label', 'date_kst']]):
            raise ValueError('Manifest differs from prior analysis')
    np.testing.assert_allclose(actual[base.ALL_COLS], old_features.set_index('file').sort_index()[base.ALL_COLS],
                               rtol=1e-14, atol=1e-12)
    return meta, inventory, features


def within_person_folds(features, n_splits=5):
    """Hold out chronological trial blocks within each person and sign.

    Each sign contributes one contiguous block per fold. Entire trials are held
    out; neighboring repeats and the same recording session may remain in fit.
    This is an easy same-person/session diagnostic, not a new-session test.
    """
    assignments = pd.Series(-1, index=features.index, dtype=int)
    for _, group in features.groupby(['person', 'label']):
        if len(group) < n_splits:
            raise ValueError('Insufficient trials for five-fold within-person diagnostic')
        order = group.sort_values(['start_kst', 'trial_id']).index.to_numpy()
        for fold, ids in enumerate(np.array_split(order, n_splits)):
            assignments.loc[ids] = fold
    if (assignments < 0).any():
        raise ValueError('Unassigned trial')
    return assignments


def trial_split_check(train, test, same_person):
    for column in ['file', 'trial_id']:
        if not set(train[column]).isdisjoint(test[column]):
            raise ValueError('Trial leakage: ' + column)
    if same_person:
        if train.person.nunique() != 1 or set(train.person) != set(test.person):
            raise ValueError('Within-person split mixes different people')
        if set(train.label) != set(base.LABELS) or set(test.label) != set(base.LABELS):
            raise ValueError('Within-person split missing classes')
    else:
        base.check_split(train, test)


def predict_rows(train, test, columns, config, protocol, representation, method, fold):
    model = base.make_model(config).fit(train[columns], train.label)
    rows = test[base.IDENTITY].copy()
    rows['prediction'] = model.predict(test[columns])
    rows['correct'] = rows.label.eq(rows.prediction)
    rows['protocol'] = protocol
    rows['representation'] = representation
    rows['method'] = method
    rows['fold'] = fold
    rows['config_id'] = config['config_id']
    rows['sensors'] = config['sensors']
    rows['train_people'] = '|'.join(base.people_order(train.person))
    rows['training_trials'] = len(train)
    return rows


def evaluate_fixed(features, progress=None):
    assignments = within_person_folds(features)
    predictions, splits = [], []
    for who in base.people_order(features.person):
        person = features[features.person.eq(who)]
        for fold in range(5):
            train = person[assignments.loc[person.index].ne(fold)]
            test = person[assignments.loc[person.index].eq(fold)]
            trial_split_check(train, test, same_person=True)
            splits.append(dict(person=who, fold=fold, train_files=train.file.tolist(), test_files=test.file.tolist(),
                               same_session_overlap=bool(set(train.session_id) & set(test.session_id))))
            for representation, feature_map in MAPS.items():
                for sensor, columns in feature_map.items():
                    cfg = dict(FIXED, sensors=sensor)
                    predictions.append(predict_rows(train, test, columns, cfg, 'same_person_blocked5',
                                                    representation, sensor, fold))
        train = features[features.person.ne(who)]
        test = person
        trial_split_check(train, test, same_person=False)
        for representation, feature_map in MAPS.items():
            for sensor, columns in feature_map.items():
                cfg = dict(FIXED, sensors=sensor)
                predictions.append(predict_rows(train, test, columns, cfg, 'heldout_person_fixed',
                                                representation, sensor, who))
        if progress:
            progress(who)
    return pd.concat(predictions, ignore_index=True), splits


def select_inner_temporal(train, configs):
    scores = []
    for config in configs:
        folds = []
        cols = TEMPORAL[config['sensors']]
        for who in base.people_order(train.person):
            fit, test = train[train.person.ne(who)], train[train.person.eq(who)]
            base.check_split(fit, test)
            model = base.make_model(config).fit(fit[cols], fit.label)
            pred = model.predict(test[cols])
            folds.append(dict(inner_person=who, trials=len(test),
                              balanced_accuracy=float(balanced_accuracy_score(test.label, pred))))
        scores.append(dict(**config, selection_score=float(np.mean([f['balanced_accuracy'] for f in folds])),
                           inner_folds=folds))
    ranked = sorted(scores, key=lambda c: (-c['selection_score'], c['config_id']))
    return {**{sensor: next(c for c in ranked if c['sensors'] == sensor) for sensor in TEMPORAL},
            'nested_selection': ranked[0]}, ranked


def evaluate_temporal_nested(features, progress=None):
    configs = base.configurations()
    predictions, searches, folds = [], [], []
    for who in base.people_order(features.person):
        train, test = features[features.person.ne(who)], features[features.person.eq(who)]
        base.check_split(train, test)
        chosen, board = select_inner_temporal(train, configs)
        searches.append(dict(outer_person=who, inner_people=base.people_order(train.person), candidates=board))
        for method in base.METHODS:
            config = chosen[method]
            rows = predict_rows(train, test, TEMPORAL[config['sensors']], config,
                                'heldout_person_nested', 'five_bins', method, who)
            predictions.append(rows)
            folds.append(dict(person=who, method=method, training_trials=len(train), test_trials=len(test),
                              correct=int(rows.correct.sum()), **{k: config[k] for k in
                              ['config_id', 'sensors', 'k', 'p', 'weights']},
                              inner_selection_score=config['selection_score']))
        if progress:
            progress(who)
    return pd.concat(predictions, ignore_index=True), searches, pd.DataFrame(folds)


def validate_predictions(rows, features):
    for _, group in rows.groupby(['protocol', 'representation', 'method']):
        if len(group) != len(features) or group.file.duplicated().any() or set(group.file) != set(features.file):
            raise ValueError('Every trial must appear once per comparison')
        if not group.correct.eq(group.label.eq(group.prediction)).all():
            raise ValueError('Incorrect correctness flag')
        for row in group.itertuples():
            is_same = row.protocol == 'same_person_blocked5'
            if (row.person in row.train_people.split('|')) != is_same:
                raise ValueError('Protocol/participant split mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a fresh diagnostic output directory')
    args.out.mkdir(parents=True)
    inputs = [args.previous / n for n in ['inventory_summary.json', 'inventory_trial_inventory.csv',
                                        'cohort_manifest.csv', 'trial_features.csv', 'outer_predictions.csv']]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    # Write comparison choices before accessing any new prediction results.
    plan = dict(exploratory_after_previous_results=True,
                warning='Post-result feature redesign on reused participants; not fresh external validation.',
                fixed_comparison=FIXED, temporal_bins=BINS, temporal_bin_min_samples=3,
                temporal_features={k: len(v) for k, v in TEMPORAL.items()},
                global_features={k: len(v) for k, v in base.FEATURES.items()},
                within_person='5 chronological blocks per person and class; no trial split across fit/test',
                between_person='All dates for one person held out; remaining eight train',
                nested_temporal_configs=base.configurations(),
                changed_conditions=['five relative-time bins instead of whole-trial statistics'],
                excluded_for_low_accuracy=False, raw_or_old_models_written=False,
                final_candidate_fitted=False, input_sha256=hashes)
    base.dump_json(args.out / 'comparison_plan.json', plan)
    shutil.copy2(Path(__file__), args.out / Path(__file__).name)
    shutil.copy2(Path(base.__file__), args.out / Path(base.__file__).name)
    meta, inventory, features = load_features(args.previous, args.repo)
    print(f'Verified {len(inventory)} raw files; identical {len(features)} paired trial cohort', flush=True)
    features.to_csv(args.out / 'trial_features.csv', index=False, encoding='utf-8-sig')
    features[base.IDENTITY + ['start_kst']].to_csv(args.out / 'cohort_manifest.csv', index=False, encoding='utf-8-sig')
    inventory[~inventory.file.isin(features.file)].to_csv(args.out / 'excluded_trials.csv', index=False, encoding='utf-8-sig')
    with threadpool_limits(limits=1):
        fixed, splits = evaluate_fixed(features, lambda who: print('Fixed comparisons: ' + who, flush=True))
        fixed.to_csv(args.out / 'fixed_predictions.csv', index=False, encoding='utf-8-sig')
        base.dump_json(args.out / 'within_person_splits.json', splits)
        temporal, searches, folds = evaluate_temporal_nested(features,
            lambda who: print('Temporal nested outer fold: ' + who, flush=True))
    temporal.to_csv(args.out / 'temporal_nested_predictions.csv', index=False, encoding='utf-8-sig')
    base.dump_json(args.out / 'temporal_inner_searches.json', searches)
    folds.to_csv(args.out / 'temporal_nested_folds.csv', index=False, encoding='utf-8-sig')
    previous = pd.read_csv(args.previous / 'outer_predictions.csv', keep_default_na=False)
    if set(previous.file) != set(features.file):
        raise ValueError('Previous predictions use a different cohort')
    previous['protocol'] = 'heldout_person_nested'
    previous['representation'] = 'global'
    previous['correct'] = previous.label.eq(previous.prediction)
    rows = pd.concat([fixed, temporal, previous], ignore_index=True)
    validate_predictions(rows, features)
    results = []
    for (protocol, representation, method), group in rows.groupby(['protocol', 'representation', 'method'], sort=False):
        results.append(dict(protocol=protocol, representation=representation, method=method,
                            **base.result_summary(group)))
    base.dump_json(args.out / 'evaluation.json', dict(source_commit=meta['commit'], exploratory=True,
                                                    raw_trials=len(inventory), paired_trials=len(features),
                                                    results=results))
    pd.DataFrame([{k: v for k, v in r.items() if k not in ['labels', 'confusion_matrix', 'people', 'per_sign']}
                  for r in results]).to_csv(args.out / 'comparison_summary.csv', index=False, encoding='utf-8-sig')
    rows.to_csv(args.out / 'all_predictions.csv', index=False, encoding='utf-8-sig')
    for p in inputs:
        if hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]:
            raise ValueError('Previous results changed during diagnostic')
    base.dump_json(args.out / 'verification.json', dict(raw_hashes_verified=len(inventory),
        identical_paired_trials=len(features), predictions=len(rows),
        same_person_folds=len(splits), outer_nested_folds=len(searches),
        previous_inputs_sha256_unchanged=True, missing_hand_trials_preserved=len(inventory) - len(features),
        no_automatic_model_replacement=True, no_new_final_candidate=True,
        experimental_data_kind='REAL', exploratory_not_external_validation=True))
    for r in results:
        print(r['protocol'], r['representation'], r['method'],
              f"{r['correct']}/{r['trials']} = {r['accuracy']*100:.2f}%", flush=True)


if __name__ == '__main__':
    main()
