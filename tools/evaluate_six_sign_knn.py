"""Reproducible six-sign app KNN: nested leave-one-participant-out evaluation.

Read a pinned, hashed inventory. Fit no scaler or setting on an outer test person.
Each source file is one trial, not one sample per sensor row. No raw CSV writes.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import tarfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

LABELS = ['괜찮다', '아니다', '무엇', '마시다', '화장실', '싫다']
ALIASES = {
    '괜찮다': '괜찮다', '괜찮아': '괜찮다',
    '아니다': '아니다', '아니다 (아니, 부인(하다))': '아니다',
    '무엇': '무엇', '무엇(어디)': '무엇', '마시다': '마시다',
    '화장실': '화장실', '싫다': '싫다', '싫다(거절)': '싫다', '싫다(거부)': '싫다',
}
SENSORS = [f'fsr{i}' for i in range(1, 7)] + ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
STATS = ['mean', 'std', 'min', 'max']
ALL_COLS = [f'{hand}_{sensor}_{stat}' for hand in ['left', 'right']
            for sensor in SENSORS for stat in STATS]
FEATURES = {
    'imu_only': [c for c in ALL_COLS if '_fsr' not in c],
    'fsr_only': [c for c in ALL_COLS if '_fsr' in c],
    'all_sensors': ALL_COLS,
}
METHODS = [*FEATURES, 'nested_selection']
IDENTITY = ['person', 'session_id', 'trial_id', 'file', 'sha256', 'label', 'date_kst']


def person(value):
    match = re.fullmatch(r'subject-?(\d+)', str(value).strip(), re.I)
    if not match:
        raise ValueError(f'Unrecognized participant: {value}')
    return 'subject' + str(int(match[1]))


def people_order(values):
    return sorted(set(values), key=lambda v: int(v.removeprefix('subject')))


def configurations():
    # Stable predeclared order breaks exact ties. No outer-test tie breaking.
    return [dict(config_id=i, sensors=s, k=k, p=p, weights=w)
            for i, (s, k, p, w) in enumerate(itertools.product(
                FEATURES, [3, 5, 7], [1, 2], ['uniform', 'distance']))]


def make_model(config):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=config['k'], p=config['p'], weights=config['weights'],
        algorithm='brute', n_jobs=1))


def extract_features(frame):
    """Summarize hands independently; device boot clocks need not be aligned."""
    for col in ['subject_id', 'session_id', 'trial_id', 'label']:
        if frame[col].astype(str).str.strip().nunique() != 1:
            raise ValueError(f'Conflicting {col} within trial')
    if set(frame.hand) != {'left', 'right'}:
        raise ValueError('Both hands required; never zero-fill a missing hand')
    if not (frame.source.eq('REAL').all() and frame.board_type.eq('XIAO').all()
            and frame.record_kind.eq('training').all()):
        raise ValueError('Expected real XIAO training capture')
    row = dict(person=person(frame.subject_id.iloc[0]),
               session_id=str(frame.session_id.iloc[0]),
               trial_id=str(frame.trial_id.iloc[0]),
               label=ALIASES[str(frame.label.iloc[0]).strip()])
    for hand in ['left', 'right']:
        part = frame[frame.hand.eq(hand)]
        values = part[SENSORS].apply(pd.to_numeric, errors='raise').to_numpy(float)
        times = pd.to_numeric(part.time_ms, errors='raise').to_numpy(float)
        seq = pd.to_numeric(part.sample_seq, errors='raise').to_numpy(float)
        if len(part) < 8 or not np.isfinite(values).all():
            raise ValueError('Insufficient or nonfinite hand data')
        if not np.isfinite(times).all() or not np.isfinite(seq).all():
            raise ValueError('Nonfinite clock/sequence')
        if (np.diff(times) <= 0).any() or (np.diff(seq) <= 0).any():
            raise ValueError('Nonmonotonic hand clock/sequence')
        if times[-1] - times[0] < 3000 or (values[:, :6] < 0).any():
            raise ValueError('Short trial or negative FSR')
        for i, sensor in enumerate(SENSORS):
            v = values[:, i]
            for stat in STATS:
                row[f'{hand}_{sensor}_{stat}'] = float(getattr(np, stat)(v))
    return row


def validate_cohort(frame):
    if frame.empty or not np.isfinite(frame[ALL_COLS].to_numpy(float)).all():
        raise ValueError('Empty or nonfinite feature cohort')
    for cols in [['person', 'session_id', 'trial_id'], ['trial_id'], ['file']]:
        if frame.duplicated(cols).any():
            raise ValueError(f'Duplicate measurement identity: {cols}')
    if frame.person.nunique() < 4:
        raise ValueError('Nested participant evaluation needs at least four people')
    if set(frame.label) != set(LABELS):
        raise ValueError('Expected all six canonical signs')
    if any(set(group.label) != set(LABELS) for _, group in frame.groupby('person')):
        raise ValueError('Every test person must have all six classes')


def check_split(train, test):
    if train.empty or test.empty:
        raise ValueError('Empty split')
    for col in ['person', 'trial_id', 'file']:
        if not set(train[col]).isdisjoint(test[col]):
            raise ValueError(f'Train/test leakage: {col}')
    # session_id alone may legitimately be shared when app Subject changes.
    if set(train.label) != set(LABELS) or set(test.label) != set(LABELS):
        raise ValueError('Split missing sign classes')


def load_cohort(inventory_dir, repo):
    meta = json.loads((inventory_dir / 'summary.json').read_text(encoding='utf-8'))
    if meta['alias_map'] != ALIASES or meta['parsing_errors']:
        raise ValueError('Inventory label rules or parsing require review')
    inventory = pd.read_csv(inventory_dir / 'trial_inventory.csv', keep_default_na=False)
    if len(inventory) != meta['files'] or set(inventory.label) != set(LABELS):
        raise ValueError('Inventory differs from summary')
    declared = {entry['path']: entry['sha256'] for entry in meta['manifest']}
    if len(declared) != len(inventory):
        raise ValueError('Repeated manifest path')
    for row in inventory.itertuples():
        if declared.get(row.file) != row.sha256:
            raise ValueError('Inventory hash differs from pinned manifest')
    for col in ['duplicate_trial_identity', 'duplicate_signal_and_time']:
        if inventory[col].astype(str).str.lower().eq('true').any():
            raise ValueError('Duplicate raw measurements require review')
    valid = inventory.basic_pass.astype(str).str.lower().eq('true')
    if (valid != inventory.hard_failures.eq('')).any():
        raise ValueError('Inconsistent quality decision')
    if int(valid.sum()) != meta['basic_pass']:
        raise ValueError('Quality counts differ from summary')
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive',
                                       meta['commit'], 'sensor-data/training'])
    rows = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for entry in inventory.itertuples():
            raw = tar.extractfile(entry.file).read()
            if hashlib.sha256(raw).hexdigest() != entry.sha256:
                raise ValueError(f'Source hash mismatch: {entry.file}')
            if str(entry.basic_pass).lower() != 'true':
                continue  # Preserve source; excluded only by audited hard failures.
            row = extract_features(pd.read_csv(io.BytesIO(raw), keep_default_na=False))
            for col in ['person', 'session_id', 'trial_id', 'label']:
                if row[col] != getattr(entry, col):
                    raise ValueError(f'Source identity mismatch: {entry.file} {col}')
            row.update(file=entry.file, sha256=entry.sha256, date_kst=entry.date_kst)
            rows.append(row)
    features = pd.DataFrame(rows).sort_values(['person', 'session_id', 'trial_id']).reset_index(drop=True)
    validate_cohort(features)
    return meta, inventory, features


def select_configs(train, configs):
    """Accept training people ONLY. Return all inner scores and stable winners."""
    scores = []
    for config in configs:
        results = []
        for who in people_order(train.person):
            fit = train[train.person.ne(who)]
            test = train[train.person.eq(who)]
            check_split(fit, test)
            cols = FEATURES[config['sensors']]
            if config['k'] > len(fit):
                raise ValueError('K exceeds inner training size')
            model = make_model(config).fit(fit[cols], fit.label)
            pred = model.predict(test[cols])
            results.append(dict(inner_person=who, trials=len(test),
                                balanced_accuracy=float(balanced_accuracy_score(test.label, pred))))
        scores.append(dict(**config, selection_score=float(np.mean([
            r['balanced_accuracy'] for r in results])), inner_folds=results))
    ranked = sorted(scores, key=lambda row: (-row['selection_score'], row['config_id']))
    winners = {'nested_selection': ranked[0]}
    for sensor in FEATURES:
        winners[sensor] = next(row for row in ranked if row['sensors'] == sensor)
    return winners, ranked


def evaluate_nested(features, configs, progress=None):
    validate_cohort(features)
    predictions, folds, selections = [], [], []
    for who in people_order(features.person):
        train = features[features.person.ne(who)].copy()
        test = features[features.person.eq(who)].copy()
        check_split(train, test)
        chosen, board = select_configs(train, configs)
        selections.append(dict(outer_person=who, inner_people=people_order(train.person),
                               candidates=board))
        for method in METHODS:
            cfg = chosen[method]
            cols = FEATURES[cfg['sensors']]
            model = make_model(cfg).fit(train[cols], train.label)
            pred = model.predict(test[cols])
            out = test[IDENTITY].copy()
            out['prediction'] = pred
            out['correct'] = out.label.eq(out.prediction)
            out['method'] = method
            out['config_id'] = cfg['config_id']
            out['sensors'] = cfg['sensors']
            out['train_people'] = '|'.join(people_order(train.person))
            predictions.append(out)
            folds.append(dict(method=method, person=who, training_trials=len(train),
                              test_trials=len(test), correct=int(out.correct.sum()),
                              accuracy=float(out.correct.mean()), **{k: cfg[k] for k in
                              ['config_id', 'sensors', 'k', 'p', 'weights']},
                              inner_selection_score=cfg['selection_score']))
        if progress:
            progress(who)
    rows = pd.concat(predictions, ignore_index=True)
    for _, group in rows.groupby('method'):
        if len(group) != len(features) or group.file.duplicated().any():
            raise ValueError('Each trial must be tested exactly once per method')
        if set(group.file) != set(features.file):
            raise ValueError('Incomplete test predictions')
    return rows, pd.DataFrame(folds), selections


def result_summary(rows):
    cm = confusion_matrix(rows.label, rows.prediction, labels=LABELS)
    people = []
    for who in people_order(rows.person):
        part = rows[rows.person.eq(who)]
        people.append(dict(person=who, trials=len(part), correct=int(part.correct.sum()),
                           accuracy=float(part.correct.mean()),
                           balanced_accuracy=float(balanced_accuracy_score(part.label, part.prediction))))
    return dict(trials=len(rows), correct=int(rows.correct.sum()), accuracy=float(rows.correct.mean()),
                participant_mean_accuracy=float(np.mean([p['accuracy'] for p in people])),
                balanced_accuracy=float(balanced_accuracy_score(rows.label, rows.prediction)),
                macro_f1=float(f1_score(rows.label, rows.prediction, labels=LABELS, average='macro')),
                labels=LABELS, confusion_matrix=cm.tolist(), people=people,
                per_sign=[dict(label=label, trials=int(cm[i].sum()), correct=int(cm[i, i]),
                               recall=float(cm[i, i] / cm[i].sum())) for i, label in enumerate(LABELS)])


def dump_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory-dir', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a fresh output directory; never overwrite results')
    meta, inventory, features = load_cohort(args.inventory_dir, args.repo)
    args.out.mkdir(parents=True)
    for name in ['summary.json', 'trial_inventory.csv', 'audit_six_sign_data.py', 'knn_evaluation_plan.md']:
        if (args.inventory_dir / name).exists():
            shutil.copy2(args.inventory_dir / name, args.out / ('inventory_' + name))
    shutil.copy2(Path(__file__), args.out / Path(__file__).name)
    features.to_csv(args.out / 'trial_features.csv', index=False, encoding='utf-8-sig')
    features[IDENTITY].to_csv(args.out / 'cohort_manifest.csv', index=False, encoding='utf-8-sig')
    excluded = inventory[~inventory.file.isin(features.file)].copy()
    excluded.to_csv(args.out / 'excluded_trials.csv', index=False, encoding='utf-8-sig')
    configs = configurations()
    dump_json(args.out / 'preregistered_settings.json', dict(
        source_commit=meta['commit'], primary_method='nested_selection', candidate_configs=configs,
        selection_criterion='Equal-person mean balanced accuracy; exact ties use config_id ascending',
        feature_statistics=STATS, std_ddof=0, shared_clock_alignment=False,
        outer='Leave one person out; every date for that person is held out',
        inner='Leave one training person out; no outer-person data permitted',
        raw_trials=len(inventory), paired_trials=len(features), excluded=len(excluded)))
    print(f'Pinned {meta["commit"]}: {len(features)} paired trials, {features.person.nunique()} people', flush=True)
    with threadpool_limits(limits=1):
        rows, folds, selections = evaluate_nested(features, configs,
            progress=lambda who: print(f'Completed outer fold {who}', flush=True))
        rows.to_csv(args.out / 'outer_predictions.csv', index=False, encoding='utf-8-sig')
        folds.to_csv(args.out / 'outer_folds.csv', index=False, encoding='utf-8-sig')
        dump_json(args.out / 'inner_searches.json', selections)
        results = {method: result_summary(rows[rows.method.eq(method)]) for method in METHODS}
        dump_json(args.out / 'evaluation.json', results)
        # Full-data group CV chooses deployment candidate only after outer evaluation.
        # Its selection score is NOT an independent accuracy of the fitted candidate.
        chosen, board = select_configs(features, configs)
        cfg = chosen['nested_selection']
        cols = FEATURES[cfg['sensors']]
        final = make_model(cfg).fit(features[cols], features.label)
        candidate_meta = dict(source_commit=meta['commit'], config={k: cfg[k] for k in
                             ['config_id', 'sensors', 'k', 'p', 'weights']},
                             trials=len(features), people=people_order(features.person), classes=LABELS,
                             independent_accuracy=None, independent_accuracy_display='미측정', deployed=False,
                             fitting='Entire paired cohort after nested evaluation',
                             warning='Outer accuracy evaluates training/selection procedure, not this full-data artifact.')
        artifact = args.out / 'six_sign_candidate_model.joblib'
        joblib.dump(dict(pipeline=final, feature_columns=cols, metadata=candidate_meta), artifact)
        restored = joblib.load(artifact)
        if restored['pipeline'].steps[-1][1].n_samples_fit_ != len(features):
            raise ValueError('Serialized candidate fit size mismatch')
        check = features.iloc[:8][cols]
        if not np.array_equal(final.predict(check), restored['pipeline'].predict(check)):
            raise ValueError('Serialized predictions differ')
        dump_json(args.out / 'candidate_manifest.json', candidate_meta)
        dump_json(args.out / 'full_data_selection.json', dict(winner=cfg, candidates=board,
                                                            performance_claim=False))
    files = [Path(__file__), args.inventory_dir / 'summary.json', args.inventory_dir / 'trial_inventory.csv']
    dump_json(args.out / 'reproducibility.json', dict(
        source_commit=meta['commit'], python=platform.python_version(), sklearn=sklearn.__version__,
        numpy=np.__version__, pandas=pd.__version__,
        files=[dict(name=f.name, sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],
        candidate_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        existing_sources_written=False, existing_models_written=False, raw_files_verified=len(inventory),
        limitations=[
            'Internal nested participant CV on nine recorded Subject IDs, not prospective external testing.',
            'Wearing position, sign correctness, calibration and shared capture clock are unverified.',
            'Participants share app sessions/devices; participant IDs must reflect actual people.',
            'No per-frame hand synchronization, temporal segmentation or sentence/sequence evaluation.',
            'Repeated trials are correlated; no trial-independent confidence interval is claimed.']))
    print(json.dumps({k: {field: v[field] for field in ['correct', 'trials', 'accuracy',
                      'participant_mean_accuracy', 'macro_f1']} for k, v in results.items()}, indent=2), flush=True)
    print('Final candidate: ' + json.dumps(candidate_meta['config']), flush=True)


if __name__ == '__main__':
    main()
