"""Combine swallowing/denial/wound in ONE classifier and evaluate held-out people.

Exploratory: labels were selected using earlier results and were recorded on
different days. Nested tuning does not make this a fresh external evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from threadpoolctl import threadpool_limits

try:
    from . import evaluate_six_sign_knn as base
except ImportError:
    import evaluate_six_sign_knn as base

LABELS = ['삼키다', '아니다', '상처']
ALIASES = {**base.ALIASES, '삼키다': '삼키다', '상처': '상처'}
IDENTITY = ['person', 'session_id', 'trial_id', 'file', 'label', 'sha256', 'source_commit', 'date_kst']


def extract(frame):
    for column in ['subject_id', 'session_id', 'trial_id', 'label']:
        if frame[column].astype(str).str.strip().nunique() != 1:
            raise ValueError('Conflicting ' + column)
    if set(frame.hand) != {'left', 'right'}:
        raise ValueError('Both hands required')
    if not (frame.source.eq('REAL').all() and frame.board_type.eq('XIAO').all()
            and frame.record_kind.eq('training').all()):
        raise ValueError('Expected real XIAO captures')
    label = ALIASES[str(frame.label.iloc[0]).strip()]
    if label not in LABELS:
        raise ValueError('Unsupported sign')
    row = dict(person=base.person(frame.subject_id.iloc[0]), session_id=str(frame.session_id.iloc[0]),
               trial_id=str(frame.trial_id.iloc[0]), label=label)
    dates = pd.to_datetime(frame.time, format='mixed', errors='raise', utc=True)
    row['date_kst'] = dates.min().tz_convert('Asia/Seoul').strftime('%Y-%m-%d')
    for hand in ['left', 'right']:
        part = frame[frame.hand.eq(hand)]
        values = part[base.SENSORS].to_numpy(float)
        times = part.time_ms.to_numpy(float)
        seq = part.sample_seq.to_numpy(float)
        if len(part) < 8 or not np.isfinite(values).all():
            raise ValueError('Short/nonfinite capture')
        if not np.isfinite(times).all() or not np.isfinite(seq).all():
            raise ValueError('Invalid clock/sequence')
        if (np.diff(times) <= 0).any() or (np.diff(seq) <= 0).any() or times[-1]-times[0] < 3000:
            raise ValueError('Invalid hand timing')
        if (values[:, :6] < 0).any():
            raise ValueError('Negative FSR')
        for i, sensor in enumerate(base.SENSORS):
            for stat in base.STATS:
                row[f'{hand}_{sensor}_{stat}'] = float(getattr(np, stat)(values[:, i]))
    return row


def complete_people(frame, labels=None):
    labels = LABELS if labels is None else labels
    return {who for who, g in frame.groupby('person') if set(g.label) == set(labels)}


def validate(frame, labels=None):
    labels = LABELS if labels is None else labels
    if frame.empty or not np.isfinite(frame[base.ALL_COLS].to_numpy(float)).all():
        raise ValueError('Empty/nonfinite cohort')
    for columns in [['file'], ['trial_id'], ['person', 'session_id', 'trial_id']]:
        if frame.duplicated(columns).any():
            raise ValueError('Duplicate measurement identity')
    if set(frame.label) != set(labels) or len(complete_people(frame, labels)) < 4:
        raise ValueError('Insufficient complete-class people')


def check_split(train, test, labels=None):
    labels = LABELS if labels is None else labels
    for col in ['person', 'file', 'trial_id']:
        if not set(train[col]).isdisjoint(test[col]):
            raise ValueError('Train/test leakage: ' + col)
    if train.empty or test.empty or set(train.label) != set(labels):
        raise ValueError('Empty split or missing training labels')
    # A partial-class test person is retained; absent recordings are not invented.
    if not set(test.label) <= set(labels):
        raise ValueError('Unexpected test labels')


def load_cohort(results_root, repo):
    old = results_root/'2026-11-08_knn_development_holdout'
    new = results_root/'2026-11-15_six_sign_nested_loso'
    old_meta = json.loads((old/'manifest.json').read_text(encoding='utf-8'))
    new_meta = json.loads((new/'inventory_summary.json').read_text(encoding='utf-8'))
    sources = [(old/'trial_features.csv', old_meta['source_commit'], old_meta['input_manifest']),
               (new/'trial_features.csv', new_meta['commit'], new_meta['manifest'])]
    records, inputs = [], [old/'manifest.json', new/'inventory_summary.json']
    for feature_file, commit, manifest in sources:
        inputs.append(feature_file)
        features = pd.read_csv(feature_file)
        features = features[features.label.isin(LABELS)].copy()
        declared = {entry['path']: entry['sha256'] for entry in manifest}
        archive = subprocess.check_output(['git', '-C', str(repo), 'archive', commit, 'sensor-data/training'])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for prior in features.to_dict('records'):
                raw = tar.extractfile(prior['file']).read()
                sha = hashlib.sha256(raw).hexdigest()
                if declared[prior['file']] != sha:
                    raise ValueError('Raw hash differs from original experiment manifest')
                row = extract(pd.read_csv(io.BytesIO(raw), keep_default_na=False))
                for column in ['person', 'session_id', 'trial_id', 'label']:
                    if row[column] != prior[column]:
                        raise ValueError('Feature/source identity mismatch: ' + column)
                np.testing.assert_allclose([row[c] for c in base.ALL_COLS],
                                           [prior[c] for c in base.ALL_COLS], rtol=1e-14, atol=1e-12)
                row.update(file=prior['file'], sha256=sha, source_commit=commit)
                records.append(row)
    frame = pd.DataFrame(records).sort_values(['person', 'session_id', 'trial_id']).reset_index(drop=True)
    validate(frame)
    if len(frame) != 539 or frame.person.nunique() != 10 or len(complete_people(frame)) != 8:
        raise ValueError('Expected pinned 539-trial, ten-person candidate snapshot')
    return frame, inputs


def select_inner(train, configs, labels=None):
    labels = LABELS if labels is None else labels
    eligible = complete_people(train, labels)
    if len(eligible) < 2:
        raise ValueError('Not enough complete-class people for tuning')
    board = []
    for config in configs:
        scores = []
        cols = base.FEATURES[config['sensors']]
        for who in base.people_order(eligible):
            fit, test = train[train.person.ne(who)], train[train.person.eq(who)]
            check_split(fit, test, labels)
            model = base.make_model(config).fit(fit[cols], fit.label)
            pred = model.predict(test[cols])
            score = float(recall_score(test.label, pred, labels=labels, average='macro', zero_division=0))
            scores.append(dict(inner_person=who, trials=len(test), balanced_accuracy=score))
        board.append(dict(**config, selection_score=float(np.mean([s['balanced_accuracy'] for s in scores])),
                          inner_folds=scores))
    board.sort(key=lambda c: (-c['selection_score'], c['config_id']))
    chosen = {sensor: next(c for c in board if c['sensors'] == sensor) for sensor in base.FEATURES}
    chosen['nested_selection'] = board[0]
    return chosen, board


def evaluate(frame, configs, progress=None, labels=None):
    labels = LABELS if labels is None else labels
    validate(frame, labels)
    outputs, folds, searches = [], [], []
    for who in base.people_order(frame.person):
        train, test = frame[frame.person.ne(who)], frame[frame.person.eq(who)]
        check_split(train, test, labels)
        chosen, board = select_inner(train, configs, labels=labels)
        searches.append(dict(outer_person=who, training_people=base.people_order(train.person),
                             inner_scored_people=base.people_order(complete_people(train, labels)), candidates=board))
        for method in base.METHODS:
            cfg = chosen[method]
            cols = base.FEATURES[cfg['sensors']]
            model = base.make_model(cfg).fit(train[cols], train.label)
            rows = test[IDENTITY].copy()
            rows['prediction'] = model.predict(test[cols])
            rows['correct'] = rows.label.eq(rows.prediction)
            rows['method'] = method
            rows['train_people'] = '|'.join(base.people_order(train.person))
            rows['config_id'] = cfg['config_id']
            rows['sensors'] = cfg['sensors']
            rows['complete_test_person'] = set(test.label) == set(labels)
            outputs.append(rows)
            folds.append(dict(person=who, method=method, test_trials=len(test), training_trials=len(train),
                              test_labels='|'.join(label for label in labels if label in set(test.label)),
                              correct=int(rows.correct.sum()), complete_test_person=bool(rows.complete_test_person.iloc[0]),
                              **{k: cfg[k] for k in ['config_id', 'sensors', 'k', 'p', 'weights']},
                              inner_score=cfg['selection_score']))
        if progress:
            progress(who)
    rows = pd.concat(outputs, ignore_index=True)
    for _, g in rows.groupby('method'):
        if len(g) != len(frame) or g.file.duplicated().any() or set(g.file) != set(frame.file):
            raise ValueError('Each source trial must be tested exactly once per method')
    return rows, pd.DataFrame(folds), searches


def summary(rows, labels=None):
    labels = LABELS if labels is None else labels
    correct = rows.label.eq(rows.prediction)
    if not correct.eq(rows.correct).all():
        raise ValueError('Incorrect prediction correctness flags')
    cm = confusion_matrix(rows.label, rows.prediction, labels=labels)
    people = []
    for who in base.people_order(rows.person):
        group = rows[rows.person.eq(who)]
        people.append(dict(person=who, labels=[s for s in labels if s in set(group.label)],
                           complete=set(group.label) == set(labels), trials=len(group),
                           correct=int(group.correct.sum()), accuracy=float(group.correct.mean())))
    complete = rows[rows.person.isin({p['person'] for p in people if p['complete']})]
    return dict(trials=len(rows), correct=int(correct.sum()), accuracy=float(correct.mean()),
                participant_mean_accuracy=float(np.mean([p['accuracy'] for p in people])),
                macro_f1=float(f1_score(rows.label, rows.prediction, labels=labels, average='macro', zero_division=0)),
                labels=labels, confusion_matrix=cm.tolist(), people=people,
                complete_people_subset=dict(people=sum(p['complete'] for p in people), trials=len(complete),
                                            correct=int(complete.correct.sum()), accuracy=float(complete.correct.mean())),
                per_sign=[dict(label=label, trials=int(cm[i].sum()), correct=int(cm[i, i]),
                               recall=float(cm[i, i]/cm[i].sum()),
                               precision=float(cm[i, i]/cm[:, i].sum()) if cm[:, i].sum() else 0)
                          for i, label in enumerate(labels)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a fresh output directory')
    frame, inputs = load_cohort(args.results_root, args.repo)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    args.out.mkdir(parents=True)
    configs = base.configurations()
    plan = dict(labels=LABELS, raw_trial_unit='One original ~5-second paired-hand capture',
                trials=len(frame), participants=base.people_order(frame.person),
                complete_people=base.people_order(complete_people(frame)),
                partial_people=base.people_order(set(frame.person)-complete_people(frame)),
                primary='nested_selection pooled over ALL 539 available trials / ten held-out people',
                inner='Equal-person macro recall, scored on complete-class training people only; partial people remain in fit',
                outer='All dates/labels of one person held out; partial-person available trials retained',
                configs=configs, tie_break='selection_score descending, config_id ascending',
                exploratory=True, labels_selected_after_prior_results=True,
                warning='Labels were captured on different dates; date/session/label confounding remains.',
                individual_prior_accuracies_not_combined=True, input_sha256=hashes)
    base.dump_json(args.out/'comparison_plan.json', plan)
    shutil.copy2(Path(__file__), args.out/Path(__file__).name)
    shutil.copy2(Path(base.__file__), args.out/Path(base.__file__).name)
    frame.to_csv(args.out/'trial_features.csv', index=False, encoding='utf-8-sig')
    frame[IDENTITY].to_csv(args.out/'cohort_manifest.csv', index=False, encoding='utf-8-sig')
    frame.groupby(['person','label']).size().unstack(fill_value=0).reindex(columns=LABELS).to_csv(
        args.out/'participant_sign_counts.csv', encoding='utf-8-sig')
    print('Verified: ONE three-class problem, 539 real trials / 10 people; no low-score exclusions', flush=True)
    with threadpool_limits(limits=1):
        predictions, folds, searches = evaluate(frame, configs,
            lambda who: print('Completed held-out participant: '+who, flush=True))
        results = {m: summary(predictions[predictions.method.eq(m)]) for m in base.METHODS}
        predictions.to_csv(args.out/'outer_predictions.csv', index=False, encoding='utf-8-sig')
        folds.to_csv(args.out/'outer_folds.csv', index=False, encoding='utf-8-sig')
        base.dump_json(args.out/'inner_searches.json', searches)
        base.dump_json(args.out/'evaluation.json', results)
        chosen, board = select_inner(frame, configs)
        config = chosen['nested_selection']
        cols = base.FEATURES[config['sensors']]
        final = base.make_model(config).fit(frame[cols], frame.label)
        metadata = dict(labels=LABELS, trials=len(frame), people=base.people_order(frame.person),
                        config={k: config[k] for k in ['config_id','sensors','k','p','weights']},
                        external_accuracy=None, external_accuracy_display='미측정', deployed=False,
                        exploratory=True, date_label_confounding_unresolved=True,
                        source_commits=sorted(frame.source_commit.unique()),
                        warning='CV evaluates the selection/training procedure, not this full-data artifact externally.')
        artifact = args.out/'three_sign_candidate_model.joblib'
        joblib.dump(dict(pipeline=final, feature_columns=cols, metadata=metadata), artifact)
        restored = joblib.load(artifact)
        assert restored['pipeline'].steps[-1][1].n_samples_fit_ == len(frame)
        np.testing.assert_array_equal(final.predict(frame.iloc[:5][cols]),
                                      restored['pipeline'].predict(frame.iloc[:5][cols]))
        base.dump_json(args.out/'candidate_manifest.json', metadata)
        base.dump_json(args.out/'full_data_selection.json', dict(winner=config, candidates=board,
                                                               independent_performance_claim=False))
    for path, sha in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha:
            raise ValueError('Old experiment source changed')
    base.dump_json(args.out/'verification.json', dict(raw_files_sha256_verified=len(frame),
        source_statistics_recomputed_and_matched=True, all_available_trials_retained=True,
        unique_trial_uuid=True, heldout_people=10, methods=4, predictions=len(predictions),
        prior_input_files_unchanged=True, previous_models_written=False, deployed=False,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        candidate_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest()))
    for method, result in results.items():
        print(method, f"{result['correct']}/{result['trials']} = {result['accuracy']*100:.2f}%", flush=True)
    print('Final candidate config:', metadata['config'], flush=True)


if __name__ == '__main__':
    main()
