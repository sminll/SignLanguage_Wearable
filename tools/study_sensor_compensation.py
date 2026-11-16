"""Audit real calibration links, then compare predeclared whole-trial transforms.

Exploratory reuse of previous held-out-person data, NOT new external validation.
Reuses each prior training-only selected KNN setting; no new parameter search,
no class-dependent transforms, no model deployment, no changes to existing CSVs.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits

try:
    from . import evaluate_six_sign_knn as base
    from . import sensor_compensation as comp
except ImportError:
    import evaluate_six_sign_knn as base
    import sensor_compensation as comp

COHORTS = {
    'three_sign': '2026-11-15_three_sign_combined_candidates',
    'six_sign': '2026-11-15_six_sign_nested_loso',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_person(value):
    value = str(value).strip()
    if re.fullmatch(r'subject-?\d+', value, re.I):
        return base.person(value)
    return value  # anonymous IDs are NOT guessed from app account or proximity


def metadata(frame, path):
    result = dict(file=path, account=path.split('/')[2])
    for col in ['subject_id', 'session_id', 'trial_id', 'label',
                'board_type', 'calibration_version', 'record_kind']:
        if col not in frame or frame[col].astype(str).nunique() != 1:
            raise ValueError('Conflicting/missing ' + col + ': ' + path)
        result[col] = str(frame[col].iloc[0])
    result['person'] = normalize_person(result['subject_id'])
    times = pd.to_datetime(frame.time, format='mixed', errors='raise', utc=True)
    result.update(start=times.min().isoformat(), end=times.max().isoformat())
    result['calibration_verified'] = bool(
        'calibration_verified' in frame and
        frame.calibration_verified.astype(str).str.lower().eq('true').all())
    return result


def rest_links(training, calibration, max_age_seconds=1800):
    """Strict metadata matches, not a claim of physical sensor calibration.

    Only explicit rest captures before this trial, same account/person/session/
    board/hand/version. Never transfer a profile across people or sessions.
    """
    keys = ['account', 'person', 'session_id', 'board_type', 'hand', 'calibration_version']
    links = []
    for trial in training:
        candidates = [c for c in calibration if c['is_rest'] and c['rows'] >= 8
                      and c['span_ms'] >= 3000 and all(c[k] == trial[k] for k in keys)
                      and 0 <= (pd.Timestamp(trial['start'])-pd.Timestamp(c['end'])).total_seconds()
                      <= max_age_seconds]
        if not candidates:
            links.append(dict(file=trial['file'], hand=trial['hand'], person=trial['person'],
                              status='no_matching_previous_rest', calibration_file=''))
            continue
        newest = max(pd.Timestamp(c['end']) for c in candidates)
        best = [c for c in candidates if pd.Timestamp(c['end']) == newest]
        if len(best) != 1:
            status = 'ambiguous_profile'
        elif not best[0]['calibration_verified']:
            status = 'metadata_match_but_unverified'
        else:
            status = 'verified_metadata_match_requires_physical_review'
        links.append(dict(file=trial['file'], hand=trial['hand'], person=trial['person'],
                          status=status, calibration_file='|'.join(c['file'] for c in best)))
    return links


def load_inputs(root):
    frames, references, folds, hashes = {}, {}, {}, {}
    for name, folder in COHORTS.items():
        directory = root / folder
        paths = [directory / n for n in ['trial_features.csv', 'outer_predictions.csv',
                                         'outer_folds.csv', 'evaluation.json']]
        hashes.update({str(p): sha(p) for p in paths})
        frames[name] = pd.read_csv(paths[0])
        if 'source_commit' not in frames[name]:
            meta_path = directory / 'inventory_summary.json'
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
            hashes[str(meta_path)] = sha(meta_path)
            frames[name]['source_commit'] = meta['commit']
        if frames[name].source_commit.isna().any():
            raise ValueError('Missing pinned source commit')
        references[name] = pd.read_csv(paths[1])
        folds[name] = pd.read_csv(paths[2])
    unique = pd.concat([f for f in frames.values()], ignore_index=True)
    for _, group in unique.groupby('file'):
        if group.sha256.nunique() != 1 or group.source_commit.nunique() != 1:
            raise ValueError('Conflicting source identity')
    unique = unique.drop_duplicates('file')
    return frames, references, folds, hashes, unique


def load_real_data(repo, unique, calibration_commit):
    feature_rows, training = {}, []
    for commit, entries in unique.groupby('source_commit'):
        archive = subprocess.check_output(['git', '-C', str(repo), 'archive', str(commit),
                                           'sensor-data/training'])
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            for i, row in enumerate(entries.itertuples(), 1):
                raw = tar.extractfile(row.file).read()
                if hashlib.sha256(raw).hexdigest() != row.sha256:
                    raise ValueError('Raw SHA mismatch')
                frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
                meta = metadata(frame, row.file)
                if meta['person'] != row.person or meta['session_id'] != row.session_id or meta['trial_id'] != row.trial_id:
                    raise ValueError('Raw/cache identity mismatch')
                aliases = {**base.ALIASES, '삼키다': '삼키다', '상처': '상처'}
                if aliases[meta['label']] != row.label:
                    raise ValueError('Raw/cache label mismatch')
                if set(frame.hand) != {'left', 'right'}:
                    raise ValueError('Missing hand')
                hands = {}
                for hand, part in frame.groupby('hand'):
                    values = part[base.SENSORS].to_numpy(float)
                    times = part.time_ms.to_numpy(float)
                    seq = part.sample_seq.to_numpy(float)
                    comp.validate(values)
                    if (not np.isfinite(times).all() or (np.diff(times) <= 0).any()
                            or not np.isfinite(seq).all() or (np.diff(seq) <= 0).any()
                            or times[-1]-times[0] < 3000):
                        raise ValueError('Invalid hand clock/sequence')
                    hands[hand] = values
                    training.append(dict(**meta, hand=hand, rows=len(part)))
                values = {rep: comp.paired_features(hands, rep) for rep in comp.REPRESENTATIONS}
                np.testing.assert_allclose(values['baseline_all'],
                    [getattr(row, c) for c in base.ALL_COLS], rtol=1e-14, atol=1e-12)
                feature_rows[row.file] = values
                if i % 200 == 0:
                    print('Raw verified', commit[:7], i, '/', len(entries), flush=True)
    archive = subprocess.check_output(['git', '-C', str(repo), 'archive', calibration_commit,
                                       'sensor-data/calibration'])
    calibration = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith('.csv'):
                continue
            raw = tar.extractfile(member).read()
            frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
            meta = metadata(frame, member.name)
            meta.update(sha256=hashlib.sha256(raw).hexdigest(),
                        is_rest=meta['label'].strip().lower() in {'휴식', 'rest', 'resting', 'neutral'})
            for hand, part in frame.groupby('hand'):
                t = pd.to_numeric(part.time_ms).to_numpy(float)
                values = part[base.SENSORS].to_numpy(float)
                calibration.append(dict(**meta, hand=hand, rows=len(part),
                    span_ms=float(t[-1]-t[0]), finite=bool(np.isfinite(values).all()),
                    fsr_zero_channels=int((values[:, :6] == 0).all(axis=0).sum())))
    return feature_rows, training, calibration


def evaluate_cohort(frame, features, previous, folds, progress=None):
    outputs = []
    for who in base.people_order(frame.person):
        train, test = frame[frame.person.ne(who)], frame[frame.person.eq(who)]
        for key in ['person', 'trial_id', 'file']:
            if set(train[key]) & set(test[key]):
                raise ValueError('Train/test identity leakage')
        for rep, params in comp.REPRESENTATIONS.items():
            reference = 'imu_only' if params.get('imu_only') else 'all_sensors'
            selection = folds[folds.person.eq(who) & folds.method.eq(reference)]
            if len(selection) != 1:
                raise ValueError('Missing unique training-only setting')
            cfg = selection.iloc[0].to_dict()
            model = base.make_model(cfg)
            x_train = np.stack([features[p][rep] for p in train.file])
            x_test = np.stack([features[p][rep] for p in test.file])
            pred = model.fit(x_train, train.label).predict(x_test)
            if rep in {'baseline_all', 'baseline_imu'}:
                old = previous[previous.person.eq(who) & previous.method.eq(reference)].set_index('file')
                np.testing.assert_array_equal(pred, old.loc[test.file, 'prediction'])
            rows = test[['person', 'session_id', 'trial_id', 'file', 'label']].copy()
            rows['prediction'], rows['correct'] = pred, test.label.to_numpy() == pred
            rows['representation'] = rep
            rows['reference_setting_family'] = reference
            rows['training_people'] = '|'.join(base.people_order(train.person))
            rows['k'], rows['p'], rows['weights'] = cfg['k'], cfg['p'], cfg['weights']
            rows['features'] = x_train.shape[1]
            outputs.append(rows)
        if progress:
            progress(who)
    result = pd.concat(outputs, ignore_index=True)
    for rep, rows in result.groupby('representation'):
        if len(rows) != len(frame) or rows.file.duplicated().any() or set(rows.file) != set(frame.file):
            raise ValueError('Trials dropped/duplicated in ' + rep)
    return result


def summarize(rows, labels):
    summaries, per_sign, per_person = [], [], []
    for rep, group in rows.groupby('representation', sort=False):
        cm = confusion_matrix(group.label, group.prediction, labels=labels)
        summaries.append(dict(representation=rep, trials=len(group), correct=int(group.correct.sum()),
                              accuracy=float(group.correct.mean()),
                              person_mean=float(group.groupby('person').correct.mean().mean()),
                              features=int(group.features.iloc[0]), confusion_matrix=cm.tolist(), labels=labels))
        for i, label in enumerate(labels):
            per_sign.append(dict(representation=rep, label=label, trials=int(cm[i].sum()),
                                 correct=int(cm[i, i]), recall=float(cm[i, i]/cm[i].sum())))
        for person, part in group.groupby('person'):
            per_person.append(dict(representation=rep, person=person, trials=len(part),
                                   correct=int(part.correct.sum()), accuracy=float(part.correct.mean())))
    return summaries, per_sign, per_person


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--calibration-commit', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh directory; no model or CSV overwrites')
    frames, references, folds, hashes, unique = load_inputs(args.results_root)
    args.out.mkdir(parents=True)
    for p in [Path(__file__), Path(comp.__file__), Path(base.__file__)]:
        hashes[str(p)] = sha(p)
    plan = dict(exploratory=True, prospective_external_validation=False,
                cohorts={k: dict(trials=len(v), people=v.person.nunique()) for k, v in frames.items()},
                calibration_commit=args.calibration_commit, representations=comp.REPRESENTATIONS,
                fsr_scale_floor_adc=comp.FSR_SCALE_FLOOR,
                setting_policy='Reuse previous inner-training-only selections by outer person; never tune new representations here',
                normalization_scope='Each completed whole trial and hand only; no other trial, label or test population statistics',
                baseline_policy='Verify original all-sensor AND IMU-only predictions exactly',
                rest_policy='Strict same-person/session/account/hand/version previous rest match; do not apply unverified records',
                no_missing_or_low_accuracy_exclusions=True, input_sha256=hashes)
    base.dump_json(args.out/'comparison_plan.json', plan)
    features, training, calibration = load_real_data(args.repo, unique, args.calibration_commit)
    links = rest_links(training, calibration)
    pd.DataFrame(calibration).to_csv(args.out/'calibration_inventory.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(links).to_csv(args.out/'calibration_links.csv', index=False, encoding='utf-8-sig')
    audit = dict(calibration_files=len({c['file'] for c in calibration}),
                 calibration_hand_records=len(calibration),
                 calibration_date_range=[min(c['start'] for c in calibration), max(c['end'] for c in calibration)],
                 explicit_rest_files=len({c['file'] for c in calibration if c['is_rest']}),
                 unique_training_trials=len(features), training_hand_records=len(training),
                 link_status=dict(Counter(c['status'] for c in links)),
                 calibration_applied=False)
    base.dump_json(args.out/'calibration_audit.json', audit)
    print('Calibration audit:', json.dumps(audit), flush=True)
    all_rows, all_scores, sign_rows, person_rows = [], [], [], []
    with threadpool_limits(limits=1):
        for name, frame in frames.items():
            rows = evaluate_cohort(frame, features, references[name], folds[name],
                                   lambda who: print(name, who, 'completed', flush=True))
            labels = list(dict.fromkeys(frame.label))
            scores, signs, people = summarize(rows, labels)
            rows['cohort'] = name
            all_rows.append(rows)
            all_scores.extend(dict(cohort=name, **s) for s in scores)
            sign_rows.extend(dict(cohort=name, **s) for s in signs)
            person_rows.extend(dict(cohort=name, **s) for s in people)
    pd.concat(all_rows, ignore_index=True).to_csv(args.out/'predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([{k:v for k,v in s.items() if k not in {'labels', 'confusion_matrix'}} for s in all_scores]).to_csv(
        args.out/'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(sign_rows).to_csv(args.out/'per_sign.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(person_rows).to_csv(args.out/'per_person.csv', index=False, encoding='utf-8-sig')
    base.dump_json(args.out/'evaluation.json', all_scores)
    for path, value in hashes.items():
        if sha(path) != value:
            raise ValueError('Existing input changed')
    base.dump_json(args.out/'verification.json', dict(raw_files_sha256_verified=len(unique),
                   all_baseline_features_and_predictions_reproduced=True,
                   heldout_person_not_in_training=True, same_trials_every_representation=True,
                   existing_inputs_unchanged=True, raw_or_model_files_written=False,
                   new_features_calibrated_to_physical_force=False, deployed=False))
    print(pd.read_csv(args.out/'comparison.csv').to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
