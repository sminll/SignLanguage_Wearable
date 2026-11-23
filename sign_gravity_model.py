"""Gravity-frame IMU sign classification, feature_version=1.

One CSV is one trial. No left/right joining across files, firmware changes,
deployment, random augmentation, or replacement of an existing model occurs.
All model selection uses training people only. Session normalization is an
explicit transductive evaluation reference, never a deployable transform.
Dependencies: numpy, pandas, scikit-learn, joblib, plus Python standard library.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 20261122
FEATURE_VERSION = 1
SENSORS = [f'fsr{i}' for i in range(1, 7)] + ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
REQUIRED = {'subject_id', 'session_id', 'trial_id', 'label', 'board_type', 'source',
            'record_kind', 'hand', 'sample_seq', 'time_ms', 'imu_ok', 'time', *SENSORS}
ALIASES = {'괜찮아':'괜찮다', '아니다 (아니, 부인(하다))':'아니다', '무엇(어디)':'무엇',
           '싫다(거부)':'싫다', '싫다(거절)':'싫다'}
SIX = ['괜찮다', '마시다', '무엇', '싫다', '아니다', '화장실']
THREE = ['교통사고', '삼키다', '상처']
CONFIGS = ['lr_c0.01', 'lr_c0.05', 'lr_c0.2', 'extratrees', 'svm_c1', 'svm_c10', 'lr0.05+extratrees']


class SkipTrial(ValueError):
    """A capture is unusable; the reason is recorded, never silently repaired."""


def canonical(label):
    value = str(label).strip()
    return ALIASES.get(value, value)


def normalize_person(value):
    value = str(value).strip()
    match = re.fullmatch(r'subject-?(\d+)', value, re.I)
    return 'subject' + str(int(match[1])) if match else value


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()


def json_write(path, value):
    # JSON readers here use utf-8-sig too, as requested for all result files.
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8-sig')


def signs_argument(value):
    if value == '6':
        return SIX[:]
    if value == '3':
        return THREE[:]
    if value == '9':
        return THREE + SIX
    signs = [canonical(s) for s in value.split(',') if s.strip()]
    if len(signs) < 2 or len(signs) != len(set(signs)):
        raise ValueError('Provide at least two distinct signs')
    return signs


def exclusion_paths(repo, explicit=None):
    excluded = set()
    metadata = Path(repo)/'data/training_exclusions.json'
    if metadata.exists():
        current = json.loads(metadata.read_text(encoding='utf-8-sig')).get('files', [])
        for item in current:
            value = item.get('path') if isinstance(item, dict) else item
            if isinstance(value, str):
                excluded.add(value.strip().replace('\\', '/').removeprefix('./'))
    if explicit:
        for line in Path(explicit).read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                excluded.add(line.strip().replace('\\', '/').removeprefix('./'))
    return excluded


def sources(repo='.', ref='origin/training-data', data_dir=None):
    if data_dir is not None:
        parent = Path(data_dir).resolve()
        directory = parent/'sensor-data/training'
        if not directory.is_dir():
            raise ValueError('--data-dir must contain sensor-data/training')
        try:
            commit = subprocess.check_output(['git', '-C', str(parent), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        except subprocess.CalledProcessError:
            commit = None
        return commit, [(p.relative_to(parent).as_posix(), p.read_bytes()) for p in sorted(directory.rglob('*.csv'))]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', '--verify', '--end-of-options', ref+'^{commit}'], text=True).strip()
    blob = subprocess.check_output(['git', '-C', str(repo), 'archive', commit, 'sensor-data/training'])
    items = []
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for item in archive:
            if item.isfile() and item.name.endswith('.csv'):
                items.append((item.name, archive.extractfile(item).read()))
    return commit, sorted(items)


def read_trial(blob):
    try:
        frame = pd.read_csv(io.BytesIO(blob), keep_default_na=False)
    except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
        raise SkipTrial('csv_parse') from exc
    if not REQUIRED <= set(frame.columns):
        raise SkipTrial('missing_columns')
    if frame.empty:
        raise SkipTrial('empty_csv')
    for column, expected in [('board_type', 'XIAO'), ('source', 'REAL'), ('record_kind', 'training')]:
        if not frame[column].astype(str).str.strip().eq(expected).all():
            raise SkipTrial('not_'+expected.lower())
    identities = {}
    for column in ['subject_id', 'session_id', 'trial_id', 'label']:
        values = frame[column].astype(str).str.strip()
        if values.nunique() != 1 or not values.iloc[0]:
            raise SkipTrial('conflicting_or_empty_'+column)
        identities[column] = values.iloc[0]
    if not set(frame.hand) <= {'left', 'right'}:
        raise SkipTrial('invalid_hand')
    frame = frame.drop_duplicates(['hand', 'sample_seq'], keep='first')
    if set(frame.hand) != {'left', 'right'}:
        raise SkipTrial('missing_hand')
    hands = []
    for name in ['left', 'right']:
        part = frame[frame.hand.eq(name)].copy()
        try:
            part['time_ms'] = pd.to_numeric(part.time_ms, errors='raise')
            part = part.sort_values('time_ms', kind='stable')
            time = part.time_ms.to_numpy(float)
            values = part[SENSORS].to_numpy(float)
            imu_ok = pd.to_numeric(part.imu_ok, errors='coerce').to_numpy(float)
        except (ValueError, TypeError) as exc:
            raise SkipTrial('nonnumeric_sensors_or_clock') from exc
        if len(part) < 30:
            raise SkipTrial('too_few_samples')
        if not np.isfinite(time).all() or not np.isfinite(values).all():
            raise SkipTrial('nonfinite_sensors_or_clock')
        if (np.diff(time) <= 0).any():
            raise SkipTrial('nonincreasing_clock')
        if time[-1]-time[0] < 3000:
            raise SkipTrial('short_duration')
        if np.diff(time).max() > 500:
            raise SkipTrial('large_time_gap')
        if np.mean(imu_ok != 1) > .05:
            raise SkipTrial('imu_invalid_over_5_percent')
        hands.append(values[:, 6:12])  # FSR checked for finiteness, not used as a feature.
    stamp = pd.to_datetime(frame.time.iloc[0], errors='coerce', utc=True)
    date = stamp.tz_convert('Asia/Seoul').date().isoformat() if not pd.isna(stamp) else ''
    metadata = dict(person=normalize_person(identities['subject_id']), session=identities['session_id'],
                    trial_id=identities['trial_id'], label=canonical(identities['label']), date=date)
    return metadata, hands


def rotation_to_up(g0):
    norm = np.linalg.norm(g0)
    if not np.isfinite(norm) or norm < 1e-12:
        raise SkipTrial('undefined_gravity_reference')
    a = g0/norm
    b = np.array([0., 0., 1.])
    cross = np.cross(a, b)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(np.dot(a, b), -1., 1.))
    if sine < 1e-12:
        return np.eye(3) if cosine >= 0 else np.diag([1., -1., -1.])
    axis = cross/sine
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    return np.eye(3) + sine*skew + (1.-cosine)*(skew@skew)


def stats(x):
    return np.concatenate([x.mean(axis=0), x.std(axis=0, ddof=0), x.min(axis=0), x.max(axis=0)])


def phased(x):
    return np.concatenate([stats(x), *[stats(part) for part in np.array_split(x, 3)]])


def hand_features(values, axis_baseline=False):
    if axis_baseline:
        return phased(values)
    acc, gyr = values[:, :3], values[:, 3:6]
    gm = np.linalg.norm(gyr, axis=1)
    quiet = gm <= np.percentile(gm, 30)
    g0 = acc[quiet].mean(axis=0)
    rotation = rotation_to_up(g0)
    A, W = acc@rotation.T, gyr@rotation.T
    gf = np.column_stack([A[:, 2], np.hypot(A[:, 0], A[:, 1]), W[:, 2], np.hypot(W[:, 0], W[:, 1]),
                          np.linalg.norm(acc-g0, axis=1), gm])
    threshold = max(10., float(np.percentile(gm, 20))*2+5.)
    active = np.flatnonzero(gm > threshold)
    start, end = 0, len(values)
    if len(active) >= 2:
        start, end = max(0, int(active[0])-2), min(len(values), int(active[-1])+3)
        if end-start < 8:
            start, end = 0, len(values)
    window = gf[start:end]
    resampled = np.column_stack([np.interp(np.linspace(0., 1., 12), np.linspace(0., 1., len(window)), window[:, j]) for j in range(6)])
    result = np.concatenate([phased(gf), phased(window), [(end-start)/len(values), float(np.mean(gm > 30))], resampled.ravel()])
    if result.shape != (266,) or not np.isfinite(result).all():
        raise SkipTrial('invalid_gravity_features')
    return result


def trial_features(hands, axis_baseline=False):
    result = np.concatenate([hand_features(hand, axis_baseline) for hand in hands])
    if result.shape != ((192,) if axis_baseline else (532,)):
        raise ValueError('Unexpected feature count')
    return result


def load_dataset(repo='.', ref='origin/training-data', data_dir=None, exclude_paths=None, signs=None, axis_baseline=False):
    commit, items = sources(repo, ref, data_dir)
    excluded = exclusion_paths(repo, exclude_paths)
    counts = Counter()
    rows, arrays = [], []
    for path, blob in items:
        if path in excluded:
            counts['excluded_current_path'] += 1
            continue
        try:
            meta, hands = read_trial(blob)
            if signs is not None and meta['label'] not in signs:
                counts['outside_requested_signs'] += 1
                continue
            vector = trial_features(hands, axis_baseline)
        except SkipTrial as exc:
            counts[str(exc)] += 1
            continue
        rows.append(dict(file=path, sha256=sha256(blob), **meta))
        arrays.append(vector)
    if not rows:
        raise ValueError('No eligible trials: '+json.dumps(dict(counts), ensure_ascii=False))
    cohort = pd.DataFrame(rows)
    if signs is not None and set(cohort.label) != set(signs):
        raise ValueError('No eligible data for signs: '+','.join(sorted(set(signs)-set(cohort.label))))
    return cohort, np.stack(arrays), dict(commit=commit, ref=ref, data_dir=str(Path(data_dir).resolve()) if data_dir else None,
        local_files_not_verified_against_git=bool(data_dir), skip_reasons=dict(counts), source_csv_count=len(items),
        excluded_paths=sorted(excluded), raw_modified=False)


def candidates():
    def lr(c):
        params = dict(C=c, max_iter=5000, random_state=SEED)
        if 'n_jobs' in inspect.signature(LogisticRegression).parameters:
            params['n_jobs'] = 1
        return make_pipeline(StandardScaler(), LogisticRegression(**params))
    trees = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=1)
    return [('lr_c0.01', lr(.01)), ('lr_c0.05', lr(.05)), ('lr_c0.2', lr(.2)), ('extratrees', trees),
            ('svm_c1', make_pipeline(StandardScaler(), SVC(C=1, gamma='scale', random_state=SEED))),
            ('svm_c10', make_pipeline(StandardScaler(), SVC(C=10, gamma='scale', random_state=SEED))),
            ('lr0.05+extratrees', VotingClassifier([('lr', lr(.05)), ('trees', clone(trees))], voting='soft', n_jobs=1))]


def select_config(X, y, people):
    if len(set(people)) < 3:
        raise ValueError('At least three training people required for GroupKFold(3)')
    splits = list(GroupKFold(3).split(X, y, people))
    results = {}
    for name, estimator in candidates():
        predicted = np.full(len(y), '', dtype=object)
        for fit, test in splits:
            if set(people[fit]) & set(people[test]):
                raise AssertionError('Person leakage in inner validation')
            fitted = clone(estimator).fit(X[fit], y[fit])
            predicted[test] = fitted.predict(X[test])
        results[name] = float(balanced_accuracy_score(y, predicted))
    # max retains insertion order on an exact tie.
    selected = max(results, key=results.get)
    return selected, results


def session_normalize(X, cohort, signs):
    for _, rows in cohort.groupby(['person', 'session'], sort=False):
        if set(rows.label) != set(signs):
            raise ValueError('--session-norm refused: sign composition differs across sessions; '+
                             f'{rows.person.iloc[0]}/{rows.session.iloc[0]} contains '+','.join(sorted(set(rows.label))))
    normalized = np.empty_like(X)
    for indices in cohort.groupby(['person', 'session'], sort=False).indices.values():
        values = X[indices]
        normalized[indices] = (values-values.mean(axis=0))/(values.std(axis=0, ddof=0)+1e-6)
    return normalized


def versions():
    return dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)


def make_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def evaluate(cohort, X, info, signs, out, session_norm=False, axis_baseline=False):
    if cohort.person.nunique() < 5:
        raise ValueError('At least five people required for outer GroupKFold(5)')
    if session_norm:
        X = session_normalize(X, cohort, signs)
    out = make_output(out)
    labels, people = cohort.label.to_numpy(str), cohort.person.to_numpy(str)
    predicted = np.full(len(labels), '', dtype=object)
    fold_ids = np.zeros(len(labels), int)
    selected_rows = []
    for fold, (fit, test) in enumerate(GroupKFold(5).split(X, labels, people), 1):
        if set(people[fit]) & set(people[test]):
            raise AssertionError('Person leakage in outer validation')
        selected, scores = select_config(X[fit], labels[fit], people[fit])
        estimator = clone(dict(candidates())[selected]).fit(X[fit], labels[fit])
        predicted[test] = estimator.predict(X[test])
        fold_ids[test] = fold
        selected_rows.append(dict(fold=fold, test_people=';'.join(sorted(set(people[test]))),
            selected_config=selected, train_people=';'.join(sorted(set(people[fit]))), **scores))
        # Checkpoint each completed outer fold; a later failure remains visible.
        pd.DataFrame(selected_rows).to_csv(out/'selected_configs.csv', index=False, encoding='utf-8-sig')
        print(f'fold {fold}/5: {selected}; test people {sorted(set(people[test]))}; '+
              f'{int(np.sum(predicted[test]==labels[test]))}/{len(test)}', flush=True)
    if (fold_ids == 0).any():
        raise AssertionError('Uncovered evaluation rows')
    predictions = cohort.copy()
    predictions['prediction'], predictions['correct'], predictions['fold'] = predicted, predicted == labels, fold_ids
    predictions.to_csv(out/'predictions.csv', index=False, encoding='utf-8-sig')
    for key, name in [('label', 'per_sign'), ('person', 'per_person')]:
        table = predictions.groupby(key, sort=True).correct.agg(trials='size', correct='sum', accuracy='mean').reset_index()
        table.to_csv(out/(name+'.csv'), index=False, encoding='utf-8-sig')
    pd.DataFrame(confusion_matrix(labels, predicted, labels=signs), index=signs, columns=signs).rename_axis('actual\\predicted').to_csv(out/'confusion.csv', encoding='utf-8-sig')
    summary = dict(**info, signs=signs, trials=len(labels), people=cohort.person.nunique(),
        correct=int(np.sum(predicted == labels)), accuracy=float(accuracy_score(labels, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(labels, predicted)), chance_level=1/len(signs),
        majority_class_baseline=float(cohort.label.value_counts(normalize=True).max()), versions=versions(),
        feature_version=FEATURE_VERSION, feature_mode='axis_phased' if axis_baseline else 'gravity_frame',
        feature_count=X.shape[1], random_state=SEED, n_jobs=1, candidate_order=CONFIGS,
        session_norm=session_norm, session_norm_uses_entire_test_session=bool(session_norm),
        procedure='Outer GroupKFold(5), default order, groups=normalized wearer. Inner GroupKFold(3) '+
            'on outer training people only; pooled inner balanced_accuracy selects candidate, first on tie. '+
            'Scalers fit within each training fold; no outer test trial used for model selection.',
        retrospective_repeatedly_analyzed_data=True, prospective_or_rewear_accuracy=None,
        deployed=False, existing_models_replaced=False, created_at_utc=datetime.now(timezone.utc).isoformat())
    json_write(out/'summary.json', summary)
    print(json.dumps(dict(correct=summary['correct'], trials=len(labels), accuracy=summary['accuracy'],
                          balanced_accuracy=summary['balanced_accuracy']), ensure_ascii=False), flush=True)
    return summary


def train(cohort, X, info, signs, out):
    if cohort.person.nunique() < 3:
        raise ValueError('At least three people required for configuration selection')
    out = make_output(out)
    labels, people = cohort.label.to_numpy(str), cohort.person.to_numpy(str)
    selected, scores = select_config(X, labels, people)
    estimator = clone(dict(candidates())[selected]).fit(X, labels)
    metadata = dict(**info, signs=signs, trials=len(cohort), people=cohort.person.nunique(),
        selected_config=selected, inner_scores=scores, feature_version=FEATURE_VERSION, feature_mode='gravity_frame',
        feature_count=X.shape[1], versions=versions(), random_state=SEED,
        training_accuracy_not_reported=True, prospective_accuracy=None, deployed=False, existing_models_replaced=False)
    joblib.dump(dict(model=estimator, info=metadata, feature_version=FEATURE_VERSION), out/'model.joblib')
    cohort.to_csv(out/'training_manifest.csv', index=False, encoding='utf-8-sig')
    json_write(out/'model_info.json', metadata)
    print('Saved a separate undeployed candidate:', out/'model.joblib', flush=True)


def predict(model_path, paths):
    # Load only your own trusted joblib artifact; this command does not train.
    bundle = joblib.load(model_path)
    if bundle.get('feature_version') != FEATURE_VERSION or bundle['info'].get('feature_mode') != 'gravity_frame':
        raise ValueError('Unsupported model feature schema')
    results = []
    for path in paths:
        try:
            blob = Path(path).read_bytes()
            meta, hands = read_trial(blob)
            vector = trial_features(hands)
            prediction = str(bundle['model'].predict(vector[None])[0])
            results.append(dict(file=str(path), sha256=sha256(blob), person=meta['person'],
                session=meta['session'], prediction=prediction, status='predicted'))
        except (SkipTrial, OSError) as exc:
            results.append(dict(file=str(path), status='skipped', reason=str(exc)))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def export_android(cohort, X, info, signs, out, repo='.'):
    """Export a fixed LR, with numeric parity fixtures; never replace a model."""
    expected = {'삼키다': 260, '아니다': 226, '상처': 239}
    if set(signs) == set(expected) and cohort.label.value_counts().to_dict() != expected:
        raise ValueError('Android demo cohort must be 725 trials: '+str(expected))
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=.05, max_iter=5000, random_state=SEED))
    estimator.fit(X, cohort.label.to_numpy(str))
    scaler, lr = estimator.steps[0][1], estimator.steps[1][1]
    model = dict(version=1, signs=lr.classes_.tolist(), feature_count=int(X.shape[1]),
        mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=lr.coef_.tolist(),
        intercept=lr.intercept_.tolist(), source_repo='sminll/SignLanguage_Wearable',
        source_ref=str(info['ref']).removeprefix('origin/'), source_commit=info['commit'],
        trained_trials=len(cohort), created=datetime.now(timezone.utc).isoformat(),
        feature_version=FEATURE_VERSION, model_type='StandardScaler + LogisticRegression(C=0.05)',
        note='삼키다·상처는 11/4~7, 아니다는 11/13~15 측정 자료입니다. '+
             '같은 날 세 동작을 함께 측정한 검증이 필요합니다. '+
             'iOS 기기 학습 KNN(k=5)과 달리 PC에서 학습한 고정 로지스틱 회귀입니다. '+
             '앱 실시간·재착용 정확도는 미측정이며 softmax는 보정된 정답 확률이 아닙니다.')
    # Choose actual examples across classes, then fill to five without duplication.
    indices = list(dict.fromkeys(int(cohort.index[cohort.label.eq(s)][0]) for s in lr.classes_))
    indices = (indices + [i for i in range(len(cohort)) if i not in indices])[:5]
    fixtures = []
    raw = {}
    for number, i in enumerate(indices):
        row = cohort.iloc[i]
        if info.get('data_dir'):
            blob = (Path(info['data_dir'])/row.file).read_bytes()
        else:
            blob = subprocess.check_output(['git', '-C', str(repo), 'show', info['commit']+':'+row.file])
        if sha256(blob) != row.sha256:
            raise ValueError('Parity fixture differs from training source: '+row.file)
        name = f'parity_{number+1}.csv'
        raw[name] = blob
        fixtures.append(dict(csv=name, source_path=row.file, sha256=row.sha256,
            features=X[i].tolist(), probabilities=estimator.predict_proba(X[i:i+1])[0].tolist(),
            prediction=str(estimator.predict(X[i:i+1])[0])))
    # All validation/training is finished before creating a fresh output directory.
    out = make_output(out)
    json_write(out/'sign_model.json', model)
    json_write(out/'parity_vectors.json', dict(feature_count=int(X.shape[1]), signs=model['signs'], vectors=fixtures,
        test_purpose_only=True, experimental_accuracy_not_measured=True))
    for name, blob in raw.items():
        (out/name).write_bytes(blob)
    print(f'Exported Android numeric model: {len(cohort)} trials; {len(fixtures)} parity captures; {out}', flush=True)
    return model


def selfcheck(cohort, items, out):
    """Real-capture invariance plus synthetic numerical edge checks, no accuracy."""
    if Path(out).exists():
        raise FileExistsError(out)
    wanted = set(cohort.file)
    rng = np.random.default_rng(SEED)
    maximum = 0.; tested = 0
    for path, blob in items:
        if path not in wanted:
            continue
        _, hands = read_trial(blob)
        for values in hands:
            reference = hand_features(values)
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            transformed = np.column_stack([values[:, :3]@q.T, values[:, 3:6]@q.T])
            actual = hand_features(transformed)
            maximum = max(maximum, float(np.max(np.abs(actual-reference))))
            np.testing.assert_allclose(actual, reference, rtol=1e-8, atol=1e-8)
            tested += 1
    np.testing.assert_array_equal(rotation_to_up([0, 0, 1]), np.eye(3))
    np.testing.assert_array_equal(rotation_to_up([0, 0, -1]), np.diag([1, -1, -1]))
    out = make_output(out)
    result = dict(real_hand_captures_checked=tested, one_random_proper_rotation_per_capture=True,
        seed=SEED, max_absolute_feature_difference=maximum, rtol=1e-8, atol=1e-8,
        rotation_invariance_passed=True, plus_z_and_minus_z_branches_passed=True,
        synthetic_values_used_for_experiment_accuracy=False)
    json_write(out/'rotation_invariance.json', result)
    print(json.dumps(result), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ['evaluate', 'train', 'selfcheck', 'export-android']:
        p = sub.add_parser(command)
        p.add_argument('--repo', default='.')
        p.add_argument('--ref', default='origin/training-data')
        p.add_argument('--data-dir', type=Path)
        p.add_argument('--exclude-paths', type=Path)
        p.add_argument('--signs', default='9')
        p.add_argument('--out', required=True, type=Path)
        if command == 'evaluate':
            p.add_argument('--session-norm', action='store_true')
            p.add_argument('--axis-baseline', action='store_true', help='Comparison only: original-axis phased features (192 dimensions)')
    p = sub.add_parser('predict')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('csv', nargs='+', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'predict':
        return predict(args.model, args.csv)
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite existing --out: '+str(args.out))
    signs = signs_argument(args.signs)
    axis = getattr(args, 'axis_baseline', False)
    cohort, X, info = load_dataset(args.repo, args.ref, args.data_dir, args.exclude_paths, signs, axis)
    print(f'Eligible: {len(cohort)} trials, {cohort.person.nunique()} people, {len(signs)} signs; commit={info["commit"]}', flush=True)
    if args.command == 'evaluate':
        return evaluate(cohort, X, info, signs, args.out, args.session_norm, axis)
    if args.command == 'train':
        return train(cohort, X, info, signs, args.out)
    if args.command == 'export-android':
        return export_android(cohort, X, info, signs, args.out, args.repo)
    _, items = sources(args.repo, args.ref, args.data_dir)
    return selfcheck(cohort, items, args.out)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileExistsError, subprocess.CalledProcessError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        raise SystemExit(2)
