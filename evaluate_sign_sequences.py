"""Frozen single-hand KNN evaluation of explicitly recorded two-window attempts.

Never trains, pairs historical sessions implicitly, or estimates joint accuracy
from marginal accuracies. See docs/sign-sequence-evaluation.md.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from train_sign_knn import SENSORS, build_features

FEATURES = [f'{s}_{stat}' for s in SENSORS for stat in ('mean', 'std', 'min', 'max')]
MANIFEST_COLUMNS = [
    'participant_id', 'attempt_id', 'step', 'actual_label', 'csv_path',
    'session_id', 'trial_id', 'hand', 'recording_id', 'window_start_ms',
    'window_end_ms', 'status',
]
RULES = {('머리', '아프다'): '머리가 아픕니다'}
MODES = ('new_participant', 'same_participant_new_session', 'exploratory')
RESULT_COLUMNS = [
    'participant_id', 'attempt_id', 'actual_1', 'predicted_1', 'correct_1',
    'actual_2', 'predicted_2', 'correct_2', 'combination_success',
    'predicted_sentence', 'valid', 'exclusion_reason', 'model_sha256',
    'evaluation_mode', 'data_kind',
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def person(value):
    """subject8 and subject-8 identify the same participant in this project."""
    value = str(value).strip()
    match = re.fullmatch(r'subject-?(\d+)', value, re.I)
    return f'subject{int(match[1])}' if match else value


def load_model(path, labels):
    data = Path(path).read_bytes()
    bundle = joblib.load(io.BytesIO(data))  # Trusted local artifacts ONLY.
    if not isinstance(bundle, dict) or 'pipeline' not in bundle:
        raise ValueError('Unsupported model artifact: expected a pipeline dictionary.')
    model = bundle['pipeline']
    classes = set(map(str, model.classes_))
    if not set(labels).issubset(classes):
        raise ValueError(f'Model does not support requested signs {labels}; classes={sorted(classes)}')
    columns = bundle.get('columns', bundle.get('feature_columns'))
    if columns != FEATURES:
        raise ValueError('This evaluator requires the legacy 48-feature single-hand model; no substitution allowed.')
    if not {'imputer', 'scaler', 'knn'}.issubset(model.named_steps):
        raise ValueError('Expected the existing imputer/scaler/KNN pipeline.')
    if set(map(str, bundle.get('classes', classes))) != classes:
        raise ValueError('Model metadata classes disagree with the fitted classifier.')
    return bundle, sha(data), sorted(classes)


def training_provenance(bundle, model_path, features_path=None):
    """Conservative training-session superset; never guesses independence."""
    sessions = set(map(str, bundle.get('training_sessions', [])))
    participants = set()
    path = Path(features_path) if features_path else Path(model_path).with_name('trial_features.csv')
    digest = None
    if path.is_file():
        data = path.read_bytes()
        frame = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
        if not {'session_id', 'subject_id'}.issubset(frame):
            raise ValueError('Training features need session_id and subject_id.')
        sessions.update(frame.session_id[frame.session_id.ne('')])
        participants.update(person(x) for x in frame.subject_id if x)
        digest = {'path': str(path.resolve()), 'sha256': sha(data), 'rows': len(frame)}
    return sessions, participants, digest


def read_window(row, manifest_dir, hand, cache, hashes):
    if row['hand'] != hand:
        raise ValueError('model_hand_mismatch')
    path = (manifest_dir / row['csv_path']).resolve()
    if path not in cache:
        data = path.read_bytes()
        hashes[str(path)] = sha(data)
        raw = pd.read_csv(io.BytesIO(data), keep_default_na=False)
        # MATLAB local CSV schema vs normalized receiver/app schema.
        if 'device_id' in raw:
            if not raw.device_id.isin(['L', 'R']).all():
                raise ValueError('unknown_device_id')
            raw = raw.rename(columns={**{f'fsr{i}_raw': f'fsr{i}' for i in range(1, 7)},
                                      **dict(zip(['ax_g','ay_g','az_g','gx_dps','gy_dps','gz_dps'], SENSORS[6:])),
                                      'seq': 'sample_seq', 'source': 'trial_id'})
            raw['hand'] = raw.device_id.map({'L': 'left', 'R': 'right'})
            raw['source'] = 'REAL'
            raw['label'] = raw.trial_id.astype(str).str.replace(r'[0-9]+$', '', regex=True)
        cache[path] = raw
    raw = cache[path]
    required = {'subject_id','session_id','trial_id','hand','time_ms','source',
                'host_elapsed_ms','trial_elapsed_ms','sample_seq', *SENSORS}
    if not required.issubset(raw):
        raise ValueError('missing_sensor_or_timing_columns')
    block = raw.loc[raw.subject_id.map(person).eq(person(row['participant_id'])) &
                    raw.session_id.astype(str).eq(row['session_id']) &
                    raw.trial_id.astype(str).eq(row['trial_id']) & raw.hand.eq(hand)].copy()
    if len(block) < 8:
        raise ValueError('missing_or_short_window')
    if not block.source.eq('REAL').all():
        raise ValueError('non_REAL_samples')
    if 'label' in block and not block.label.eq(row['actual_label']).all():
        raise ValueError('raw_label_mismatch')
    if block.sample_seq.duplicated().any():
        raise ValueError('duplicate_sample_sequence')
    for c in [*SENSORS, 'time_ms','host_elapsed_ms','trial_elapsed_ms','sample_seq']:
        block[c] = pd.to_numeric(block[c], errors='coerce')
    if not np.isfinite(block[[*SENSORS,'time_ms','host_elapsed_ms','trial_elapsed_ms','sample_seq']].to_numpy()).all():
        raise ValueError('missing_or_nonfinite_sensor_values')
    if (block[SENSORS[:6]] < 0).any().any():
        raise ValueError('negative_FSR')
    if 'imu_ok' in block and not pd.to_numeric(block.imu_ok, errors='coerce').eq(1).all():
        raise ValueError('IMU_not_valid')
    clock = block.trial_elapsed_ms.to_numpy(float)
    device = block.time_ms.to_numpy(float)
    host = block.host_elapsed_ms.to_numpy(float)
    seq = block.sample_seq.to_numpy(float)
    if min(device[0], host[0]) < 0:
        raise ValueError('negative_sensor_clock')
    if any(np.any(np.diff(v) <= 0) for v in (clock, device, host, seq)):
        raise ValueError('nonmonotonic_clock_or_sequence')
    if np.any(seq != np.floor(seq)) or np.any(seq < 1):
        raise ValueError('invalid_sample_sequence')
    # Boundary coverage and maximum internal gap are predeclared, not tuned on predictions.
    if clock[0] < 0 or clock[0] > 500 or clock[-1] < 4500 or clock[-1] >= 5000:
        raise ValueError('incomplete_5s_window')
    if np.max(np.diff(clock)) > 500:
        raise ValueError('sensor_gap_over_500ms')
    # A reboot/clock drift must not create a seemingly complete host-time window.
    if np.max(np.abs((device-device[0])-(clock-clock[0]))) > 500:
        raise ValueError('device_host_clock_disagreement')
    if np.ptp(host-clock) > 1:
        raise ValueError('host_trial_clock_disagreement')
    block['subject_id'] = row['participant_id']
    block['label'] = row['actual_label']
    features, rejected = build_features(block)
    if len(features) != 1 or not rejected.empty:
        raise ValueError('feature_extraction_failed')
    return features[FEATURES]


def evaluate(manifest_path, model_path, output, *, labels=('머리', '아프다'),
             hand='left', mode='exploratory', data_kind='real', training_features=None):
    if mode not in MODES or data_kind not in ('real', 'synthetic'):
        raise ValueError('Invalid evaluation mode or data kind.')
    if len(labels) != 2:
        raise ValueError('Exactly two ordered signs are required.')
    if hand != 'left':
        raise ValueError('The verified legacy models were trained on left-hand data only.')
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new directory. No files overwritten.')
    bundle, model_hash, classes = load_model(model_path, labels)
    sessions, participants, provenance = training_provenance(bundle, model_path, training_features)
    if mode != 'exploratory' and (not sessions or not participants):
        raise ValueError('Independent evaluation requires training session AND participant provenance.')
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = pd.read_csv(io.BytesIO(manifest_bytes), dtype=str, keep_default_na=False)
    if not set(MANIFEST_COLUMNS).issubset(manifest):
        raise ValueError(f'Manifest requires columns: {MANIFEST_COLUMNS}')
    if 'data_kind' in manifest and not manifest.data_kind.eq(data_kind).all():
        raise ValueError('Manifest data_kind disagrees with --data-kind; synthetic data cannot be reported as real.')
    for col in MANIFEST_COLUMNS:
        manifest[col] = manifest[col].str.strip()
    if (manifest.participant_id.eq('') | manifest.attempt_id.eq('')).any():
        raise ValueError('Every row needs a participant and attempt ID, including failed attempts.')
    manifest['participant_id'] = manifest.participant_id.map(person)
    cache, hashes, results = {}, {}, []
    reused = manifest.duplicated(['session_id','trial_id','hand'], keep=False)
    overlap = set(manifest.session_id) & sessions
    if overlap and mode != 'exploratory':
        raise ValueError(f'Training/evaluation session overlap: {sorted(overlap)}')
    for (participant, attempt), rows in manifest.groupby(['participant_id','attempt_id'], sort=False):
        result = dict.fromkeys(RESULT_COLUMNS, '')
        result.update(participant_id=participant, attempt_id=attempt, valid=False,
                      model_sha256=model_hash, evaluation_mode=mode, data_kind=data_kind)
        try:
            if len(rows) != 2 or set(rows.step) != {'1','2'}:
                raise ValueError('missing_or_duplicate_step')
            rows = rows.set_index('step', drop=False).loc[['1','2']]
            for step in (1,2):
                result[f'actual_{step}'] = rows.loc[str(step), 'actual_label']
            if tuple(rows.actual_label) != tuple(labels):
                raise ValueError('unexpected_actual_sign_order')
            if rows[MANIFEST_COLUMNS].eq('').any().any():
                raise ValueError('missing_attempt_metadata')
            if not rows.status.eq('complete').all():
                raise ValueError('capture_not_complete:' + '|'.join(rows.status))
            if reused.loc[manifest.participant_id.eq(participant) & manifest.attempt_id.eq(attempt)].any():
                raise ValueError('window_reused_in_multiple_steps_or_attempts')
            if rows.recording_id.nunique() != 1:
                raise ValueError('different_recording_clocks')
            start = pd.to_numeric(rows.window_start_ms, errors='coerce').to_numpy(float)
            end = pd.to_numeric(rows.window_end_ms, errors='coerce').to_numpy(float)
            if not np.isfinite(np.r_[start,end]).all() or start[0] < 0:
                raise ValueError('invalid_window_schedule')
            if not np.allclose(end-start,5000,rtol=0,atol=1) or abs(start[1]-end[0]-3000)>1:
                raise ValueError('not_5s_rest3s_5s_schedule')
            if mode == 'new_participant' and participant in participants:
                raise ValueError('participant_was_used_in_training')
            if mode == 'same_participant_new_session' and participant not in participants:
                raise ValueError('participant_not_in_training')
            features = [read_window(row, manifest_path.parent, hand, cache, hashes)
                        for _, row in rows.iterrows()]
            prediction = bundle['pipeline'].predict(pd.concat(features, ignore_index=True))
            for step, label in enumerate(prediction, 1):
                result[f'predicted_{step}'] = str(label)
                result[f'correct_{step}'] = str(label) == labels[step-1]
            result['combination_success'] = result['correct_1'] and result['correct_2']
            result['predicted_sentence'] = RULES.get(tuple(map(str, prediction)), '')
            result['valid'] = True
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            result['exclusion_reason'] = str(exc)
        results.append(result)
    frame = pd.DataFrame(results, columns=RESULT_COLUMNS)
    valid = frame.loc[frame.valid.eq(True)]
    n = len(valid)
    success = int(valid.combination_success.eq(True).sum())
    summary = {
        'status': ('미측정' if not n else '테스트용 가상 데이터' if data_kind=='synthetic' else '측정 완료'),
        'data_kind': data_kind, 'evaluation_mode': mode,
        'evaluation_method': 'fixed_windows_5s_rest3s_5s_single_hand_frozen_KNN',
        'independence': 'not_claimed' if mode=='exploratory' or data_kind=='synthetic' else 'session_disjoint; participant_relation_checked',
        'model': str(Path(model_path).resolve()), 'model_sha256': model_hash,
        'model_classes': classes, 'model_hand': hand,
        'training_features_provenance': provenance,
        'training_sessions_conservative_superset': sorted(sessions),
        'training_participants': sorted(participants), 'overlapping_sessions': sorted(overlap),
        'ordered_truth': list(labels), 'total_attempts': len(frame), 'valid_attempts': n,
        'excluded_attempts': len(frame)-n, 'successes': success, 'failures': n-success,
        'first_sign_accuracy_pct': 100*float(valid.correct_1.eq(True).mean()) if n else None,
        'second_sign_accuracy_pct': 100*float(valid.correct_2.eq(True).mean()) if n else None,
        'combination_accuracy_pct': 100*success/n if n else None,
        'exclusion_reasons': dict(Counter(frame.loc[frame.valid.eq(False), 'exclusion_reason'])),
        'manifest_sha256': sha(manifest_bytes), 'input_sha256': hashes,
        'evaluator_sha256': sha(Path(__file__).read_bytes()),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__,
                     'matplotlib': matplotlib.__version__},
        'quality_policy': {'min_rows':8,'first_sample_max_ms':500,'last_sample_min_ms':4500,
                           'max_gap_ms':500,'device_clock_tolerance_ms':500},
    }
    if sha(Path(model_path).read_bytes()) != model_hash:
        raise ValueError('Model changed during evaluation.')
    for filename, digest in hashes.items():
        if sha(Path(filename).read_bytes()) != digest:
            raise ValueError(f'Sensor input changed during evaluation: {filename}')
    output.mkdir(parents=True)
    # Include the exact pairing plan and per-file hashes for reproducibility.
    (output/'manifest_snapshot.csv').write_bytes(manifest_bytes)
    frame.to_csv(output/'attempt_results.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([summary | {'exclusion_reasons': json.dumps(summary['exclusion_reasons'],ensure_ascii=False)}]).to_csv(
        output/'summary.csv', index=False, encoding='utf-8-sig')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    by_person = []
    for participant, rows in frame.groupby('participant_id'):
        good = rows.loc[rows.valid.eq(True)]
        by_person.append(dict(participant_id=participant, total_attempts=len(rows), valid_attempts=len(good),
                              successes=int(good.combination_success.eq(True).sum()),
                              combination_accuracy_pct=100*float(good.combination_success.eq(True).mean()) if len(good) else None))
    pd.DataFrame(by_person, columns=['participant_id','total_attempts','valid_attempts','successes','combination_accuracy_pct']).to_csv(
        output/'participant_summary.csv',index=False,encoding='utf-8-sig')
    rate = '미측정' if not n else f'{success}/{n} = {100*success/n:.2f}%'
    text = (f'# 고정 구간 두 수어 조합 평가\n\n상태: {summary["status"]}\n\n'
            f'실제 순서: {" + ".join(labels)} / 모델 손: {hand}\n\n'
            f'평가 구분: {mode}; {summary["independence"]}\n\n'
            f'전체 {len(frame)}회, 유효 {n}회, 제외 {len(frame)-n}회\n\n'
            f'조합 정확도: {rate}; 성공 {success}회, 실패 {n-success}회\n\n'
            f'첫 수어: {str(summary["first_sign_accuracy_pct"])+"%" if n else "미측정"} / '
            f'두 번째: {str(summary["second_sign_accuracy_pct"])+"%" if n else "미측정"}\n\n'
            f'제외 사유: {json.dumps(summary["exclusion_reasons"],ensure_ascii=False)}\n\n'
            f'모델: `{summary["model"]}`\n\nSHA256: `{model_hash}`\n\n'
            '각 수어와 조합은 동일한 유효 시도를 분모로 사용합니다. 오인식은 실패로 집계합니다.\n\n'
            '두 손 결합·자동 동작 구간 분할·휴식 거부·자유로운 연속 인식 성능은 평가하지 않습니다. '
            '이 프로그램은 학습하거나 모델을 배포하지 않습니다. 독립성은 제공된 학습 이력의 정확성에 의존합니다.\n')
    if data_kind == 'synthetic':
        text += '\n테스트 전용 가상 데이터입니다. 포스터 실험 결과로 사용하지 마세요.\n'
    (output/'report.md').write_text(text, encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.set_ylim(0,105)
    ax.set_ylabel('Accuracy (%)')
    if n:
        values = [summary[k] for k in ('first_sign_accuracy_pct','second_sign_accuracy_pct','combination_accuracy_pct')]
        bars = ax.bar(['Step 1','Step 2','Both correct'], values, color=['#748ba8','#748ba8','#2166ac'])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+1, f'{value:.2f}%', ha='center')
        ax.set_title(f'{"SYNTHETIC TEST ONLY" if data_kind=="synthetic" else "Measured fixed-window attempts"}\n'
                     f'{mode} | valid={n}, excluded={len(frame)-n} | both={success}/{n}')
    else:
        ax.text(.5,.5,'NOT MEASURED\nNo valid paired attempts',transform=ax.transAxes,ha='center')
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(output/'combination_accuracy.png', dpi=200)
    plt.close(fig)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--first', default='머리')
    p.add_argument('--second', default='아프다')
    p.add_argument('--model-hand', choices=['left'], default='left')
    p.add_argument('--mode', choices=MODES, default='exploratory')
    p.add_argument('--data-kind', choices=['real','synthetic'], default='real')
    p.add_argument('--training-features', type=Path)
    args = p.parse_args(argv)
    result = evaluate(args.manifest, args.model, args.output, labels=(args.first,args.second),
                      hand=args.model_hand, mode=args.mode, data_kind=args.data_kind,
                      training_features=args.training_features)
    attempts = pd.read_csv(args.output/'attempt_results.csv', keep_default_na=False)
    for _, row in attempts.iterrows():
        prefix = f'{row.participant_id} / {row.attempt_id}'
        if str(row.valid).lower() == 'true':
            print(f'{prefix} | 실제: {row.actual_1} + {row.actual_2} | '
                  f'AI: {row.predicted_1} + {row.predicted_2} | '
                  f'{"성공" if str(row.combination_success).lower()=="true" else "실패"} | {row.predicted_sentence}')
        else:
            print(f'{prefix} | 제외: {row.exclusion_reason}')
    score = '미측정' if result['combination_accuracy_pct'] is None else f'{result["combination_accuracy_pct"]:.2f}%'
    print(f'{result["status"]}: both={result["successes"]}/{result["valid_attempts"]}; '
          f'excluded={result["excluded_attempts"]}; accuracy={score}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f'평가 중단: {exc}', file=sys.stderr)
        sys.exit(1)
