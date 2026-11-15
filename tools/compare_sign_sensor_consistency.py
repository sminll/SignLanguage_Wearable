"""Direct between-person FSR/IMU waveform comparison, not a KNN accuracy score.

Resample each hand on its own relative capture progress; do not synchronize board
boot clocks. Compare equal-person templates and preserve all original captures.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np
import pandas as pd

try:
    from . import evaluate_six_sign_knn as base
except ImportError:
    import evaluate_six_sign_knn as base

POINTS = 50
SHAPE_MIN_RMS = .05


def waveform(frame):
    if set(frame.hand) != {'left', 'right'}:
        raise ValueError('Missing hand; never zero-fill')
    waves, flags = [], {}
    for hand in ['left', 'right']:
        part = frame[frame.hand.eq(hand)]
        values = part[base.SENSORS].to_numpy(float)
        times = part.time_ms.to_numpy(float)
        if len(part) < 8 or not np.isfinite(values).all() or not np.isfinite(times).all():
            raise ValueError('Invalid sensor capture')
        if (np.diff(times) <= 0).any() or times[-1]-times[0] < 3000:
            raise ValueError('Invalid hand time')
        t = (times-times[0])/(times[-1]-times[0])
        waves.append(np.column_stack([np.interp(np.linspace(0, 1, POINTS), t, values[:,i])
                                      for i in range(len(base.SENSORS))]))
        # Zero flags are measured on every ORIGINAL row, not on interpolation.
        flags[hand+'_fsr_all_zero'] = bool((values[:,:6] == 0).all())
        flags[hand+'_fsr_zero_channels'] = int((values[:,:6] == 0).all(axis=0).sum())
        flags[hand+'_fsr_constant_channels'] = int((np.ptp(values[:,:6], axis=0) == 0).sum())
    return np.stack(waves), flags


def rms_distance(a, b):
    return float(np.sqrt(np.mean((a-b)**2)))


def shape_cosine(a, b):
    # Subtract each channel's temporal mean. Constant/inactive traces are not
    # awarded perfect similarity just because both are all zero.
    a = a-a.mean(axis=1, keepdims=True)
    b = b-b.mean(axis=1, keepdims=True)
    if np.sqrt(np.mean(a*a)) < SHAPE_MIN_RMS or np.sqrt(np.mean(b*b)) < SHAPE_MIN_RMS:
        return None
    return float(np.clip(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)), -1, 1))


def compare_templates(templates):
    keys = sorted(templates)
    stack = np.stack([templates[k] for k in keys])
    # Descriptive pooled scale: one weight per person x sign template, not one
    # weight per frame/trial. NOT a scaler used to fit/evaluate a classifier.
    scale = stack.std(axis=(0,2), ddof=0)
    scale[scale < 1e-12] = 1
    normalized = {k: templates[k]/scale[:,None,:] for k in keys}
    labels = sorted({k[0] for k in keys})
    grand = {label: np.mean([v for (s,_),v in normalized.items() if s == label], axis=0) for label in labels}
    summaries, pairs = [], []
    groups = {'FSR':slice(0,6), 'IMU':slice(6,12)}
    for label in labels:
        members = [(who,v) for (s,who),v in normalized.items() if s == label]
        row = dict(label=label, people=len(members), pairs=len(members)*(len(members)-1)//2)
        for name, columns in groups.items():
            distances, shapes = [], []
            for (who_a,a),(who_b,b) in itertools.combinations(members,2):
                distance = rms_distance(a[:,:,columns], b[:,:,columns])
                shape = shape_cosine(a[:,:,columns], b[:,:,columns])
                distances.append(distance)
                if shape is not None:
                    shapes.append(shape)
                pairs.append(dict(label=label, sensor=name, person_a=who_a, person_b=who_b,
                                  scaled_waveform_distance=distance, centered_shape_cosine=shape,
                                  left_distance=rms_distance(a[0,:,columns],b[0,:,columns]),
                                  right_distance=rms_distance(a[1,:,columns],b[1,:,columns])))
            other = sorted((rms_distance(grand[label][:,:,columns],grand[s][:,:,columns]),s)
                           for s in labels if s != label)
            typical = float(np.median(distances))
            gap, nearest = other[0]
            row.update({name+'_person_distance':typical, name+'_nearest_other_gap':gap,
                        name+'_nearest_other_sign':nearest,
                        name+'_difference_ratio':typical/gap if gap > 1e-12 else None,
                        name+'_shape_cosine_median':float(np.median(shapes)) if shapes else None,
                        name+'_dynamic_pairs':len(shapes)})
        summaries.append(row)
    return summaries, pairs, scale


def load(root, repo):
    old = root/'2026-11-08_knn_development_holdout'
    new = root/'2026-11-15_six_sign_nested_loso'
    specs = [(old/'trial_features.csv', old/'manifest.json', 'source_commit','input_manifest','기존3종'),
             (new/'trial_features.csv', new/'inventory_summary.json','commit','manifest','새6종')]
    trials, waves, hashes, sources = [], [], {}, []
    for feature_path, meta_path, commit_key, manifest_key, cohort in specs:
        for path in [feature_path,meta_path]:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        features = pd.read_csv(feature_path)
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        commit = metadata[commit_key]
        declared = {r['path']:r['sha256'] for r in metadata[manifest_key]}
        blob = subprocess.check_output(['git','-C',str(repo),'archive',commit,'sensor-data/training'])
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            for expected in features.to_dict('records'):
                raw = tar.extractfile(expected['file']).read()
                sha = hashlib.sha256(raw).hexdigest()
                if sha != declared[expected['file']]:
                    raise ValueError('Raw file hash mismatch')
                frame = pd.read_csv(io.BytesIO(raw), keep_default_na=False)
                for col in ['subject_id','session_id','trial_id','label']:
                    if frame[col].astype(str).str.strip().nunique() != 1:
                        raise ValueError('Conflicting raw identity')
                label = str(frame.label.iloc[0]).strip()
                label = base.ALIASES.get(label,label)
                if label != expected['label'] or base.person(frame.subject_id.iloc[0]) != expected['person']:
                    raise ValueError('Raw label/person mismatch')
                if str(frame.session_id.iloc[0]) != expected['session_id'] or str(frame.trial_id.iloc[0]) != expected['trial_id']:
                    raise ValueError('Raw session/trial mismatch')
                if not (frame.source.eq('REAL').all() and frame.board_type.eq('XIAO').all()):
                    raise ValueError('Expected real XIAO')
                wave, flags = waveform(frame)
                # Confirm these are exactly the original cohort's sensor values.
                for hand,part in frame.groupby('hand'):
                    for sensor in base.SENSORS:
                        values = part[sensor].to_numpy(float)
                        for stat in base.STATS:
                            if not np.isclose(getattr(np,stat)(values), expected[f'{hand}_{sensor}_{stat}'], rtol=1e-12,atol=1e-10):
                                raise ValueError('Original statistics no longer match')
                trials.append(dict(file=expected['file'],sha256=sha,source_commit=commit,cohort=cohort,
                                   person=expected['person'],label=label,trial_id=expected['trial_id'],**flags))
                waves.append(wave)
        sources.append(dict(cohort=cohort,commit=commit,trials=len(features)))
        print(f'Verified {cohort}: {len(features)} original paired captures',flush=True)
    trial_frame = pd.DataFrame(trials)
    if trial_frame.file.duplicated().any() or trial_frame.trial_id.duplicated().any():
        raise ValueError('Duplicate capture')
    if len(trials) != 1603 or trial_frame.label.nunique() != 9:
        raise ValueError('Unexpected pinned cohort')
    templates = {(label,who):np.mean([waves[i] for i in group.index],axis=0)
                 for (label,who),group in trial_frame.groupby(['label','person'])}
    return trial_frame, templates, hashes, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--repo',type=Path,default=Path.cwd())
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh output directory')
    args.out.mkdir(parents=True)
    plan = dict(points=POINTS,phase_alignment='Independent relative progress per hand; no DTW or clock synchronization',
                average='Mean repeated trials per person/sign; each person one template',
                absolute_metric='Median pairwise RMS waveform distance after common per-channel scaling',
                comparison='Ratio of same-sign between-person distance to nearest other-sign grand-template gap',
                descriptive_not_accuracy=True,shape_min_rms=SHAPE_MIN_RMS,
                candidate_rule='FSR and IMU ratios both below 1, no capture with both hands entirely zero FSR',
                rule_is_exploratory_not_validated=True,zero_hand_alone_is_not_exclusion=True)
    (args.out/'comparison_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    trials, templates, hashes, sources = load(args.results_root,args.repo)
    summaries,pairs,scale = compare_templates(templates)
    for row in summaries:
        group = trials[trials.label.eq(row['label'])]
        row.update(trials=len(group),cohort=group.cohort.iloc[0],
                   left_zero=int(group.left_fsr_all_zero.sum()),right_zero=int(group.right_fsr_all_zero.sum()),
                   both_zero=int((group.left_fsr_all_zero & group.right_fsr_all_zero).sum()))
        matches = {s:row[s+'_difference_ratio'] is not None and row[s+'_difference_ratio'] < 1 for s in ['FSR','IMU']}
        row['sensor_candidate'] = all(matches.values()) and row['both_zero'] == 0
        row['pattern_note'] = ('두 센서 기준 통과' if row['sensor_candidate'] else
                               '한 센서만 기준 통과' if any(matches.values()) else '두 센서 기준 미충족')
    result = pd.DataFrame(summaries)
    result.to_csv(args.out/'수어별_센서일관성.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(pairs).to_csv(args.out/'사람쌍별_파형비교.csv',index=False,encoding='utf-8-sig')
    trials.to_csv(args.out/'회차별_FSR상태.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(scale,index=['left','right'],columns=base.SENSORS).to_csv(args.out/'channel_scales.csv',encoding='utf-8-sig')
    np.savez_compressed(args.out/'person_templates.npz',waveforms=np.stack(list(templates.values())),
                        labels=np.array([k[0] for k in templates]),people=np.array([k[1] for k in templates]))
    for filename,sha in hashes.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha:
            raise ValueError('Original source changed')
    meta = dict(plan=plan,sources=sources,input_sha256=hashes,raw_files_verified=len(trials),
                person_sign_templates=len(templates),sensor_pair_comparisons=len(pairs),results=summaries,
                input_data_or_models_modified=False,new_model_fitted=False,
                limitations=['Sign and date/session/wearing conditions are confounded.',
                             'A flat or inactive trace is not automatically a bad capture.',
                             'Relative resampling does not align gesture phases or synchronize hands.',
                             'Low distance/ratio does not guarantee classifier accuracy or real-time performance.',
                             'Different signs have different participant coverage; analysis is exploratory.'])
    (args.out/'summary.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    shutil.copy2(Path(__file__),args.out/Path(__file__).name)
    print(result[['label','people','trials','FSR_person_distance','IMU_person_distance',
                  'FSR_difference_ratio','IMU_difference_ratio','left_zero','right_zero','pattern_note']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
