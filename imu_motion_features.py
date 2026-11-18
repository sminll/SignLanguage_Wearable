"""Magnitude augmentation and conservative within-trial motion normalization.

No rest calibration, physical orientation calibration, clock synchronization or
streaming segmentation is claimed. All transformations use one completed trial.
"""
from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pandas as pd

import group_model_study as base
import sensor_preprocessing as prep


VARIANTS = ('original', 'magnitude', 'aligned', 'magnitude_aligned')


@dataclass(frozen=True)
class MotionPolicy:
    gyro_floor_dps: float = 10.
    noise_multiplier: float = 3.
    pad_ms: float = 250.
    min_motion_span_ms: float = 200.
    min_window_ms: float = 1000.
    min_window_samples: int = 8
    max_resample_gap_ms: float = 250.
    resample_points: int = 32

    def __post_init__(self):
        if (not np.isfinite(list(asdict(self).values())).all()
            or min(self.gyro_floor_dps,self.noise_multiplier,self.min_window_ms,
                   self.min_window_samples,self.max_resample_gap_ms,self.resample_points)<=0
            or self.pad_ms<0 or self.min_motion_span_ms<0
            or self.min_window_samples<3 or self.resample_points<6
            or int(self.min_window_samples)!=self.min_window_samples
            or int(self.resample_points)!=self.resample_points):
            raise ValueError('Invalid motion policy')


def magnitudes(values):
    values=np.asarray(values,float)
    if values.ndim!=2 or values.shape[1]!=12 or not np.isfinite(values).all():
        raise ValueError('Finite 12-channel sensor matrix required')
    return np.column_stack([np.linalg.norm(values[:,6:9],axis=1),
                            np.linalg.norm(values[:,9:12],axis=1)])


def motion_window(values, times_ms, policy=MotionPolicy()):
    """Select first-to-last sustained gyro activity with context padding.

    Quiet statistics come from this trial, not a claimed rest calibration.
    Still hands, isolated spikes and unreliable windows fall back to the full
    original samples. A fallback is not a rejected trial or a zero-filled hand.
    """
    values=np.asarray(values,float)
    times=np.asarray(times_ms,float)
    if (times.ndim!=1 or len(times)!=len(values) or len(times)<3
        or not np.isfinite(times).all() or (np.diff(times)<=0).any()):
        raise ValueError('Strictly increasing finite time required')
    gyro=magnitudes(values)[:,1]
    low=gyro[gyro<=np.quantile(gyro,.4)]
    quiet=float(np.quantile(gyro,.2))
    mad=float(np.median(np.abs(low-np.median(low))))
    threshold=max(policy.gyro_floor_dps,quiet+policy.noise_multiplier*mad)
    active=gyro>threshold
    sustained=active.copy()
    sustained[0] &= active[1]
    sustained[-1] &= active[-2]
    sustained[1:-1] &= active[:-2]|active[2:]
    indices=np.flatnonzero(sustained)
    reason='no_sustained_activity'
    start,end=0,len(times)-1
    if len(indices)>=2 and times[indices[-1]]-times[indices[0]]>=policy.min_motion_span_ms:
        start=max(0,int(np.searchsorted(times,times[indices[0]]-policy.pad_ms,side='right'))-1)
        end=min(len(times)-1,int(np.searchsorted(times,times[indices[-1]]+policy.pad_ms)))
        if end-start+1<policy.min_window_samples or times[end]-times[start]<policy.min_window_ms:
            reason='short_window'
        elif np.diff(times[start:end+1]).max()>policy.max_resample_gap_ms:
            reason='resample_gap_too_large'
        else:
            reason='aligned'
    if reason!='aligned':
        start,end=0,len(times)-1
        selected_values=values.copy()
        selected_times=times.copy()
    else:
        selected_times=np.linspace(times[start],times[end],policy.resample_points)
        selected_values=np.column_stack([np.interp(selected_times,times[start:end+1],
            values[start:end+1,i]) for i in range(12)])
    info=dict(status=reason,threshold_dps=threshold,start_index=start,end_index=end,
        input_samples=len(times),output_samples=len(selected_times),
        original_span_ms=float(times[-1]-times[0]),selected_span_ms=float(times[end]-times[start]),
        window_fraction=float((times[end]-times[start])/(times[-1]-times[0])),
        active_fraction=float(sustained.mean()),resampled=reason=='aligned')
    return selected_values,selected_times,info


def add_stats(target, prefix, values, sensors):
    for i,sensor in enumerate(sensors):
        for stat in prep.STATS:
            target[f'{prefix}{sensor}_{stat}']=float(getattr(np,stat)(values[:,i]))


def features(hands, recipe, policy=MotionPolicy()):
    original=base.trial_features(hands,recipe)
    result={name:dict(original) for name in VARIANTS}
    logs=[]
    for hand in ['left','right']:
        capture=hands[hand]
        values,_=prep.process_hand(capture,recipe)
        aligned,times,info=motion_window(values,capture.times_ms,policy)
        logs.append(dict(hand=hand,**info))
        whole_mag=magnitudes(values)
        for variant in ['magnitude','magnitude_aligned']:
            add_stats(result[variant],f'{hand}_',whole_mag,['accel_norm','gyro_norm'])
        for variant in VARIANTS:
            use_alignment=variant in ['aligned','magnitude_aligned']
            part=aligned if use_alignment else values
            part_times=times if use_alignment else capture.times_ms
            for phase,mask in enumerate(base.phase_masks(part_times)):
                if use_alignment:
                    add_stats(result[variant],f'phase{phase}_{hand}_',part[mask],prep.SENSORS)
                if variant in ['magnitude','magnitude_aligned']:
                    add_stats(result[variant],f'phase{phase}_{hand}_',magnitudes(part[mask]),['accel_norm','gyro_norm'])
            if use_alignment:
                # Full-window axis features remain present; include actual time
                # information which time normalization would otherwise discard.
                result[variant][f'{hand}_motion_window_fraction']=info['window_fraction']
                result[variant][f'{hand}_motion_active_fraction']=info['active_fraction']
    return result,logs


def build_tables(repo,ref,cohort,reference,policy=MotionPolicy()):
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','--verify','--end-of-options',ref+'^{commit}'],text=True).strip()
    archive=subprocess.check_output(['git','-C',str(repo),'archive',commit,'sensor-data/training'])
    wanted=set(cohort.file)
    blobs={}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.name in wanted:
                blobs[member.name]=tar.extractfile(member).read()
    if set(blobs)!=wanted or cohort.file.duplicated().any() or cohort.duplicated(['person','session_id','trial_id']).any():
        raise ValueError('Missing/duplicate frozen cohort files or identities')
    records={(recipe,variant):[] for recipe in base.RECIPES for variant in VARIANTS}
    logs=[]
    for row in cohort.itertuples():
        blob=blobs[row.file]
        if hashlib.sha256(blob).hexdigest()!=row.sha256:
            raise ValueError('Raw file hash changed: '+row.file)
        hands=prep.read_capture(pd.read_csv(io.BytesIO(blob)),account=Path(row.file).parent.name,
            source_path=row.file,source_sha256=row.sha256)
        if set(hands)!={'left','right'}:
            raise ValueError('Missing hand is never filled')
        for hand in hands.values():
            if (hand.person,hand.session_id,hand.trial_id,base.ALIASES.get(hand.label,hand.label))!=(
                    row.person,row.session_id,row.trial_id,row.label):
                raise ValueError('Frozen identity mismatch')
        for recipe_name,recipe in base.RECIPES.items():
            variants,detail=features(hands,recipe,policy)
            for variant,record in variants.items():
                records[recipe_name,variant].append(record)
            logs.extend(dict(file=row.file,person=row.person,label=row.label,recipe=recipe_name,**r) for r in detail)
    tables={key:pd.DataFrame(rows) for key,rows in records.items()}
    for recipe,old in reference.items():
        actual=tables[recipe,'original']
        if list(actual)!=list(old):
            raise ValueError('Baseline feature schema differs')
        np.testing.assert_allclose(actual.to_numpy(),old.to_numpy(),rtol=1e-12,atol=1e-10)
    return commit,tables,pd.DataFrame(logs)


def matrices(tables):
    X,columns={},{ }
    for (recipe,variant),table in tables.items():
        for sensors in ['imu','all']:
            if recipe=='denoise_normalize' and sensors=='imu':
                continue
            name=f'{recipe}__{variant}__{sensors}'
            cols=[c for c in table if sensors=='all' or '_fsr' not in c]
            X[name]=table[cols].to_numpy(float)
            if not np.isfinite(X[name]).all():
                raise ValueError('Nonfinite feature matrix')
            columns[name]=cols
    return X,columns
