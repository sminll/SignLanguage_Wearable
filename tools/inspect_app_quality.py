"""Read immutable app CSV blobs from a Git ref; never train or edit originals."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pandas as pd


def git(*args):
    return subprocess.check_output(['git', *args])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ref',default='origin/training-data')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Choose a new output directory')
    sha=git('rev-parse',args.ref).decode().strip()
    paths=git('ls-tree','-r','--name-only',sha,'sensor-data/training').decode().splitlines()
    paths=[x for x in paths if x.endswith('.csv')]
    archive=tarfile.open(fileobj=io.BytesIO(git('archive',sha,'sensor-data/training')))
    sensors=[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
    trials=[]; channels=[]; manifests=[]; allrows=[]
    for path in paths:
        blob=archive.extractfile(path).read()
        d=pd.read_csv(io.BytesIO(blob))
        required=['subject_id','session_id','trial_id','label','hand','time','time_ms','sample_seq',*sensors]
        missing=sorted(set(required)-set(d))
        if missing:
            trials.append(dict(file=path,status='exclude',reason='missing_columns:'+','.join(missing)))
            continue
        manifests.append(dict(path=path,sha256=hashlib.sha256(blob).hexdigest(),rows=len(d)))
        d['file']=path; allrows.append(d)
        hard=[]; flags=[]
        if d[['subject_id','session_id','trial_id','label','hand']].isna().any().any(): hard.append('missing_identity')
        for k in ['subject_id','session_id','trial_id','label']:
            if d[k].nunique()!=1: hard.append('conflicting_'+k)
        if set(d.hand)!= {'left','right'}: hard.append('missing_or_invalid_hand')
        if not d.get('record_kind',pd.Series(dtype=str)).eq('training').all(): flags.append('record_kind_unverified')
        if 'calibration_verified' not in d or not d.calibration_verified.astype(str).str.lower().eq('true').all(): flags.append('calibration_unverified')
        flags.append('gesture_type_unrecorded' if 'gesture_type' not in d else 'gesture_type_requires_confirmation')
        flags.append('shared_capture_clock_unverified')
        item=dict(file=path,label=str(d.label.iloc[0]),subject_id=str(d.subject_id.iloc[0]),
                  session_id=str(d.session_id.iloc[0]),trial_id=str(d.trial_id.iloc[0]),rows=len(d))
        for hand,b in d.groupby('hand',sort=True):
            vals=b[sensors].apply(pd.to_numeric,errors='coerce')
            tm=pd.to_numeric(b.time_ms,errors='coerce').to_numpy(float)
            seq=pd.to_numeric(b.sample_seq,errors='coerce').to_numpy(float)
            stamps=pd.to_datetime(b.time,utc=True,errors='coerce')
            gaps=np.diff(tm); seqdiff=np.diff(seq)
            span=float(tm[-1]-tm[0]) if len(tm)>1 else 0
            if len(b)<8: hard.append(hand+':too_few_samples')
            if not np.isfinite(vals.to_numpy()).all(): hard.append(hand+':nonfinite_sensor')
            if not np.isfinite(tm).all() or not np.isfinite(seq).all() or stamps.isna().any(): hard.append(hand+':invalid_time_or_seq')
            if (gaps<=0).any(): hard.append(hand+':device_time_not_increasing')
            if (seqdiff<=0).any(): hard.append(hand+':seq_not_increasing')
            if span<3000: hard.append(hand+':span_under_3s')
            if len(gaps) and gaps.max()>500: flags.append(hand+':gap_over_500ms')
            if (seqdiff>1).any(): flags.append(hand+':seq_gaps_within_trial')
            if (vals[sensors[:6]]<0).any().any(): hard.append(hand+':negative_fsr')
            if (stamps.diff().dt.total_seconds().dropna()<0).any(): flags.append(hand+':host_timestamp_reversed')
            item.update({hand+'_rows':len(b),hand+'_span_ms':span,
                hand+'_max_gap_ms':float(gaps.max()) if len(gaps) else None,
                hand+'_median_interval_ms':float(np.median(gaps)) if len(gaps) else None,
                hand+'_seq_missing_within_trial':int(np.maximum(seqdiff-1,0).sum()),
                hand+'_imu_bad':int((pd.to_numeric(b.imu_ok,errors='coerce')!=1).sum()) if 'imu_ok' in b else None})
            if item[hand+'_imu_bad']: flags.append(hand+':imu_not_ok')
            for sensor in sensors:
                v=vals[sensor].to_numpy(float)
                constant=bool(np.ptp(v)==0)
                if constant: flags.append(hand+':constant_'+sensor)
                channels.append(dict(file=path,label=item['label'],hand=hand,sensor=sensor,
                    min=float(np.nanmin(v)),max=float(np.nanmax(v)),mean=float(np.nanmean(v)),
                    std=float(np.nanstd(v)),zero_fraction=float((v==0).mean()),
                    constant=constant,near_4095_fraction=float((v>=4000).mean()) if sensor.startswith('fsr') else None))
        item.update(status='exclude' if hard else 'review',basic_capture_pass=not bool(hard),
                    reason=';'.join(hard),review_flags=';'.join(sorted(set(flags))))
        trials.append(item)
    archive.close()
    combined=pd.concat(allrows,ignore_index=True)
    key=['subject_id','session_id','hand','sample_seq']
    duplicate_count=int(combined.duplicated(key,keep=False).sum())
    duplicate_keys=combined[combined.duplicated(key,keep=False)]
    for t in trials:
        if t['file'] in set(duplicate_keys.file):
            t['status']='review';t['review_flags']+=';cross_file_duplicate_seq_key'
    result=pd.DataFrame(trials)
    args.out.mkdir(parents=True)
    result.to_csv(args.out/'trial_quality.csv',index=False)
    pd.DataFrame(channels).to_csv(args.out/'channel_quality.csv',index=False)
    duplicate_keys.to_csv(args.out/'duplicate_keys.csv',index=False)
    summary=dict(commit=sha,files=len(paths),rows=len(combined),
        labels=combined.groupby('label').agg(files=('file','nunique'),rows=('label','size')).to_dict('index'),
        subjects=combined.subject_id.unique().tolist(),sessions=combined.session_id.unique().tolist(),
        status_counts=result.status.value_counts().to_dict(),
        basic_capture_pass=int(result.basic_capture_pass.sum()),cross_file_duplicate_key_rows=duplicate_count,
        manifest=manifests,
        limits=['Timestamp semantics and calibration algorithm are not confirmed by the app source.',
                'Constant FSR alone does not prove hardware failure. 4000 is descriptive, not a calibrated cutoff.',
                'No training, no deletions, no raw CSV modifications. A basic pass is not verified gesture correctness.'])
    (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='manifest'},ensure_ascii=True,indent=2))
    print(result.groupby('label')[['left_rows','right_rows','left_span_ms','right_span_ms','left_max_gap_ms','right_max_gap_ms']].agg(['min','max']).to_string())
    print('Review flags:',result.review_flags.str.split(';').explode().value_counts().to_dict())


if __name__=='__main__': main()
