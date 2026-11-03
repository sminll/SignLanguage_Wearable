"""Read-only, exploratory quality review of the two legacy health sessions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SESSIONS = {
    's20261102_202453_620_757740130': 'health_original_posture_unconfirmed',
    's20261102_203328_055_743132468': 'health_seated',
}
SENSORS = [f'fsr{i}' for i in range(1, 7)] + ['ax', 'ay', 'az', 'gx', 'gy', 'gz']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/sensor_dataset_v2.csv'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit('Output exists; choose a new directory.')
    snapshot = args.input.read_bytes()
    import io
    raw = pd.read_csv(io.BytesIO(snapshot))
    data = raw[raw.session_id.isin(SESSIONS)].copy()
    assert set(data.session_id) == set(SESSIONS), 'Expected both sessions'
    for col in SENSORS + ['time_ms', 'sample_seq']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['trial_number'] = data.trial_id.astype(str).str.extract(r'(\d+)$')[0].astype(int)
    overview, trials, features = [], [], []
    for (sid, hand), g in data.groupby(['session_id', 'hand'], sort=True):
        seq = g.sample_seq.to_numpy(float)
        times = g.time_ms.to_numpy(float)
        overview.append(dict(session_id=sid, condition=SESSIONS[sid], hand=hand,
            samples=len(g), trials=sorted(g.trial_number.unique().tolist()),
            missing_seq_count=int(np.maximum(np.diff(seq) - 1, 0).sum()),
            duplicate_seq_count=int(g.sample_seq.duplicated().sum()),
            nonincreasing_seq=int((np.diff(seq) <= 0).sum()),
            nonincreasing_device_time=int((np.diff(times) <= 0).sum()),
            first_seq=float(seq[0]), last_seq=float(seq[-1]),
            labels=g.label.unique().tolist(), subjects=g.subject_id.unique().tolist()))
        for trial in range(1, 21):
            b = g[g.trial_number == trial]
            vals = b[SENSORS].to_numpy(float)
            tm = b.time_ms.to_numpy(float)
            gaps = np.diff(tm)
            hard, review = [], []
            if len(b) < 8: hard.append('too_few_samples')
            if len(b) and not np.isfinite(vals).all(): hard.append('nonfinite_sensor_values')
            span = float(tm[-1] - tm[0]) if len(tm) > 1 else 0
            max_gap = float(gaps.max()) if len(gaps) else None
            if span < 3000: hard.append('device_span_under_3s')
            if len(gaps) and (gaps <= 0).any(): hard.append('nonincreasing_device_time')
            if max_gap is not None and max_gap > 500: review.append('device_gap_over_500ms')
            if len(b):
                fsr = b[SENSORS[:6]].to_numpy(float)
                if (fsr < 0).any(): hard.append('negative_fsr')
                if np.all(np.ptp(fsr, axis=0) == 0): review.append('all_fsr_channels_constant')
                if np.all(np.ptp(vals[:, 6:], axis=0) == 0): review.append('all_imu_channels_constant')
            status = 'exclude' if hard else ('review' if review else 'candidate_unverified_timing')
            trials.append(dict(session_id=sid,condition=SESSIONS[sid],hand=hand,trial=trial,
                samples=len(b),device_span_ms=span,max_device_gap_ms=max_gap,
                median_device_interval_ms=float(np.median(gaps)) if len(gaps) else None,
                status=status,reasons=';'.join(hard+review)))
            if len(b):
                f = dict(session_id=sid,condition=SESSIONS[sid],hand=hand,trial=trial)
                for sensor in SENSORS:
                    v=b[sensor].to_numpy(float)
                    for stat in ['mean','std','min','max']:
                        f[f'{sensor}_{stat}']=float(getattr(np,stat)(v))
                features.append(f)
    summaries=[]
    for (sid,hand),g in data.groupby(['session_id','hand']):
        for sensor in SENSORS:
            v=g[sensor].to_numpy(float)
            summaries.append(dict(session_id=sid,condition=SESSIONS[sid],hand=hand,sensor=sensor,
                finite_fraction=float(np.isfinite(v).mean()),min=float(np.nanmin(v)),
                median=float(np.nanmedian(v)),max=float(np.nanmax(v)),std=float(np.nanstd(v)),
                zero_fraction=float((v==0).mean()),unique_values=int(g[sensor].nunique()),
                over_50_fraction=float((v>50).mean()) if sensor.startswith('fsr') else None))
    feature_df=pd.DataFrame(features)
    comparisons=[]
    a,b=SESSIONS
    for hand in ['left','right']:
        for sensor in SENSORS:
            col=f'{sensor}_mean'
            x=feature_df[(feature_df.session_id==a)&(feature_df.hand==hand)][col].to_numpy()
            y=feature_df[(feature_df.session_id==b)&(feature_df.hand==hand)][col].to_numpy()
            pooled=float(np.sqrt((np.var(x,ddof=1)+np.var(y,ddof=1))/2))
            delta=float(y.mean()-x.mean())
            comparisons.append(dict(hand=hand,sensor=sensor,original_trial_mean=float(x.mean()),
                seated_trial_mean=float(y.mean()),delta=delta,
                standardized_difference=delta/pooled if pooled>0 else None))
    paired=[]
    # Descriptive flag only: session and condition are confounded, with one
    # recording per condition. Never automatically delete low-pressure trials.
    shift_flags=[c for c in comparisons if c['standardized_difference'] is not None
                 and abs(c['standardized_difference']) >= 3]
    for sid in SESSIONS:
        for trial in range(1,21):
            sides=[t for t in trials if t['session_id']==sid and t['trial']==trial]
            statuses=[t['status'] for t in sides]
            paired.append(dict(session_id=sid,trial=trial,
                status='exclude' if 'exclude' in statuses else ('review' if 'review' in statuses else 'candidate_unverified_timing'),
                training_decision='exclude' if 'exclude' in statuses else 'review_before_combining',
                review_reason='session_signal_shift;no_shared_clock',
                gesture_type='both',canonical_label='health',
                timing='unknown_no_shared_clock',
                max_hand_duration_difference_ms=abs(sides[0]['device_span_ms']-sides[1]['device_span_ms'])))
    report=dict(source_sha256=hashlib.sha256(snapshot).hexdigest(),samples=len(data),
        overview=overview,trial_status_counts=dict(Counter(t['status'] for t in trials)),
        paired_trial_status_counts=dict(Counter(t['status'] for t in paired)),
        training_decision_counts=dict(Counter(t['training_decision'] for t in paired)),
        descriptive_large_session_shifts=shift_flags,
        notes=['No common host clock; no precise alignment or clock offset inference.',
               'Candidate means basic signal checks only, not verified gesture correctness.',
               'FSR >50 is a descriptive ADC cutoff, not a calibrated pass/fail threshold.',
               'Original posture is unconfirmed. One session per condition; differences cannot establish a posture effect.',
               'Two conditions share one health label; no sign classification accuracy or model training.'])
    assert args.input.read_bytes()==snapshot, 'Input changed during analysis'
    args.out.mkdir(parents=True)
    for name,records in [('trial_quality',trials),('paired_trial_decisions',paired),
                         ('sensor_summary',summaries),('condition_comparison',comparisons),
                         ('trial_features',features)]:
        pd.DataFrame(records).to_csv(args.out/f'{name}.csv',index=False)
    (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,hand in enumerate(['left','right']):
        for col,(sensor,unit) in enumerate([('fsr2','raw ADC'),('az','recorded acceleration units')]):
            ax=axes[row,col]
            for sid,style,label in [(a,'o-','Health (original posture unconfirmed)'),(b,'s-','Health (seated)')]:
                block=feature_df[(feature_df.session_id==sid)&(feature_df.hand==hand)]
                ax.plot(block.trial,block[f'{sensor}_mean'],style,label=label,markersize=3)
            ax.set_title(f'{hand}: {sensor} trial mean');ax.set_ylabel(unit)
            ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.25)
            if row==1: ax.set_xlabel('Trial number (separate recording sequences)')
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Descriptive session comparison: no shared clock, no causal posture inference')
    fig.tight_layout();fig.savefig(args.out/'session_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(report,ensure_ascii=True,indent=2))
    print('Largest descriptive differences (no causal inference):')
    print(pd.DataFrame(comparisons).assign(magnitude=lambda d:d.standardized_difference.abs()).sort_values('magnitude',ascending=False).head(10).to_string(index=False))


if __name__=='__main__':
    main()
