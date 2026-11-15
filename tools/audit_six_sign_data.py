"""Read-only inventory of six-sign app CSVs pinned to a Git commit. No fitting."""
from pathlib import Path
import argparse
import collections
import hashlib
import io
import json
import re
import subprocess
import tarfile

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out',type=Path,required=True)
parser.add_argument('--commit',required=True,help='Pinned data Git revision')
args = parser.parse_args()
OUT = args.out.resolve()
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'summary.json').exists():
    raise FileExistsError('This snapshot already exists; choose a fresh directory.')
ALIASES = {
    '괜찮다':'괜찮다', '괜찮아':'괜찮다', '아니다':'아니다',
    '아니다 (아니, 부인(하다))':'아니다', '무엇':'무엇', '무엇(어디)':'무엇',
    '마시다':'마시다', '화장실':'화장실', '싫다':'싫다',
    '싫다(거절)':'싫다', '싫다(거부)':'싫다',
}
SENSORS = [f'fsr{i}' for i in range(1,7)] + ['ax','ay','az','gx','gy','gz']
REQUIRED = set(['subject_id','session_id','trial_id','label','hand','time','time_ms','sample_seq','source',*SENSORS])
git = lambda *args: subprocess.check_output(['git',*args])
commit = git('rev-parse',args.commit).decode().strip()
blob = git('archive',commit,'sensor-data/training')
records, manifest, parsing_errors = [], [], []
label_inventory = collections.Counter()
with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
    for member in archive:
        if not member.isfile() or not member.name.endswith('.csv'):
            continue
        data = archive.extractfile(member).read()
        try:
            frame = pd.read_csv(io.BytesIO(data),encoding='utf-8-sig',keep_default_na=False)
        except Exception as exc:
            parsing_errors.append(dict(file=member.name,error=str(exc)))
            continue
        if 'label' not in frame or frame.empty:
            parsing_errors.append(dict(file=member.name,error='empty_or_missing_label'))
            continue
        raw_labels = sorted(frame.label.astype(str).str.strip().unique())
        for label in raw_labels:
            label_inventory[label] += 1
        if not any(label in ALIASES for label in raw_labels):
            continue
        hard, flags = [], []
        row = dict(file=member.name,sha256=hashlib.sha256(data).hexdigest(),rows=len(frame),
                   raw_label='|'.join(raw_labels),label=ALIASES.get(raw_labels[0],''))
        manifest.append(dict(path=member.name,sha256=row['sha256'],rows=len(frame)))
        missing = sorted(REQUIRED-set(frame.columns))
        if missing:
            row.update(hard_failures='missing_columns:'+'|'.join(missing),review_flags='',basic_pass=False)
            records.append(row)
            continue
        for name in ['subject_id','session_id','trial_id','label']:
            if frame[name].astype(str).str.strip().nunique()!=1:
                hard.append('conflicting_'+name)
        raw_subject = str(frame.subject_id.iloc[0]).strip()
        match = re.fullmatch(r'subject-?(\d+)',raw_subject,re.I)
        row.update(raw_subject=raw_subject,person='subject'+str(int(match[1])) if match else raw_subject,
                   session_id=str(frame.session_id.iloc[0]),trial_id=str(frame.trial_id.iloc[0]))
        if not match: hard.append('unrecognized_person')
        if frame[['subject_id','session_id','trial_id']].astype(str).eq('').any().any(): hard.append('missing_identity')
        if not frame.source.eq('REAL').all(): hard.append('non_REAL')
        if set(frame.hand) != {'left','right'}: hard.append('missing_or_invalid_hand')
        row['board'] = '|'.join(sorted(frame.board_type.astype(str).unique())) if 'board_type' in frame else 'unrecorded'
        row['calibration_verified'] = '|'.join(sorted(frame.calibration_verified.astype(str).str.lower().unique())) if 'calibration_verified' in frame else 'unrecorded'
        row['record_kind'] = '|'.join(sorted(frame.record_kind.astype(str).unique())) if 'record_kind' in frame else 'unrecorded'
        if row['board'] != 'XIAO': flags.append('board_requires_review')
        if row['calibration_verified'] != 'true': flags.append('calibration_unverified')
        if row['record_kind'] != 'training': flags.append('record_kind_requires_review')
        row['shared_clock_columns'] = all(c in frame for c in ['host_elapsed_ms','trial_elapsed_ms'])
        if not row['shared_clock_columns']: flags.append('shared_capture_clock_unverified')
        dates = pd.to_datetime(frame.time,format='mixed',errors='coerce',utc=True)
        if dates.isna().any(): hard.append('invalid_host_time')
        if dates.notna().any():
            row.update(date_kst=dates.min().tz_convert('Asia/Seoul').strftime('%Y-%m-%d'),
                       start_kst=dates.min().tz_convert('Asia/Seoul').isoformat(),
                       end_kst=dates.max().tz_convert('Asia/Seoul').isoformat())
        for hand, part in frame.groupby('hand',sort=False):
            values = part[SENSORS].apply(pd.to_numeric,errors='coerce')
            times = pd.to_numeric(part.time_ms,errors='coerce').to_numpy(float)
            sequence = pd.to_numeric(part.sample_seq,errors='coerce').to_numpy(float)
            gaps = np.diff(times)
            row[hand+'_rows'] = len(part)
            row[hand+'_span_ms'] = float(times[-1]-times[0]) if len(times)>1 else 0.
            row[hand+'_max_gap_ms'] = float(gaps.max()) if len(gaps) else 0.
            if len(part)<8: hard.append(hand+':too_few_rows')
            if not np.isfinite(values.to_numpy()).all(): hard.append(hand+':nonfinite_sensor')
            if not np.isfinite(times).all() or not np.isfinite(sequence).all(): hard.append(hand+':invalid_clock_or_sequence')
            if (gaps<=0).any(): hard.append(hand+':nonmonotonic_device_clock')
            if (np.diff(sequence)<=0).any(): hard.append(hand+':nonmonotonic_sequence')
            if (np.diff(sequence)>1).any(): flags.append(hand+':sequence_gaps')
            if (values[SENSORS[:6]]<0).any().any(): hard.append(hand+':negative_FSR')
            if row[hand+'_span_ms']<3000: hard.append(hand+':span_below_3s')
            if row[hand+'_span_ms']<4500: flags.append(hand+':span_below_4p5s')
            if row[hand+'_max_gap_ms']>500: flags.append(hand+':gap_above_500ms')
            if 'imu_ok' in part and not pd.to_numeric(part.imu_ok,errors='coerce').eq(1).all(): flags.append(hand+':IMU_not_ok')
            fsr = values[SENSORS[:6]].to_numpy(float)
            row[hand+'_FSR_all_zero'] = bool((fsr==0).all())
            row[hand+'_FSR_constant_channels'] = int((np.ptp(fsr,axis=0)==0).sum())
            row[hand+'_FSR_zero_fraction'] = float((fsr==0).mean())
            row[hand+'_FSR_high_fraction'] = float((fsr>=4000).mean())
            if row[hand+'_FSR_all_zero']: flags.append(hand+':all_FSR_zero')
            if row[hand+'_FSR_constant_channels']: flags.append(hand+':constant_FSR_channels')
            if row[hand+'_FSR_high_fraction']>.5: flags.append(hand+':mostly_high_FSR_ADC')
        signals = frame[['hand','time','time_ms',*SENSORS]]
        row['signal_fingerprint'] = hashlib.sha256(pd.util.hash_pandas_object(signals,index=False).to_numpy().tobytes()).hexdigest()
        row.update(hard_failures=';'.join(sorted(set(hard))),review_flags=';'.join(sorted(set(flags))),basic_pass=not hard)
        records.append(row)

trials = pd.DataFrame(records)
if trials.empty:
    raise ValueError('No matching six-sign files found.')
identity = ['person','session_id','trial_id']
trials['duplicate_trial_identity'] = trials.duplicated(identity,keep=False)
trials['duplicate_signal_and_time'] = trials.duplicated('signal_fingerprint',keep=False)
by_person = trials.groupby(['person','label']).size().unstack(fill_value=0)
by_day = trials.groupby(['date_kst','person','label']).size().reset_index(name='trials')
flags = collections.Counter(flag for line in trials.review_flags for flag in line.split(';') if flag)
hard = collections.Counter(reason for line in trials.hard_failures for reason in line.split(';') if reason)
summary = dict(commit=commit,files=len(trials),participants=int(trials.person.nunique()),rows=int(trials.rows.sum()),
               by_person=by_person.to_dict('index'),by_date=trials.groupby('date_kst').size().to_dict(),
               by_label=trials.groupby('label').size().to_dict(),raw_label_counts=trials.raw_label.value_counts().to_dict(),
               basic_pass=int(trials.basic_pass.sum()),hard_failure_reasons=dict(hard),review_flags=dict(flags),
               boards=trials.board.value_counts().to_dict(),calibration_verified=trials.calibration_verified.value_counts().to_dict(),
               duplicate_identity_files=int(trials.duplicate_trial_identity.sum()),duplicate_signal_files=int(trials.duplicate_signal_and_time.sum()),
               alias_map=ALIASES,all_repository_labels=dict(label_inventory),parsing_errors=parsing_errors,
               label_confirmation='User confirmed refusal aliases on 2026-11-13 and fine/denial aliases on 2026-11-15.',
               audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               manifest=manifest,training_performed=False,
               limits=['Structural checks do not confirm correct sign execution or wearing position.',
                       'Low/constant FSR alone is not an exclusion criterion.',
                       'Calibration and timestamp meaning require validation before sample-wise two-hand alignment.'])
trials.to_csv(OUT/'trial_inventory.csv',index=False,encoding='utf-8-sig')
by_person.to_csv(OUT/'participant_sign_counts.csv',encoding='utf-8-sig')
by_day.to_csv(OUT/'date_participant_sign_counts.csv',index=False,encoding='utf-8-sig')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
if Path(__file__).resolve() != OUT/'audit_six_sign_data.py':
    (OUT/'audit_six_sign_data.py').write_bytes(Path(__file__).read_bytes())
print('COMMIT',commit)
print('SIX_SIGN_INVENTORY',len(trials),'files;',trials.person.nunique(),'people;',int(trials.rows.sum()),'sensor rows')
print(by_person.to_string())
print('BY_DAY')
print(trials.groupby(['date_kst','person']).size().to_string())
print('BASIC_CAPTURE_PASS',int(trials.basic_pass.sum()),'/',len(trials))
print('HARD_FAILURES',dict(hard))
print('REVIEW_FLAGS',dict(flags))
print('BOARDS',summary['boards'],'CALIBRATION',summary['calibration_verified'])
print('DUPLICATE_IDENTITIES',summary['duplicate_identity_files'],'DUPLICATE_SIGNALS',summary['duplicate_signal_files'])
print('ALIASED_RAW_LABELS',summary['raw_label_counts'])
print('PARSING_ERRORS',parsing_errors[:5])
print('OTHER_SIMILAR_LABELS',{k:v for k,v in label_inventory.items() if k not in ALIASES and any(word in k for word in ['괜찮','아니','무엇','어디','마시','화장실','싫'])})
print('Saved',OUT)
