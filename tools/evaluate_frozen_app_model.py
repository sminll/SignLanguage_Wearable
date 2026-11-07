"""Evaluate newly added app sessions with a frozen saved model; never fit."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--after-ref',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Output exists; select a new directory')
    modelpath=args.model_dir/'model.joblib'
    before=modelpath.read_bytes();modelhash=hashlib.sha256(before).hexdigest()
    saved=joblib.load(io.BytesIO(before));model=saved['pipeline'];cols=saved['feature_columns']
    assert len(cols)==96 and model.steps[-1][1].n_samples_fit_==120
    assert set(saved['metadata']['subjects'])=={'subject8','subject10'}
    meta=json.loads((args.quality_dir/'summary.json').read_text(encoding='utf-8'))
    changed=subprocess.check_output(['git','diff','--name-only','--diff-filter=A',args.after_ref,meta['commit'],'--','sensor-data/training']).decode().splitlines()
    changed=set(changed)
    quality=pd.read_csv(args.quality_dir/'trial_quality.csv')
    selected=quality[quality.file.isin(changed)].copy()
    assert set(selected.subject_id)=={'subject-1','subject-6','subject-9'},'New participants changed; review selection'
    training=pd.read_csv(args.model_dir/'training_trials.csv')
    assert set(selected.file).isdisjoint(training.file)
    assert set(selected.subject_id).isdisjoint(training.subject_id)
    passed=selected[selected.basic_capture_pass.eq(True)].copy()
    assert len(passed)>0
    files={f['path']:f for f in meta['manifest'] if f['path'] in set(passed.file)}
    archive=subprocess.check_output(['git','archive',meta['commit'],'sensor-data/training'])
    sensors=[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
    records=[];relationships=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for path,f in files.items():
            blob=tar.extractfile(path).read();assert hashlib.sha256(blob).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(blob))
            assert d[['subject_id','session_id','trial_id','label']].nunique().eq(1).all()
            assert set(d.hand)=={'left','right'} and d.source.eq('REAL').all()
            r={k:d[k].iloc[0] for k in ['subject_id','session_id','trial_id','label']};r['file']=path
            for hand in ['left','right']:
                b=d[d.hand==hand]
                assert len(b)>=8
                for sensor in sensors:
                    v=b[sensor].to_numpy(float);assert np.isfinite(v).all()
                    for stat in ['mean','std','min','max']:r[f'{hand}_{sensor}_{stat}']=float(getattr(np,stat)(v))
            records.append(r)
    features=pd.DataFrame(records).sort_values(['subject_id','label','session_id','trial_id']).reset_index(drop=True)
    assert not features.duplicated(['subject_id','session_id','trial_id']).any()
    assert set(cols)<=set(features)
    labels=list(model.classes_);assert set(features.label)<=set(labels)
    # Deliberately no fit, fit_transform, or partial_fit in this evaluation tool.
    pred=model.predict(features[cols])
    results=features[['subject_id','session_id','trial_id','file','label']].copy()
    results['prediction']=pred;results['correct']=results.label.eq(pred)
    args.out.mkdir(parents=True)
    features.to_csv(args.out/'trial_features.csv',index=False)
    results.to_csv(args.out/'predictions.csv',index=False)
    selected.to_csv(args.out/'selected_trial_quality.csv',index=False)
    channels=pd.read_csv(args.quality_dir/'channel_quality.csv').merge(selected[['file','subject_id']],on='file',validate='many_to_one')
    channel_summary=channels.groupby(['subject_id','label','hand','sensor']).agg(
        mean_of_trial_means=('mean','mean'),max=('max','max'),
        mean_zero_fraction=('zero_fraction','mean'),constant_trials=('constant','sum'))
    channel_summary.to_csv(args.out/'sensor_quality_by_subject.csv')
    summaries=[]
    for person,g in results.groupby('subject_id'):
        folder=args.out/person;folder.mkdir()
        cm=confusion_matrix(g.label,g.prediction,labels=labels)
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(folder/'confusion.csv')
        report=dict(subject_id=person,trials=len(g),correct=int(g.correct.sum()),
            accuracy=float(accuracy_score(g.label,g.prediction)),
            balanced_accuracy=float(balanced_accuracy_score(g.label,g.prediction)),
            labels=labels,confusion_matrix=cm.tolist(),sessions=g.session_id.unique().tolist())
        for label in labels:
            b=g[g.label==label];report[label+'_correct']=int(b.correct.sum());report[label+'_trials']=len(b)
        (folder/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        summaries.append(report)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['labels','confusion_matrix','sessions']} for r in summaries]).to_csv(args.out/'participant_comparison.csv',index=False)
    assert modelpath.read_bytes()==before,'Frozen model changed'
    manifest=dict(source_commit=meta['commit'],new_files_since=args.after_ref,
        model_path=str(modelpath),model_sha256=modelhash,training_participants=['subject8','subject10'],
        train_trials=120,new_trial_count=len(selected),evaluated_trial_count=len(features),
        capture_rejected_count=int((~selected.basic_capture_pass).sum()),
        all_correct=int(results.correct.sum()),all_trials=len(results),
        pooled_accuracy=float(results.correct.mean()),fitting_performed=False,
        feature_count=96,k=3,model_unchanged=True,participants=summaries,
        warnings=['Participant IDs treated as recorded; real identity and protocol adherence require confirmation.',
                  'subject-1 retake selected by new files, not combined with initial subject-1 sessions.',
                  'FSR/calibration/timestamp quality concerns remain; these are exploratory predictions.',
                  'Incomplete 19-trial class is reported as observed, not silently filled or deleted.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['warnings','participants']},ensure_ascii=True,indent=2))
    print(pd.DataFrame([{k:v for k,v in r.items() if k not in ['labels','confusion_matrix','sessions']} for r in summaries]).to_string(index=False))
    print('PASS: new-file selection, raw hashes, no training overlap, immutable 120-trial baseline; no fitting.')


if __name__=='__main__': main()
