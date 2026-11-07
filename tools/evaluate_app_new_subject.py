"""Fit only app subjects 1+2 and evaluate a new subject without updating it."""
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--test-subject',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Output exists')
    assert args.test_subject not in {'subject-1','subject-2'}
    meta=json.loads((args.quality_dir/'summary.json').read_text(encoding='utf-8'))
    quality=pd.read_csv(args.quality_dir/'trial_quality.csv')
    selected=quality[quality.subject_id.isin(['subject-1','subject-2',args.test_subject])]
    assert len(selected)>120 and selected.basic_capture_pass.all()
    excluded_path=Path(__file__).resolve().parents[1]/'data/training_exclusions.json'
    excluded=json.loads(excluded_path.read_text(encoding='utf-8')) if excluded_path.exists() else {'files':[]}
    blocked={(f['path'],f['sha256']) for f in excluded['files']}
    trainfiles=set(selected[selected.subject_id.isin(['subject-1','subject-2'])].file)
    assert not any((f['path'],f['sha256']) in blocked for f in meta['manifest'] if f['path'] in trainfiles), 'Training files still excluded'
    sensors=[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
    cols=[f'{hand}_{sensor}_{stat}' for hand in ['left','right'] for sensor in sensors for stat in ['mean','std','min','max']]
    archive=subprocess.check_output(['git','archive',meta['commit'],'sensor-data/training'])
    records=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for f in meta['manifest']:
            if f['path'] not in set(selected.file): continue
            blob=tar.extractfile(f['path']).read(); assert hashlib.sha256(blob).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(blob))
            assert d[['subject_id','session_id','trial_id','label']].nunique().eq(1).all()
            assert set(d.hand)=={'left','right'}
            rec={key:d[key].iloc[0] for key in ['subject_id','session_id','trial_id','label']};rec['file']=f['path']
            for hand in ['left','right']:
                b=d[d.hand==hand];assert len(b)>=8
                for sensor in sensors:
                    v=b[sensor].to_numpy(float);assert np.isfinite(v).all()
                    for stat in ['mean','std','min','max']:rec[f'{hand}_{sensor}_{stat}']=float(getattr(np,stat)(v))
            records.append(rec)
    features=pd.DataFrame(records);assert not features.duplicated(['subject_id','session_id','trial_id']).any()
    train=features[features.subject_id.isin(['subject-1','subject-2'])]
    test=features[features.subject_id.eq(args.test_subject)]
    assert len(train)==120 and train.groupby(['subject_id','label']).size().eq(20).all()
    assert set(train.subject_id).isdisjoint(test.subject_id)
    model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),KNeighborsClassifier(n_neighbors=3))
    model.fit(train[cols],train.label)
    pred=model.predict(test[cols]);labels=sorted(train.label.unique());assert set(test.label)<=set(labels)
    result=test[['subject_id','session_id','trial_id','file','label']].copy()
    result['prediction']=pred;result['correct']=result.label.eq(result.prediction)
    metadata=dict(source_commit=meta['commit'],train_subjects=['subject-1','subject-2'],train_trials=120,
        test_subject=args.test_subject,test_trials=len(test),test_labels=sorted(test.label.unique()),
        correct=int(result.correct.sum()),accuracy=float(accuracy_score(test.label,pred)),
        labels=labels,confusion_matrix=confusion_matrix(test.label,pred,labels=labels).tolist(),
        status='reinstated_by_user',quality='sensor_review_pending',deployed=False,
        test_used_in_training=False,feature_count=96,k=3,
        limitations=['Trial-level independent hand summaries; precise shared measurement clock unverified.',
                    'Sensor/calibration review issues remain unresolved.',
                    'If test labels cover fewer than all classes, score is partial-class recall, not overall three-class accuracy.'])
    args.out.mkdir(parents=True)
    features.to_csv(args.out/'trial_features.csv',index=False);result.to_csv(args.out/'heldout_predictions.csv',index=False)
    pd.DataFrame(metadata['confusion_matrix'],index=labels,columns=labels).to_csv(args.out/'confusion_matrix.csv')
    (args.out/'evaluation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    joblib.dump(dict(pipeline=model,feature_columns=cols,metadata=metadata),args.out/'subject1_subject2_reinstated_model.joblib')
    reloaded=joblib.load(args.out/'subject1_subject2_reinstated_model.joblib')
    assert reloaded['pipeline'].steps[-1][1].n_samples_fit_==120
    assert np.array_equal(reloaded['pipeline'].predict(test[cols]),pred)
    print(json.dumps(metadata,ensure_ascii=True,indent=2))


if __name__=='__main__': main()
