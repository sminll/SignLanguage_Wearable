"""Frozen subject-1 -> subject-2 evaluation, followed by a separate full fit."""
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
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pipeline():
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),KNeighborsClassifier(n_neighbors=3))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Output already exists')
    meta=json.loads((args.quality_dir/'summary.json').read_text(encoding='utf-8'))
    exclusions_path=Path(__file__).resolve().parents[1]/'data'/'training_exclusions.json'
    if exclusions_path.exists():
        exclusions=json.loads(exclusions_path.read_text(encoding='utf-8'))
        blocked={(f['path'],f['sha256']) for f in exclusions['files']}
        if any((f['path'],f['sha256']) in blocked for f in meta['manifest']):
            raise SystemExit('User-excluded measurements present. Do not retrain without explicit reinstatement.')
    quality=pd.read_csv(args.quality_dir/'trial_quality.csv')
    assert quality.basic_capture_pass.all(), 'Review failed capture before running'
    archive=subprocess.check_output(['git','archive',meta['commit'],'sensor-data/training'])
    sensors=[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
    cols=[f'{hand}_{sensor}_{stat}' for hand in ['left','right'] for sensor in sensors for stat in ['mean','std','min','max']]
    records=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for f in meta['manifest']:
            blob=tar.extractfile(f['path']).read()
            assert hashlib.sha256(blob).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(blob))
            for key,b in d.groupby(['subject_id','session_id','trial_id'],sort=True):
                assert b.label.nunique()==1 and set(b.hand)=={'left','right'}
                rec=dict(zip(['subject_id','session_id','trial_id'],key))
                rec.update(label=b.label.iloc[0],file=f['path'])
                for hand in ['left','right']:
                    side=b[b.hand==hand]
                    assert len(side)>=8
                    for sensor in sensors:
                        v=side[sensor].to_numpy(float);assert np.isfinite(v).all()
                        for stat in ['mean','std','min','max']:
                            rec[f'{hand}_{sensor}_{stat}']=float(getattr(np,stat)(v))
                records.append(rec)
    features=pd.DataFrame(records)
    assert not features.duplicated(['subject_id','session_id','trial_id']).any()
    assert set(features.subject_id)=={'subject-1','subject-2'}
    train=features[features.subject_id=='subject-1'];test=features[features.subject_id=='subject-2']
    assert len(train)==len(test)==60
    labels=sorted(train.label.unique());assert labels==sorted(test.label.unique()) and len(labels)==3
    assert train.label.value_counts().eq(20).all() and test.label.value_counts().eq(20).all()
    model=pipeline();model.fit(train[cols],train.label)
    pred=model.predict(test[cols])
    matrix=confusion_matrix(test.label,pred,labels=labels)
    predictions=test[['subject_id','session_id','trial_id','label','file']].copy()
    predictions['prediction']=pred;predictions['correct']=predictions.label==pred
    report=dict(source_commit=meta['commit'],training_subject='subject-1',test_subject='subject-2',
        train_trials=60,test_trials=60,labels=labels,correct=int((pred==test.label).sum()),
        accuracy=float(accuracy_score(test.label,pred)),confusion_matrix=matrix.tolist(),
        classification_report=classification_report(test.label,pred,labels=labels,output_dict=True,zero_division=0),
        feature_count=len(cols),k=3,weights='uniform',distance='euclidean',
        protocol='Independent hand statistics concatenated per trial; no sample-level alignment asserted.',
        limitations=['Calibration and sensor review flags remain unresolved; exploratory model, not deployed.',
                    'App subject identities not linked to historical MATLAB subject IDs.',
                    'One person per split; no hyperparameter tuning on subject-2.',
                    'Both persons share app session ID; this is a participant holdout, not independent-device/session validation.',
                    'Combined model has seen both people and has no independent reported accuracy.'])
    args.out.mkdir(parents=True)
    features.to_csv(args.out/'trial_features.csv',index=False)
    predictions.to_csv(args.out/'subject2_predictions.csv',index=False)
    pd.DataFrame(matrix,index=labels,columns=labels).to_csv(args.out/'subject2_confusion.csv')
    (args.out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Persist the holdout results and frozen model BEFORE full-data fit.
    joblib.dump(dict(pipeline=model,feature_columns=cols,metadata=report),args.out/'subject1_only_model.joblib')
    combined=pipeline();combined.fit(features[cols],features.label)
    combined_meta=dict(source_commit=meta['commit'],subjects=['subject-1','subject-2'],trials=120,
        independent_accuracy=None,deployed=False,quality='review_required',feature_count=len(cols))
    joblib.dump(dict(pipeline=combined,feature_columns=cols,metadata=combined_meta),args.out/'combined_candidate_model.joblib')
    (args.out/'combined_manifest.json').write_text(json.dumps(combined_meta,indent=2)+'\n',encoding='utf-8')
    # Reload artifacts to verify serialization, fixed column order and model sizes.
    frozen=joblib.load(args.out/'subject1_only_model.joblib')
    assert np.array_equal(frozen['pipeline'].predict(test[cols]),pred)
    full=joblib.load(args.out/'combined_candidate_model.joblib')
    assert full['pipeline'].steps[-1][1].n_samples_fit_==120
    assert frozen['pipeline'].steps[-1][1].n_samples_fit_==60
    print(json.dumps({k:report[k] for k in ['labels','correct','accuracy','confusion_matrix','feature_count']},ensure_ascii=True,indent=2))
    print('Verified serialized models: frozen=60, combined=120; evaluation recorded before combined fit.')


if __name__=='__main__': main()
