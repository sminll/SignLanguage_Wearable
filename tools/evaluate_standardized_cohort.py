"""Fixed KNN participant holdouts and initial-cohort ablation; no tuning."""
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SENSORS=[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
COLS=[f'{hand}_{sensor}_{stat}' for hand in ['left','right'] for sensor in SENSORS for stat in ['mean','std','min','max']]
INITIAL=['subject-1','subject-2']
TODAY=['subject8','subject10']


def make_model():
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                         KNeighborsClassifier(n_neighbors=3,weights='uniform'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists(): raise SystemExit('Use a fresh output directory')
    meta=json.loads((args.quality_dir/'summary.json').read_text(encoding='utf-8'))
    quality=pd.read_csv(args.quality_dir/'trial_quality.csv')
    selected=quality[quality.subject_id.isin(INITIAL+TODAY)]
    assert set(selected.subject_id)==set(INITIAL+TODAY)
    assert selected.basic_capture_pass.all(), 'Failed basic quality: inspect before inclusion'
    assert selected.groupby(['subject_id','label']).size().eq(20).all()
    excluded_path=Path(__file__).resolve().parents[1]/'data/training_exclusions.json'
    blocked=set()
    if excluded_path.exists():
        ex=json.loads(excluded_path.read_text(encoding='utf-8'))
        blocked={(f['path'],f['sha256']) for f in ex['files']}
    files={f['path']:f for f in meta['manifest'] if f['path'] in set(selected.file)}
    assert not any((f['path'],f['sha256']) in blocked for f in files.values()), 'User-excluded input'
    archive=subprocess.check_output(['git','archive',meta['commit'],'sensor-data/training'])
    records=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for path,f in files.items():
            blob=tar.extractfile(path).read()
            assert hashlib.sha256(blob).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(blob))
            assert d[['subject_id','session_id','trial_id','label']].nunique().eq(1).all()
            assert d.source.eq('REAL').all() and d.record_kind.eq('training').all()
            assert set(d.hand)=={'left','right'}
            row={k:d[k].iloc[0] for k in ['subject_id','session_id','trial_id','label']}
            row.update(file=path,sha256=f['sha256'])
            for hand in ['left','right']:
                b=d[d.hand.eq(hand)]
                assert len(b)>=8
                for sensor in SENSORS:
                    v=b[sensor].to_numpy(float);assert np.isfinite(v).all()
                    for stat in ['mean','std','min','max']:
                        row[f'{hand}_{sensor}_{stat}']=float(getattr(np,stat)(v))
            records.append(row)
    features=pd.DataFrame(records).sort_values(['subject_id','label','trial_id']).reset_index(drop=True)
    assert len(features)==240
    assert not features.duplicated(['subject_id','session_id','trial_id']).any()
    labels=sorted(features.label.unique());assert len(labels)==3
    assert features.groupby(['subject_id','label']).size().eq(20).all()
    # All folds and KNN settings are declared before inspecting predictions.
    specs=[('8_to_10',['subject8'],'subject10'),
           ('10_to_8',['subject10'],'subject8'),
           ('initial_plus_8_to_10',INITIAL+['subject8'],'subject10'),
           ('initial_plus_10_to_8',INITIAL+['subject10'],'subject8')]
    args.out.mkdir(parents=True)
    features.to_csv(args.out/'trial_features.csv',index=False)
    rows=[];predictions={}
    for name,train_people,test_person in specs:
        train=features[features.subject_id.isin(train_people)]
        test=features[features.subject_id.eq(test_person)]
        assert test_person not in train_people and set(train.file).isdisjoint(test.file)
        assert len(train)==len(train_people)*60 and len(test)==60
        model=make_model();model.fit(train[COLS],train.label)
        pred=model.predict(test[COLS]);cm=confusion_matrix(test.label,pred,labels=labels)
        fold=args.out/name;fold.mkdir()
        result=test[['subject_id','session_id','trial_id','file','label']].copy()
        result['prediction']=pred;result['correct']=result.label.eq(pred)
        result.to_csv(fold/'predictions.csv',index=False)
        train[['subject_id','session_id','trial_id','file','label']].to_csv(fold/'training_trials.csv',index=False)
        pd.DataFrame(cm,index=labels,columns=labels).to_csv(fold/'confusion.csv')
        report=dict(name=name,train_subjects=train_people,test_subject=test_person,
            train_trials=len(train),test_trials=60,correct=int(result.correct.sum()),
            accuracy=float(accuracy_score(test.label,pred)),labels=labels,confusion_matrix=cm.tolist(),
            classification_report=classification_report(test.label,pred,labels=labels,output_dict=True,zero_division=0),
            source_commit=meta['commit'],k=3,features=96,placement_protocol_verified=False,
            status='exploratory_participant_holdout')
        (fold/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        joblib.dump(dict(pipeline=model,feature_columns=COLS,metadata=report),fold/'model.joblib')
        reloaded=joblib.load(fold/'model.joblib')
        assert np.array_equal(reloaded['pipeline'].predict(test[COLS]),pred)
        assert reloaded['pipeline'].steps[-1][1].n_samples_fit_==len(train)
        # Prove scaler uses only training rows, including for mixed cohorts.
        assert np.allclose(reloaded['pipeline'].steps[1][1].mean_,train[COLS].mean().to_numpy())
        predictions[name]=result
        summary={k:report[k] for k in ['name','train_trials','test_subject','test_trials','correct','accuracy']}
        for i,label in enumerate(labels):summary[label+'_recall']=float(cm[i,i]/cm[i].sum())
        rows.append(summary)
    comparisons=[]
    for base,extra in [('8_to_10','initial_plus_8_to_10'),('10_to_8','initial_plus_10_to_8')]:
        a,b=predictions[base],predictions[extra]
        assert a.file.tolist()==b.file.tolist() and a.label.tolist()==b.label.tolist()
        comparisons.append(dict(baseline=base,initial_added=extra,
            correct_delta=int(b.correct.sum()-a.correct.sum()),
            accuracy_delta=float(b.correct.mean()-a.correct.mean()),
            newly_correct=int((~a.correct & b.correct).sum()),newly_wrong=int((a.correct & ~b.correct).sum())))
    pd.DataFrame(rows).to_csv(args.out/'comparison.csv',index=False)
    (args.out/'initial_data_effect.json').write_text(json.dumps(comparisons,indent=2)+'\n',encoding='utf-8')
    # Only after every independent evaluation is persisted, fit the new baseline.
    combined=features[features.subject_id.isin(TODAY)];assert len(combined)==120
    model=make_model();model.fit(combined[COLS],combined.label)
    candidate=args.out/'subject8_subject10_combined';candidate.mkdir()
    candidate_meta=dict(source_commit=meta['commit'],subjects=TODAY,train_trials=120,
        independent_accuracy=None,placement_protocol_verified=False,sensor_quality_review_pending=True,
        excludes_initial_subjects=True,excludes_subject5=True,deployed=False,k=3,features=96)
    joblib.dump(dict(pipeline=model,feature_columns=COLS,metadata=candidate_meta),candidate/'model.joblib')
    combined[['subject_id','session_id','trial_id','file','label']].to_csv(candidate/'training_trials.csv',index=False)
    (candidate/'manifest.json').write_text(json.dumps(candidate_meta,indent=2)+'\n',encoding='utf-8')
    assert joblib.load(candidate/'model.joblib')['pipeline'].steps[-1][1].n_samples_fit_==120
    manifest=dict(source_commit=meta['commit'],labels=labels,input_files=list(files.values()),
        protocol='96 independent-hand trial summary features; no precise sample alignment claimed.',
        split='Whole participant held out; preprocessing trained on training participant(s) only.',
        original_data_unchanged=True,models_deployed=False,
        warnings=['Placement standardization is a planned protocol, not verified from CSV.',
                  'Results from 2 people are exploratory; new people needed for confirmation.',
                  'Initial participants were already analyzed in earlier experiments.',
                  'No tuning on held-out outcomes; reusing folds for tuning requires fresh final test people.',
                  'Final combined model has seen 8 and 10; holdout scores are not its independent accuracy.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))
    print(json.dumps(comparisons,indent=2))
    print('PASS: source hashes, participant separation, identical paired test sets, train-only scaler, serialized models.')


if __name__=='__main__': main()
