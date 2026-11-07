"""Participant LOSO across app sessions, preserving IDs and protocol cohorts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SENSORS=[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
COLS=[f'{hand}_{sensor}_{stat}' for hand in ['left','right'] for sensor in SENSORS for stat in ['mean','std','min','max']]


def canonical_id(value):
    match=re.fullmatch(r'subject-?([0-9]+)',str(value).strip(),flags=re.I)
    if not match:raise ValueError(f'Not a confirmed numeric participant ID: {value}')
    return 'subject'+str(int(match[1]))


def pipeline():
    return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),KNeighborsClassifier(n_neighbors=3))


def metrics(predictions,labels):
    cm=confusion_matrix(predictions.label,predictions.prediction,labels=labels)
    recalls=np.divide(np.diag(cm),cm.sum(axis=1),out=np.zeros(len(labels),dtype=float),where=cm.sum(axis=1)>0)
    return dict(trials=len(predictions),correct=int(predictions.correct.sum()),accuracy=float(predictions.correct.mean()),
        labels_present=int(predictions.label.nunique()),confusion_matrix=cm.tolist(),
        per_label={label:dict(correct=int(cm[i,i]),trials=int(cm[i].sum()),recall=float(recalls[i]) if cm[i].sum() else None) for i,label in enumerate(labels)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--initial-quality-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('Choose a fresh output directory')
    meta=json.loads((args.quality_dir/'summary.json').read_text(encoding='utf-8'))
    initial_meta=json.loads((args.initial_quality_dir/'summary.json').read_text(encoding='utf-8'))
    initial={(x['path'],x['sha256']) for x in initial_meta['manifest']}
    assert len(initial)==120, 'Expected explicitly identified initial subject-1/2 recordings'
    q=pd.read_csv(args.quality_dir/'trial_quality.csv')
    assert q.basic_capture_pass.all(), 'Hard quality failures require explicit selection'
    exclusions=Path(__file__).resolve().parents[1]/'data/training_exclusions.json'
    blocked=set()
    if exclusions.exists():
        blocked={(f['path'],f['sha256']) for f in json.loads(exclusions.read_text(encoding='utf-8'))['files']}
    archive=subprocess.check_output(['git','archive',meta['commit'],'sensor-data/training'])
    records=[];excluded=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for f in meta['manifest']:
            if (f['path'],f['sha256']) in blocked:
                excluded.append(f['path']);continue
            blob=tar.extractfile(f['path']).read();assert hashlib.sha256(blob).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(blob))
            assert d[['subject_id','session_id','trial_id','label']].nunique().eq(1).all()
            assert d.source.eq('REAL').all() and d.record_kind.eq('training').all()
            assert set(d.hand)=={'left','right'}
            r={key:d[key].iloc[0] for key in ['subject_id','session_id','trial_id','label']}
            r.update(person=canonical_id(r['subject_id']),file=f['path'],sha256=f['sha256'],
                cohort='initial_position_uncontrolled' if (f['path'],f['sha256']) in initial else 'later_position_unverified')
            for hand in ['left','right']:
                b=d[d.hand==hand];assert len(b)>=8
                for sensor in SENSORS:
                    v=b[sensor].to_numpy(float);assert np.isfinite(v).all()
                    for stat in ['mean','std','min','max']:r[f'{hand}_{sensor}_{stat}']=float(getattr(np,stat)(v))
            records.append(r)
    features=pd.DataFrame(records).sort_values(['person','session_id','trial_id']).reset_index(drop=True)
    assert not features.duplicated(['person','session_id','trial_id']).any()
    assert len(features)==439 and features.cohort.eq('initial_position_uncontrolled').sum()==120
    labels=sorted(features.label.unique());assert len(labels)==3
    args.out.mkdir(parents=True)
    features.to_csv(args.out/'trial_features.csv',index=False)
    features[['subject_id','person']].drop_duplicates().to_csv(args.out/'subject_id_mapping.csv',index=False)
    features.groupby(['person','cohort','label']).size().reset_index(name='trials').to_csv(args.out/'cohort_inventory.csv',index=False)
    all_predictions=[];fold_rows=[]
    # Same k and preprocessing in every fold, declared before predictions.
    for mode in ['all_sessions','without_initial']:
        cohort=features if mode=='all_sessions' else features[features.cohort.ne('initial_position_uncontrolled')]
        for person in sorted(cohort.person.unique(),key=lambda x:int(x[7:])):
            train=cohort[cohort.person.ne(person)];test=cohort[cohort.person.eq(person)]
            assert len(test)>0 and person not in set(train.person)
            assert set(train.file).isdisjoint(test.file)
            assert set(train.label)==set(labels)
            model=pipeline();model.fit(train[COLS],train.label)
            pred=model.predict(test[COLS])
            result=test[['subject_id','person','session_id','trial_id','file','cohort','label']].copy()
            result['prediction']=pred;result['correct']=result.label.eq(pred);result['mode']=mode
            all_predictions.append(result)
            report=metrics(result,labels)
            folder=args.out/mode/person;folder.mkdir(parents=True)
            report.update(person=person,mode=mode,train_people=sorted(train.person.unique()),train_trials=len(train),
                source_commit=meta['commit'],k=3,feature_count=96,exploratory=True)
            result.to_csv(folder/'predictions.csv',index=False)
            train[['subject_id','person','session_id','trial_id','file','cohort','label']].to_csv(folder/'training_trials.csv',index=False)
            pd.DataFrame(report['confusion_matrix'],index=labels,columns=labels).to_csv(folder/'confusion.csv')
            (folder/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
            joblib.dump(dict(pipeline=model,feature_columns=COLS,metadata=report),folder/'model.joblib')
            assert np.allclose(model.steps[1][1].mean_,train[COLS].mean().to_numpy())
            saved=joblib.load(folder/'model.joblib')['pipeline']
            assert saved.steps[-1][1].n_samples_fit_==len(train)
            assert np.array_equal(saved.predict(test[COLS]),pred)
            row={k:report[k] for k in ['mode','person','train_trials','trials','correct','accuracy','labels_present']}
            for label in labels:
                row[label+'_correct']=report['per_label'][label]['correct'];row[label+'_trials']=report['per_label'][label]['trials']
            fold_rows.append(row)
    predictions=pd.concat(all_predictions,ignore_index=True)
    predictions.to_csv(args.out/'all_predictions.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(args.out/'fold_summary.csv',index=False)
    summaries={mode:metrics(g,labels) for mode,g in predictions.groupby('mode')}
    for mode in summaries:
        folds=[r for r in fold_rows if r['mode']==mode]
        summaries[mode]['participant_macro_accuracy']=float(np.mean([r['accuracy'] for r in folds]))
        # subject5 has only traffic; also report subjects with all 3 labels.
        complete={r['person'] for r in folds if r['labels_present']==3}
        summaries[mode]['complete_label_participants']=metrics(predictions[(predictions['mode']==mode)&predictions.person.isin(complete)],labels)
    # Fair comparison: all-session models scored only on the same later trials.
    a=predictions[(predictions['mode']=='all_sessions')&predictions.cohort.ne('initial_position_uncontrolled')].set_index('file').sort_index()
    b=predictions[predictions['mode']=='without_initial'].set_index('file').sort_index()
    assert a.index.tolist()==b.index.tolist() and a.person.tolist()==b.person.tolist() and a.label.tolist()==b.label.tolist()
    comparisons=[]
    for person in a.person.unique():
        x=a[a.person==person];y=b[b.person==person]
        comparisons.append(dict(person=person,trials=len(x),with_initial_correct=int(x.correct.sum()),
            without_initial_correct=int(y.correct.sum()),delta_correct=int(x.correct.sum()-y.correct.sum())))
    pd.DataFrame(comparisons).to_csv(args.out/'matched_later_trial_comparison.csv',index=False)
    summaries['matched_later_trials']=dict(with_initial=metrics(a,labels),without_initial=metrics(b,labels),
        warning='Held-out person excluded across all dates, including their own initial sessions.')
    # New combined candidate only after every fold has been evaluated and saved.
    combined=pipeline();combined.fit(features[COLS],features.label)
    candidate_meta=dict(source_commit=meta['commit'],trials=len(features),people=sorted(features.person.unique()),
        k=3,features=96,independent_accuracy=None,deployed=False,mixed_position_protocols=True,
        sensor_quality_review_pending=True,warning='LOSO score describes held-out fold models, not this all-data model.')
    joblib.dump(dict(pipeline=combined,feature_columns=COLS,metadata=candidate_meta),args.out/'all_people_candidate_model.joblib')
    assert joblib.load(args.out/'all_people_candidate_model.joblib')['pipeline'].steps[-1][1].n_samples_fit_==439
    (args.out/'candidate_manifest.json').write_text(json.dumps(candidate_meta,indent=2)+'\n',encoding='utf-8')
    later=features[features.cohort.ne('initial_position_uncontrolled')]
    later_model=pipeline();later_model.fit(later[COLS],later.label)
    later_meta={**candidate_meta,'trials':len(later),'people':sorted(later.person.unique()),
                'mixed_position_protocols':False,'placement_protocol_verified':False,
                'excludes_initial':True,'warning':'Later cohort excludes known initial recordings; actual placement still unverified. LOSO score is not this model independent accuracy.'}
    joblib.dump(dict(pipeline=later_model,feature_columns=COLS,metadata=later_meta),args.out/'without_initial_candidate_model.joblib')
    assert joblib.load(args.out/'without_initial_candidate_model.joblib')['pipeline'].steps[-1][1].n_samples_fit_==319
    (args.out/'without_initial_candidate_manifest.json').write_text(json.dumps(later_meta,indent=2)+'\n',encoding='utf-8')
    manifest=dict(source_commit=meta['commit'],initial_source_commit=initial_meta['commit'],labels=labels,
        identity_rule='User confirmed: subject8 == subject-8; integer suffix identifies same person across dates.',
        trials=len(features),participants=features.person.nunique(),excluded_files=excluded,
        summaries=summaries,feature_count=96,k=3,
        limitations=['FSR and calibration concerns remain; basic quality pass is not verified gesture correctness.',
                    'Later cohort placement adherence is not verified; never call this a controlled causal comparison.',
                    'Repeated prior exploratory evaluations mean these scores are not a pristine final test.',
                    'subject5 has only one label; report per-subject/class coverage alongside pooled accuracy.',
                    'No hyperparameter tuning in this run, no deployment, originals and prior models preserved.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(pd.DataFrame(fold_rows).to_string(index=False))
    print(json.dumps(summaries,ensure_ascii=True,indent=2))
    print('PASS: all raw hashes, normalized person separation, train-only scaling, model reloads, matched evaluation trials.')


if __name__=='__main__':main()
