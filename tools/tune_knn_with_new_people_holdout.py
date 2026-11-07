"""Select KNN only on old people, evaluate Nov 7 people, then fit new candidate."""
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import re
import subprocess
import tarfile

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from threadpoolctl import threadpool_limits


SENSORS=[f'fsr{i}' for i in range(1,7)]+['ax','ay','az','gx','gy','gz']
COLS=[f'{hand}_{sensor}_{stat}' for hand in ['left','right'] for sensor in SENSORS for stat in ['mean','std','min','max']]
FEATURES=dict(all_sensors=COLS,no_left_fsr=[c for c in COLS if not c.startswith('left_fsr')],
              imu_only=[c for c in COLS if '_fsr' not in c])


def person(value):
    match=re.fullmatch(r'subject-?(\d+)',str(value),flags=re.I)
    if not match:raise ValueError('Unconfirmed participant ID: '+str(value))
    return 'subject'+str(int(match[1]))


def make_model(config):
    scaler=StandardScaler() if config['scaler']=='standard' else RobustScaler()
    return make_pipeline(SimpleImputer(strategy='median'),scaler,
        KNeighborsClassifier(n_neighbors=config['k'],weights=config['weights'],p=config['p'],algorithm='brute',n_jobs=1))


def report_results(rows,labels):
    cm=confusion_matrix(rows.label,rows.prediction,labels=labels)
    people=[]
    for who,g in rows.groupby('person'):
        people.append(dict(person=who,trials=len(g),correct=int(g.correct.sum()),accuracy=float(g.correct.mean())))
    return dict(trials=len(rows),correct=int(rows.correct.sum()),accuracy=float(rows.correct.mean()),
        labels=labels,confusion_matrix=cm.tolist(),people=people,
        per_label={label:dict(trials=int(cm[i].sum()),correct=int(cm[i,i]),recall=float(cm[i,i]/cm[i].sum()) if cm[i].sum() else None) for i,label in enumerate(labels)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    p.add_argument('--initial-quality-dir',type=Path,required=True)
    p.add_argument('--previous-commit',required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('Choose a fresh output directory')
    meta=json.loads((args.quality_dir/'summary.json').read_text(encoding='utf-8'))
    initial_meta=json.loads((args.initial_quality_dir/'summary.json').read_text(encoding='utf-8'))
    initial={(f['path'],f['sha256']) for f in initial_meta['manifest']};assert len(initial)==120
    quality=pd.read_csv(args.quality_dir/'trial_quality.csv')
    quality=quality[quality.label.isin(['삼키다','상처','교통사고'])].copy()
    new_paths=set(subprocess.check_output(['git','diff','--name-only','--diff-filter=A',args.previous_commit,meta['commit'],'--','sensor-data/training']).decode().splitlines())
    new_paths.intersection_update(set(quality.file))
    assert len(new_paths)==240
    allowed=set(quality[quality.basic_capture_pass].file)
    assert len(new_paths-allowed)==0, 'New data has hard capture failures; inspect selection before training'
    blocked=set();exclusions=Path(__file__).resolve().parents[1]/'data/training_exclusions.json'
    if exclusions.exists():blocked={(f['path'],f['sha256']) for f in json.loads(exclusions.read_text(encoding='utf-8'))['files']}
    selected=[f for f in meta['manifest'] if (f['path'],f['sha256']) not in initial|blocked and f['path'] in allowed]
    archive=subprocess.check_output(['git','archive',meta['commit'],'sensor-data/training'])
    records=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for f in selected:
            blob=tar.extractfile(f['path']).read();assert hashlib.sha256(blob).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(blob))
            assert d[['subject_id','session_id','trial_id','label']].nunique().eq(1).all()
            assert d.source.eq('REAL').all() and d.record_kind.eq('training').all() and set(d.hand)=={'left','right'}
            row={k:d[k].iloc[0] for k in ['subject_id','session_id','trial_id','label']}
            row.update(person=person(row['subject_id']),file=f['path'],split='new_holdout' if f['path'] in new_paths else 'development')
            for hand in ['left','right']:
                b=d[d.hand==hand]
                for sensor in SENSORS:
                    v=b[sensor].to_numpy(float);assert len(v)>=8 and np.isfinite(v).all()
                    for stat in ['mean','std','min','max']:row[f'{hand}_{sensor}_{stat}']=float(getattr(np,stat)(v))
            records.append(row)
    features=pd.DataFrame(records).sort_values(['person','session_id','trial_id']).reset_index(drop=True)
    assert not features.duplicated(['person','session_id','trial_id']).any()
    dev=features[features.split.eq('development')].copy();holdout=features[features.split.eq('new_holdout')].copy()
    assert len(dev)==319 and len(holdout)==240
    assert set(dev.person)=={'subject1','subject5','subject6','subject8','subject9','subject10'}
    assert set(holdout.person)=={'subject2','subject3','subject4','subject7'}
    assert set(dev.person).isdisjoint(holdout.person)
    labels=sorted(dev.label.unique());assert len(labels)==3 and set(holdout.label)==set(labels)
    assert holdout.groupby(['person','label']).size().eq(20).all()
    args.out.mkdir(parents=True)
    features.to_csv(args.out/'trial_features.csv',index=False)
    features[['subject_id','person','session_id','trial_id','file','label','split']].to_csv(args.out/'split_manifest.csv',index=False)
    # Equal-person, equal-class selection on 5 complete-label development people.
    # subject5 remains usable for training, but its one-class test fold is not
    # allowed to dominate three-class hyperparameter selection.
    complete={who for who,g in dev.groupby('person') if g.label.nunique()==3}
    assert len(complete)==5
    configs=[dict(sensors=s,scaler=sc,k=k,weights=w,p=p) for s,sc,k,w,p in itertools.product(
        FEATURES,['standard','robust'],[1,3,5,7,9],['uniform','distance'],[1,2])]
    leaderboard=[]
    with threadpool_limits(limits=1):
        for index,cfg in enumerate(configs):
            scores=[];total_correct=0;total=0
            cols=FEATURES[cfg['sensors']]
            for who in sorted(complete):
                tr=dev[dev.person.ne(who)];te=dev[dev.person.eq(who)]
                assert set(tr.person).isdisjoint(te.person)
                model=make_model(cfg);model.fit(tr[cols],tr.label);prediction=model.predict(te[cols])
                scores.append(float(balanced_accuracy_score(te.label,prediction)))
                total_correct+=int((prediction==te.label).sum());total+=len(te)
            leaderboard.append(dict(config_id=index,**cfg,selection_score=float(np.mean(scores)),
                development_correct=total_correct,development_trials=total))
            if (index+1)%30==0:print(f'Development selection: {index+1}/{len(configs)} settings checked',flush=True)
        board=pd.DataFrame(leaderboard).sort_values(['selection_score','config_id'],ascending=[False,True])
        board.to_csv(args.out/'development_leaderboard.csv',index=False)
        winner=int(board.iloc[0].config_id);config=configs[winner];cols=FEATURES[config['sensors']]
        selection=dict(config=config,selection_score=float(board.iloc[0].selection_score),
            criterion='Mean per-participant balanced accuracy on 5 complete-class development people',
            cv_test_trials=299,new_holdout_used=False,source_commit=meta['commit'])
        (args.out/'locked_selection.json').write_text(json.dumps(selection,indent=2)+'\n',encoding='utf-8')
        print('Locked selection BEFORE new people evaluation: '+json.dumps(config),flush=True)
        selected_model=make_model(config);selected_model.fit(dev[cols],dev.label)
        joblib.dump(dict(pipeline=selected_model,feature_columns=cols,metadata=selection),args.out/'selected_development_model.joblib')
        # Only now generate outcomes on the four new people.
        baseline_bytes=args.baseline.read_bytes();baseline=joblib.load(io.BytesIO(baseline_bytes))
        assert baseline['pipeline'].steps[-1][1].n_samples_fit_==319
        assert len(baseline['feature_columns'])==96
        results={};allpred=[]
        for name,model,model_cols in [('frozen_baseline',baseline['pipeline'],baseline['feature_columns']),('selected_knn',selected_model,cols)]:
            pred=model.predict(holdout[model_cols])
            rows=holdout[['subject_id','person','session_id','trial_id','file','label']].copy()
            rows['prediction']=pred;rows['correct']=rows.label.eq(pred);rows['model']=name
            results[name]=report_results(rows,labels);allpred.append(rows)
            rows.to_csv(args.out/f'{name}_holdout_predictions.csv',index=False)
            pd.DataFrame(results[name]['confusion_matrix'],index=labels,columns=labels).to_csv(args.out/f'{name}_confusion.csv')
        pd.concat(allpred,ignore_index=True).to_csv(args.out/'holdout_predictions.csv',index=False)
        (args.out/'external_evaluation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        assert args.baseline.read_bytes()==baseline_bytes
        assert np.array_equal(joblib.load(args.out/'selected_development_model.joblib')['pipeline'].predict(holdout[cols]),allpred[1].prediction)
        # New candidate follows the pre-holdout selection, regardless of whether
        # the test score improves. Never choose a second config on these labels.
        combined=make_model(config);combined.fit(features[cols],features.label)
        candidate_meta=dict(source_commit=meta['commit'],config=config,people=sorted(features.person.unique()),
            trials=len(features),initial_trials_excluded=120,independent_accuracy=None,deployed=False,
            sensor_quality_review_pending=True,protocol_adherence_verified=False,
            warning='New 240 trials are now in candidate training. External scores belong to the earlier 319-trial model, not this final candidate.')
        joblib.dump(dict(pipeline=combined,feature_columns=cols,metadata=candidate_meta),args.out/'all_current_candidate_model.joblib')
        assert joblib.load(args.out/'all_current_candidate_model.joblib')['pipeline'].steps[-1][1].n_samples_fit_==559
        (args.out/'candidate_manifest.json').write_text(json.dumps(candidate_meta,indent=2)+'\n',encoding='utf-8')
    manifest=dict(source_commit=meta['commit'],selected_config=config,selection=selection,
        baseline_sha256=hashlib.sha256(baseline_bytes).hexdigest(),development_people=sorted(dev.person.unique()),
        new_people=sorted(holdout.person.unique()),development_trials=319,new_trials=240,candidate_trials=559,
        evaluations=results,input_manifest=selected,
        limitations=['Development scores are selection scores, not unbiased final performance.',
                    'Old development data has been repeatedly explored; new Nov 7 people kept outside selection.',
                    'Pressure/calibration/placement/timing concerns remain; no automatic deletion due to low FSR.',
                    'Only 4 new people evaluated. Replication on future participants is needed.',
                    'Final candidate learned all 559 trials after holdout evaluation; it needs a fresh test participant.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(config=config,evaluations=results),ensure_ascii=True,indent=2))
    print('PASS: immutable source/model hashes; participant disjoint holdout; selection locked before external outcomes; 319/559 serialized models.')


if __name__=='__main__':main()
