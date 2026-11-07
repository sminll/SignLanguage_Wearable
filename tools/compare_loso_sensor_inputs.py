"""Post-hoc sensor ablation on identical participant-held-out trial sets."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def score(g,labels):
    cm=confusion_matrix(g.label,g.prediction,labels=labels)
    return dict(trials=len(g),correct=int(g.correct.sum()),accuracy=float(g.correct.mean()),
        confusion_matrix=cm.tolist(),per_label={label:dict(trials=int(cm[i].sum()),correct=int(cm[i,i])) for i,label in enumerate(labels)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('Output exists')
    source=args.source_run/'trial_features.csv';raw=source.read_bytes()
    f=pd.read_csv(source);f=f[f.cohort.ne('initial_position_uncontrolled')].copy()
    assert len(f)==319 and f.person.nunique()==6
    labels=sorted(f.label.unique())
    cols=[f'{hand}_{sensor}_{stat}' for hand in ['left','right']
          for sensor in [*[f'fsr{i}' for i in range(1,7)],'ax','ay','az','gx','gy','gz']
          for stat in ['mean','std','min','max']]
    modes=dict(all_sensors=cols,no_left_fsr=[c for c in cols if not c.startswith('left_fsr')],
        imu_only=[c for c in cols if '_fsr' not in c])
    assert [len(c) for c in modes.values()]==[96,72,48]
    assert np.isfinite(f[cols].to_numpy()).all()
    prior=pd.read_csv(args.source_run/'all_predictions.csv')
    prior=prior[prior['mode'].eq('without_initial')].set_index('file').sort_index()
    folds=[];predictions=[]
    args.out.mkdir(parents=True)
    for mode,features in modes.items():
        for person in sorted(f.person.unique(),key=lambda s:int(s[7:])):
            train=f[f.person.ne(person)];test=f[f.person.eq(person)]
            assert set(train.person).isdisjoint(test.person) and set(train.file).isdisjoint(test.file)
            model=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),KNeighborsClassifier(n_neighbors=3))
            model.fit(train[features],train.label);pred=model.predict(test[features])
            assert np.allclose(model.steps[1][1].mean_,train[features].mean().to_numpy())
            result=test[['subject_id','person','session_id','trial_id','file','label']].copy()
            result['prediction']=pred;result['correct']=result.label.eq(pred);result['mode']=mode
            predictions.append(result)
            report=score(result,labels)
            row=dict(mode=mode,person=person,features=len(features),train_trials=len(train),
                trials=len(test),correct=report['correct'],accuracy=report['accuracy'],labels_present=test.label.nunique())
            for label in labels:row[label+'_correct']=report['per_label'][label]['correct'];row[label+'_trials']=report['per_label'][label]['trials']
            folds.append(row)
            out=args.out/mode/person;out.mkdir(parents=True)
            result.to_csv(out/'predictions.csv',index=False)
            train[['person','file','label']].to_csv(out/'training_trials.csv',index=False)
            pd.DataFrame(report['confusion_matrix'],index=labels,columns=labels).to_csv(out/'confusion.csv')
    preds=pd.concat(predictions,ignore_index=True)
    preds.to_csv(args.out/'predictions.csv',index=False)
    pd.DataFrame(folds).to_csv(args.out/'fold_summary.csv',index=False)
    baseline=preds[preds['mode'].eq('all_sensors')].set_index('file').sort_index()
    assert prior.index.tolist()==baseline.index.tolist()
    assert prior.prediction.tolist()==baseline.prediction.tolist(), 'Baseline did not reproduce'
    assert baseline.correct.sum()==200
    summaries={}
    for mode,g in preds.groupby('mode'):
        matching=g.set_index('file').sort_index()
        assert matching.index.tolist()==baseline.index.tolist() and matching.person.tolist()==baseline.person.tolist()
        report=score(g,labels)
        full_people=g.groupby('person').label.nunique();full_people=set(full_people[full_people.eq(3)].index)
        report['three_label_participants']=score(g[g.person.isin(full_people)],labels)
        report['participant_macro_accuracy']=float(g.groupby('person').correct.mean().mean())
        report['newly_correct_vs_all']=int((~baseline.correct & matching.correct).sum())
        report['newly_wrong_vs_all']=int((baseline.correct & ~matching.correct).sum())
        summaries[mode]=report
    meta=json.loads((args.source_run/'manifest.json').read_text(encoding='utf-8'))
    manifest=dict(source_commit=meta['source_commit'],source_feature_sha256=hashlib.sha256(raw).hexdigest(),
        trials=319,labels=labels,k=3,feature_columns=modes,summaries=summaries,
        baseline_reproduced=True,models_replaced=False,
        warnings=['Post-hoc diagnostic on previously evaluated participants, not fresh final validation.',
                  'No rows deleted: only input channels are changed.',
                  'subject5 has only traffic; complete-label participant aggregate is also reported.',
                  'All-zero values may be physiological/contact/sensor effects, not automatic hardware failure.'])
    (args.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert source.read_bytes()==raw
    print(pd.DataFrame(folds).to_string(index=False))
    print(json.dumps(summaries,ensure_ascii=True,indent=2))
    print('PASS: baseline predictions reproduced, identical 319 held-out trials, participant isolation, train-only scaling, source unchanged.')


if __name__=='__main__':main()
