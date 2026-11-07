"""Fixed-configuration LOSO and separately labelled training resubstitution."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits


def summarize(g,labels):
    cm=confusion_matrix(g.label,g.prediction,labels=labels)
    return dict(trials=len(g),correct=int(g.correct.sum()),accuracy=float(g.correct.mean()),
        confusion_matrix=cm.tolist(),per_label={label:dict(trials=int(cm[i].sum()),correct=int(cm[i,i])) for i,label in enumerate(labels)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise SystemExit('Choose a new output directory')
    before=args.model.read_bytes();raw=args.features.read_bytes()
    saved=joblib.load(args.model);pipeline=saved['pipeline'];cols=saved['feature_columns']
    f=pd.read_csv(args.features);labels=list(pipeline.classes_)
    assert len(f)==559 and f.person.nunique()==10 and len(cols)==48
    assert pipeline.steps[-1][1].n_samples_fit_==559
    assert not f.duplicated(['person','session_id','trial_id']).any()
    assert np.isfinite(f[cols].to_numpy()).all()
    args.out.mkdir(parents=True)
    folds=[];predictions=[]
    with threadpool_limits(limits=1):
        # Settings are cloned from saved candidate, never searched here.
        for who in sorted(f.person.unique(),key=lambda s:int(s[7:])):
            tr=f[f.person.ne(who)];te=f[f.person.eq(who)]
            assert set(tr.person).isdisjoint(te.person) and set(tr.file).isdisjoint(te.file)
            model=clone(pipeline);model.fit(tr[cols],tr.label)
            pred=model.predict(te[cols])
            g=te[['subject_id','person','session_id','trial_id','file','label']].copy()
            g['prediction']=pred;g['correct']=g.label.eq(pred);predictions.append(g)
            assert np.allclose(model.steps[1][1].mean_,tr[cols].mean().to_numpy())
            r=summarize(g,labels)
            folds.append(dict(person=who,train_trials=len(tr),trials=len(te),correct=r['correct'],accuracy=r['accuracy'],labels_present=te.label.nunique()))
            folder=args.out/who;folder.mkdir()
            g.to_csv(folder/'predictions.csv',index=False)
            tr[['person','file','label']].to_csv(folder/'training_trials.csv',index=False)
            pd.DataFrame(r['confusion_matrix'],index=labels,columns=labels).to_csv(folder/'confusion.csv')
        training=f[['person','file','label']].copy()
        training['prediction']=pipeline.predict(f[cols]);training['correct']=training.label.eq(training.prediction)
    predictions=pd.concat(predictions,ignore_index=True)
    assert len(predictions)==559 and predictions.file.nunique()==559
    predictions.to_csv(args.out/'loso_predictions.csv',index=False)
    pd.DataFrame(folds).to_csv(args.out/'fold_summary.csv',index=False)
    training.to_csv(args.out/'training_resubstitution_predictions.csv',index=False)
    complete={r['person'] for r in folds if r['labels_present']==3}
    report=dict(model_path=str(args.model),model_sha256=hashlib.sha256(before).hexdigest(),
        features_sha256=hashlib.sha256(raw).hexdigest(),labels=labels,config=saved['metadata']['config'],
        loso=summarize(predictions,labels),complete_class_people=summarize(predictions[predictions.person.isin(complete)],labels),
        participant_macro_accuracy=float(np.mean([r['accuracy'] for r in folds])),folds=folds,
        training_resubstitution=summarize(training,labels),untouched_external_accuracy=None,
        warnings=['LOSO refits 10 temporary models. It is an internal estimate, not a new independent test of the saved combined model.',
                  'Settings were previously chosen using part of these people; post-selection CV may be optimistic.',
                  'Training resubstitution measures seen data and is not generalization accuracy.',
                  'subject5 has only one sign; complete-class participant aggregate is supplied.',
                  'No settings changed, no candidate overwritten, no deployment.'])
    (args.out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert args.model.read_bytes()==before and args.features.read_bytes()==raw
    print(pd.DataFrame(folds).to_string(index=False))
    print(json.dumps({k:report[k] for k in ['loso','complete_class_people','training_resubstitution','participant_macro_accuracy']},ensure_ascii=True,indent=2))
    print('PASS: participant-separated 10 folds, 559 unique test trials, fixed configuration, training-only scaling, original candidate unchanged.')


if __name__=='__main__':main()
