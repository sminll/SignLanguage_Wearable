"""Nested fixed-split SVM ablations for magnitude and motion-window features."""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix,f1_score
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import group_model_study as base
import imu_motion_features as motion
import sensor_preprocessing as prep


def configs_for(X):
    result=[]
    for name in X:
        variant=name.split('__')[1]
        for C,gamma in itertools.product([.1,1.,10.,100.],['scale',.001,.01,.1]):
            result.append(dict(id=len(result),representation=name,layout=variant,algorithm='svm',C=C,gamma=gamma))
    return result


def rank(board):
    return sorted(board,key=lambda cfg:(-cfg['selection_score'],cfg['id']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    previous=args.previous
    source=json.loads((previous/'source_manifest.json').read_text(encoding='utf-8'))
    cohort=pd.read_csv(previous/'cohort_manifest.csv')
    if len(cohort)!=1170 or cohort.person.nunique()!=10 or cohort.label.nunique()!=6:
        raise ValueError('Expected same six-sign cohort of 1170 paired trials')
    reference={name:pd.read_csv(previous/(name+'_features.csv')) for name in base.RECIPES}
    commit,tables,alignment=motion.build_tables(args.repo,source['commit'],cohort,reference)
    X,columns=motion.matrices(tables)
    configs=configs_for(X)
    old_splits=json.loads((previous/'split_manifest.json').read_text(encoding='utf-8'))
    old_choices=pd.read_csv(previous/'selected_configs.csv')
    old_predictions=pd.read_csv(previous/'predictions.csv')
    outer=base.splits(cohort,5,base.SEED)
    for (fit,test),old in zip(outer,old_splits):
        assert list(fit)==old['train_indices'] and list(test)==old['test_indices']
    args.out.mkdir(parents=True)
    paths=[previous/p for p in ['cohort_manifest.csv','source_manifest.json','split_manifest.json',
        'selected_configs.csv','predictions.csv',*[n+'_features.csv' for n in base.RECIPES]]]
    paths += [Path(__file__),Path(base.__file__),Path(motion.__file__),Path(prep.__file__)]
    base.dump(args.out/'plan.json',dict(commit=commit,outer_folds=5,inner_folds=3,seed=base.SEED,
        comparison='Same historical cohort and exact previous outer/inner splits',
        primary='train_selected_features; select features/settings using only inner CV',
        selection='Equal-person mean class recall, tie by predetermined ID',
        variants=list(motion.VARIANTS),configs=configs,motion_policy=asdict(motion.MotionPolicy()),
        input_hashes={str(p):base.digest(p) for p in paths},features=columns,
        numpy=np.__version__,sklearn=sklearn.__version__,models_replaced=False,
        unseen_future_accuracy=None,rest_calibration_applied=False,loso_used=False))
    cohort.to_csv(args.out/'cohort_manifest.csv',index=False,encoding='utf-8-sig')
    alignment.to_csv(args.out/'alignment_audit.csv',index=False,encoding='utf-8-sig')
    for (recipe,variant),table in tables.items():
        table.to_csv(args.out/f'features_{recipe}_{variant}.csv',index=False,encoding='utf-8-sig')
    print(f'Input: {len(cohort)} trials, 10 people, 6 signs; {len(configs)} SVM configurations',flush=True)
    predictions,selected_rows,logs=[],[],[]
    def predict(method,cfg,train,test,fold):
        fit,val=train.index.to_numpy(),test.index.to_numpy()
        model=base.make_model(cfg).fit(X[cfg['representation']][fit],train.label)
        np.testing.assert_allclose(model.named_steps['scaler'].mean_,X[cfg['representation']][fit].mean(axis=0))
        pred=model.predict(X[cfg['representation']][val])
        rows=test.copy()
        rows['prediction'],rows['correct']=pred,pred==test.label.to_numpy()
        rows['method'],rows['outer_fold']=method,fold
        rows['training_people']='|'.join(sorted(set(train.person)))
        rows['representation']=cfg['representation']
        predictions.append(rows)
        return rows
    with threadpool_limits(limits=1):
        for fold,(fit,val) in enumerate(outer,1):
            train,test=cohort.iloc[fit],cohort.iloc[val]
            print(f'Outer {fold}/5: {len(train)} train, {len(test)} test',flush=True)
            board,inner=base.search(train,X,configs,progress=lambda f:print(f'  inner {f}/3 done',flush=True))
            assert inner==old_splits[fold-1]['inner']
            ranked=rank(board)
            selected={f'svm_{v}':next(c for c in ranked if c['layout']==v) for v in motion.VARIANTS}
            selected['train_selected_features']=ranked[0]
            pd.DataFrame(board).to_csv(args.out/f'outer_{fold}_search.csv',index=False,encoding='utf-8-sig')
            base.dump(args.out/f'outer_{fold}_selected_before_test.json',selected)
            for name,cfg in selected.items():
                selected_rows.append(dict(method=name,outer_fold=fold,**cfg))
                rows=predict(name,cfg,train,test,fold)
                if name=='svm_original':
                    prior=old_predictions[old_predictions.method.eq('svm_temporal') & old_predictions.outer_fold.eq(fold)]
                    pd.testing.assert_frame_equal(rows[['file','prediction']].reset_index(drop=True),prior[['file','prediction']].reset_index(drop=True))
            old=old_choices[old_choices.method.eq('svm_temporal') & old_choices.outer_fold.eq(fold)].iloc[0]
            old_recipe,_,old_sensor=old.representation.split('__')
            gamma=old.gamma if old.gamma=='scale' else float(old.gamma)
            # Same old settings and sensors isolate feature changes from retuning.
            for variant in ['magnitude','aligned','magnitude_aligned']:
                cfg=dict(algorithm='svm',representation=f'{old_recipe}__{variant}__{old_sensor}',C=float(old.C),gamma=gamma)
                predict('fixed_settings_'+variant,cfg,train,test,fold)
            original=old_recipe+'__original__'+old_sensor
            cfg=dict(algorithm='svm',representation=original,C=float(old.C),gamma=gamma)
            rows=predict('previous_svm_replay',cfg,train,test,fold)
            prior=old_predictions[old_predictions.method.eq('svm_temporal') & old_predictions.outer_fold.eq(fold)]
            pd.testing.assert_frame_equal(rows[['file','prediction']].reset_index(drop=True),prior[['file','prediction']].reset_index(drop=True))
            knn=dict(algorithm='knn',representation='raw__original__imu',k=3,p=1,weights='uniform')
            # The old fixed KNN uses whole-window IMU, not temporal features.
            knn_name='fixed_whole_imu'
            X[knn_name]=reference['raw'][[c for c in prep.FEATURES if '_fsr' not in c]].to_numpy()
            knn['representation']=knn_name
            rows=predict('fixed_existing_knn',knn,train,test,fold)
            prior=old_predictions[old_predictions.method.eq('fixed_existing_imu_knn') & old_predictions.outer_fold.eq(fold)]
            pd.testing.assert_frame_equal(rows[['file','prediction']].reset_index(drop=True),prior[['file','prediction']].reset_index(drop=True))
            # Remove this control so it never participates in feature selection.
            del X[knn_name]
            logs.append(dict(outer_fold=fold,train_indices=[int(i) for i in fit],test_indices=[int(i) for i in val],
                train_people=sorted(set(train.person)),test_people=sorted(set(test.person)),inner=inner))
            base.dump(args.out/'split_manifest.json',logs)
    data=pd.concat(predictions,ignore_index=True)
    if any(r.person in r.training_people.split('|') for r in data.itertuples()):
        raise ValueError('Person leakage')
    labels=sorted(set(cohort.label))
    summary=[]
    for name,part in data.groupby('method'):
        assert len(part)==1170 and part.file.nunique()==1170 and set(part.file)==set(cohort.file)
        assert np.array_equal(part.correct,part.label.eq(part.prediction))
        per_person_class=part.groupby(['person','label']).correct.mean().groupby('person').mean()
        summary.append(dict(method=name,correct=int(part.correct.sum()),trials=len(part),accuracy=float(part.correct.mean()),
            equal_person_class_recall=float(per_person_class.mean()),macro_f1=float(f1_score(part.label,part.prediction,average='macro')),
            confusion_matrix=confusion_matrix(part.label,part.prediction,labels=labels).tolist()))
    data.to_csv(args.out/'predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(selected_rows).to_csv(args.out/'selected_configs.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).drop(columns='confusion_matrix').to_csv(args.out/'summary.csv',index=False,encoding='utf-8-sig')
    for key,name in [('label','per_sign.csv'),('person','per_person.csv'),('outer_fold','per_fold.csv')]:
        data.groupby(['method',key]).correct.agg(['size','sum','mean']).reset_index().to_csv(args.out/name,index=False,encoding='utf-8-sig')
    base.dump(args.out/'metrics.json',dict(labels=labels,results=summary))
    base.dump(args.out/'verification.json',dict(cohort_unchanged=True,baseline_features_match=True,
        previous_svm_predictions_match=True,previous_knn_predictions_match=True,
        exact_previous_splits=True,people_disjoint=True,scaler_training_only=True,
        selection_saved_before_test=True,each_trial_once_per_method=True,
        training_future_accuracy=None,models_saved=False))
    print(pd.DataFrame(summary)[['method','correct','trials','accuracy']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
