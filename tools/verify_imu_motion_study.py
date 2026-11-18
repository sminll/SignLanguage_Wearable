"""Independently recount stored predictions, splits and model-selection choices."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--results',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--test-log',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh verification output required')
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    root=args.results
    plan=read(root/'plan.json')
    cohort=pd.read_csv(root/'cohort_manifest.csv')
    old_cohort=pd.read_csv(args.previous/'cohort_manifest.csv')
    pd.testing.assert_frame_equal(cohort,old_cohort)
    logs=read(root/'split_manifest.json')
    old_logs=read(args.previous/'split_manifest.json')
    assert logs==old_logs and len(logs)==5
    universe=set(range(len(cohort)))
    tested=[]
    for row in logs:
        train,test=set(row['train_indices']),set(row['test_indices'])
        assert not train&test and train|test==universe
        assert len(row['test_people'])==2 and len(row['train_people'])==8
        assert not set(row['test_people'])&set(row['train_people'])
        assert len(row['inner'])==3
        seen=[]
        for inner in row['inner']:
            fit,val=set(inner['train_indices']),set(inner['validation_indices'])
            assert not fit&val and fit|val==train and not (fit|val)&test
            assert not set(inner['train_people'])&set(inner['validation_people'])
            seen.extend(val)
        assert sorted(seen)==sorted(train)
        tested.extend(test)
        board=pd.read_csv(root/f"outer_{row['outer_fold']}_search.csv")
        assert len(board)==len(plan['configs'])==320 and board.id.nunique()==320
        selections=read(root/f"outer_{row['outer_fold']}_selected_before_test.json")
        for name,cfg in selections.items():
            available=board if name=='train_selected_features' else board[board.layout.eq(cfg['layout'])]
            # Some pandas float parsers differ in their last bit. Treat scores
            # within 1e-14 as equal only for this audit, preserving planned ID.
            best=available.selection_score.max()
            chosen=available[available.id.eq(cfg['id'])].iloc[0]
            assert abs(chosen.selection_score-best)<1e-14
    assert sorted(tested)==sorted(universe)
    pred=pd.read_csv(root/'predictions.csv')
    summary=pd.read_csv(root/'summary.csv')
    metrics=read(root/'metrics.json')
    assert len(summary)==10 and len(pred)==11700
    for r in summary.itertuples():
        part=pred[pred.method.eq(r.method)]
        assert len(part)==1170 and part.file.nunique()==1170 and set(part.file)==set(cohort.file)
        assert int(part.label.eq(part.prediction).sum())==r.correct
        assert np.array_equal(part.correct,part.label.eq(part.prediction))
        assert not any(x.person in x.training_people.split('|') for x in part.itertuples())
        original=next(x for x in metrics['results'] if x['method']==r.method)
        np.testing.assert_array_equal(confusion_matrix(part.label,part.prediction,labels=metrics['labels']),original['confusion_matrix'])
    old=pd.read_csv(args.previous/'predictions.csv')
    for current,previous in [('previous_svm_replay','svm_temporal'),('svm_original','svm_temporal'),
                             ('fixed_existing_knn','fixed_existing_imu_knn')]:
        a=pred[pred.method.eq(current)].set_index('file').prediction.sort_index()
        b=old[old.method.eq(previous)].set_index('file').prediction.sort_index()
        pd.testing.assert_series_equal(a,b)
    for name,digest in plan['input_hashes'].items():
        path=Path(name)
        if not path.is_absolute():
            path=args.repo/path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,str(path)
    log=args.test_log.read_text(encoding='utf-8')
    count=int(re.search(r'Ran (\d+) tests',log)[1])
    assert count==81 and re.search(r'^OK\s*$',log,re.M) and not re.search(r'^FAILED',log,re.M)
    result=dict(automated_tests_passed=count,outer_folds=5,inner_fold_count_per_outer=3,
        each_test_people=2,total_prediction_rows=len(pred),methods=10,
        exact_previous_cohort_and_splits=True,baseline_predictions_identical=True,
        choice_uses_inner_score=True,all_counts_recounted=True,all_confusions_recounted=True,
        input_and_experiment_code_hashes_match=True,existing_models_replaced=False,
        unseen_future_accuracy=None,known_historical_data=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
