"""Nested 5-person-group folds: KNN/SVM and completed-window temporal features.

This is an exploratory historical-data study, never a deployed model update.
Outer and inner fold counts are fixed, not optimized on their accuracy.
"""
from collections import Counter
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import sensor_preprocessing as prep


SEED = 20261116
RECIPES = {'raw': prep.Recipe(), 'denoise': prep.Recipe(denoise=True),
           'denoise_normalize': prep.Recipe(denoise=True, normalize=True)}
ALIASES = {'괜찮아': '괜찮다', '아니다 (아니, 부인(하다))': '아니다',
           '무엇(어디)': '무엇', '싫다(거부)': '싫다', '싫다(거절)': '싫다'}


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_masks(times):
    """Three equal-duration bins on each hand's own elapsed device clock.

    Does not imply clock synchronization or dynamic gesture alignment.
    """
    times = np.asarray(times, float)
    if times.ndim != 1 or len(times) < 6 or not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Finite strictly increasing device time required')
    phase = np.minimum(((times-times[0])/(times[-1]-times[0])*3).astype(int), 2)
    masks = [phase == i for i in range(3)]
    if any(mask.sum() < 2 for mask in masks):
        raise ValueError('At least two samples per temporal bin required')
    return masks


def trial_features(hands, recipe):
    whole, _ = prep.extract_features(hands, recipe)
    values = dict(whole)
    for hand in ['left', 'right']:
        capture = hands[hand]
        processed, _ = prep.process_hand(capture, recipe)
        for phase, mask in enumerate(phase_masks(capture.times_ms)):
            for i, sensor in enumerate(prep.SENSORS):
                for stat in prep.STATS:
                    values[f'phase{phase}_{hand}_{sensor}_{stat}'] = float(getattr(np, stat)(processed[mask, i]))
    return values


def build_tables(repo, ref, cohort):
    """Only the frozen eligible cohort; verify raw Git blob hashes and identity."""
    repo = Path(repo)
    if cohort.empty or cohort.file.duplicated().any() or cohort.duplicated(['person','session_id','trial_id']).any():
        raise ValueError('Unique paired trials required')
    commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','--verify','--end-of-options',ref+'^{commit}'],text=True).strip()
    archive = subprocess.check_output(['git','-C',str(repo),'archive',commit,'sensor-data/training'])
    wanted = set(cohort.file)
    blobs = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if member.name in wanted:
                blobs[member.name] = tar.extractfile(member).read()
    if set(blobs) != wanted:
        raise ValueError('Frozen cohort file missing from input commit')
    records = {name: [] for name in RECIPES}
    for row in cohort.itertuples():
        blob = blobs[row.file]
        if hashlib.sha256(blob).hexdigest() != row.sha256:
            raise ValueError('Raw file hash changed: '+row.file)
        hands = prep.read_capture(pd.read_csv(io.BytesIO(blob)), account=Path(row.file).parent.name,
                                  source_path=row.file, source_sha256=row.sha256)
        if set(hands) != {'left','right'}:
            raise ValueError('Missing hand')
        for hand in hands.values():
            if (hand.person,hand.session_id,hand.trial_id,ALIASES.get(hand.label,hand.label)) != (
                    row.person,row.session_id,row.trial_id,row.label):
                raise ValueError('Frozen cohort identity or label differs')
        for name, recipe in RECIPES.items():
            records[name].append(trial_features(hands, recipe))
    return commit, {name:pd.DataFrame(rows) for name,rows in records.items()}


def matrices(tables):
    result, columns = {}, {}
    for table, layouts, sensors in itertools.product(tables, ['whole','temporal'], ['imu','fsr','all']):
        if table == 'denoise_normalize' and sensors == 'imu':
            continue  # FSR normalization cannot change an IMU-only matrix.
        cols = [c for c in tables[table] if (layouts == 'temporal' or not c.startswith('phase'))
                and (sensors == 'all' or ('_fsr' in c) == (sensors == 'fsr'))]
        name = f'{table}__{layouts}__{sensors}'
        result[name] = tables[table][cols].to_numpy(float)
        if not np.isfinite(result[name]).all():
            raise ValueError('Nonfinite features')
        columns[name] = cols
    return result, columns


def configurations(representations):
    result = []
    for representation in representations:
        layout = representation.split('__')[1]
        for k,p,weights in itertools.product([1,3,5,7,11,15,21,31],[1,2],['uniform','distance']):
            result.append(dict(id=len(result), representation=representation, layout=layout,
                               algorithm='knn', k=k, p=p, weights=weights))
        for C,gamma in itertools.product([.1,1.,10.,100.],['scale',.001,.01,.1]):
            result.append(dict(id=len(result), representation=representation, layout=layout,
                               algorithm='svm', C=C, gamma=gamma))
    return result


def make_model(config):
    classifier = (KNeighborsClassifier(n_neighbors=config['k'],p=config['p'],weights=config['weights'],
                                      algorithm='brute',n_jobs=1) if config['algorithm'] == 'knn'
                  else SVC(C=config['C'],gamma=config['gamma'],kernel='rbf',cache_size=256))
    return Pipeline([('scaler',StandardScaler()),('classifier',classifier)])


def splits(cohort, count, seed):
    if not 2 <= count < cohort.person.nunique():
        raise ValueError('Require fewer folds than people; no LOSO fallback')
    result, seen = [], []
    for fit,val in GroupKFold(n_splits=count,shuffle=True,random_state=seed).split(cohort,cohort.label,cohort.person):
        train, test = cohort.iloc[fit],cohort.iloc[val]
        if set(train.person)&set(test.person) or set(train.file)&set(test.file):
            raise ValueError('Group leakage')
        if not set(test.label) <= set(train.label):
            raise ValueError('Test sign absent from training')
        result.append((fit,val))
        seen.extend(val)
    if sorted(seen) != list(range(len(cohort))):
        raise ValueError('Trials omitted or repeated')
    return result


def scoring_weights(cohort):
    people = cohort.person.nunique()
    classes = cohort.groupby('person').label.nunique()
    sizes = cohort.groupby(['person','label']).size()
    w = np.array([1/(people*classes[r.person]*sizes[r.person,r.label]) for r in cohort.itertuples()])
    np.testing.assert_allclose(w.sum(),1)
    return w


def knn_predictions(train, y, test, configs, verify=False):
    """Share neighbor queries, retaining sklearn tie handling at k boundaries."""
    classes, encoded = np.unique(y,return_inverse=True)
    result = {}
    query = min(max(c['k'] for c in configs)+1,len(train))
    if max(c['k'] for c in configs)>len(train):
        raise ValueError('Not enough training samples')
    for p in [1,2]:
        model = KNeighborsClassifier(p=p,algorithm='brute',n_jobs=1).fit(train,y)
        distance,index = model.kneighbors(test,n_neighbors=query)
        labels = encoded[index]
        with np.errstate(divide='ignore'):
            weight = 1/distance
        zeros = distance==0
        weight[zeros.any(axis=1)] = zeros[zeros.any(axis=1)]
        votes = {'uniform':np.stack([np.cumsum(labels==i,axis=1) for i in range(len(classes))],axis=2),
                 'distance':np.stack([np.cumsum(weight*(labels==i),axis=1) for i in range(len(classes))],axis=2)}
        for cfg in configs:
            if cfg['p'] != p:
                continue
            k = cfg['k']
            pred = classes[votes[cfg['weights']][:,k-1].argmax(axis=1)].copy()
            tied = np.isclose(distance[:,k-1],distance[:,k],rtol=1e-12,atol=1e-12) if k<query else np.zeros(len(test),bool)
            model.set_params(n_neighbors=k,weights=cfg['weights'])
            if tied.any():
                pred[tied] = model.predict(test[tied])
            if verify:
                np.testing.assert_array_equal(pred,model.predict(test))
            result[cfg['id']] = pred
    return result


def search(cohort, X, configs, count=3, seed=SEED, progress=None, verify=False):
    y = cohort.label.to_numpy()
    weights = scoring_weights(cohort)
    scores, counts = np.zeros(len(configs)),np.zeros(len(configs),int)
    logs = []
    for fold,(fit,val) in enumerate(splits(cohort,count,seed),1):
        logs.append(dict(fold=fold,train_indices=[int(i) for i in cohort.index[fit]],
            validation_indices=[int(i) for i in cohort.index[val]],
            train_people=sorted(set(cohort.iloc[fit].person)),validation_people=sorted(set(cohort.iloc[val].person))))
        for name in X:
            entries = [c for c in configs if c['representation']==name]
            if not entries:
                continue
            xt, xv = X[name][cohort.index[fit]], X[name][cohort.index[val]]
            scaler = StandardScaler().fit(xt)
            np.testing.assert_allclose(scaler.mean_,xt.mean(axis=0))
            xt,xv = scaler.transform(xt),scaler.transform(xv)
            knns = [c for c in entries if c['algorithm']=='knn']
            predictions = knn_predictions(xt,y[fit],xv,knns,verify=verify and fold==1) if knns else {}
            for cfg in entries:
                pred = predictions[cfg['id']] if cfg['algorithm']=='knn' else SVC(
                    C=cfg['C'],gamma=cfg['gamma'],kernel='rbf',cache_size=256).fit(xt,y[fit]).predict(xv)
                correct = pred==y[val]
                scores[cfg['id']] += float(weights[val]@correct)
                counts[cfg['id']] += int(correct.sum())
        if progress:
            progress(fold)
    board = [dict(**cfg,selection_score=float(scores[cfg['id']]),correct=int(counts[cfg['id']]),trials=len(cohort)) for cfg in configs]
    return board,logs


def choices(board):
    ranked = sorted(board,key=lambda c:(-c['selection_score'],c['id']))
    selected = {}
    for layout in ['whole','temporal']:
        for algorithm in ['knn','svm']:
            selected[f'{algorithm}_{layout}'] = next(c for c in ranked if c['layout']==layout and c['algorithm']==algorithm)
    selected['selected_whole'] = next(c for c in ranked if c['layout']=='whole')
    selected['selected_any'] = ranked[0]
    return selected


def evaluate(cohort, X, configs, out, progress=print):
    out = Path(out)
    outer = splits(cohort,5,SEED)
    dump(out/'evaluation_plan.json',dict(outer_folds=5,inner_folds=3,seed=SEED,configs=configs,
        criterion='Equal-person mean class recall, then predetermined configuration ID',
        temporal='Whole-trial stats plus three equal-duration local-device-time bins',
        folds_optimized=False,loso_used=False,independent_new_data=False,models_replaced=False))
    predictions, logs, picks = [], [], []
    for fold,(fit,test) in enumerate(outer,1):
        train,evaluation = cohort.iloc[fit],cohort.iloc[test]
        progress(f'Outer {fold}/5: training {len(train)} trials, testing {len(evaluation)} trials',flush=True)
        board, inner = search(train,X,configs,progress=lambda f:progress(f'  inner {f}/3 complete',flush=True),verify=fold==1)
        selected = choices(board)
        pd.DataFrame(board).to_csv(out/f'outer_{fold}_search.csv',index=False,encoding='utf-8-sig')
        dump(out/f'outer_{fold}_selected_before_test.json',selected)
        logs.append(dict(outer_fold=fold,train_indices=[int(i) for i in train.index],
            test_indices=[int(i) for i in evaluation.index],train_people=sorted(set(train.person)),
            test_people=sorted(set(evaluation.person)),inner=inner))
        for name,cfg in selected.items():
            model = make_model(cfg).fit(X[cfg['representation']][fit],train.label)
            np.testing.assert_allclose(model.named_steps['scaler'].mean_,X[cfg['representation']][fit].mean(axis=0))
            pred = model.predict(X[cfg['representation']][test])
            rows = evaluation.copy()
            rows['prediction'],rows['correct'] = pred,pred==evaluation.label.to_numpy()
            rows['method'],rows['outer_fold'] = name,fold
            rows['training_people'] = '|'.join(sorted(set(train.person)))
            rows['config_id'],rows['representation'],rows['algorithm'] = cfg['id'],cfg['representation'],cfg['algorithm']
            predictions.append(rows)
            picks.append(dict(outer_fold=fold,method=name,**cfg))
        # Fixed baseline on identical folds: the existing raw IMU KNN settings.
        cfg = dict(algorithm='knn',representation='raw__whole__imu',k=3,p=1,weights='uniform')
        model = make_model(cfg).fit(X[cfg['representation']][fit],train.label)
        pred = model.predict(X[cfg['representation']][test])
        rows = evaluation.copy()
        rows['prediction'],rows['correct'] = pred,pred==evaluation.label.to_numpy()
        rows['method'],rows['outer_fold'],rows['config_id'] = 'fixed_existing_imu_knn',fold,-1
        rows['representation'],rows['algorithm'] = cfg['representation'],cfg['algorithm']
        rows['training_people'] = '|'.join(sorted(set(train.person)))
        predictions.append(rows)
        dump(out/'split_manifest.json',logs)
    data = pd.concat(predictions,ignore_index=True)
    for method,group in data.groupby('method'):
        if len(group)!=len(cohort) or group.file.duplicated().any() or set(group.file)!=set(cohort.file):
            raise ValueError('Incomplete or duplicated outer predictions')
    if any(row.person in row.training_people.split('|') for row in data.itertuples()):
        raise ValueError('Test person present in training')
    if not np.array_equal(data.correct.to_numpy(),data.label.eq(data.prediction).to_numpy()):
        raise ValueError('Wrong correctness counts')
    summary = []
    labels = sorted(set(cohort.label))
    for method,group in data.groupby('method'):
        person_class = group.groupby(['person','label']).correct.mean().groupby('person').mean()
        summary.append(dict(method=method,trials=len(group),correct=int(group.correct.sum()),
            accuracy=float(group.correct.mean()),equal_person_class_recall=float(person_class.mean()),
            macro_f1=float(f1_score(group.label,group.prediction,average='macro')),
            outer_fold_accuracy_std=float(group.groupby('outer_fold').correct.mean().std()),
            confusion_matrix=confusion_matrix(group.label,group.prediction,labels=labels).tolist()))
    data.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(picks).to_csv(out/'selected_configs.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(summary).drop(columns='confusion_matrix').to_csv(out/'summary.csv',index=False,encoding='utf-8-sig')
    for key,name in [('label','per_sign.csv'),('person','per_person.csv'),('outer_fold','per_fold.csv')]:
        data.groupby(['method',key]).correct.agg(['size','sum','mean']).reset_index().to_csv(out/name,index=False,encoding='utf-8-sig')
    dump(out/'metrics.json',dict(labels=labels,results=summary))
    dump(out/'verification.json',dict(outer_folds=5,inner_folds=3,outer_predictions_per_method=len(cohort),
        each_trial_predicted_once=True,people_disjoint=True,scaler_training_only=True,
        selection_saved_before_test=True,missing_hands_zero_filled=False,models_saved=False,
        external_accuracy=None,known_historical_data=True))
    return summary
