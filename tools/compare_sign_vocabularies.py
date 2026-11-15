"""Exploratory comparison of original 3, +denial 4, and five possible 5-sign sets.

Every vocabulary is fitted as ONE classifier. Label choice from these results is
post-selection internal exploration, not fresh external validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

try:
    from . import evaluate_three_sign_candidates as tri
    from . import evaluate_six_sign_knn as base
except ImportError:
    import evaluate_three_sign_candidates as tri
    import evaluate_six_sign_knn as base

ORIGINAL = ['교통사고','삼키다','상처']
FOUR = ORIGINAL + ['아니다']
EXTRAS = ['괜찮다','무엇','마시다','화장실','싫다']


def vocabularies():
    return [dict(name='original3',labels=ORIGINAL),dict(name='plus_denial4',labels=FOUR)] + [
        dict(name=f'five_{i+1}',labels=FOUR+[sign],extra=sign) for i,sign in enumerate(EXTRAS)]


def verified_features(root):
    prior = root/'2026-11-15_direct_sensor_consistency'
    meta = json.loads((prior/'summary.json').read_text(encoding='utf-8'))
    for filename,sha in meta['input_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != sha:
            raise ValueError('Cached source differs from raw-verified sensor audit')
    if meta['raw_files_verified'] != 1603:
        raise ValueError('Unexpected raw verification snapshot')
    paths = [root/'2026-11-08_knn_development_holdout/trial_features.csv',
             root/'2026-11-15_six_sign_nested_loso/trial_features.csv']
    frame = pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    frame['person'] = frame.person.map(base.person)
    frame['date_kst'] = frame.date_kst.fillna('')
    verified = pd.read_csv(prior/'회차별_FSR상태.csv')
    if set(frame.file) != set(verified.file) or frame.file.duplicated().any():
        raise ValueError('Cached/raw-verified cohorts differ')
    frame = frame.drop(columns=['sha256','source_commit'],errors='ignore').merge(
        verified[['file','person','label','trial_id','sha256','source_commit']], on='file',
        validate='one_to_one',suffixes=('','_verified'))
    for name in ['person','label','trial_id']:
        if not frame[name].eq(frame[name+'_verified']).all():
            raise ValueError('Raw/cached identity mismatch')
    if len(frame) != 1603 or frame.label.nunique() != 9 or frame.trial_id.duplicated().any():
        raise ValueError('Unexpected pooled measurements')
    if not np.isfinite(frame[base.ALL_COLS].to_numpy(float)).all():
        raise ValueError('Nonfinite cached sensor feature')
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [*paths,prior/'summary.json',prior/'회차별_FSR상태.csv']}
    return frame,hashes


def choose_five(rows):
    # Protect against a high overall score hiding a failed fifth class.
    return sorted(rows,key=lambda r:(-r['minimum_sign_recall'],-r['macro_recall'],
                                      -r['accuracy'],r['name']))[0]


def evaluate_one(frame,spec,out,configs):
    labels=spec['labels']
    data=frame[frame.label.isin(labels)].copy()
    tri.validate(data,labels)
    out.mkdir()
    data[tri.IDENTITY].to_csv(out/'cohort_manifest.csv',index=False,encoding='utf-8-sig')
    data.groupby(['person','label']).size().unstack(fill_value=0).reindex(columns=labels).to_csv(
        out/'participant_sign_counts.csv',encoding='utf-8-sig')
    pred,folds,searches=tri.evaluate(data,configs,labels=labels)
    results={method:tri.summary(pred[pred.method.eq(method)],labels) for method in base.METHODS}
    pred.to_csv(out/'outer_predictions.csv',index=False,encoding='utf-8-sig')
    folds.to_csv(out/'outer_folds.csv',index=False,encoding='utf-8-sig')
    base.dump_json(out/'inner_searches.json',searches)
    base.dump_json(out/'evaluation.json',results)
    chosen,board=tri.select_inner(data,configs,labels=labels)
    cfg=chosen['nested_selection'];cols=base.FEATURES[cfg['sensors']]
    model=base.make_model(cfg).fit(data[cols],data.label)
    metadata=dict(vocabulary=spec,config={k:cfg[k] for k in ['config_id','sensors','k','p','weights']},
                  people=base.people_order(data.person),trials=len(data),exploratory=True,
                  external_accuracy=None,deployed=False,date_label_confounding_unresolved=True)
    path=out/'candidate_model.joblib'
    joblib.dump(dict(pipeline=model,feature_columns=cols,metadata=metadata),path)
    restored=joblib.load(path)
    assert restored['pipeline'].steps[-1][1].n_samples_fit_==len(data)
    np.testing.assert_array_equal(model.predict(data.iloc[:5][cols]),restored['pipeline'].predict(data.iloc[:5][cols]))
    base.dump_json(out/'candidate_manifest.json',metadata)
    base.dump_json(out/'full_data_selection.json',dict(winner=cfg,candidates=board))
    primary=results['nested_selection']
    return dict(**spec,trials=len(data),people=data.person.nunique(),
                correct=primary['correct'],accuracy=primary['accuracy'],
                macro_recall=float(np.mean([r['recall'] for r in primary['per_sign']])),
                minimum_sign_recall=min(r['recall'] for r in primary['per_sign']),
                macro_f1=primary['macro_f1'],per_sign=primary['per_sign'],
                participant_mean_accuracy=primary['participant_mean_accuracy'],
                complete_people=primary['complete_people_subset'],
                candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Use a fresh comparison directory')
    frame,hashes=verified_features(args.results_root)
    args.out.mkdir(parents=True)
    specs=vocabularies();configs=base.configurations()
    base.dump_json(args.out/'comparison_plan.json',dict(
        vocabularies=specs,settings=configs,primary_method='nested_selection',
        outer='Every date/trial of one person excluded; all partial-person available trials retained',
        inner='Equal-person macro recall on training people with all current labels; partial people still fit',
        five_selection='Highest worst-class recall, then macro recall, pooled accuracy, stable name order',
        label_selection_uses_outer_results=True,exploratory_not_external=True,
        source_raw_files_previously_verified=1603,input_sha256=hashes,
        warning='Repeated exploration of same people and date/label confounding remain; no confirmatory claim.'))
    for p in [Path(__file__),Path(tri.__file__),Path(base.__file__)]:
        shutil.copy2(p,args.out/p.name)
    rows=[]
    with threadpool_limits(limits=1):
        for spec in specs:
            row=evaluate_one(frame,spec,args.out/spec['name'],configs)
            rows.append(row)
            base.dump_json(args.out/'completed_results.json',rows)
            print(spec['name'],'/'.join(spec['labels']),f"{row['correct']}/{row['trials']} = {row['accuracy']*100:.2f}%",
                  f"weakest sign {row['minimum_sign_recall']*100:.2f}%",flush=True)
    # Test the original four on EXACTLY the same 739 captures in all added-sign experiments.
    reference=pd.read_csv(args.out/'plus_denial4/outer_predictions.csv')
    reference=reference[reference.method.eq('nested_selection')]
    reference_ids=set(reference.file)
    assert len(reference_ids)==739
    for row in rows:
        if 'extra' not in row:
            continue
        p=pd.read_csv(args.out/row['name']/'outer_predictions.csv')
        p=p[p.method.eq('nested_selection')]
        core=p[p.label.isin(FOUR)]
        assert set(core.file)==reference_ids and not core.file.duplicated().any()
        row['base_four_correct']=int(core.label.eq(core.prediction).sum())
        row['base_four_trials']=len(core)
        row['base_four_accuracy']=float(core.label.eq(core.prediction).mean())
        row['extra_recall']=next(s['recall'] for s in row['per_sign'] if s['label']==row['extra'])
    winner=choose_five([r for r in rows if 'extra' in r])
    base.dump_json(args.out/'summary.json',dict(results=rows,chosen_five=winner,
                                             chosen_using_same_internal_predictions=True,
                                             independent_final_accuracy=None))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['per_sign','complete_people','labels']}
                  | dict(labels='/'.join(r['labels'])) for r in rows]).to_csv(
        args.out/'vocabulary_comparison.csv',index=False,encoding='utf-8-sig')
    for filename,sha in hashes.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==sha
    base.dump_json(args.out/'verification.json',dict(vocabularies=len(rows),
        all_available_measurements_kept=True,source_inputs_unchanged=True,
        base_four_same_test_trials=739,models_deployed=False,old_models_written=False,
        chosen_five=winner['name'],choice_is_exploratory=True))
    print('Chosen five for review:',winner['name'],'/'.join(winner['labels']),flush=True)


if __name__ == '__main__':
    main()
