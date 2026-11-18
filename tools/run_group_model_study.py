"""Run the fixed-cohort nested KNN/SVM temporal comparison in a new directory."""
import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import group_model_study as study
import sensor_preprocessing as prep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--reference-raw',type=Path,required=True)
    parser.add_argument('--reference-denoise',type=Path,required=True)
    parser.add_argument('--ref',default='4d4d24a4d672c78d934bf120a59a193c2340637e')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Fresh output directory required')
    cohort = pd.read_csv(args.cohort)
    if len(cohort)!=1170 or cohort.person.nunique()!=10 or cohort.label.nunique()!=6:
        raise ValueError('Expected pinned 10-person six-sign cohort of 1170 paired trials')
    commit,tables = study.build_tables(args.repo,args.ref,cohort)
    for name,path in [('raw',args.reference_raw),('denoise',args.reference_denoise)]:
        cached = pd.read_csv(path)
        pd.testing.assert_frame_equal(cached[cohort.columns],cohort)
        np.testing.assert_allclose(tables[name][prep.FEATURES],cached[prep.FEATURES],rtol=1e-12,atol=1e-10)
    X,columns = study.matrices(tables)
    configs = study.configurations(X)
    args.out.mkdir(parents=True)
    cohort.to_csv(args.out/'cohort_manifest.csv',index=False,encoding='utf-8-sig')
    paths = [args.cohort,args.reference_raw,args.reference_denoise,Path(__file__),Path(study.__file__),Path(prep.__file__)]
    study.dump(args.out/'source_manifest.json',dict(commit=commit,input_hashes={str(p):study.digest(p) for p in paths},
        policy=asdict(prep.Policy()),numpy=np.__version__,sklearn=sklearn.__version__,
        whole_features_match_previous=True,feature_columns=columns,configurations=len(configs)))
    for name,table in tables.items():
        table.to_csv(args.out/(name+'_features.csv'),index=False,encoding='utf-8-sig')
    print(f'Frozen input: {len(cohort)} paired trials, 10 people, 6 signs; {len(configs)} configurations',flush=True)
    with threadpool_limits(limits=1):
        summary = study.evaluate(cohort,X,configs,args.out)
    print(pd.DataFrame(summary)[['method','correct','trials','accuracy']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
