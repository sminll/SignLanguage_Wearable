"""Rebuild derived reference tables from the published cohort and pinned Git CSVs.

Creates no model and runs no accuracy evaluation. Raw files are read-only.
"""
import argparse
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import group_model_study as study
import sensor_preprocessing as prep


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--ref',default='4d4d24a4d672c78d934bf120a59a193c2340637e')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a fresh directory; previous inputs are preserved')
    cohort=pd.read_csv(args.cohort)
    if len(cohort)!=1170 or cohort.person.nunique()!=10 or cohort.label.nunique()!=6:
        raise ValueError('Expected the published 1170-trial, ten-person six-sign cohort')
    commit,tables=study.build_tables(args.repo,args.ref,cohort)
    args.out.mkdir(parents=True)
    shutil.copyfile(args.cohort,args.out/'cohort_manifest.csv')
    for name in ['raw','denoise']:
        reference=pd.concat([cohort.reset_index(drop=True),tables[name][prep.FEATURES].reset_index(drop=True)],axis=1)
        reference.to_csv(args.out/(name+'_features.csv'),index=False,encoding='utf-8-sig')
    study.dump(args.out/'rebuild_manifest.json',dict(commit=commit,trials=len(cohort),participants=10,labels=6,
        cohort_sha256=study.digest(args.cohort),raw_blob_hashes_verified=True,
        code_hashes={p.name:study.digest(p) for p in [Path(__file__),Path(study.__file__),Path(prep.__file__)]},
        numpy=np.__version__,sklearn=sklearn.__version__,models_saved=False,accuracy_measured=False))
    print(f'Rebuilt raw/denoise reference tables: {len(cohort)} trials from {commit}')


if __name__=='__main__':
    main()
