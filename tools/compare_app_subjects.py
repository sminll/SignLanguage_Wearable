"""Compare immutable app subjects; descriptive signal review, no training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quality-dir',type=Path,required=True)
    args=p.parse_args(); out=args.quality_dir
    result_path=out/'subject_signal_comparison.csv'
    if result_path.exists(): raise SystemExit('Comparison already exists')
    meta=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    archive=subprocess.check_output(['git','archive',meta['commit'],'sensor-data/training'])
    dfs=[]
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for f in meta['manifest']:
            blob=tar.extractfile(f['path']).read()
            assert hashlib.sha256(blob).hexdigest()==f['sha256']
            d=pd.read_csv(io.BytesIO(blob));d['file']=f['path'];dfs.append(d)
    raw=pd.concat(dfs,ignore_index=True)
    stats=[]
    for (subject,label,hand),g in raw.groupby(['subject_id','label','hand']):
        fsr=[f'fsr{i}' for i in range(1,7)]
        corr=g[fsr].corr().to_numpy()[np.triu_indices(6,1)]
        pertrial=[]
        for _,b in g.groupby('trial_id'):
            pertrial.append(np.nanmedian(b[fsr].corr().to_numpy()[np.triu_indices(6,1)]))
        row=dict(subject_id=subject,label=label,hand=hand,rows=len(g),trials=g.trial_id.nunique(),
            fsr5_zero_fraction=float((g.fsr5==0).mean()),fsr5_max=float(g.fsr5.max()),
            pooled_fsr_correlation_min=float(np.nanmin(corr)),
            pooled_fsr_correlation_median=float(np.nanmedian(corr)),
            within_trial_fsr_correlation_median=float(np.nanmedian(pertrial)),
            within_trial_accel_std_mean=float(g.groupby('trial_id')[['ax','ay','az']].std(ddof=0).to_numpy().mean()),
            within_trial_gyro_std_mean=float(g.groupby('trial_id')[['gx','gy','gz']].std(ddof=0).to_numpy().mean()))
        for sensor in fsr+['ax','ay','az']:
            row[sensor+'_mean']=float(g[sensor].mean())
        stats.append(row)
    table=pd.DataFrame(stats);table.to_csv(result_path,index=False)
    quality=pd.read_csv(out/'trial_quality.csv')
    quality.groupby(['subject_id','label']).agg(files=('file','count'),rows=('rows','sum'),
        basic_pass=('basic_capture_pass','sum')).to_csv(out/'subject_quality_summary.csv')
    print(table.to_string(index=False))
    print(quality.groupby(['subject_id','label']).agg(files=('file','count'),rows=('rows','sum'),basic_pass=('basic_capture_pass','sum')).to_string())
    print('Verified all source blob hashes; no models fitted or source data written.')


if __name__=='__main__': main()
