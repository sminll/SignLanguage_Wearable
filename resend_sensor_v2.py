"""Retry a new-format MATLAB CSV safely using server-side idempotency."""
import csv
import json
import re
import argparse
from pathlib import Path
from tkinter import Tk,filedialog
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

ENDPOINT='http://127.0.0.1:5000/sensor'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',default=ENDPOINT)
    args=parser.parse_args()
    gui=Tk();gui.withdraw()
    try:
        name=filedialog.askopenfilename(title='MATLAB 새 수집 CSV 선택',filetypes=[('CSV','*.csv')])
    finally:gui.destroy()
    if not name:print('취소했습니다.');return
    with Path(name).open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader)
    required=['source','device_id','seq','time_ms','subject_id','session_id']
    if not rows or any(k not in rows[0] for k in required):
        raise ValueError('Subject/Session이 기록된 최신 MATLAB CSV만 지원합니다.')
    meta={(r['subject_id'],r['session_id'],r['device_id']) for r in rows}
    if len(meta) != 1:
        raise ValueError('한 CSV 안에 Subject/Session/손이 여러 개입니다. 전송을 중단합니다.')
    extra = ['gesture_type', 'host_elapsed_ms', 'trial_elapsed_ms']
    if any(k in rows[0] for k in extra):
        if not all(k in rows[0] for k in extra):
            raise ValueError('양손 CSV 메타데이터가 일부 누락됐습니다.')
        if not args.endpoint.endswith('/sensor'):
            raise ValueError('전송 주소는 /sensor로 끝나야 합니다.')
        with urlopen(args.endpoint[:-len('/sensor')]+'/health',timeout=5) as response:
            health=json.load(response)
        if 3 not in health.get('schema_versions',[]):
            raise ValueError('서버가 v3 수신을 지원하지 않습니다. CSV는 보존됩니다.')
    print('전송 대상:',meta,' / 행:',len(rows))
    if input('진행하려면 YES: ').strip()!='YES':return
    saved=duplicates=0
    for i,r in enumerate(rows,1):
        label=re.sub(r'[0-9]+$', '', r['source'])
        # Match MATLAB conversion, including optional IMU null.
        data={'source':'REAL','hand':('left' if r['device_id']=='L' else 'right'),
              'trial_id':r['source'],'label':label,'time_ms':float(r['time_ms']),
              'subject_id':r['subject_id'],'session_id':r['session_id'],'seq':int(float(r['seq']))}
        for j in range(1,7):data[f'fsr{j}']=float(r[f'fsr{j}_raw'])
        for a,b in zip(['ax_g','ay_g','az_g','gx_dps','gy_dps','gz_dps'],['ax','ay','az','gx','gy','gz']):
            v=r[a]
            data[b]=None if v in ('','NaN','nan','null') else float(v)
        extra = ['gesture_type', 'host_elapsed_ms', 'trial_elapsed_ms']
        if any(k in r for k in extra):
            if not all(k in r for k in extra):
                raise ValueError('양손 CSV 메타데이터가 일부 누락됐습니다.')
            data.update(gesture_type=r['gesture_type'],
                        host_elapsed_ms=float(r['host_elapsed_ms']),
                        trial_elapsed_ms=float(r['trial_elapsed_ms']))
        req=Request(args.endpoint,data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode('utf-8'),headers={'Content-Type':'application/json'},method='POST')
        try:
            with urlopen(req,timeout=10) as res: out=json.load(res)
        except (HTTPError,URLError,TimeoutError) as e:
            print(f'{i}번째 전송 중단: {e} / 이전 저장분은 중복 방지됨');return
        if out['status']=='duplicate':duplicates+=1
        else:saved+=1
        if i%100==0:print(i,'건 확인')
    print('완료! 신규 저장:',saved,'/ 중복 건너뜀:',duplicates)


if __name__=='__main__':main()
