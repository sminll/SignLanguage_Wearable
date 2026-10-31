"""Inspect and safely correct a single session in sensor_dataset_v2.csv.

STOP the Flask receiver and MATLAB sender before running this script.
Usage:
    python correct_session.py list
    python correct_session.py edit SESSION_ID subject4 새수어이름
"""
import csv
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from prepare_sensor_v2 import MASTER,FIELDS


def load():
    if not MASTER.exists(): raise ValueError('통합 CSV가 없습니다.')
    with MASTER.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=FIELDS: raise ValueError('CSV 열 구조가 예상과 다릅니다.')
        return list(reader)


def main():
    if len(sys.argv)==2 and sys.argv[1]=='list':
        counts=Counter((r['session_id'],r['subject_id'],r['label'],r['hand']) for r in load())
        print('session_id | subject_id | label | hand | count')
        for parts,n in counts.items(): print(' | '.join([*parts,str(n)]))
        return
    if len(sys.argv)!=5 or sys.argv[1]!='edit':
        print('사용법: python correct_session.py list')
        print('       python correct_session.py edit SESSION_ID subject4 수정수어명')
        return
    _,_,session,subject,label=sys.argv
    if subject not in {f'subject{i}' for i in range(1,11)}: raise ValueError('subject1~subject10만 허용')
    label=label.strip()
    if not label or len(label)>80 or label[-1:].isdigit():
        raise ValueError('수어명은 비어 있으면 안 되고 회차 숫자로 끝나면 안 됩니다.')
    if not re.fullmatch(r'[A-Za-z0-9_-]{6,80}',session): raise ValueError('session_id 형식 오류')
    rows=load()
    matches=[r for r in rows if r['session_id']==session]
    if not matches: raise ValueError('해당 session_id 데이터가 없습니다.')
    if len({(r['subject_id'],r['label']) for r in matches})!=1:
        raise ValueError('세션에 서로 다른 Subject/수어 이름이 섞여 있습니다. 자동 정정하지 않습니다.')
    if any(r['source']!='REAL' for r in matches): raise ValueError('TEST 행은 정정할 수 없습니다.')
    old_subject=matches[0]['subject_id'];old_label=matches[0]['label']
    print(f'대상: {session} / {len(matches)}행')
    print(f'{old_subject} / {old_label} -> {subject} / {label}')
    if input('정말 수정하시겠습니까? YES 입력: ').strip()!='YES':
        print('취소했습니다.');return
    for row in matches:
        m=re.fullmatch(re.escape(old_label)+r'(\d+)',row['trial_id'])
        if not m: raise ValueError(f"잘못된 회차명 {row['trial_id']}; 수정 중단")
        row['subject_id']=subject
        row['label']=label
        row['trial_id']=f'{label}{m.group(1)}'
    backupdir=MASTER.parent/'backups'
    backupdir.mkdir(exist_ok=True)
    backup=backupdir/f'{MASTER.stem}_before_edit_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}.csv'
    shutil.copy2(MASTER,backup)
    temp=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8-sig',newline='',dir=MASTER.parent,prefix='.edit_',suffix='.csv',delete=False) as f:
            temp=Path(f.name)
            w=csv.DictWriter(f,fieldnames=FIELDS)
            w.writeheader();w.writerows(rows)
        os.replace(temp,MASTER)
    finally:
        if temp and temp.exists():temp.unlink()
    print(f'정정 완료: {len(matches)}행 / 자동 백업: {backup}')
    print('주의: MATLAB 원본 CSV는 그대로입니다. KNN에는 새 통합 CSV만 사용하세요.')


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,csv.Error) as e:
        print(f'오류: {e}');sys.exit(1)
