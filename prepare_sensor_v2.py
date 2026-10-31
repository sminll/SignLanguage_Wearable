"""Safely create a new 21-column master dataset from two historical CSVs.

Leaves both historical CSV files byte-for-byte untouched. Run once, with
receive_sensor.py stopped. Safe to re-run: refuses to replace an existing master.
"""
import csv
import os
import tempfile
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / 'data'
RAW = DATA / 'received_sensor.csv'
TAGGED = DATA / 'received_sensor_with_subject.csv'
MASTER = DATA / 'sensor_dataset_v2.csv'
BASE_FIELDS = ['time', 'source', 'hand', 'trial_id', 'label', 'time_ms'] + [
    *(f'fsr{i}' for i in range(1, 7)), 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
FIELDS = BASE_FIELDS + ['subject_id', 'session_id', 'sample_seq']


def load(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or any(row.get(None) for row in rows):
        raise ValueError(f'잘못된 CSV 구조: {path}')
    return fields, rows


def migrate():
    raw_fields, raw = load(RAW)
    tagged_fields, tagged = load(TAGGED)
    if MASTER.exists():
        master_fields, current = load(MASTER)
        if master_fields != FIELDS or len(current) < len(raw):
            raise ValueError('새 통합 파일이 과거 원본보다 짧거나 구조가 다릅니다. 덮어쓰지 않았습니다.')
        for idx, (a, b) in enumerate(zip(raw, current), 1):
            if any(a[k] != b[k] for k in BASE_FIELDS):
                raise ValueError(f'{idx}번째 원본 행이 통합 파일과 다릅니다. 덮어쓰지 않았습니다.')
        if len(tagged) != len(raw):
            raise ValueError('기존 Subject CSV와 원본 CSV 행 수가 다릅니다. 수동 확인이 필요합니다.')
        for idx,(a,b) in enumerate(zip(tagged,current),1):
            if a['subject_id'] != b['subject_id']:
                raise ValueError(f'{idx}번째 Subject가 다릅니다. 수동 확인이 필요합니다.')
        print(f'기존 {len(raw):,}행이 새 통합 파일과 일치합니다. 이미 준비됐으므로 덮어쓰지 않았습니다.')
        return
    if raw_fields != BASE_FIELDS or tagged_fields != BASE_FIELDS + ['subject_id']:
        raise ValueError('원본 두 CSV의 열 구조가 예상과 다릅니다. 변경하지 않았습니다.')
    if len(raw) != len(tagged):
        raise ValueError('CSV 행 수가 달라 변환을 중단합니다.')
    result=[]
    last_key=None
    session_num=0
    for idx, (a,b) in enumerate(zip(raw,tagged),1):
        if any(a[k]!=b[k] for k in BASE_FIELDS):
            raise ValueError(f'{idx}번째 행의 내용이 달라 변환을 중단합니다.')
        subject=b['subject_id'].strip()
        if a['source']=='TEST':
            if subject:
                raise ValueError(f'{idx}번째 TEST 행에 subject가 지정돼 있습니다.')
            key=('TEST', idx)
        else:
            if subject not in {f'subject{i}' for i in range(1,11)}:
                raise ValueError(f'{idx}번째 실제 데이터의 Subject가 비어 있거나 잘못됐습니다.')
            key=(subject,a['label'],a['hand'])
        if key!=last_key:
            session_num+=1
            last_key=key
        result.append({**a,'subject_id':subject,'session_id':f'legacy_{session_num:04d}','sample_seq':''})
    DATA.mkdir(parents=True,exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8-sig',newline='',dir=DATA,prefix='.v2_',suffix='.tmp',delete=False) as f:
            temporary=Path(f.name)
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(result)
        # O_EXCL to avoid overwriting (no concurrent migration expected).
        with MASTER.open('xb') as dest,temporary.open('rb') as src:
            shutil_copy(src,dest)
    finally:
        if temporary and temporary.exists(): temporary.unlink()
    counts=Counter(row['subject_id'] or '(TEST)' for row in result)
    print(f'새 통합 데이터 생성 완료: {MASTER.name} / {len(result):,}행')
    print('기존 원본 2개 파일은 수정하지 않았습니다.')
    for subject,n in sorted(counts.items()): print(f'  {subject}: {n:,}행')
    print(f'기존 측정 묶음: {session_num}개 (TEST 포함)')


def shutil_copy(src,dst):
    import shutil
    shutil.copyfileobj(src,dst)


if __name__=='__main__':
    migrate()
