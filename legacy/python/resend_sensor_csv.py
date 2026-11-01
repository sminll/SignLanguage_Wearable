"""Send a saved 17-column MATLAB sensor CSV to the existing local Flask receiver.

Usage: place beside receive_sensor.py, run `python resend_sensor_csv.py`.
Does not edit existing project code or source CSV. Skips records already in
`data/received_sensor.csv` as checked before posting.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from tkinter import Tk, filedialog
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parents[2]
DEST = BASE / 'data' / 'received_sensor.csv'
URL = 'http://127.0.0.1:5000/sensor'
REQUIRED = ['source', 'device_id', 'time_ms'] + [f'fsr{i}_raw' for i in range(1, 7)] + [
    'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps']


def normalize_number(value):
    value = str(value).strip()
    if value == '' or value.lower() in {'nan', 'inf', '-inf', 'infinity', '-infinity'}:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def payload_from_raw(r):
    trial = r['source'].strip()
    hand_code = r['device_id'].strip().upper()
    if hand_code not in {'L', 'R'}:
        raise ValueError(f'손 구분 오류: {hand_code!r}')
    label = re.sub(r'\d+$', '', trial)
    if not label:
        raise ValueError(f'동작명을 확인하세요: {trial!r}')
    p = {'source': 'REAL', 'hand': 'left' if hand_code == 'L' else 'right',
         'trial_id': trial, 'label': label, 'time_ms': int(float(r['time_ms']))}
    for i in range(1, 7):
        p[f'fsr{i}'] = normalize_number(r[f'fsr{i}_raw'])
        if p[f'fsr{i}'] is None:
            raise ValueError(f'FSR{i} 센서값 없음: {trial}')
    for axis, raw in [('ax', 'ax_g'), ('ay', 'ay_g'), ('az', 'az_g'),
                      ('gx', 'gx_dps'), ('gy', 'gy_dps'), ('gz', 'gz_dps')]:
        p[axis] = normalize_number(r[raw])
    return p


def sample_key(p):
    fields = ['source', 'hand', 'trial_id', 'label', 'time_ms'] + [f'fsr{i}' for i in range(1,7)] + ['ax','ay','az','gx','gy','gz']
    def clean(v):
        if v is None or v == '':
            return None
        try:
            n = float(v)
            return round(n, 8) if math.isfinite(n) else None
        except (TypeError, ValueError):
            return str(v).strip()
    return tuple(clean(p.get(k)) for k in fields)


def main():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    selected = filedialog.askopenfilename(
        title='MATLAB에서 저장한 아프다 CSV 선택',
        filetypes=[('CSV files', '*.csv')])
    root.destroy()
    if not selected:
        print('파일을 선택하지 않아 종료했습니다.')
        return
    source_path = Path(selected).resolve()
    if source_path == DEST.resolve():
        raise ValueError('received_sensor.csv가 아니라 MATLAB sensor_data 폴더의 원본 CSV를 선택하세요.')
    with source_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = [k for k in REQUIRED if k not in (reader.fieldnames or [])]
        if missing:
            raise ValueError('MATLAB CSV 열 누락: ' + ', '.join(missing))
        records = [payload_from_raw(r) for r in reader]
    if not records:
        print('CSV에 전송할 데이터가 없습니다.')
        return
    print(f'원본 CSV: {source_path.name} / {len(records)}개')
    print(f'저장 위치: {DEST}')
    existing = set()
    if DEST.is_file():
        with DEST.open('r', encoding='utf-8-sig', newline='') as f:
            for r in csv.DictReader(f):
                existing.add(sample_key(r))
    skipped = 0
    sent = 0
    for idx, p in enumerate(records, start=1):
        key = sample_key(p)
        if key in existing:
            skipped += 1
            continue
        data = json.dumps(p, ensure_ascii=False, allow_nan=False).encode('utf-8')
        req = Request(URL, data=data, headers={'Content-Type': 'application/json'}, method='POST')
        try:
            with urlopen(req, timeout=10) as response:
                if not (200 <= response.status < 300):
                    raise RuntimeError(f'HTTP {response.status}')
        except Exception as e:
            print(f'전송 중단: 원본 {idx}번째 행, 확인된 전송 {sent}개. 오류: {e}')
            print('서버에 저장됐지만 응답이 유실됐을 수도 있으니 확인 후 재실행하세요.')
            return
        sent += 1
        existing.add(key)
        if sent % 100 == 0:
            print(f'{sent}개 전송 확인')
    print(f'완료! 새로 전송 {sent}개 / 기존 중복 건너뜀 {skipped}개')


if __name__ == '__main__':
    main()
