"""Receive MATLAB FSR/IMU data with explicit subject/session identification.

Requires `python prepare_sensor_v2.py` once before first run. Older
received_sensor*.csv files are never modified by this receiver.
"""
from flask import Flask, request
from datetime import datetime
from pathlib import Path
from threading import Lock
from collections import Counter
import csv
import math
import re
import argparse

from prepare_sensor_v2 import MASTER, FIELDS

app = Flask(__name__)
save_lock = Lock()
SUBJECTS = {f'subject{i}' for i in range(1, 11)}
SESSION_RE = re.compile(r'^[A-Za-z0-9_-]{6,80}$')
SENSOR_KEYS = [*(f'fsr{i}' for i in range(1, 7)), 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
_seen = {}
_count = 0


def signature(row):
    """Compare a repeated sample with the data that was originally saved."""
    values = ['subject_id', 'label', 'trial_id', 'time_ms'] + SENSOR_KEYS
    return tuple(
        '' if row.get(k) in (None, '') else
        float(row[k]) if k == 'time_ms' or k in SENSOR_KEYS else str(row[k])
        for k in values
    )


def read_existing():
    if not MASTER.exists():
        raise RuntimeError('새 데이터 파일이 없습니다. 먼저 python prepare_sensor_v2.py 를 실행하세요.')
    keys = {}
    with MASTER.open(encoding='utf-8-sig',newline='') as f:
        dr=csv.DictReader(f)
        if dr.fieldnames!=FIELDS:
            raise RuntimeError('새 CSV의 헤더가 예상과 다릅니다. 서버 시작 중단.')
        for line in dr:
            if line['sample_seq']:
                keys[(line['session_id'],line['hand'],line['sample_seq'])] = signature(line)
    return keys


def validate(data):
    if not isinstance(data,dict):
        raise ValueError('JSON 객체가 필요합니다.')
    if data.get('source')!='REAL':
        raise ValueError('source=REAL 데이터만 수신합니다.')
    subject=data.get('subject_id')
    session=data.get('session_id')
    label=data.get('label')
    trial=data.get('trial_id')
    hand=data.get('hand')
    if subject not in SUBJECTS:
        raise ValueError('subject_id는 subject1~subject10 중 하나여야 합니다.')
    if not isinstance(session,str) or not SESSION_RE.fullmatch(session) or session.startswith('legacy_'):
        raise ValueError('session_id가 없거나 유효하지 않습니다.')
    if hand not in ('left','right'):
        raise ValueError('hand는 left 또는 right 여야 합니다.')
    if not isinstance(label,str) or not label.strip() or label != label.strip() or len(label)>80:
        raise ValueError('label이 비어 있거나 공백/길이가 잘못됐습니다.')
    if not isinstance(trial,str) or not re.fullmatch(re.escape(label)+r'([1-9]|1[0-9]|20)',trial):
        raise ValueError('trial_id는 수어 이름 + 1~20회차여야 합니다.')
    seq=data.get('seq')
    if isinstance(seq,bool) or not isinstance(seq,(int,float)) or not math.isfinite(seq) or seq<1 or int(seq)!=seq:
        raise ValueError('seq는 양의 정수여야 합니다.')
    time_ms=data.get('time_ms')
    if isinstance(time_ms,bool) or not isinstance(time_ms,(int,float)) or not math.isfinite(time_ms) or time_ms<0:
        raise ValueError('time_ms가 유효하지 않습니다.')
    row={k:'' for k in FIELDS}
    row.update(time=datetime.now().isoformat(timespec='microseconds'),source='REAL',hand=hand,
               trial_id=trial,label=label,time_ms=int(time_ms),subject_id=subject,
               session_id=session,sample_seq=int(seq))
    for name in SENSOR_KEYS:
        value=data.get(name)
        if value is None and name in ('ax','ay','az','gx','gy','gz'):
            row[name]=''  # IMU null remains missing, never fake zero.
            continue
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise ValueError(f'{name} 센서값이 유효하지 않습니다.')
        if name.startswith('fsr') and (value<0 or value>4095):
            raise ValueError(f'{name} FSR 값은 0~4095 사이여야 합니다.')
        row[name]=value
    return row


@app.route('/health',methods=['GET'])
def health():
    return {'status':'ok','dataset':MASTER.name},200


@app.route('/sensor',methods=['POST'])
def receive_sensor():
    global _count
    try:
        row=validate(request.get_json(silent=True))
    except ValueError as e:
        return {'error':str(e)},400
    key=(row['session_id'],row['hand'],str(row['sample_seq']))
    with save_lock:
        if key in _seen:
            if _seen[key] != signature(row):
                return {'error':'기존과 다른 데이터가 동일 session_id/seq로 들어왔습니다. 전송을 중단하고 확인하세요.'},409
            return {'status':'duplicate','session_id':row['session_id']},200
        with MASTER.open('a',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            writer.writerow(row)
            f.flush()
        _seen[key] = signature(row)
        _count+=1
        if _count%100==0:
            print(f'센서 데이터 {_count}행 추가 수신 (최근: {row["subject_id"]} / {row["label"]})',flush=True)
    return {'status':'saved','session_id':row['session_id']},200


def start_receiver(host='127.0.0.1'):
    global _seen
    _seen=read_existing()
    print(f'수신 대기: {MASTER} / 기존 고유 샘플키 {len(_seen)}개',flush=True)
    print(f'연결 확인: http://{host}:5000/health',flush=True)
    app.run(host=host,port=5000,debug=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='MATLAB 센서 데이터 수신 서버')
    parser.add_argument('--host', default='127.0.0.1',
                        help='수신 주소. 원격 수신 시 이 PC의 Tailscale IPv4 주소를 입력하세요.')
    start_receiver(parser.parse_args().host)
